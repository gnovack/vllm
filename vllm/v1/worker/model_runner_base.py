from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List
import numpy as np
import torch
import torch.nn as nn

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.config import VllmConfig
from vllm.inputs import INPUT_REGISTRY
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sampling_params import SamplingType
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType, cdiv, is_pin_memory_available
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.engine.mm_input_mapper import MMInputMapperClient
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput

class ModelRunnerBase(ABC):
    """
    Model runner interface that abstracts a particular hardware and/or type of
    model.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]
    
        self.is_multimodal_model = model_config.is_multimodal_model
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Model-related.
        self.num_attn_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_query_heads = model_config.get_num_attention_heads(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()
        self.hidden_size = model_config.get_hidden_size()

        # Multi-modal data support
        self.input_registry = INPUT_REGISTRY
        self.mm_registry = MULTIMODAL_REGISTRY

        # NOTE: Initialized input mapper is only used for processing dummy
        # multimodal data into multimodal kwargs for GPU memory profiling.
        self.mm_input_mapper_profiling = MMInputMapperClient(self.model_config)
        self.mm_input_mapper_profiling.use_cache = False

        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=model_config,
            scheduler_config=scheduler_config,
        )
        self.max_num_encoder_input_tokens = encoder_compute_budget
        self.encoder_cache_size = encoder_cache_size

        # Lazy initialization
        # self.model: nn.Module  # Set after load_model
        self.kv_caches: List[torch.Tensor] = []
        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: Dict[str, Dict[int, torch.Tensor]] = {}

        # Request states.
        self.requests: Dict[str, CachedRequestState] = {}
        # Persistent batch.
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=model_config.get_vocab_size(),
        )

        # Persistent buffers for CUDA graphs.
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=self.device)
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.model_config.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = torch.zeros((3, self.max_num_tokens + 1),
                                               dtype=torch.int64,
                                               device=self.device)
            self.mrope_positions_cpu = torch.zeros(
                (3, self.max_num_tokens + 1),
                dtype=torch.int64,
                device="cpu",
                pin_memory=self.pin_memory)

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device)

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        self.arange_np = np.arange(max(self.max_num_reqs + 1,
                                       self.max_model_len,
                                       self.max_num_tokens),
                                   dtype=np.int32)
        # NOTE(woosuk): These tensors are "stateless", i.e., they are literally
        # a faster version of creating a new tensor every time. Thus, we should
        # not make any assumptions about the values in these tensors.
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.input_ids_np = self.input_ids_cpu.numpy()
        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.positions_np = self.positions_cpu.numpy()
        self.slot_mapping_cpu = torch.zeros(self.max_num_tokens,
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=self.pin_memory)
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()
        self.query_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=self.pin_memory)
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()
        self.seq_lens_cpu = torch.zeros(self.max_num_reqs,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=self.pin_memory)
        self.seq_lens_np = self.seq_lens_cpu.numpy()
    
    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        # Remove stopped requests from the cached states.
        # Keep the states of the preempted requests.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

        # Remove the requests from the persistent batch.
        stopped_req_ids = set().union(
            scheduler_output.preempted_req_ids,
            scheduler_output.finished_req_ids,
        )
        removed_req_indices: List[int] = []
        for req_id in stopped_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Update the states of the running requests.
        for req_data in scheduler_output.scheduled_running_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]
            req_index = self.input_batch.req_id_to_index[req_id]

            # Update the num_computed_tokens.
            req_state.num_computed_tokens = req_data.num_computed_tokens
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                req_data.num_computed_tokens)

            # Update the block table.
            num_new_blocks = len(req_data.new_block_ids)
            if num_new_blocks == 0:
                continue
            start_index = len(req_state.block_ids)
            req_state.block_ids.extend(req_data.new_block_ids)
            self.input_batch.block_table.append_row(req_index, start_index,
                                                    req_data.new_block_ids)

        req_ids_to_add: List[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt=new_req_data.prompt,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
            )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.model_config.uses_mrope:
                image_grid_thw = []
                video_grid_thw = []
                for mm_input in self.requests[req_id].mm_inputs:
                    if mm_input.get("image_grid_thw") is not None:
                        image_grid_thw.extend(
                            mm_input["image_grid_thw"].tolist())
                    if mm_input.get("video_grid_thw") is not None:
                        video_grid_thw.extend(
                            mm_input["video_grid_thw"].tolist())

                hf_config = self.model_config.hf_config

                self.requests[req_id].mrope_positions, \
                    self.requests[req_id].mrope_position_delta = \
                    MRotaryEmbedding.get_input_positions_tensor(
                        self.requests[req_id].prompt_token_ids,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        image_token_id=hf_config.image_token_id,
                        video_token_id=hf_config.video_token_id,
                        vision_start_token_id=hf_config.vision_start_token_id,
                        vision_end_token_id=hf_config.vision_end_token_id,
                        spatial_merge_size=hf_config.vision_config.
                        spatial_merge_size,
                    )

            req_ids_to_add.append(req_id)

        # Update the cached states of the resumed requests.
        for res_req_data in scheduler_output.scheduled_resumed_reqs:
            req_id = res_req_data.req_id
            req_state = self.requests[req_id]

            req_state.block_ids = res_req_data.block_ids
            req_state.num_computed_tokens = res_req_data.num_computed_tokens
            req_ids_to_add.append(req_id)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)
    
    def get_model(self) -> nn.Module:
        return self.model

    @abstractmethod
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        raise NotImplementedError()
    
    @abstractmethod
    def load_model(self) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def profile_run(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def capture_model(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        raise NotImplementedError()
    
    def get_kv_cache_spec(self) -> KVCacheSpec:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each 
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache 
            format. Layers that do not need KV cache are not included.
        """

        forward_ctx = self.vllm_config.compilation_config.static_forward_context
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: KVCacheSpec = {}
        for layer_name, attn_module in forward_ctx.items():
            # TODO: Support other attention modules, e.g., sliding window,
            # cross-attention, MLA.
            assert isinstance(attn_module, Attention)
            if attn_module.attn_type == AttentionType.DECODER:
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=attn_module.dtype,
                )
            elif attn_module.attn_type in (AttentionType.ENCODER,
                                           AttentionType.ENCODER_ONLY):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")

        return kv_cache_spec