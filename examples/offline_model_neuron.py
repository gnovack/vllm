import os
import tempfile

from vllm import SamplingParams
from vllm.attention.backends.neuron_attn import NeuronAttentionBackend
# from vllm.config import VllmConfig
# from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment
)
from vllm.engine.arg_utils import EngineArgs
# from vllm.model_executor.layers.logits_processor import _prune_hidden_states
from vllm.model_executor.model_loader import get_model

import torch
# import torch_neuronx
# import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from vllm.model_executor.sampling_metadata import SamplingMetadata
# from vllm.neuron.compiler import neuron_argmax

# creates XLA hlo graphs for all the context length buckets.
os.environ['NEURON_CONTEXT_LENGTH_BUCKETS'] = "128,512,1024,2048"
# creates XLA hlo graphs for all the token gen buckets.
os.environ['NEURON_TOKEN_GEN_BUCKETS'] = "128,512,1024,2048"

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=1)

# Create an LLM.
config = EngineArgs(
    model="/root/workspace/gnovack/models/llama-3.2-1b-instruct",
    max_num_seqs=8,
    # The max_model_len and block_size arguments are required to be same as
    # max sequence length when targeting neuron device.
    # Currently, this is a known limitation in continuous batching support
    # in transformers-neuronx.
    # TODO(liangfu): Support paged-attention in transformers-neuronx.
    max_model_len=128,
    block_size=128,
    # The device can be automatically detected when AWS Neuron SDK is installed.
    # The device argument can be either unspecified for automated detection,
    # or explicitly assigned.
    device="neuron",
    tensor_parallel_size=1,
    disable_async_output_proc=True
)

temp_file = tempfile.mkstemp()[1]

init_distributed_environment(
    world_size=1,
    rank=0,
    local_rank=0,
    distributed_init_method=f"file://{temp_file}",
    backend="gloo",
)
ensure_model_parallel_initialized(
    1,
    1,
)

attn_backend = NeuronAttentionBackend
vllm_config = config.create_engine_config()
device = xm.xla_device()
model = get_model(vllm_config=vllm_config)
model = model.eval().to(device)
model.logits_processor.to(device)
num_layers = len(model.model.layers)

xm.wait_device_ops()

def forward(
        input_ids,
        positions,
        kv_caches,
        attn_metadata,
        intermediate_tensors,
        inputs_embeds,
        sampling_metadata
    ):
    hidden_states = model(
        input_ids,
        positions,
        kv_caches=kv_caches,
        attn_metadata=attn_metadata,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds
    )
    
    return hidden_states

compiled_model = torch.compile(forward,
    backend="openxla",
    fullgraph=True,
    dynamic=False
)

batch_size = 1
seq_len = 128

token_ids = torch.zeros((batch_size, seq_len),
                        dtype=torch.int32)
position_ids = torch.arange(0, 128, dtype=torch.int32).unsqueeze(0)
slot_mapping = torch.zeros((batch_size, seq_len),
                            dtype=torch.int64)
input_lens = torch.ones((batch_size, ),
                        dtype=torch.int32)

attn_metadata = attn_backend.make_metadata(
    num_prefills=batch_size,
    num_prefill_tokens=batch_size * seq_len,
    num_decode_tokens=0,
    slot_mapping=slot_mapping,
    multi_modal_placeholder_index_maps=None,
    block_tables=None,
    context_lens=None,
    effective_query_lens=None,
)

cache_shape = attn_backend.get_kv_cache_shape(
    num_blocks=10_000,
    block_size = 32,
    num_kv_heads=model.config.num_key_value_heads,
    head_size=model.config.head_dim
)

# Calculate the positions to sample from.
start_indicies = torch.arange(batch_size, dtype=torch.int32) * seq_len
logits_indices = start_indicies + input_lens - 1

sampling_metadata = SamplingMetadata(
    seq_groups=[],
    selected_token_indices=logits_indices.to(device),
    categorized_sample_indices={},
    num_prompts=attn_metadata.num_prefills,
)
kv_caches = [torch.zeros(cache_shape) for _ in range(num_layers)]

output = compiled_model(
    token_ids.to(device),
    position_ids.to(device),
    kv_caches=[x.to(device) for x in kv_caches],
    attn_metadata=attn_metadata,
    intermediate_tensors=None,
    inputs_embeds=None,
    sampling_metadata=sampling_metadata
)
print(output)