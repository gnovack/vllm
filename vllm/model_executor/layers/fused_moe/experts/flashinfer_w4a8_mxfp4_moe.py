# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A8 MXFP4 MoE experts backed by the FlashInfer CuTe-DSL Hopper (SM90) kernel.

MXFP4 weights x FP8-e4m3 activations: the activation is BF16/FP16 on the way in
and the kernel quantizes it to FP8 internally (per-token), so from vLLM's side
this is a W4A16-style backend (no prepare-stage activation quant). The kernel
also fuses routing gather/scatter, SwiGLU, and the top-k reduction, so this is a
modular experts impl that does everything in ``apply()`` and finalizes with a
no-op reduce.

The kernel is vendored under ``..flashinfer_w4a8_mxfp4`` from FlashInfer PR #3516.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.flashinfer_w4a8_mxfp4 import (
    has_flashinfer_w4a8_mxfp4,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)
from vllm.platforms import current_platform


class FlashInferCuteDslW4A8Mxfp4Experts(mk.FusedMoEExpertsModular):
    """Modular MXFP4-weight / FP8-activation MoE experts (Hopper SM90).

    Wraps the vendored ``w4a8_mxfp4_moe`` which takes pre-routed
    ``(topk_ids, topk_weights)`` and returns the already-reduced output, so the
    modular finalize is a no-op.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        **kwargs,
    ):
        super().__init__(moe_config, quant_config)
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.hidden_dim = moe_config.hidden_dim
        self.hidden_dim_unpadded = (
            moe_config.hidden_dim_unpadded or moe_config.hidden_dim
        )
        self.local_num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability(90)
            and has_flashinfer_w4a8_mxfp4()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # MXFP4 weights; activation arrives unquantized (bf16) and the kernel
        # casts to FP8 internally, so there is no vLLM-side activation key.
        return (weight_key, activation_key) == (kMxfp4Static, None)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # Plain SwiGLU (silu(gate) * up); clamped-SwiGLU params are not wired up.
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        # No expert-parallel all-to-all path wired up yet; caller shards experts.
        return not moe_parallel_config.use_all2all_kernels

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        # The kernel quantizes the activation to FP8 itself.
        return True

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # The kernel already applies routing weights and reduces over top-k.
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Intermediates are managed by the flashinfer kernel.
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, self.hidden_dim_unpadded)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        from vllm.model_executor.layers.fused_moe.flashinfer_w4a8_mxfp4.w4a8_mxfp4_moe import (  # noqa: E501
            w4a8_mxfp4_moe,
        )

        assert self.w1_scale is not None and self.w2_scale is not None

        w4a8_mxfp4_moe(
            input=hidden_states,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            fc1_expert_weights=w1,
            fc2_expert_weights=w2,
            output_dtype=output.dtype,
            quant_scales=[self.w1_scale, self.w2_scale],
            output=output,
        )

        return output
