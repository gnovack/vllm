# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.platforms import current_platform


def _torch_topk_softplus_sqrt(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    e_score_correction_bias: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
):
    scores = F.softplus(gating_output.float()).sqrt()
    original_scores = scores
    if e_score_correction_bias is not None:
        scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
    else:
        scores_for_choice = scores

    if hash_indices_table is not None:
        assert input_ids is not None
        topk_ids = hash_indices_table[input_ids.long()]
    else:
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=True)[1]

    topk_weights = original_scores.gather(1, topk_ids.long())
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def test_sqrtsoftplus_bias_uses_deepseek_v4_routing_method():
    assert (
        get_routing_method_type(
            scoring_func="sqrtsoftplus",
            top_k=8,
            renormalize=True,
            num_expert_group=None,
            has_e_score_bias=True,
        )
        == RoutingMethodType.DeepseekV4
    )
    assert (
        get_routing_method_type(
            scoring_func="sqrtsoftplus",
            top_k=8,
            renormalize=False,
            num_expert_group=None,
            has_e_score_bias=True,
        )
        == RoutingMethodType.Unspecified
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="This test is skipped on non-CUDA platform.",
)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk_softplus_sqrt_nan_inf_clamp(
    bad_value: float,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    num_tokens = 4
    hidden_size = 1024
    num_experts = 256
    topk = 6
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    gating_output[1:, :] = bad_value

    topk_weights, topk_ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=None,
        topk=topk,
        renormalize=True,
    )

    topk_weights_ref, topk_ids_ref = _torch_topk_softplus_sqrt(
        gating_output=gating_output[:1],
        topk=topk,
        renormalize=True,
        routed_scaling_factor=1.0,
    )
    torch.testing.assert_close(topk_ids[:1], topk_ids_ref, atol=0, rtol=0)
    torch.testing.assert_close(topk_weights[:1], topk_weights_ref, atol=2e-2, rtol=1e-2)

    for row in range(1, num_tokens):
        row_ids = topk_ids[row]
        valid_ids = row_ids[(row_ids >= 0) & (row_ids < num_experts)]
        assert valid_ids.unique().numel() == valid_ids.numel(), (
            f"Row {row} has duplicate valid expert IDs {row_ids.tolist()} "
            f"(bad_value={bad_value})"
        )
        invalid_mask = row_ids == -1
        assert torch.isfinite(topk_weights[row]).all(), (
            f"Row {row} has non-finite weights {topk_weights[row].tolist()} "
            f"(bad_value={bad_value})"
        )
        assert (topk_weights[row][invalid_mask] == 0).all(), (
            f"Row {row} has non-zero sentinel weights "
            f"{topk_weights[row].tolist()} (bad_value={bad_value})"
        )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="This test is skipped on non-CUDA platform.",
)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk_softplus_sqrt_hash_nan_inf_clamp(
    bad_value: float,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    num_tokens = 4
    hidden_size = 1024
    num_experts = 256
    topk = 6
    vocab_size = 32
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    gating_output[1:, :] = bad_value
    hash_indices_table = torch.stack(
        [
            (torch.arange(topk, dtype=torch.int64) + token_id) % num_experts
            for token_id in range(vocab_size)
        ]
    ).to(device="cuda")
    input_ids = torch.arange(num_tokens, dtype=torch.int64, device="cuda")

    topk_weights, topk_ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=None,
        topk=topk,
        renormalize=True,
        indices_type=torch.int64,
        input_tokens=input_ids,
        hash_indices_table=hash_indices_table,
    )

    topk_weights_ref, topk_ids_ref = _torch_topk_softplus_sqrt(
        gating_output=gating_output[:1],
        topk=topk,
        renormalize=True,
        routed_scaling_factor=1.0,
        input_ids=input_ids[:1],
        hash_indices_table=hash_indices_table,
    )
    torch.testing.assert_close(
        topk_ids[:1], topk_ids_ref.to(topk_ids.dtype), atol=0, rtol=0
    )
    torch.testing.assert_close(topk_weights[:1], topk_weights_ref, atol=2e-2, rtol=1e-2)

    for row in range(1, num_tokens):
        assert torch.equal(topk_ids[row], torch.full_like(topk_ids[row], -1)), (
            f"Row {row} should contain only sentinel IDs (bad_value={bad_value})"
        )
        assert torch.isfinite(topk_weights[row]).all(), (
            f"Row {row} has non-finite weights {topk_weights[row].tolist()} "
            f"(bad_value={bad_value})"
        )
        assert (topk_weights[row] == 0).all(), (
            f"Row {row} has non-zero weights {topk_weights[row].tolist()} "
            f"(bad_value={bad_value})"
        )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="This test is skipped on non-CUDA platform.",
)
@pytest.mark.parametrize("num_tokens", [1, 33, 128])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("num_experts", [128, 256, 384, 512])
@pytest.mark.parametrize("topk", [6, 8, 16])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 1.5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk_softplus_sqrt(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    e_score_correction_bias = torch.randn(
        (num_experts,), dtype=torch.float32, device="cuda"
    )

    topk_weights_ref, topk_ids_ref = _torch_topk_softplus_sqrt(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
    )

    topk_weights, topk_ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=e_score_correction_bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )

    # Different kernels may return the topk experts in different orders when
    # scores tie; sort by expert id before comparing.
    sorted_ref_ids, idx_ref = topk_ids_ref.sort(dim=-1)
    sorted_ids, idx_ops = topk_ids.sort(dim=-1)
    torch.testing.assert_close(sorted_ref_ids, sorted_ids, atol=0, rtol=0)

    sorted_w_ref = topk_weights_ref.gather(1, idx_ref)
    sorted_w = topk_weights.gather(1, idx_ops)
    torch.testing.assert_close(sorted_w_ref, sorted_w, atol=2e-2, rtol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="This test is skipped on non-CUDA platform.",
)
@pytest.mark.parametrize("num_tokens", [1, 33, 128])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("num_experts", [256, 384, 512])
@pytest.mark.parametrize("topk", [6, 8, 16])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk_softplus_sqrt_hash(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    vocab_size = 1024
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    # Per-token fixed expert selection: for each vocab id pick `topk` distinct
    # experts.
    hash_indices_table = torch.stack(
        [torch.randperm(num_experts)[:topk] for _ in range(vocab_size)]
    ).to(device="cuda", dtype=torch.int32)
    input_ids = torch.randint(
        0, vocab_size, (num_tokens,), dtype=torch.int32, device="cuda"
    )

    topk_weights_ref, topk_ids_ref = _torch_topk_softplus_sqrt(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        input_ids=input_ids,
        hash_indices_table=hash_indices_table,
    )

    topk_weights, topk_ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=None,
        topk=topk,
        renormalize=renormalize,
        input_tokens=input_ids,
        hash_indices_table=hash_indices_table,
        routed_scaling_factor=routed_scaling_factor,
    )

    sorted_ref_ids, idx_ref = topk_ids_ref.sort(dim=-1)
    sorted_ids, idx_ops = topk_ids.sort(dim=-1)
    torch.testing.assert_close(sorted_ref_ids, sorted_ids, atol=0, rtol=0)

    sorted_w_ref = topk_weights_ref.gather(1, idx_ref)
    sorted_w = topk_weights.gather(1, idx_ops)
    torch.testing.assert_close(sorted_w_ref, sorted_w, atol=2e-2, rtol=1e-2)
