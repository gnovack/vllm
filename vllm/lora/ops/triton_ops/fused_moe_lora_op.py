# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import supports_pdl, supports_tma

_LORA_PTR_DICT: dict[tuple[int, ...], torch.tensor] = {}


def _get_ptr(lora_weights: list[torch.Tensor], device: torch.device):
    """
    `_LORA_PTR_DICT` collects the required information during `profile_run`,
    After this, it remains constant and subsequent usage is through LUT.
    Refer to:
    https://github.com/triton-lang/triton/blob/release/3.1.x/python/tutorials/08-grouped-gemm.py
    """
    key = tuple(lora_weight.data_ptr() for lora_weight in lora_weights)

    if (ptr_tensor := _LORA_PTR_DICT.get(key)) is not None:
        return ptr_tensor

    tensor_ptrs = []
    for lora_weight in lora_weights:
        tensor_ptrs.append(lora_weight.data_ptr())
    ptr_tensor = torch.tensor(tensor_ptrs, device=device, dtype=torch.uint64)

    _LORA_PTR_DICT[key] = ptr_tensor
    return _LORA_PTR_DICT.get(key)


def _set_triton_allocator(device: torch.device):
    def alloc_fn(size: int, alignment: int, stream: int | None):
        return torch.empty(size, device=device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)

@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, group_size_m, max_loras, num_tiles_per_lora):

    lora_idx = tile_id // num_tiles_per_lora
    lora_tile_id = tile_id % num_tiles_per_lora
    group_id = lora_tile_id // num_pid_in_group
    first_pid_m = group_id * group_size_m
    
    trimmed_group_size_m = min(num_pid_m - first_pid_m, group_size_m)
    pid_m = first_pid_m + (lora_tile_id % trimmed_group_size_m)

    pid_n = (lora_tile_id % num_pid_in_group) // trimmed_group_size_m
    return lora_idx, pid_m, pid_n

def triton_autotune_expand_configs():
    BLOCK_SIZE_M = [128]
    BLOCK_SIZE_N = [128]
    BLOCK_SIZE_K = [16]
    num_warps = [4]
    num_stages = [3]

    gpu_sms = 132
    NUM_SMS = [gpu_sms*10]
    GROUP_SIZE_M = [1]
    SPLIT_K = [1]

    # return cartesian product of all possible configs
    configs = []
    for bm in BLOCK_SIZE_M:
        for bn in BLOCK_SIZE_N:
            for bk in BLOCK_SIZE_K:
                for nw in num_warps:
                    for ns in num_stages:
                        for sms in NUM_SMS:
                            for gsm in GROUP_SIZE_M:
                                for sk in SPLIT_K:
                                    configs.append(triton.Config({
                                        'BLOCK_SIZE_M': bm,
                                        'BLOCK_SIZE_N': bn, 
                                        'BLOCK_SIZE_K': bk,
                                        'GROUP_SIZE_M': gsm,
                                        'SPLIT_K': sk,
                                        'NUM_SMS': sms,
                                    }, num_warps=nw, num_stages=ns, pre_hook=matmul_tma_set_block_size_hook))
    return configs
    


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    
    if "a_desc" in nargs and nargs["a_desc"] is not None:
        nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    
    if "b_desc" in nargs and nargs["b_desc"] is not None:
        nargs["b_desc"].block_shape = [1, 1, BLOCK_N, BLOCK_K]

# @triton.autotune(
#     configs=triton_autotune_expand_configs(),
#     key=["N", "K", "EM"],
#     cache_results=False,
# )
@triton.jit(
    do_not_specialize=[
        "num_valid_tokens",
        "EM",
        "stride_tl",
        "stride_el",
    ]
)
def _fused_moe_lora_kernel_persistent(
    # Input pointers
    a_ptr,
    a_desc,
    b_ptr,
    b_desc,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    lora_ids_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    # Other dims
    num_valid_tokens,
    num_experts,
    adapter_enabled,
    top_k,
    # Strides
    stride_am,
    stride_ak,
    stride_bl,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_el,
    stride_tl,
    # Meta
    ADD_INPUTS: tl.constexpr,
    max_loras: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    IS_PRIMARY: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr
):
    c_type = c_ptr.dtype.element_ty
    tile_id = tl.program_id(axis=0)
    num_blocks_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)
    num_tiles_per_lora = num_blocks_m * num_blocks_n
    num_tiles = max_loras * num_tiles_per_lora
    num_pid_in_group = GROUP_SIZE_M * num_blocks_n

    m_block = tl.arange(0, BLOCK_SIZE_M).to(tl.int32)
    n_block = tl.arange(0, BLOCK_SIZE_N).to(tl.int32)
    k_block = tl.arange(0, BLOCK_SIZE_K)
    
    # Load all lora ids at the start
    # lora_ids_array = tl.load(lora_ids_ptr + tl.arange(0, max_loras), cache_modifier='.ca')
    
    # while tile_id < num_tiles:
    for tile_id in tl.range(tile_id, num_tiles, NUM_SMS):

        lora_idx, tile_m_idx, tile_n_idx = _compute_pid(
            tile_id // SPLIT_K, num_pid_in_group, num_blocks_m, GROUP_SIZE_M, max_loras, num_tiles_per_lora,
        )
        lora_id = tl.load(lora_ids_ptr + lora_idx)
        
        if lora_id != -1:
            
            moe_enabled = tl.load(adapter_enabled + lora_id)
            
            if moe_enabled != 0:
                
                num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_id)

                m_start = tile_m_idx * BLOCK_SIZE_M
                n_start = tile_n_idx * BLOCK_SIZE_N
                splitk_start = tile_id % SPLIT_K

                if m_start < num_tokens_post_padded:
                    
                    # offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
                    # offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)

                    expert_index = lora_id * stride_el + tile_m_idx
                    expert_id = tl.load(expert_ids_ptr + expert_index, expert_index < max_loras * stride_el, -1)
                    
                    if expert_id != -1:
                        
                        offs_token_id = tile_m_idx * BLOCK_SIZE_M + m_block
                        token_index = stride_tl * lora_id + offs_token_id
                        offs_m = tl.load(
                            sorted_token_ids_ptr + token_index,
                            mask=token_index < max_loras * stride_tl,
                            other=num_valid_tokens,
                        )

                        offs_n = n_start + n_block
                        offs_k = splitk_start * BLOCK_SIZE_K + k_block

                        slice_id = 0
                        
                        if b_desc is None:
                            cur_b_ptr = tl.load(b_ptr + slice_id).to(tl.pointer_type(c_ptr.dtype.element_ty))
                            b_ptrs = (
                                cur_b_ptr
                                + lora_id * stride_bl
                                + expert_id * stride_be
                                + offs_k[:, None] * stride_bk
                                + offs_n[None, :] * stride_bn
                            )

                        if a_desc is not None:
                            offs_am = slice_id * tl.cdiv(EM, top_k) * top_k + m_start // top_k
                        else:
                            a_ptrs = a_ptr + (
                                offs_m[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
                            )
                        
                        if IS_PRIMARY:
                            # GDC launch dependents hints the runtime system to launch dependent kernels.
                            tl.extra.cuda.gdc_launch_dependents()
                        
                        # Create masks for bounds checking
                        mask_m = offs_m < num_valid_tokens
                        mask_n = offs_n < N

                        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                        for ki in tl.range(num_tiles_k):

                            # offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                            cur_k_offset = ki * (BLOCK_SIZE_K * SPLIT_K)

                            if b_desc is None or a_desc is None:
                                mask_k = offs_k < (K - cur_k_offset)

                            if b_desc is not None:
                                b = (
                                    b_desc.load([lora_id, expert_id, n_start, splitk_start * BLOCK_SIZE_K + cur_k_offset])
                                    .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K).T
                                )
                            else:
                                mask_b = mask_n[None, :] & mask_k[:, None]
                                b = tl.load(b_ptrs, mask=mask_b, other=0.0)
                                b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

                            if not IS_PRIMARY:
                                tl.extra.cuda.gdc_wait()
                            if a_desc is not None:
                                a = a_desc.load([offs_am, splitk_start * BLOCK_SIZE_K + cur_k_offset])
                            else:
                                mask_a = mask_m[:, None] & mask_k[None, :]
                                a = tl.load(a_ptrs, mask=mask_a, other=0.0)
                                a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak

                            accumulator += tl.dot(a, b)
                        
                        # tile_id_c += NUM_SMS
                        # _, tile_m_idx, tile_n_idx = _compute_pid(
                        #     tile_id_c, num_pid_in_group, num_blocks_m, GROUP_SIZE_M, max_loras, num_tiles_per_lora,
                        # )

                        # offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                        # offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

                        # mask_m = offs_m < num_valid_tokens
                        # mask_n = offs_n < N
                        mask_c = mask_m[:, None] & mask_n[None, :]

                        if MUL_ROUTED_WEIGHT:
                            moe_weight = tl.load(topk_weights_ptr + offs_m, mask=mask_m, other=0.0)
                            accumulator = accumulator * moe_weight[:, None]

                        c = accumulator.to(c_type)
                        # Store output (C) with bounds checking
                        if a_desc is not None:
                            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
                        else:
                            c_ptrs = c_ptr + offs_token_id[:, None] * stride_cm + offs_n[None, :] * stride_cn

                        if SPLIT_K > 1 or ADD_INPUTS:
                            tl.atomic_add(c_ptrs, c, mask=mask_c, sem="relaxed")
                        else:
                            tl.store(c_ptrs, c, mask=mask_c)

        #     tile_id += NUM_SMS
        # else:
        #     tile_id += num_tiles_per_lora


@triton.jit(
    do_not_specialize=[
        "num_valid_tokens",
        "EM",
        "stride_tl",
        "stride_el",
        "slice_a_size",
        "slice_c_size",
    ]
)
def _fused_moe_lora_kernel(
    a_ptr,
    a_desc,
    b_ptr,
    b_desc,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    num_experts,
    lora_ids,
    adapter_enabled,
    max_loras,  # <<< PR2: rename, used for masks when grid axis-2 != max_loras
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_bl,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_tl,
    stride_el,
    slice_a_size,
    slice_c_size,
    # Meta-parameters
    num_slice_a: tl.constexpr,
    num_slice_c: tl.constexpr,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    USE_B_L2_CACHE: tl.constexpr,  # new, enable .ca load for B
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
    IS_PRIMARY: tl.constexpr,
    USE_TMA: tl.constexpr,
    sorted_c: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)
    lora_id = tl.load(lora_ids + lora_idx)

    if lora_id == -1:
        # Early exit for the no-lora case.
        return
    moe_enabled = tl.load(adapter_enabled + lora_id)
    if moe_enabled == 0:
        # Early exit for the no moe lora case.
        return
    # The grid's axis-2 dimension is max_loras + 1 to accommodate the -1 sentinel.
    # This guard ensures we don't access sorted_token_ids / expert_ids /
    # num_tokens_post_padded beyond their allocated bounds if an invalid
    # lora_id somehow appears. Although the caller should pass correct
    # max_loras, defensive programming prevents accidental out-of-bounds.
    if lora_id >= max_loras:
        return
    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    # calculate pid_m,pid_n
    pid_sk = pid % SPLIT_K
    pid_m_n = pid // SPLIT_K
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_m_n // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_m_n % num_pid_in_group) % group_size_m)
    pid_n = (pid_m_n % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_id)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    # get the expert_id to process curr shard
    ind = lora_id * stride_el + pid_m
    expert_id = tl.load(expert_ids_ptr + ind, ind < max_loras * stride_el, -1)
    if expert_id == -1:
        return

    cur_c_ptr = c_ptr + (slice_id % num_slice_c) * slice_c_size

    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int32)
    token_ind = stride_tl * lora_id + offs_token_id
    offs_token = tl.load(
        sorted_token_ids_ptr + token_ind,
        mask=token_ind < max_loras * stride_tl,
        other=num_valid_tokens,
    )
    token_mask = offs_token < num_valid_tokens

    if USE_TMA and not sorted_c:
        # Expand path - with TMA enabled, we load from A using TMA
        tl.static_assert(a_desc is not None, "a_desc should not be none")
        offs_am = slice_id * tl.cdiv(EM, top_k) * top_k + pid_m * BLOCK_SIZE_M // top_k
        offs_ak = pid_sk * BLOCK_SIZE_K
    else:
        # Shrink path - load hidden states based on order defined in
        # 'sorted_token_ids_ptr' then store them in c_ptr in this same sorted order
        tl.static_assert(a_desc is None, "a_desc must be none")
        cur_a_ptr = a_ptr + (slice_id % num_slice_a) * slice_a_size
        a_ptrs = cur_a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )

    if USE_TMA:
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_bk = pid_sk * BLOCK_SIZE_K
        if b_desc is None:
            cur_b_ptr = tl.load(b_ptr + slice_id).to(
                tl.pointer_type(c_ptr.dtype.element_ty)
            )

            # Note(@gnovack) - Allocation of TMA descriptors on-device
            # can cause conflicts when running in parallel via PDL
            if USE_GDC and not IS_PRIMARY:
                tl.extra.cuda.gdc_wait()

            b_desc = tl.make_tensor_descriptor(
                cur_b_ptr,
                shape=[max_loras, num_experts, N, K],
                strides=[stride_bl, stride_be, stride_bn, stride_bk],
                block_shape=[1, 1, BLOCK_SIZE_N, BLOCK_SIZE_K],
            )
    else:
        cur_b_ptr = tl.load(b_ptr + slice_id).to(
            tl.pointer_type(c_ptr.dtype.element_ty)
        )
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int32)
        b_ptrs = (
            cur_b_ptr
            + lora_id * stride_bl
            + expert_id * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

    if USE_GDC and IS_PRIMARY:
        # GDC launch dependents hints the runtime system to launch dependent kernels.
        tl.extra.cuda.gdc_launch_dependents()

    # accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if USE_GDC and not IS_PRIMARY:
        tl.extra.cuda.gdc_wait()

    for k in range(0, grid_k):
        cur_k_offset = k * (BLOCK_SIZE_K * SPLIT_K)
        k_remaining = K - cur_k_offset
        # pre-fetch lora weight
        if b_desc is not None:
            b = (
                b_desc.load([lora_id, expert_id, offs_bn, offs_bk + cur_k_offset])
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                .T
            )
        else:
            # add (offs_bn < N) mask; optional .ca for B
            b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)
            if USE_B_L2_CACHE:
                b = tl.load(b_ptrs, mask=b_mask, other=0.0, cache_modifier=".ca")
            else:
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

        if a_desc is not None:
            a = a_desc.load([offs_am, offs_ak + cur_k_offset])
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                other=0.0,
            )
            a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak

        accumulator += tl.dot(a, b)

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(c_ptr.dtype.element_ty)
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # When sorted_c is true, store the output in c_ptr using token order defined
    # in sorted_token_ids_ptr; otherwise, use the original token order from the prompt
    if sorted_c:
        c_ptrs = (
            cur_c_ptr
            + stride_cm * offs_token_id[:, None]
            + stride_cn * offs_cn[None, :]
        )
    else:
        c_ptrs = (
            cur_c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        )
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        if ADD_INPUTS:
            prev = tl.load(c_ptrs, mask=c_mask, other=0.0)
            tl.store(c_ptrs, prev + accumulator, mask=c_mask)
        else:
            tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")


@torch.inference_mode()
def _fused_moe_lora_shrink(
    a_intermediate_cache1: torch.Tensor,
    # (num_slices, num_tokens, top_k_num, max_lora_rank)
    qcurr_hidden_states: torch.Tensor,  # (num_tokens, K,)
    lora_a_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, max_lora_rank, K,),...]
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)
    sorted_token_ids: torch.Tensor,  # (max_loras, _)
    expert_ids: torch.Tensor,  # (max_loras, _ ,)
    num_tokens_post_padded: torch.Tensor,  # (max_loras, )
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    ## adding for kernel
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    mul_routed_weight: bool = False,
    use_gdc: bool = False,
    use_tma: bool = False,
    start: torch.cuda.Event | None = None,
    end: torch.cuda.Event | None = None,
) -> None:
    w1_lora_a_stacked = lora_a_stacked[0]
    shrink_config = {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 4,
        "SPLIT_K": 8,
        "launch_pdl": use_gdc,
        # "USE_GDC": use_gdc,
        # "launch_pdl": use_gdc,  # triton kernel metadata
        # "USE_TMA": use_tma,
    }

    b_ptr = _get_ptr(lora_a_stacked, device)

    NUM_SMS = torch.cuda.get_device_properties(w1_lora_a_stacked.device).multi_processor_count * 10
    grid = lambda META: (META["NUM_SMS"], 1, 1)

    a_desc = b_desc = None
    if use_tma and num_slices == 1:
        b_desc = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
            lora_a_stacked[0],
            [1, 1, shrink_config["BLOCK_SIZE_N"], shrink_config["BLOCK_SIZE_K"]],
        )

    _fused_moe_lora_kernel_persistent[grid](
        qcurr_hidden_states,
        a_desc,
        b_ptr,
        b_desc,
        a_intermediate_cache1,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        lora_ids,
        N=N,
        K=K,
        EM=EM,
        stride_am=qcurr_hidden_states.stride(0),
        stride_ak=qcurr_hidden_states.stride(1),
        stride_el=expert_ids.stride(0),
        stride_tl=sorted_token_ids.stride(0),
        stride_bl=w1_lora_a_stacked.stride(0),
        stride_be=w1_lora_a_stacked.stride(1),
        stride_bk=w1_lora_a_stacked.stride(3),
        stride_bn=w1_lora_a_stacked.stride(2),
        stride_cm=a_intermediate_cache1.stride(2),
        stride_cn=a_intermediate_cache1.stride(3),
        num_valid_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k_num,
        adapter_enabled=adapter_enabled,
        max_loras=lora_a_stacked[0].shape[0],
        ADD_INPUTS=False,
        NUM_SMS=NUM_SMS,
        IS_PRIMARY=True,
        MUL_ROUTED_WEIGHT=False,
        **shrink_config,
    )
    # grid = lambda META: (
    #     split_k
    #     * triton.cdiv(EM, META["BLOCK_SIZE_M"])
    #     * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    #     len(lora_a_stacked),
    #     ## max_loras + 1 to handle the no-lora case (lora_id == -1)
    #     lora_a_stacked[0].shape[0] + 1,
    # )

    
    

    # _fused_moe_lora_kernel[grid](
    #     qcurr_hidden_states,
    #     a_desc,
    #     b_ptr,
    #     b_desc,
    #     a_intermediate_cache1,
    #     topk_weights,
    #     sorted_token_ids,
    #     expert_ids,
    #     num_tokens_post_padded,
    #     N,
    #     K,
    #     EM,
    #     num_tokens,
    #     num_experts,
    #     lora_ids,
    #     adapter_enabled,
    #     lora_a_stacked[0].shape[0],
    #     qcurr_hidden_states.stride(0),
    #     qcurr_hidden_states.stride(1),
    #     w1_lora_a_stacked.stride(0),
    #     w1_lora_a_stacked.stride(1),
    #     w1_lora_a_stacked.stride(3),
    #     w1_lora_a_stacked.stride(2),
    #     a_intermediate_cache1.stride(2),
    #     a_intermediate_cache1.stride(3),
    #     sorted_token_ids.stride(0),
    #     expert_ids.stride(0),
    #     slice_a_size=qcurr_hidden_states.numel(),
    #     slice_c_size=a_intermediate_cache1.numel() // num_slices,
    #     num_slice_a=1,
    #     num_slice_c=num_slices,
    #     top_k=1 if mul_routed_weight else top_k_num,
    #     MUL_ROUTED_WEIGHT=False,
    #     ADD_INPUTS=False,
    #     USE_B_L2_CACHE=True,
    #     sorted_c=use_tma,
    #     IS_PRIMARY=True,
    #     **shrink_config,
    # )


@torch.inference_mode()
def _fused_moe_lora_expand(
    output: torch.Tensor,  # (num_tokens, top_k_num, N*len(lora_a_stacked),)
    a_intermediate_cache1: torch.Tensor,  # (num_slices, M, top_k_num, max_lora_rank)
    lora_b_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, max_lora_rank, K,),...]
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)
    sorted_token_ids: torch.Tensor,  # (max_loras, _)
    expert_ids: torch.Tensor,  # (max_loras, _ ,)
    num_tokens_post_padded: torch.Tensor,  # (max_loras, )
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    ## adding for kernel
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    max_lora_rank: int,
    w1_output_dim_size: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    mul_routed_weight: bool = False,
    offset: int = 0,
    use_gdc: bool = False,
    use_tma: bool = False,
    start: torch.cuda.Event | None = None,
    end: torch.cuda.Event | None = None,
) -> None:
    b_ptr = _get_ptr(lora_b_stacked, device)
    K = max_lora_rank
    N = w1_output_dim_size

    w1_lora_b_stacked = lora_b_stacked[0]

    a_intermediate_cache1 = a_intermediate_cache1.view(
        -1, a_intermediate_cache1.shape[3]
    )

    expand_config = {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 16,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
        "SPLIT_K": 1,  # Set split_k = 1 for expand calls
        # "USE_GDC": use_gdc,
        "launch_pdl": use_gdc,  # triton kernel metadata
        # "USE_TMA": use_tma,
    }

    

    NUM_SMS = torch.cuda.get_device_properties(w1_lora_b_stacked.device).multi_processor_count * 10
    grid = lambda META: (META["NUM_SMS"], 1, 1)

    # Fast path: directly accumulate into the corresponding slice interval of output.
    out_view = output[:, :, offset : offset + num_slices * N]
    slice_c_size = N * out_view.stride(2)
    a_desc = b_desc = None
    if use_tma:
        a_desc = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
            a_intermediate_cache1,
            [expand_config["BLOCK_SIZE_M"], expand_config["BLOCK_SIZE_K"]],
        )
        if num_slices == 1:
            b_desc = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
                lora_b_stacked[0],
                [1, 1, expand_config["BLOCK_SIZE_N"], expand_config["BLOCK_SIZE_K"]],
            )
    else:
        b_desc = None
    
    _fused_moe_lora_kernel_persistent[grid](
        a_intermediate_cache1,
        a_desc,
        b_ptr,
        b_desc,
        out_view,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        lora_ids,
        N=N,
        K=K,
        EM=EM,
        stride_am=a_intermediate_cache1.stride(0),
        stride_ak=a_intermediate_cache1.stride(1),
        stride_el=expert_ids.stride(0),
        stride_tl=sorted_token_ids.stride(0),
        stride_bl=w1_lora_b_stacked.stride(0),
        stride_be=w1_lora_b_stacked.stride(1),
        stride_bk=w1_lora_b_stacked.stride(3),
        stride_bn=w1_lora_b_stacked.stride(2),
        stride_cm=out_view.stride(1),
        stride_cn=out_view.stride(2),
        num_valid_tokens=num_tokens,
        num_experts=num_experts,
        top_k=1,
        adapter_enabled=adapter_enabled,
        max_loras=lora_b_stacked[0].shape[0],
        ADD_INPUTS=True,
        NUM_SMS=NUM_SMS,
        IS_PRIMARY=False,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        **expand_config,
    )

    # grid = lambda META: (
    #     triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    #     len(lora_b_stacked),
    #     ## max_loras + 1 to handle the no-lora case (lora_id == -1)
    #     lora_b_stacked[0].shape[0] + 1,
    # )

    
    # _fused_moe_lora_kernel[grid](
    #     a_intermediate_cache1,
    #     a_desc,
    #     b_ptr,
    #     b_desc,
    #     out_view,
    #     topk_weights,
    #     sorted_token_ids,
    #     expert_ids,
    #     num_tokens_post_padded,
    #     N,
    #     K,
    #     EM,
    #     num_tokens,
    #     num_experts,
    #     lora_ids,
    #     adapter_enabled,
    #     lora_b_stacked[0].shape[0],
    #     a_intermediate_cache1.stride(0),
    #     a_intermediate_cache1.stride(1),
    #     w1_lora_b_stacked.stride(0),
    #     w1_lora_b_stacked.stride(1),
    #     w1_lora_b_stacked.stride(3),
    #     w1_lora_b_stacked.stride(2),
    #     out_view.stride(1),
    #     out_view.stride(2),
    #     sorted_token_ids.stride(0),
    #     expert_ids.stride(0),
    #     slice_a_size=a_intermediate_cache1.numel() // num_slices,
    #     slice_c_size=slice_c_size,
    #     num_slice_a=num_slices,
    #     num_slice_c=num_slices,
    #     top_k=1,
    #     MUL_ROUTED_WEIGHT=mul_routed_weight,
    #     ADD_INPUTS=True,
    #     USE_B_L2_CACHE=True,
    #     sorted_c=False,
    #     IS_PRIMARY=False,
    #     **expand_config,
    # )
    


@torch.inference_mode()
def _fused_moe_lora(
    output: torch.Tensor,  # (num_tokens, top_k_num, N*len(lora_a_stacked),)
    qcurr_hidden_states: torch.Tensor,  # (num_tokens, K,)
    lora_a_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, max_lora_rank, K,),...]
    lora_b_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, N, max_lora_rank,),...]
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)
    sorted_token_ids: torch.Tensor,  # (max_loras, _)
    expert_ids: torch.Tensor,  # (max_loras, _ ,)
    num_tokens_post_padded: torch.Tensor,  # (max_loras, )
    max_lora_rank: int,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    shrink_block_size_m: int,
    shrink_block_size_n: int,
    shrink_block_size_k: int,
    shrink_group_size_m: int,
    shrink_num_warps: int,
    shrink_num_stages: int,
    shrink_split_k: int,
    expand_block_size_m: int,
    expand_block_size_n: int,
    expand_block_size_k: int,
    expand_group_size_m: int,
    expand_num_warps: int,
    expand_num_stages: int,
    expand_split_k: int,
    mul_routed_weight: bool = False,
    fully_sharded: bool = False,
    offset: int = 0,
    start: torch.cuda.Event | None = None,
    end: torch.cuda.Event | None = None,
) -> None:
    assert len(lora_a_stacked) == len(lora_b_stacked) > 0
    assert (
        sorted_token_ids.dim()
        == expert_ids.dim()
        == topk_weights.dim()
        == qcurr_hidden_states.dim()
        == 2
    )
    assert (
        sorted_token_ids.shape[0]
        == expert_ids.shape[0]
        == num_tokens_post_padded.shape[0]
    )
    assert output.shape[0] == topk_weights.shape[0]
    assert top_k_num == topk_weights.shape[1]
    device = qcurr_hidden_states.device
    num_slices = len(lora_a_stacked)
    w1_lora_b_stacked = lora_b_stacked[0]
    num_experts = lora_a_stacked[0].shape[1]
    N = max_lora_rank
    M = topk_weights.shape[0]
    EM = sorted_token_ids.shape[1]
    K = qcurr_hidden_states.shape[1]
    num_tokens = M * top_k_num
    w1_output_dim_size = w1_lora_b_stacked.shape[2]

    # TMA is not currently compatiple with fully_sharded due to the non-determinism
    # of token id sorting across ranks.
    use_tma = supports_tma(device) and not fully_sharded

    if use_tma and num_slices > 1:
        # if num_slices > 1, we construct TMA descriptors for
        # LoRA weights within the kernel, which requires us to first set an allocator
        _set_triton_allocator(device)

    a_intermediate_cache1 = torch.zeros(
        (num_slices, triton.cdiv(EM, top_k_num), top_k_num, max_lora_rank),
        dtype=output.dtype,
        device=device,
    )

    use_gdc = supports_pdl(device) and not fully_sharded
    torch.cuda.nvtx.range_push("fused_moe_lora_shrink")
    _fused_moe_lora_shrink(
        a_intermediate_cache1,
        qcurr_hidden_states,
        lora_a_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k_num,
        lora_ids,
        adapter_enabled,
        ## adding for kernel
        device,
        N,
        M,
        EM,
        K,
        num_tokens,
        num_experts,
        num_slices,
        shrink_block_size_m,
        shrink_block_size_n,
        shrink_block_size_k,
        shrink_group_size_m,
        shrink_num_warps,
        shrink_num_stages,
        shrink_split_k,
        mul_routed_weight,
        use_gdc=use_gdc,
        use_tma=use_tma,
        start=start,
        end=end,
    )
    torch.cuda.nvtx.range_pop()

    if fully_sharded:
        if max_lora_rank == w1_lora_b_stacked.shape[-1]:
            a_intermediate_cache1 = tensor_model_parallel_all_reduce(
                a_intermediate_cache1
            )
        else:
            a_intermediate_cache1 = tensor_model_parallel_all_gather(
                a_intermediate_cache1
            )

            # reset max_lora_rank to the full rank after allgather
            max_lora_rank = a_intermediate_cache1.shape[-1]

    torch.cuda.nvtx.range_push("fused_moe_lora_expand")
    _fused_moe_lora_expand(
        output,
        a_intermediate_cache1,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k_num,
        lora_ids,
        adapter_enabled,
        ## adding for kernel
        device,
        N,
        M,
        EM,
        K,
        num_tokens,
        num_experts,
        num_slices,
        max_lora_rank,
        w1_output_dim_size,
        expand_block_size_m,
        expand_block_size_n,
        expand_block_size_k,
        expand_group_size_m,
        expand_num_warps,
        expand_num_stages,
        expand_split_k,
        mul_routed_weight,
        offset,
        use_gdc=use_gdc,
        use_tma=use_tma,
    )
    torch.cuda.nvtx.range_pop()


def _fused_moe_lora_fake(
    output: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    max_lora_rank: int,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    shrink_block_size_m: int,
    shrink_block_size_n: int,
    shrink_block_size_k: int,
    shrink_group_size_m: int,
    shrink_num_warps: int,
    shrink_num_stages: int,
    shrink_split_k: int,
    expand_block_size_m: int,
    expand_block_size_n: int,
    expand_block_size_k: int,
    expand_group_size_m: int,
    expand_num_warps: int,
    expand_num_stages: int,
    expand_split_k: int,
    mul_routed_weight: bool = False,
) -> None:
    return


def _fused_moe_lora_shrink_fake(
    a_intermediate_cache1: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    mul_routed_weight: bool = False,
    use_gdc: bool = False,
    use_tma: bool = False,
) -> None:
    return


def _fused_moe_lora_expand_fake(
    output: torch.Tensor,
    a_intermediate_cache1: torch.Tensor,
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k_num: int,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    device: torch.device,
    N: int,
    M: int,
    EM: int,
    K: int,
    num_tokens: int,
    num_experts: int,
    num_slices: int,
    max_lora_rank: int,
    w1_output_dim_size: int,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    group_size_m: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    mul_routed_weight: bool = False,
    offset: int = 0,
    use_gdc: bool = False,
    use_tma: bool = False,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="fused_moe_lora",
        op_func=_fused_moe_lora,
        mutates_args=["output"],
        fake_impl=_fused_moe_lora_fake,
    )

    direct_register_custom_op(
        op_name="fused_moe_lora_shrink",
        op_func=_fused_moe_lora_shrink,
        mutates_args=["a_intermediate_cache1"],
        fake_impl=_fused_moe_lora_shrink_fake,
    )

    direct_register_custom_op(
        op_name="fused_moe_lora_expand",
        op_func=_fused_moe_lora_expand,
        mutates_args=["output"],
        fake_impl=_fused_moe_lora_expand_fake,
    )

    fused_moe_lora = torch.ops.vllm.fused_moe_lora
    fused_moe_lora_shrink = torch.ops.vllm.fused_moe_lora_shrink
    fused_moe_lora_expand = torch.ops.vllm.fused_moe_lora_expand

except AttributeError:
    fused_moe_lora = _fused_moe_lora
    fused_moe_lora_shrink = _fused_moe_lora_shrink
    fused_moe_lora_expand = _fused_moe_lora_expand
