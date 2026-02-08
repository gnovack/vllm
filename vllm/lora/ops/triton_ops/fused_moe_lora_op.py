# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import supports_pdl, supports_tma, use_persistent



def _set_triton_allocator(device: torch.device):
    def alloc_fn(size: int, alignment: int, stream: int | None):
        return torch.empty(size, device=device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)


@triton.jit
def _get_tile_details(tile_id, num_pid_in_group, num_pid_m, group_size_m, num_tiles_per_lora, 
                      lora_ids_ptr, adapter_enabled_ptr, num_tokens_post_padded_ptr, BLOCK_SIZE_M,
                      stride_el, expert_ids_ptr, MAX_LORAS, MAX_EXPERT_INDEX):
    lora_idx = tile_id // num_tiles_per_lora
    lora_id = tl.load(lora_ids_ptr + lora_idx)
    if lora_id == -1 or lora_id >= MAX_LORAS:
        return -1, -1, -1, -1, -1, False

    lora_tile_id = tile_id % num_tiles_per_lora
    group_id = lora_tile_id // num_pid_in_group
    first_pid_m = group_id * group_size_m
    trimmed_group_size_m = min(num_pid_m - first_pid_m, group_size_m)
    pid_m = first_pid_m + (lora_tile_id % trimmed_group_size_m)
    m_start = pid_m * BLOCK_SIZE_M
    expert_index = lora_id * stride_el + pid_m
    
    moe_enabled = tl.load(adapter_enabled_ptr + lora_id)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_id)
    expert_id = tl.load(expert_ids_ptr + expert_index, expert_index < MAX_EXPERT_INDEX, -1)
    if moe_enabled == 0:
        return -1, -1, -1, -1, -1, False
    
    if m_start >= num_tokens_post_padded:
        return -1, -1, -1, -1, -1, False

    if expert_id == -1:
        return -1, -1, -1, -1, -1, False

    pid_n = (lora_tile_id % num_pid_in_group) // trimmed_group_size_m
    return lora_id, pid_m, pid_n, m_start, expert_id, True


@triton.jit(
    do_not_specialize=[
        "num_valid_tokens",
        "EM",
        "stride_tl",
        "stride_el",
        "NUM_BLOCKS_M",
        "NUM_TILES_PER_LORA",
        "MAX_EXPERT_INDEX",
        "SLICE_A_SIZE",
        "MAX_TOKEN_IDX",
    ]
)
def _fused_moe_lora_kernel_persistent(
    # Input pointers
    a_ptr, a_desc,
    b_ptr, b_desc_0, b_desc_1,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    lora_ids_ptr,
    token_lora_mapping_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    # Other dims
    num_valid_tokens,
    num_experts,
    adapter_enabled,
    top_k_num,
    token_mapping_factor,
    # Strides
    stride_am, stride_ak,
    stride_bl, stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_el,
    stride_tl,
    # slice size
    slice_c_size,
    # non-const meta
    NUM_BLOCKS_M,
    NUM_TILES_PER_LORA,
    MAX_EXPERT_INDEX,
    SLICE_A_SIZE,
    MAX_TOKEN_IDX,
    # Fakes
    slice_a_size,
    num_slice_a,
    num_slice_c,
    max_loras, # fake max_loras
    naive_block_assignment,
    USE_B_L2_CACHE,
    sort_c,
    # Meta
    ADD_INPUTS: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    IS_PRIMARY: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    USE_TMA: tl.constexpr,
    USE_GDC: tl.constexpr,
    NUM_BLOCKS_N: tl.constexpr,
    NUM_TILE_K: tl.constexpr,
    NUM_TILES: tl.constexpr,
    NUM_SLICES: tl.constexpr,
):
    tl.static_assert(NUM_SLICES <= 2, "num_slices > 2 is not supported")
    c_type = c_ptr.dtype.element_ty
    
    pid = tl.program_id(axis=0)
    slice_id = pid // (NUM_SMS // NUM_SLICES)
    pid = pid % (NUM_SMS // NUM_SLICES)
    
    num_blocks_m = NUM_BLOCKS_M
    num_blocks_n = NUM_BLOCKS_N
    num_tiles_k = NUM_TILE_K
    num_tiles_per_lora = NUM_TILES_PER_LORA
    num_tiles = NUM_TILES
    num_pid_in_group = GROUP_SIZE_M * num_blocks_n

    m_block = tl.arange(0, BLOCK_SIZE_M)
    n_block = tl.arange(0, BLOCK_SIZE_N)
    k_block = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in tl.range(pid, num_tiles, NUM_SMS//NUM_SLICES):

        lora_id, tile_m_idx, tile_n_idx, m_start, expert_id, is_active = _get_tile_details(
            tile_id // SPLIT_K, num_pid_in_group, num_blocks_m, GROUP_SIZE_M, num_tiles_per_lora, lora_ids_ptr, 
            adapter_enabled, num_tokens_post_padded_ptr, BLOCK_SIZE_M, stride_el, expert_ids_ptr, MAX_LORAS, MAX_EXPERT_INDEX
        )
        n_start = tile_n_idx * BLOCK_SIZE_N
        k_start = (tile_id % SPLIT_K) * BLOCK_SIZE_K
        if is_active:
                
            offs_token_id = m_start + m_block
            token_index = stride_tl * lora_id + offs_token_id
            offs_m = tl.load(
                sorted_token_ids_ptr + token_index,
                mask=token_index < MAX_TOKEN_IDX,
                other=num_valid_tokens,
            )
            offs_n = n_start + n_block
            offs_k = k_start + k_block

            if a_desc is not None:
                offs_am = slice_id * SLICE_A_SIZE + m_start // token_mapping_factor
            else:
                a_ptrs = a_ptr + (
                    offs_m[:, None] // token_mapping_factor * stride_am + offs_k[None, :] * stride_ak
                )
            
            if IS_PRIMARY:
                # GDC launch dependents hints the runtime system to launch dependent kernels.
                tl.extra.cuda.gdc_launch_dependents()
            
            # Create masks for bounds checking
            mask_m = offs_m < num_valid_tokens
            mask_n = offs_n < N

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            if not IS_PRIMARY:
                tl.extra.cuda.gdc_wait()

            for ki in tl.range(num_tiles_k):

                cur_k_offset = ki * (BLOCK_SIZE_K * SPLIT_K)

                if a_desc is None:
                    mask_k = offs_k < (K - cur_k_offset)

                if NUM_SLICES == 1:
                    b = (
                        b_desc_0.load([lora_id, expert_id, n_start, k_start + cur_k_offset])
                        .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K).T
                    )
                else:
                    if slice_id == 0:
                        b = (
                            b_desc_0.load([lora_id, expert_id, n_start, k_start + cur_k_offset])
                            .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K).T
                        )
                    else:
                        b = (
                            b_desc_1.load([lora_id, expert_id, n_start, k_start + cur_k_offset])
                            .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K).T
                        )    

                if a_desc is not None:
                    a = a_desc.load([offs_am, k_start + cur_k_offset])
                else:
                    mask_a = mask_m[:, None] & mask_k[None, :]
                    a = tl.load(a_ptrs, mask=mask_a, other=0.0)
                    a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak

                accumulator += tl.dot(a, b)

            if MUL_ROUTED_WEIGHT:
                moe_weight = tl.load(topk_weights_ptr + offs_m, mask=mask_m, other=0.0)
                accumulator = accumulator * moe_weight[:, None]

            offs_cm = offs_token_id if a_desc is None else offs_m
            cur_c_ptr = c_ptr + (slice_id % NUM_SLICES) * slice_c_size
            c_ptrs = cur_c_ptr + offs_cm[:, None] * stride_cm + offs_n[None, :] * stride_cn

            c = accumulator.to(c_type)
            mask_c = mask_m[:, None] & mask_n[None, :]
            if SPLIT_K == 1:
                if ADD_INPUTS:
                    # prev = tl.load(c_ptrs, mask=mask_c, other=0.0)
                    # tl.store(c_ptrs, prev + c, mask=mask_c)
                    tl.atomic_add(c_ptrs, c, mask=mask_c, sem="relaxed")
                else:
                    tl.store(c_ptrs, c, mask=mask_c)
            else:
                tl.atomic_add(c_ptrs, c, mask=mask_c, sem="relaxed")


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
def _fused_moe_lora_kernel_tma(
    a_ptr,
    a_desc,
    b_ptr,
    b_desc_0,
    b_desc_1,  # unused, for signature compatibility
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    token_lora_mapping_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    num_experts,
    top_k_num,
    lora_ids_ptr,
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
    # top_k_num or 1 depending on input token
    # is expanded by top_k or not
    token_mapping_factor: tl.constexpr,
    # whether use naive block assignment
    naive_block_assignment: tl.constexpr,
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
    sort_c: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)
    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    # calculate pid_m,pid_n
    lora_idx = tl.program_id(axis=2)
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

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)

    # Get lora_id
    lora_id = _get_lora_id(
        lora_ids_ptr,
        token_lora_mapping_ptr,
        lora_idx,
        pid_m,
        top_k_num,
        naive_block_assignment,
    )
    if lora_id == -1:
        return
    moe_enabled = tl.load(adapter_enabled + lora_id)
    if moe_enabled == 0:
        return
    if lora_id >= max_loras:
        return

    # Non-naive only: check num_tokens_post_padded
    if not naive_block_assignment:
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_id)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return

    # Get expert_id
    expert_id = _get_expert_id(
        expert_ids_ptr,
        lora_id,
        pid_m,
        stride_el,
        max_loras,
        naive_block_assignment,
    )
    if expert_id == -1:
        return

    # Get token offsets
    offs_token = _get_token_offs(
        sorted_token_ids_ptr,
        lora_id,
        pid_m,
        offs,
        stride_tl,
        max_loras,
        num_valid_tokens,
        naive_block_assignment,
        BLOCK_SIZE_M,
    )
    # get a_ptr,b_ptr,c_ptr
    cur_a_ptr = a_ptr + (slice_id % num_slice_a) * slice_a_size
    cur_b_ptr = tl.load(b_ptr + slice_id).to(tl.pointer_type(c_ptr.dtype.element_ty))
    cur_c_ptr = c_ptr + (slice_id % num_slice_c) * slice_c_size

    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    token_mask = offs_token < num_valid_tokens

    if USE_TMA and a_desc is not None:
        # Expand path - with TMA enabled, we load from A using TMA
        pid_m_offset = 1 if naive_block_assignment else BLOCK_SIZE_M
        offs_am = (
            slice_id * tl.cdiv(EM, top_k_num) * top_k_num
            + pid_m * pid_m_offset // token_mapping_factor
        )
        offs_ak = pid_sk * BLOCK_SIZE_K
    else:
        # Shrink path - load hidden states based on order defined in
        # 'sorted_token_ids_ptr' then store them in c_ptr in this same sorted order
        tl.static_assert(a_desc is None, "a_desc must be none")
        a_ptrs = cur_a_ptr + (
            offs_token[:, None] // token_mapping_factor * stride_am
            + offs_k[None, :] * stride_ak
        )

    b_desc = b_desc_0
    if USE_TMA:
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_bk = pid_sk * BLOCK_SIZE_K
        if b_desc is None:
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

    # When sort_c is true, store the output in c_ptr using token order defined
    # in sorted_token_ids_ptr; otherwise, use the original token order from the prompt
    if sort_c:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
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


@triton.jit
def _get_lora_id(
    lora_ids,
    token_lora_mapping_ptr,
    lora_idx,
    pid_m,
    top_k_num,
    naive_block_assignment: tl.constexpr,
):
    """Returns lora_id"""
    if naive_block_assignment:
        token_idx = pid_m // top_k_num
        return tl.load(token_lora_mapping_ptr + token_idx)
    else:
        return tl.load(lora_ids + lora_idx)


@triton.jit
def _get_expert_id(
    expert_ids_ptr,
    lora_id,
    pid_m,
    stride_el,
    max_loras,
    naive_block_assignment: tl.constexpr,
):
    """Returns expert_id"""
    if naive_block_assignment:
        return tl.load(expert_ids_ptr + pid_m)
    else:
        ind = lora_id * stride_el + pid_m
        return tl.load(expert_ids_ptr + ind, ind < max_loras * stride_el, -1)


@triton.jit
def _get_token_offs(
    sorted_token_ids_ptr,
    lora_id,
    pid_m,
    offs,
    stride_tl,
    max_loras,
    num_valid_tokens,
    naive_block_assignment: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Returns token offsets"""
    if naive_block_assignment:
        return tl.where(offs == 0, pid_m, num_valid_tokens)
    else:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        token_ind = stride_tl * lora_id + offs_token_id
        return tl.load(
            sorted_token_ids_ptr + token_ind, token_ind < max_loras * stride_tl, 0
        )


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


def _adjust_kernel_inputs(
    num_active_loras: int,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
):
    """
    helper function to adjust kernel inputs when sorted_token_ids is None
    """
    if sorted_token_ids is None:
        stride_tl = 0
        stride_el = 0
        grid_lora_dim = 1
    else:
        stride_tl = sorted_token_ids.stride(0)
        stride_el = expert_ids.stride(0)
        grid_lora_dim = num_active_loras
    return grid_lora_dim, stride_tl, stride_el


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
    b_desc_0,  # unused, for signature compatibility
    b_desc_1,  # unused, for signature compatibility
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    token_lora_mapping_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    num_experts,
    top_k_num,
    lora_ids_ptr,
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
    # top_k_num or 1 depending on input token
    # is expanded by top_k or not
    token_mapping_factor: tl.constexpr,
    # whether use naive block assignment
    naive_block_assignment: tl.constexpr,
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
    sort_c: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)
    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    # calculate pid_m,pid_n
    lora_idx = tl.program_id(axis=2)
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

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)

    # Get lora_id
    lora_id = _get_lora_id(
        lora_ids_ptr,
        token_lora_mapping_ptr,
        lora_idx,
        pid_m,
        top_k_num,
        naive_block_assignment,
    )
    if lora_id == -1:
        return
    moe_enabled = tl.load(adapter_enabled + lora_id)
    if moe_enabled == 0:
        return
    if lora_id >= max_loras:
        return

    # Non-naive only: check num_tokens_post_padded
    if not naive_block_assignment:
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_id)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return

    # Get expert_id
    expert_id = _get_expert_id(
        expert_ids_ptr,
        lora_id,
        pid_m,
        stride_el,
        max_loras,
        naive_block_assignment,
    )
    if expert_id == -1:
        return

    # Get token offsets
    offs_token = _get_token_offs(
        sorted_token_ids_ptr,
        lora_id,
        pid_m,
        offs,
        stride_tl,
        max_loras,
        num_valid_tokens,
        naive_block_assignment,
        BLOCK_SIZE_M,
    )
    # get a_ptr,b_ptr,c_ptr
    cur_a_ptr = a_ptr + (slice_id % num_slice_a) * slice_a_size
    cur_b_ptr = tl.load(b_ptr + slice_id).to(tl.pointer_type(c_ptr.dtype.element_ty))
    cur_c_ptr = c_ptr + (slice_id % num_slice_c) * slice_c_size

    # remove modulo wrap-around
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int32)
    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    token_mask = offs_token < num_valid_tokens

    # get a_ptrs,b_ptrs
    a_ptrs = cur_a_ptr + (
        offs_token[:, None] // token_mapping_factor * stride_am
        + offs_k[None, :] * stride_ak
    )

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
        k_remaining = K - k * (BLOCK_SIZE_K * SPLIT_K)
        # GDC wait waits for ALL programs in the prior kernel to complete
        # before continuing.
        # pre-fetch lora weight
        # add (offs_bn < N) mask; optional .ca for B
        b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)
        if USE_B_L2_CACHE:
            b = tl.load(b_ptrs, mask=b_mask, other=0.0, cache_modifier=".ca")
        else:
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        if USE_GDC and not IS_PRIMARY:
            tl.extra.cuda.gdc_wait()
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(c_ptr.dtype.element_ty)
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = cur_c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
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
    sorted_token_ids: torch.Tensor | None,  # (max_loras, _)
    expert_ids: torch.Tensor,  # (max_loras, _ ,) or (num_tokens * top_k,)
    num_tokens_post_padded: torch.Tensor | None,  # (max_loras, )
    token_lora_mapping: torch.Tensor,
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
    num_active_loras: int,
    mul_routed_weight: bool = False,
    use_gdc: bool = False,
    use_tma: bool = False,
) -> None:
    w1_lora_a_stacked = lora_a_stacked[0]
    shrink_config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "SPLIT_K": split_k,
        "USE_GDC": use_gdc,
        "launch_pdl": use_gdc,  # triton kernel metadata
        "USE_TMA": use_tma,
    }

    b_ptr = _get_ptr(lora_a_stacked, device)

    grid_lora_dim, stride_tl, stride_el = _adjust_kernel_inputs(
        num_active_loras, sorted_token_ids, expert_ids
    )
    grid = lambda META: (
        split_k
        * triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        len(lora_a_stacked),
        grid_lora_dim,
    )

    a_desc = None
    b_desc_0 = None
    b_desc_1 = None
    
    persistent = use_persistent(device)
    if persistent and sorted_token_ids is not None:
        assert num_slices <= 2, "num_slices > 2 is not supported"
        b_desc_0 = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
            lora_a_stacked[0],
            [1, 1, shrink_config["BLOCK_SIZE_N"], shrink_config["BLOCK_SIZE_K"]],
        )
        if num_slices == 2:
            b_desc_1 = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
                lora_a_stacked[1],
                [1, 1, shrink_config["BLOCK_SIZE_N"], shrink_config["BLOCK_SIZE_K"]],
            )

        NUM_SMS = torch.cuda.get_device_properties(w1_lora_a_stacked.device).multi_processor_count * 8
        grid = lambda META: (META["NUM_SMS"], 1, 1)
        num_tiles_per_lora = triton.cdiv(EM, block_size_m) * triton.cdiv(N, block_size_n)

        shrink_config['MAX_LORAS'] = lora_a_stacked[0].shape[0]
        
        shrink_config['NUM_SMS'] = NUM_SMS
        shrink_config.pop("num_warps")
        shrink_config.pop("num_stages")
        
        shrink_config['NUM_BLOCKS_M'] = triton.cdiv(EM, block_size_m)
        shrink_config['NUM_BLOCKS_N'] = triton.cdiv(N, block_size_n)
        shrink_config['NUM_TILE_K'] = triton.cdiv(K, block_size_k * split_k)
        shrink_config['NUM_TILES_PER_LORA'] = num_tiles_per_lora
        shrink_config['NUM_SLICES'] = num_slices
        shrink_config['NUM_TILES'] = num_tiles_per_lora * num_active_loras
        shrink_config['MAX_EXPERT_INDEX'] = lora_a_stacked[0].shape[0]*expert_ids.stride(0)
        shrink_config['SLICE_A_SIZE'] = qcurr_hidden_states.numel()
        shrink_config['MAX_TOKEN_IDX'] = lora_a_stacked[0].shape[0] * sorted_token_ids.stride(0)
        kernel = _fused_moe_lora_kernel_persistent
    elif use_tma:
        if num_slices == 1:
            b_desc_0 = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
                lora_a_stacked[0],
                [1, 1, shrink_config["BLOCK_SIZE_N"], shrink_config["BLOCK_SIZE_K"]],
            )

        kernel = _fused_moe_lora_kernel_tma
    else:
        kernel = _fused_moe_lora_kernel

    kernel[grid](
        a_ptr=qcurr_hidden_states,
        a_desc=a_desc,
        b_ptr=b_ptr,
        b_desc_0=b_desc_0,
        b_desc_1=b_desc_1,
        c_ptr=a_intermediate_cache1,
        topk_weights_ptr=topk_weights,
        sorted_token_ids_ptr=sorted_token_ids,
        expert_ids_ptr=expert_ids,
        num_tokens_post_padded_ptr=num_tokens_post_padded,
        token_lora_mapping_ptr=token_lora_mapping,
        N=N,
        K=K,
        EM=EM,
        num_valid_tokens=num_tokens,
        num_experts=num_experts,
        top_k_num=top_k_num,
        lora_ids_ptr=lora_ids,
        adapter_enabled=adapter_enabled,
        max_loras=lora_a_stacked[0].shape[0],
        stride_am=qcurr_hidden_states.stride(0),
        stride_ak=qcurr_hidden_states.stride(1),
        stride_bl=w1_lora_a_stacked.stride(0),
        stride_be=w1_lora_a_stacked.stride(1),
        stride_bk=w1_lora_a_stacked.stride(3),
        stride_bn=w1_lora_a_stacked.stride(2),
        stride_cm=a_intermediate_cache1.stride(2),
        stride_cn=a_intermediate_cache1.stride(3),
        stride_tl=stride_tl,
        stride_el=stride_el,
        slice_a_size=qcurr_hidden_states.numel(),
        slice_c_size=a_intermediate_cache1.numel() // num_slices,
        num_slice_a=1,
        num_slice_c=num_slices,
        token_mapping_factor=1 if mul_routed_weight else top_k_num,
        naive_block_assignment=sorted_token_ids is None,
        MUL_ROUTED_WEIGHT=False,
        ADD_INPUTS=False,
        USE_B_L2_CACHE=True,
        sort_c=use_tma and sorted_token_ids is not None,
        IS_PRIMARY=True,
        **shrink_config,
    )


@torch.inference_mode()
def _fused_moe_lora_expand(
    output: torch.Tensor,  # (num_tokens, top_k_num, N*len(lora_a_stacked),)
    a_intermediate_cache1: torch.Tensor,  # (num_slices, M, top_k_num, max_lora_rank)
    lora_b_stacked: list[
        torch.Tensor
    ],  # [(max_loras, num_experts, max_lora_rank, K,),...]
    topk_weights: torch.Tensor,  # (num_tokens, top_k_num)
    sorted_token_ids: torch.Tensor | None,  # (max_loras, _)
    expert_ids: torch.Tensor,  # (max_loras, _ ,) or (num_tokens * top_k,)
    num_tokens_post_padded: torch.Tensor | None,  # (max_loras, )
    token_lora_mapping: torch.Tensor,
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
    num_active_loras: int,
    mul_routed_weight: bool = False,
    offset: int = 0,
    use_gdc: bool = False,
    use_tma: bool = False,
) -> None:
    b_ptr = _get_ptr(lora_b_stacked, device)
    K = max_lora_rank
    N = w1_output_dim_size

    w1_lora_b_stacked = lora_b_stacked[0]

    a_intermediate_cache1 = a_intermediate_cache1.view(
        -1, a_intermediate_cache1.shape[3]
    )

    expand_config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "SPLIT_K": 1,  # Set split_k = 1 for expand calls
        "USE_GDC": use_gdc,
        "launch_pdl": use_gdc,  # triton kernel metadata
        "USE_TMA": use_tma,
    }

    grid_lora_dim, stride_tl, stride_el = _adjust_kernel_inputs(
        num_active_loras, sorted_token_ids, expert_ids
    )

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        len(lora_b_stacked),
        grid_lora_dim,
    )

    # Fast path: directly accumulate into the corresponding slice interval of output.
    out_view = output[:, :, offset : offset + num_slices * N]
    slice_c_size = N * out_view.stride(2)
    a_desc = None
    b_desc_0 = None
    b_desc_1 = None
    persistent = use_persistent(device)
    if persistent and sorted_token_ids is not None:
        assert num_slices <= 2, "num_slices > 2 is not supported"
        a_desc = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
            a_intermediate_cache1,
            [expand_config["BLOCK_SIZE_M"], expand_config["BLOCK_SIZE_K"]],
        )
        b_desc_0 = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
            lora_b_stacked[0],
            [1, 1, expand_config["BLOCK_SIZE_N"], expand_config["BLOCK_SIZE_K"]],
        )
        if num_slices == 2:
            b_desc_1 = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
                lora_b_stacked[1],
                [1, 1, expand_config["BLOCK_SIZE_N"], expand_config["BLOCK_SIZE_K"]],
            )

        NUM_SMS = torch.cuda.get_device_properties(w1_lora_b_stacked.device).multi_processor_count * 16
        grid = lambda META: (META["NUM_SMS"], 1, 1)
        num_tiles_per_lora = triton.cdiv(EM, block_size_m) * triton.cdiv(N, block_size_n)
        expand_config['MAX_LORAS'] = lora_b_stacked[0].shape[0]
        
        expand_config['NUM_SMS'] = NUM_SMS
        expand_config.pop("num_warps")
        expand_config.pop("num_stages")

        expand_config['NUM_BLOCKS_M'] = triton.cdiv(EM, block_size_m)
        expand_config['NUM_BLOCKS_N'] = triton.cdiv(N, block_size_n)
        expand_config['NUM_TILE_K'] = triton.cdiv(K, block_size_k * split_k)
        expand_config['NUM_TILES_PER_LORA'] = num_tiles_per_lora
        expand_config['NUM_SLICES'] = num_slices
        expand_config['NUM_TILES'] = num_tiles_per_lora * num_active_loras
        expand_config['MAX_EXPERT_INDEX'] = lora_b_stacked[0].shape[0]*expert_ids.stride(0)
        expand_config['SLICE_A_SIZE'] = triton.cdiv(EM, top_k_num) * top_k_num
        expand_config['MAX_TOKEN_IDX'] = lora_b_stacked[0].shape[0] * sorted_token_ids.stride(0)
        
        kernel = _fused_moe_lora_kernel_persistent
    elif use_tma:
        kernel = _fused_moe_lora_kernel_tma
        if sorted_token_ids is not None:
            a_desc = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
                a_intermediate_cache1,
                [expand_config["BLOCK_SIZE_M"], expand_config["BLOCK_SIZE_K"]],
            )
        if num_slices == 1:
            b_desc_0 = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
                lora_b_stacked[0],
                [1, 1, expand_config["BLOCK_SIZE_N"], expand_config["BLOCK_SIZE_K"]],
            )
    else:
        kernel = _fused_moe_lora_kernel

    kernel[grid](
        a_ptr=a_intermediate_cache1,
        a_desc=a_desc,
        b_ptr=b_ptr,
        b_desc_0=b_desc_0,
        b_desc_1=b_desc_1,
        c_ptr=out_view,
        topk_weights_ptr=topk_weights,
        sorted_token_ids_ptr=sorted_token_ids,
        expert_ids_ptr=expert_ids,
        num_tokens_post_padded_ptr=num_tokens_post_padded,
        token_lora_mapping_ptr=token_lora_mapping,
        N=N,
        K=K,
        EM=EM,
        num_valid_tokens=num_tokens,
        num_experts=num_experts,
        top_k_num=top_k_num,
        lora_ids_ptr=lora_ids,
        adapter_enabled=adapter_enabled,
        max_loras=lora_b_stacked[0].shape[0],
        stride_am=a_intermediate_cache1.stride(0),
        stride_ak=a_intermediate_cache1.stride(1),
        stride_bl=w1_lora_b_stacked.stride(0),
        stride_be=w1_lora_b_stacked.stride(1),
        stride_bk=w1_lora_b_stacked.stride(3),
        stride_bn=w1_lora_b_stacked.stride(2),
        stride_cm=out_view.stride(1),
        stride_cn=out_view.stride(2),
        stride_tl=stride_tl,
        stride_el=stride_el,
        slice_a_size=a_intermediate_cache1.numel() // num_slices,
        slice_c_size=slice_c_size,
        num_slice_a=num_slices,
        num_slice_c=num_slices,
        token_mapping_factor=1,
        naive_block_assignment=sorted_token_ids is None,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        ADD_INPUTS=True,
        USE_B_L2_CACHE=True,
        sort_c=False,
        IS_PRIMARY=False,
        **expand_config,
    )


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
    sorted_token_ids: torch.Tensor | None,  # (max_loras, _)
    expert_ids: torch.Tensor,  # (max_loras, _ ,) or (num_tokens * top_k,)
    num_tokens_post_padded: torch.Tensor | None,  # (max_loras, )
    token_lora_mapping: torch.Tensor,
    max_lora_rank: int,
    top_k_num: int,
    lora_ids: torch.Tensor,
    num_active_loras: int,
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
) -> None:
    assert len(lora_a_stacked) == len(lora_b_stacked) > 0
    assert topk_weights.dim() == qcurr_hidden_states.dim() == 2
    if sorted_token_ids is None:
        assert expert_ids.dim() == 1
    else:
        assert sorted_token_ids is not None
        assert num_tokens_post_padded is not None
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
    K = qcurr_hidden_states.shape[1]
    num_tokens = M * top_k_num
    w1_output_dim_size = w1_lora_b_stacked.shape[2]
    assert shrink_block_size_m == expand_block_size_m
    EM = (
        sorted_token_ids.shape[1]
        if sorted_token_ids is not None
        else num_tokens * shrink_block_size_m
    )

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
    _fused_moe_lora_shrink(
        a_intermediate_cache1,
        qcurr_hidden_states,
        lora_a_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
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
        num_active_loras,
        mul_routed_weight,
        use_gdc=use_gdc,
        use_tma=use_tma,
    )

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

    _fused_moe_lora_expand(
        output,
        a_intermediate_cache1,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
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
        num_active_loras,
        mul_routed_weight,
        offset,
        use_gdc=use_gdc,
        use_tma=use_tma,
    )


def _fused_moe_lora_fake(
    output: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor | None,
    token_lora_mapping: torch.Tensor,
    max_lora_rank: int,
    top_k_num: int,
    lora_ids: torch.Tensor,
    num_active_loras: int,
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
) -> None:
    return


def _fused_moe_lora_shrink_fake(
    a_intermediate_cache1: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor | None,
    token_lora_mapping: torch.Tensor,
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
    num_active_loras: int,
    mul_routed_weight: bool = False,
    use_gdc: bool = False,
) -> None:
    return


def _fused_moe_lora_expand_fake(
    output: torch.Tensor,
    a_intermediate_cache1: torch.Tensor,
    b_intermediate_cache1: torch.Tensor,
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor | None,
    token_lora_mapping: torch.Tensor,
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
    num_active_loras: int,
    mul_routed_weight: bool = False,
    offset: int = 0,
    use_gdc: bool = False,
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
