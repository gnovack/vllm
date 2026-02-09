#!/usr/bin/env python3
"""Standalone benchmark for fused_moe_lora kernel."""

import argparse
import random
import torch
from vllm import _custom_ops as ops
from vllm.lora.ops.triton_ops.utils import get_lora_op_configs, load_lora_op_config, use_persistent, supports_tma
from vllm.lora.ops.triton_ops import fused_moe_lora
import triton
from vllm import envs

import os

if os.environ.get('VLLM_USE_PERSISTENT_KERNEL', '1') == '1':
    os.environ['VLLM_TUNED_CONFIG_FOLDER'] = '/root/workspace/gnovack/lora-profiling/kernel-tuner/configs/gpt-oss-120b-persistent'
else:
    os.environ['VLLM_TUNED_CONFIG_FOLDER'] = '/root/workspace/gnovack/lora-profiling/kernel-tuner/configs/gpt-oss-120b-tma'


def round_up(x, base):
    return ((x + base - 1) // base) * base


def CEILDIV(x, y):
    return (x + y - 1) // y


def assign_loras_to_tokens(num_tokens, num_sequences, max_loras):
    tokens_per_seq = num_tokens // num_sequences
    remainder = num_tokens % num_sequences
    token_lora_mapping = torch.empty(num_tokens, dtype=torch.int32)
    start = 0
    for seq_idx in range(num_sequences):
        end = start + tokens_per_seq + (1 if seq_idx < remainder else 0)
        lora_id = random.randint(0, max_loras - 1)
        token_lora_mapping[start:end] = lora_id
        start = end
    return token_lora_mapping


def assign_experts_to_tokens(num_tokens, num_experts, top_k_num):
    expert_indices = torch.empty((num_tokens, top_k_num), dtype=torch.int32)
    for i in range(num_tokens):
        expert_indices[i] = torch.randperm(num_experts)[:top_k_num]
    expert_weights = torch.rand((num_tokens, top_k_num), dtype=torch.float32)
    expert_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True)
    return expert_indices, expert_weights


def _normalize_keys(config: dict[str, int | None]) -> dict[str, int | None]:
        normalized_config = {}
        for key, value in config.items():
            if key.islower():
                if key.startswith("block_"):
                    normalized_key = "BLOCK_SIZE_" + key.split("_")[-1].upper()
                else:
                    normalized_key = key.upper()
            else:
                normalized_key = key
            normalized_config[normalized_key] = value
        return normalized_config


@triton.testing.perf_report(
triton.testing.Benchmark(
    x_names=['num_tokens'],  # Argument names to use as an x-axis for the plot.
    x_vals=[1,2,4,8,16,32,64,128,256,512,1024,1536,2048,3072,4096,6144,8192],  # Different possible values for `x_name`.e`.
    line_arg='kernel',  # Argument name whose value corresponds to a different line in the plot.
    # line_vals=['main', 'tma', 'persistent'],
    # line_names=['Main', 'TMA', 'Persistent'],
    line_vals=['main', 'main-tuned', 'tma', 'tma-tuned', 'persistent'],
    line_names=['Main', 'Main (tuned)', 'TMA', 'TMA (tuned)', 'Persistent'],
    y_log=True,
    x_log=True,
    # line_names=['Main', 'Main (Tuned)', 'TMA', 'TMA (Tuned)', 'Persistent'],  # Label name for the lines.
    styles=[('indianred', '-'), ('indianred', ':'), ('steelblue', '-'), ('steelblue', ':'), ('blueviolet', '-')],  # Line styles.
    ylabel='us',  # Label name for the y-axis.
    plot_name='fused-moe-lora-performance',  # Name for the plot. Used also as a file name for saving the plot.
    args={
        "max_loras": 8,
        "max_lora_rank": 32,
        "dtype": torch.bfloat16,
        "device": "cuda:0",
        "model": "qwen3-coder"
        # "model": "gpt-oss-120b"
    }
))
def triton_bench_kernel(num_tokens, max_loras, max_lora_rank, dtype, device, kernel, model):

    op_prefix = "w13"
    if model == 'qwen3-coder':
        tuned_config = '/root/workspace/gnovack/lora-profiling/kernel-tuner/configs/new-coder'
        top_k_num = 8
        num_experts = 128
        N = 768 if op_prefix == 'w13' else 2048
        K = 2048 if op_prefix == 'w13' else 768
        num_slices = 2 if op_prefix == "w13" else 1
    elif model == 'gpt-oss-120b':
        tuned_config = '/root/workspace/gnovack/lora-profiling/kernel-tuner/configs/gpt-oss-120b'
        top_k_num = 4
        num_experts = 128
        N = 5888 if op_prefix == 'w13' else 3072
        K = 3072 if op_prefix == 'w13' else 2944
        num_slices = 1
    else:
        raise ValueError("unknown model")

    if kernel.startswith('persistent'):
        os.environ['USE_PERSISTENT'] = '1'
        os.environ['USE_TMA'] = '1'
    elif kernel.startswith('tma'):
        os.environ['USE_PERSISTENT'] = '0'
        os.environ['USE_TMA'] = '1'
    elif kernel.startswith('main'):
        os.environ['USE_PERSISTENT'] = '0'
        os.environ['USE_TMA'] = '0'
    
    if kernel.endswith('tuned'):
        os.environ['VLLM_TUNED_CONFIG_FOLDER'] = tuned_config
        envs.VLLM_TUNED_CONFIG_FOLDER = tuned_config
    else:
        os.environ['VLLM_TUNED_CONFIG_FOLDER'] = '/root/workspace/gnovack/lora-profiling/kernel-tuner/configs/fake'
        envs.VLLM_TUNED_CONFIG_FOLDER = '/root/workspace/gnovack/lora-profiling/kernel-tuner/configs/fake'


    torch.set_default_device(device)
    get_lora_op_configs.cache_clear()
    load_lora_op_config.cache_clear()
    use_persistent.cache_clear()
    supports_tma.cache_clear()

    shrink_config = get_lora_op_configs(
        op_type=f"fused_moe_lora_{op_prefix}_shrink",
        max_loras=max_loras,
        batch=num_tokens,
        hidden_size=K,
        rank=max_lora_rank,
        num_slices=num_slices,
        moe_intermediate_size=N,
    )
    expand_config = get_lora_op_configs(
        op_type=f"fused_moe_lora_{op_prefix}_expand",
        max_loras=max_loras,
        batch=num_tokens,
        hidden_size=K,  # lora_a_stacked.shape[-1],
        rank=max_lora_rank,
        num_slices=num_slices,
        moe_intermediate_size=N,  # lora_b_stacked.shape[-2],
    )
    shrink_config = _normalize_keys(shrink_config)
    expand_config = _normalize_keys(expand_config)
    block_size = shrink_config["BLOCK_SIZE_M"]
    
    # Generate data
    num_sequences = 8
    active_loras = 8
    topk_ids, topk_weights = assign_experts_to_tokens(num_tokens, num_experts, top_k_num)
    token_lora_mapping = assign_loras_to_tokens(num_tokens, num_sequences, max_loras)
    
    # Initialize tensors
    lora_a_stacked = [torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype) for _ in range(num_slices)]
    lora_b_stacked = [torch.rand((max_loras, num_experts, N, max_lora_rank), dtype=dtype) for _ in range(num_slices)]
    hidden_states = torch.rand((num_tokens, K), dtype=dtype)
    output = torch.zeros((num_tokens, top_k_num, N*2), dtype=dtype)
    
    # Prepare kernel inputs
    max_num_active_experts = min(topk_ids.numel(), num_experts)
    max_num_tokens_padded = topk_ids.numel() + max_num_active_experts * (block_size - 1)
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size)
    
    sorted_token_ids = torch.empty((max_loras * max_num_tokens_padded,), dtype=torch.int32)
    expert_ids = torch.empty((max_loras * max_num_m_blocks,), dtype=torch.int32)
    num_tokens_post_padded = torch.empty((max_loras,), dtype=torch.int32)
    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32)
    lora_ids = torch.arange(max_loras + 2, dtype=torch.int32)
    
    ops.moe_lora_align_block_size(
        topk_ids, token_lora_mapping, num_experts, block_size, max_loras,
        max_num_tokens_padded, max_num_m_blocks, sorted_token_ids, expert_ids,
        num_tokens_post_padded, adapter_enabled, lora_ids
    )
    
    expert_ids = expert_ids.view(max_loras, -1)
    sorted_token_ids = sorted_token_ids.view(max_loras, -1)

    bench_fn = lambda: fused_moe_lora(output, hidden_states, lora_a_stacked, lora_b_stacked, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded, token_lora_mapping, max_lora_rank,
        top_k_num, lora_ids, active_loras, adapter_enabled, shrink_config["BLOCK_SIZE_M"],
        shrink_config["BLOCK_SIZE_N"], shrink_config["BLOCK_SIZE_K"], shrink_config["GROUP_SIZE_M"],
        shrink_config["NUM_WARPS"], shrink_config["NUM_STAGES"], shrink_config.get("SPLIT_K", 1),
        expand_config["BLOCK_SIZE_M"], expand_config["BLOCK_SIZE_N"], expand_config["BLOCK_SIZE_K"],
        expand_config["GROUP_SIZE_M"], expand_config["NUM_WARPS"], expand_config["NUM_STAGES"],
        expand_config.get("SPLIT_K", 1), False, fully_sharded=False, offset=0)
    
    print(f"Benchmarking kernel {kernel} with {num_tokens} tokens")
    ms = triton.testing.do_bench_cudagraph(bench_fn, rep=50, return_mode='median')
    return ms * 1000


if __name__ == "__main__":

    triton_bench_kernel.run(print_data=True, save_path='fused-moe-lora-comparison')
    # triton_bench_kernel.run(print_data=True)
