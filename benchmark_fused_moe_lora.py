#!/usr/bin/env python3
"""Standalone benchmark for fused_moe_lora kernel."""

import argparse
import random
import torch
from vllm import _custom_ops as ops
from vllm.lora.ops.triton_ops.utils import get_lora_op_configs, load_lora_op_config
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


def benchmark_kernel(num_tokens, top_k_num, num_experts, max_loras, N, K, 
                     max_lora_rank, block_size, dtype, device, num_warmup=10, num_iters=100):
    torch.set_default_device(device)
    # torch.manual_seed(42)
    # random.seed(42)
    op_prefix = "w13"

    shrink_config = get_lora_op_configs(
        op_type=f"fused_moe_lora_{op_prefix}_shrink",
        max_loras=max_loras,
        batch=num_tokens,
        hidden_size=K,
        rank=max_lora_rank,
        num_slices=1,
        moe_intermediate_size=N,
    )
    expand_config = get_lora_op_configs(
        op_type=f"fused_moe_lora_{op_prefix}_expand",
        max_loras=max_loras,
        batch=num_tokens,
        hidden_size=K,  # lora_a_stacked.shape[-1],
        rank=max_lora_rank,
        num_slices=1,
        moe_intermediate_size=N,  # lora_b_stacked.shape[-2],
    )
    
    # Generate data
    num_sequences = min(num_tokens, 8)
    topk_ids, topk_weights = assign_experts_to_tokens(num_tokens, num_experts, top_k_num)
    token_lora_mapping = assign_loras_to_tokens(num_tokens, num_sequences, max_loras)
    
    # Initialize tensors
    lora_a_stacked = [torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)]
    lora_b_stacked = [torch.rand((max_loras, num_experts, N, max_lora_rank), dtype=dtype)]
    hidden_states = torch.rand((num_tokens, K), dtype=dtype)
    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    
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
    
    config = {"BLOCK_SIZE_M":block_size, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
              "GROUP_SIZE_M": 1, "NUM_WARPS": 4, "NUM_STAGES": 4, "SPLIT_K": 8}
    
    print("Shrink config", shrink_config)
    
    # Warmup
    for _ in range(num_warmup):
        fused_moe_lora(output, hidden_states, lora_a_stacked, lora_b_stacked, topk_weights,
                       sorted_token_ids, expert_ids, num_tokens_post_padded, max_lora_rank,
                       top_k_num, lora_ids, adapter_enabled, shrink_config["BLOCK_SIZE_M"],
                       shrink_config["BLOCK_SIZE_N"], shrink_config["BLOCK_SIZE_K"], shrink_config["GROUP_SIZE_M"],
                       shrink_config["num_warps"], shrink_config["num_stages"], shrink_config.get("SPLIT_K", 1),
                       expand_config["BLOCK_SIZE_M"], expand_config["BLOCK_SIZE_N"], expand_config["BLOCK_SIZE_K"],
                       expand_config["GROUP_SIZE_M"], expand_config["num_warps"], expand_config["num_stages"],
                       expand_config.get("SPLIT_K", 1), False, fully_sharded=False, offset=0)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.cudart().cudaProfilerStart()
    start.record()
    for _ in range(num_iters):
        fused_moe_lora(output, hidden_states, lora_a_stacked, lora_b_stacked, topk_weights,
                       sorted_token_ids, expert_ids, num_tokens_post_padded, max_lora_rank,
                       top_k_num, lora_ids, adapter_enabled, config["BLOCK_SIZE_M"],
                       shrink_config["BLOCK_SIZE_N"], shrink_config["BLOCK_SIZE_K"], shrink_config["GROUP_SIZE_M"],
                       shrink_config["num_warps"], shrink_config["num_stages"], shrink_config.get("SPLIT_K", 8),
                       expand_config["BLOCK_SIZE_M"], expand_config["BLOCK_SIZE_N"], expand_config["BLOCK_SIZE_K"],
                       expand_config["GROUP_SIZE_M"], expand_config["num_warps"], expand_config["num_stages"],
                       expand_config.get("SPLIT_K", 1), False, fully_sharded=False, offset=0)
    end.record()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    avg_time_ms = triton.testing.do_bench_cudagraph(bench_fn)
    return avg_time_ms


@triton.testing.perf_report(
triton.testing.Benchmark(
    x_names=['num_tokens'],  # Argument names to use as an x-axis for the plot.
    x_vals=[1,2,4,8,16,32,64,128,256,1024,8192],  # Different possible values for `x_name`.
    # x_vals=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192],  # Different possible values for `x_name`.
    line_arg='kernel',  # Argument name whose value corresponds to a different line in the plot.
    # line_vals=['base'],  # Possible values for `line_arg`.
    # line_names=['No TMA'],  # Label name for the lines.
    line_vals=['base', 'persistent'],  # Possible values for `line_arg`.
    line_names=['Base', 'Persistent'],  # Label name for the lines.
    styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('yellow', '-')],  # Line styles.
    ylabel='us',  # Label name for the y-axis.
    plot_name='fused-moe-lora-performance',  # Name for the plot. Used also as a file name for saving the plot.
    args={
        "top_k_num": 8,
        "num_experts": 128,
        "max_loras": 32,
        "N": 2048,
        "K": 768,
        "max_lora_rank": 32,
        "dtype": torch.bfloat16,
        "device": "cuda:0",
        "num_slices": 2
    },  # Values for function arguments not in `x_names` and `y_name`.
))
def triton_bench_kernel(num_tokens, top_k_num, num_experts, max_loras, N, K,  max_lora_rank, dtype, device, kernel, num_slices):

    if kernel == 'persistent':
        os.environ['VLLM_USE_PERSISTENT_KERNEL'] = '1'
        os.environ['DISABLE_TMA'] = '0'
        os.environ['VLLM_TUNED_CONFIG_FOLDER'] = '/root/workspace/gnovack/lora-profiling/kernel-tuner/configs/gpt-oss-120b-persistent'
        envs.VLLM_TUNED_CONFIG_FOLDER = '/root/workspace/gnovack/lora-profiling/kernel-tuner/configs/gpt-oss-120b-persistent'
        # os.environ['VLLM_TUNED_CONFIG_FOLDER'] = '/root/workspace/gnovack/lora-profiling/kernel-tuner/configs/gpt-oss-120b-old'
        # envs.VLLM_TUNED_CONFIG_FOLDER = '/root/workspace/gnovack/lora-profiling/kernel-tuner/configs/gpt-oss-120b-old'
    elif kernel == 'tma':
        os.environ['VLLM_USE_PERSISTENT_KERNEL'] = '0'
        os.environ['DISABLE_TMA'] = '0'
        os.environ['VLLM_TUNED_CONFIG_FOLDER'] = '/root/workspace/gnovack/lora-profiling/kernel-tuner/configs/gpt-oss-120b-persistent'
        envs.VLLM_TUNED_CONFIG_FOLDER = '/root/workspace/gnovack/lora-profiling/kernel-tuner/configs/gpt-oss-120b-persistent'
    elif kernel == 'base':
        os.environ['VLLM_USE_PERSISTENT_KERNEL'] = '0'
        os.environ['DISABLE_TMA'] = '1'
        # os.environ['VLLM_TUNED_CONFIG_FOLDER'] = '/root/workspace/gnovack/lora-profiling/kernel-tuner/configs/gpt-oss-120b-old'
        os.environ['VLLM_TUNED_CONFIG_FOLDER'] = '/root/workspace/gnovack/lora-profiling/vllm/vllm/lora/ops/triton_ops/qwen3-coder-30B-A3B'
        # envs.VLLM_TUNED_CONFIG_FOLDER = '/root/workspace/gnovack/lora-profiling/kernel-tuner/configs/gpt-oss-120b-old'
        envs.VLLM_TUNED_CONFIG_FOLDER = '/root/workspace/gnovack/lora-profiling/vllm/vllm/lora/ops/triton_ops/qwen3-coder-30B-A3B'
    elif kernel == 'untuned':
        os.environ['VLLM_USE_PERSISTENT_KERNEL'] = '0'
        os.environ['DISABLE_TMA'] = '1'


    torch.set_default_device(device)
    op_prefix = "w13"
    get_lora_op_configs.cache_clear()
    load_lora_op_config.cache_clear()

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
    block_size = shrink_config["BLOCK_SIZE_M"]
    
    # Generate data
    num_sequences = 1
    topk_ids, topk_weights = assign_experts_to_tokens(num_tokens, num_experts, top_k_num)
    token_lora_mapping = assign_loras_to_tokens(num_tokens, num_sequences, max_loras)
    
    # Initialize tensors
    lora_a_stacked = [torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype) for _ in range(num_slices)]
    lora_b_stacked = [torch.rand((max_loras, num_experts, N, max_lora_rank), dtype=dtype) for _ in range(num_slices)]
    hidden_states = torch.rand((num_tokens, K), dtype=dtype)
    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    
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
        sorted_token_ids, expert_ids, num_tokens_post_padded, max_lora_rank,
        top_k_num, lora_ids, adapter_enabled, shrink_config["BLOCK_SIZE_M"],
        shrink_config["BLOCK_SIZE_N"], shrink_config["BLOCK_SIZE_K"], shrink_config["GROUP_SIZE_M"],
        shrink_config["num_warps"], shrink_config["num_stages"], shrink_config.get("SPLIT_K", 8),
        expand_config["BLOCK_SIZE_M"], expand_config["BLOCK_SIZE_N"], expand_config["BLOCK_SIZE_K"],
        expand_config["GROUP_SIZE_M"], expand_config["num_warps"], expand_config["num_stages"],
        expand_config.get("SPLIT_K", 1), False, fully_sharded=False, offset=0)
    ms = triton.testing.do_bench_cudagraph(bench_fn, rep=50)
    return ms * 1000


if __name__ == "__main__":

    triton_bench_kernel.run(print_data=True, show_plots=False)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--num-tokens", type=int, default=2)
    # parser.add_argument("--top-k-num", type=int, default=4)
    # parser.add_argument("--num-experts", type=int, default=128)
    # parser.add_argument("--max-loras", type=int, default=8)
    # parser.add_argument("--N", type=int, default=5888)
    # parser.add_argument("--K", type=int, default=3072)
    # parser.add_argument("--max-lora-rank", type=int, default=32)
    # parser.add_argument("--block-size", type=int, default=16)
    # parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    # parser.add_argument("--device", type=str, default="cuda:0")
    # parser.add_argument("--num-warmup", type=int, default=10)
    # parser.add_argument("--num-iters", type=int, default=30)
    # args = parser.parse_args()
    
    # dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    # print(f"Benchmarking fused_moe_lora kernel:")
    # print(f"  num_tokens={args.num_tokens}, top_k={args.top_k_num}, experts={args.num_experts}")
    # print(f"  max_loras={args.max_loras}, N={args.N}, K={args.K}, rank={args.max_lora_rank}")
    # print(f"  dtype={args.dtype}, device={args.device}")
    
    # avg_time = benchmark_kernel(
    #     args.num_tokens, args.top_k_num, args.num_experts, args.max_loras,
    #     args.N, args.K, args.max_lora_rank, args.block_size, dtype, args.device,
    #     args.num_warmup, args.num_iters
    # )
    
    # print(f"\nResults:")
    # print(f"  Average time: {avg_time * 1000:.3f} us")
    # print(f"  Throughput: {args.num_tokens / avg_time * 1000:.1f} tokens/sec")
