# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark mhc_fused_post_pre using DeepSeek V4 Flash tensor shapes."""

import argparse
import json
import math
import time
from pathlib import Path

import torch


DEFAULT_CONFIG = Path(
    "/home/george/.cache/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-V4-Flash/snapshots/"
    "fd53f944496234770ba80e15004f9b6d269a71f5/config.json"
)

SUPPORTED_TILE_SPLITS = (
    (1, 1),
    (1, 2),
    (1, 4),
    (1, 8),
    (2, 1),
    (2, 2),
    (2, 4),
    (2, 8),
    (3, 1),
    (3, 2),
    (3, 4),
    (4, 1),
    (4, 2),
    (1, 16),
    (2, 16),
    (4, 16),
    (6, 16),
    (8, 16),
    (12, 8),
    (12, 16),
)


def parse_token_counts(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_tile_splits(value: str) -> list[tuple[int, int]]:
    if value == "all":
        return list(SUPPORTED_TILE_SPLITS)

    pairs = []
    for item in value.split(","):
        tile_n, n_splits = (int(part) for part in item.split(":"))
        pair = (tile_n, n_splits)
        if pair not in SUPPORTED_TILE_SPLITS:
            raise argparse.ArgumentTypeError(
                f"Unsupported tile split {item}. Use one of "
                f"{SUPPORTED_TILE_SPLITS}."
            )
        pairs.append(pair)
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--token-counts", type=parse_token_counts,
                        default=parse_token_counts("1,2,4,8,16,32,64"))
    parser.add_argument("--tile-splits", type=parse_tile_splits,
                        default=parse_tile_splits("all"),
                        help=("Comma-separated tile_n:n_splits pairs, or "
                              "'all' for every supported kernel variant."))
    parser.add_argument("--include-unfused", action="store_true",
                        help=("Also benchmark the post + DeepGEMM + pre path "
                              "used above the fused-token threshold."))
    parser.add_argument("--variants", default="all",
                        choices=("cuda", "tilelang", "all"),
                        help=("Which fused kernel implementation(s) to "
                              "benchmark across the tile-split grid."))
    parser.add_argument("--eager-warmup", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=100,
                        help="Number of op invocations captured in the graph.")
    parser.add_argument("--replays", type=int, default=10,
                        help="Number of timed CUDA graph replays.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_mhc_module(include_unfused: bool):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the MHC fused benchmark.")

    try:
        import vllm._custom_ops as _custom_ops  # noqa: F401
        import vllm._tilelang_ops as tl_ops
    except ImportError as exc:
        raise SystemExit(
            "Could not import vLLM custom ops or the tilelang ops module. "
            "Make sure vLLM is built and tilelang is installed."
        ) from exc

    tf32_hc_prenorm_gemm = None
    if include_unfused:
        try:
            from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm
        except ImportError as exc:
            raise SystemExit(
                "Could not import tf32_hc_prenorm_gemm for the unfused path."
            ) from exc

    return tl_ops, tf32_hc_prenorm_gemm


def make_inputs(config: dict, num_tokens: int, device, generator):
    hidden_size = int(config["hidden_size"])
    hc_mult = int(config["hc_mult"])
    mix_hc = hc_mult * (2 + hc_mult)
    hc_dim = hc_mult * hidden_size

    residual = torch.randn(
        num_tokens,
        hc_mult,
        hidden_size,
        device=device,
        generator=generator,
    )
    x = torch.randn(num_tokens, hidden_size, device=device, generator=generator)
    residual = (0.25 * residual).to(torch.bfloat16).contiguous()
    x = (0.25 * x).to(torch.bfloat16).contiguous()

    post = torch.sigmoid(
        torch.randn(num_tokens, hc_mult, 1, device=device, generator=generator)
    )
    post = (2.0 * post).contiguous()

    comb = torch.softmax(
        torch.randn(num_tokens, hc_mult, hc_mult, device=device,
                    generator=generator),
        dim=-1,
    )
    for _ in range(2):
        comb = comb / comb.sum(-2, keepdim=True)
        comb = comb / comb.sum(-1, keepdim=True)
    comb = comb.contiguous()

    fn = torch.randn(
        mix_hc,
        hc_dim,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    fn = (fn / math.sqrt(hc_dim)).contiguous()

    hc_scale = torch.randn(3, dtype=torch.float32, device=device,
                           generator=generator)
    hc_base = torch.randn(mix_hc, dtype=torch.float32, device=device,
                          generator=generator)
    hc_scale = (0.25 * hc_scale).contiguous()
    hc_base = (0.25 * hc_base).contiguous()

    return x, residual, post, comb, fn, hc_scale, hc_base


TILELANG_N_THR = 256


def tilelang_config_valid(hidden_size: int, n_splits: int) -> bool:
    """The tilelang fused kernel requires (hidden / n_splits) % n_thr == 0
    because each thread strides through h_per_split by n_thr elements."""
    if hidden_size % n_splits != 0:
        return False
    return (hidden_size // n_splits) % TILELANG_N_THR == 0


def run_configured_fused_post_pre(
    tl_ops,
    config: dict,
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    tile_n: int,
    n_splits: int,
    variant: str = "cuda",
):
    # Mirror mhc_fused_post_pre's fused pmap/GEMM path, but keep tile_n and
    # n_splits configurable so the benchmark can search kernel variants.
    # `variant` selects between the CUDA `mhc_fused_pmap_gemm_fma_ksplit` op
    # and the tilelang `mhc_fused_tilelang` kernel.
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x_flat = x.view(num_tokens, hidden_size)
    post_flat = post.view(num_tokens, hc_mult)
    comb_flat = comb.view(num_tokens, hc_mult, hc_mult)

    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )
    residual_cur = torch.empty_like(residual_flat)
    post_mix_cur = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    if variant == "tilelang":
        tl_ops.mhc_fused_tilelang(
            comb_flat,
            residual_flat,
            post_flat,
            x_flat,
            fn.view(hc_mult3, hc_mult, hidden_size),
            gemm_out_mul,
            gemm_out_sqrsum,
            residual_cur,
            hc_mult,
            hidden_size,
            hc_mult3,
            TILELANG_N_THR,
            256,
            tile_n,
            n_splits,
        )
    else:
        raise ValueError(f"Unknown variant: {variant!r}")
    tl_ops.mhc_pre_big_fuse_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_cur,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
        hidden_size,
        float(config["rms_norm_eps"]),
        float(config["hc_eps"]),
        float(config["hc_eps"]),
        2.0,
        int(config["hc_sinkhorn_iters"]),
        n_splits,
        hc_mult,
    )
    return (
        residual_cur.view(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.view(*outer_shape, hc_mult, 1),
        comb_mix_cur.view(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.view(*outer_shape, hidden_size),
    )


def run_unfused_post_pre(
    tl_ops,
    tf32_hc_prenorm_gemm,
    config: dict,
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    n_splits: int,
):
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )
    residual_cur = torch.empty_like(residual_flat)
    post_mix_cur = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    tl_ops.mhc_post_tilelang(
        comb,
        residual,
        post.squeeze(-1),
        x,
        residual_cur,
        hc_mult,
        hidden_size,
    )
    tf32_hc_prenorm_gemm(
        residual_cur.view(num_tokens, hc_mult * hidden_size),
        fn,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )
    tl_ops.mhc_pre_big_fuse_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_cur,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
        hidden_size,
        float(config["rms_norm_eps"]),
        float(config["hc_eps"]),
        float(config["hc_eps"]),
        2.0,
        int(config["hc_sinkhorn_iters"]),
        n_splits,
        hc_mult,
    )
    return (
        residual_cur.view(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.view(*outer_shape, hc_mult, 1),
        comb_mix_cur.view(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.view(*outer_shape, hidden_size),
    )


def capture_graph(run, device, eager_warmup: int, graph_iters: int):
    side_stream = torch.cuda.Stream(device=device)
    current_stream = torch.cuda.current_stream(device)
    side_stream.wait_stream(current_stream)
    with torch.cuda.stream(side_stream):
        for _ in range(eager_warmup):
            run()
    current_stream.wait_stream(side_stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    static_outputs = None
    with torch.cuda.graph(graph):
        for _ in range(graph_iters):
            static_outputs = run()
    return graph, static_outputs


def benchmark_one(run, device, eager_warmup: int, warmup: int, iters: int,
                  replays: int) -> float:
    with torch.inference_mode():
        graph, static_outputs = capture_graph(run, device, eager_warmup, iters)
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize(device)

    assert static_outputs is not None
    return start.elapsed_time(end) / (iters * replays)


def get_unfused_n_splits(tl_ops, num_tokens: int, hidden_size: int,
                         hc_mult: int) -> int:
    block_k = 64
    block_m = 64
    hc_hidden_size = hc_mult * hidden_size
    grid_size = (num_tokens + block_m - 1) // block_m
    return tl_ops.compute_num_split(block_k, hc_hidden_size, grid_size)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tl_ops, tf32_hc_prenorm_gemm = load_mhc_module(args.include_unfused)

    with args.config.open() as f:
        config = json.load(f)

    hidden_size = int(config["hidden_size"])
    hc_mult = int(config["hc_mult"])
    mix_hc = hc_mult * (2 + hc_mult)
    hc_dim = hc_mult * hidden_size
    if any(hidden_size % n_splits != 0 for _, n_splits in args.tile_splits):
        raise SystemExit("All requested n_splits must divide hidden_size.")
    if any(mix_hc % tile_n != 0 for tile_n, _ in args.tile_splits):
        raise SystemExit("All requested tile_n values must divide fn rows.")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"config: {args.config}")
    print(f"hidden_size={hidden_size} hc_mult={hc_mult} "
          f"fn_shape=({mix_hc}, {hc_dim}) "
          f"sinkhorn_iters={config['hc_sinkhorn_iters']}")
    print(f"cuda_graph_iters={args.iters} replays={args.replays} "
          f"eager_warmup={args.eager_warmup} replay_warmup={args.warmup}")
    print(f"include_unfused={args.include_unfused} variants={args.variants}")
    print("tokens, variant, tile_n, n_splits, latency_ms, latency_us, rank")

    if args.variants == "all":
        active_variants = ("tilelang",)
    else:
        active_variants = (args.variants,)

    best_per_variant: dict[tuple[int, str], tuple[float, int, int]] = {}

    wall_start = time.perf_counter()
    for num_tokens in args.token_counts:
        inputs = make_inputs(config, num_tokens, device, generator)
        x, residual, post, comb, fn, hc_scale, hc_base = inputs
        results = []
        for variant in active_variants:
            for tile_n, n_splits in args.tile_splits:
                if variant == "tilelang" and not tilelang_config_valid(
                    hidden_size, n_splits
                ):
                    # Tilelang kernel needs (hidden/n_splits) % n_thr == 0.
                    continue

                def run_fused(variant=variant, tile_n=tile_n, n_splits=n_splits):
                    return run_configured_fused_post_pre(
                        tl_ops,
                        config,
                        x,
                        residual,
                        post,
                        comb,
                        fn,
                        hc_scale,
                        hc_base,
                        tile_n,
                        n_splits,
                        variant=variant,
                    )

                latency_ms = benchmark_one(
                    run_fused,
                    device,
                    args.eager_warmup,
                    args.warmup,
                    args.iters,
                    args.replays,
                )
                results.append((latency_ms, variant, tile_n, n_splits))

                key = (num_tokens, variant)
                if (
                    key not in best_per_variant
                    or latency_ms < best_per_variant[key][0]
                ):
                    best_per_variant[key] = (latency_ms, tile_n, n_splits)

        if args.include_unfused:
            unfused_n_splits = get_unfused_n_splits(
                tl_ops, num_tokens, hidden_size, hc_mult)

            def run_unfused():
                return run_unfused_post_pre(
                    tl_ops,
                    tf32_hc_prenorm_gemm,
                    config,
                    x,
                    residual,
                    post,
                    comb,
                    fn,
                    hc_scale,
                    hc_base,
                    unfused_n_splits,
                )

            latency_ms = benchmark_one(
                run_unfused,
                device,
                args.eager_warmup,
                args.warmup,
                args.iters,
                args.replays,
            )
            results.append((latency_ms, "unfused", "-", unfused_n_splits))

        for rank, (latency_ms, variant, tile_n, n_splits) in enumerate(
                sorted(results, key=lambda result: result[0]), start=1):
            print(f"{num_tokens}, {variant}, {tile_n}, {n_splits}, "
                  f"{latency_ms * 1000:.3f}")
    wall_s = time.perf_counter() - wall_start
    print(f"total_wall_s={wall_s:.3f}")

    if best_per_variant:
        print()
        print("# Best (tile_n, n_splits) per variant per token count:")
        print("tokens, variant, tile_n, n_splits, latency_us")
        for num_tokens in args.token_counts:
            for variant in active_variants:
                key = (num_tokens, variant)
                if key not in best_per_variant:
                    continue
                latency_ms, tile_n, n_splits = best_per_variant[key]
                print(f"{num_tokens}, {variant}, {tile_n}, {n_splits}, "
                      f"{latency_ms * 1000:.3f}")


if __name__ == "__main__":
    main()
