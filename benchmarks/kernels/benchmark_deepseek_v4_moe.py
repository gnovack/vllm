# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone micro-benchmark for the ``DeepseekV4MoE`` module.

This runs a single ``DeepseekV4MoE`` block in isolation (i.e. without building
the full DeepSeek-V4 model / engine) so the MoE forward can be profiled and
iterated on quickly.

It builds a real :class:`~vllm.config.VllmConfig` from a model path via
``EngineArgs`` (so every sub-config the module and the underlying ``FusedMoE``
expect is present), initializes a single-rank distributed environment, then
instantiates the module with dummy weights and times its forward pass.

``--model`` only needs to resolve to a real DeepSeek-V4 ``config.json`` -- no
checkpoint is loaded (``load_format="dummy"``). Point it at the HF repo id, a
local checkpoint directory, or a directory containing just ``config.json``.
This drives the *real* DeepSeek-V4-Flash shapes (4096 hidden, 256 experts,
top-6) and its quantization (block-FP8 / fp4 experts).

Dummy weights are generated exactly the way vLLM's ``DummyModelLoader`` does
(quantization-aware, including FP8 and packed-fp4 experts), so the *outputs
are meaningless* -- this measures latency/throughput only, not correctness.

Example
-------
    # Default backend, real DeepSeek-V4-Flash config (block-FP8 experts):
    .venv/bin/python benchmarks/kernels/benchmark_deepseek_v4_moe.py \
        --model deepseek-ai/DeepSeek-V4-Flash --trust-remote-code \
        --num-tokens 1024 --layer-idx 5

    # Tensor parallel across 2 GPUs (one process per rank via mp.spawn):
    .venv/bin/python benchmarks/kernels/benchmark_deepseek_v4_moe.py \
        --model deepseek-ai/DeepSeek-V4-Flash --trust-remote-code \
        --tensor-parallel-size 2 --num-tokens 1024 --layer-idx 5

    # Data + expert parallel: DP=2 token shards x TP=2, EP across all 4 GPUs:
    .venv/bin/python benchmarks/kernels/benchmark_deepseek_v4_moe.py \
        --model deepseek-ai/DeepSeek-V4-Flash --trust-remote-code \
        --data-parallel-size 2 --tensor-parallel-size 2 --enable-expert-parallel \
        --num-tokens 2048 --layer-idx 5

    # fp4 MegaMoE / DeepGEMM backend with expert parallel across 4 GPUs. The
    # real checkpoint's experts are fp4; this path needs expert parallel and
    # SM100 (B200) GPUs with DeepGEMM:
    .venv/bin/python benchmarks/kernels/benchmark_deepseek_v4_moe.py \
        --model deepseek-ai/DeepSeek-V4-Flash --trust-remote-code \
        --tensor-parallel-size 4 --enable-expert-parallel \
        --moe-backend deep_gemm_mega_moe --num-tokens 1024 --layer-idx 5

Multi-GPU runs spawn one process per rank (total ranks = TP x DP), with layout
``global_rank = dp_rank * tp_size + tp_rank``:
  * ``--tensor-parallel-size N`` TP-shards the dense weights over N ranks.
  * ``--data-parallel-size M`` splits the global ``--num-tokens`` batch evenly
    across M DP groups (the TP ranks within a group replicate their shard).
  * ``--enable-expert-parallel`` shards the experts (EP) across the whole
    TP x DP world and dispatches tokens to them via all-to-all -- the usual way
    DP scales MoE, so pair ``-dp`` with this. Without it, vLLM folds DP into the
    MoE's tensor-parallel group (experts intermediate-sharded across TP x DP),
    which is not the typical MoE configuration.
The collectives keep ranks in lockstep; rank 0 reports its own latency.
Single-node only (no PP, no multi-node DP).

Pick ``--layer-idx`` below ``num_hash_layers`` (3 in the real config) to
benchmark a hash-routing MoE layer instead of a gated/routed one.
"""

import contextlib
import json

import torch

from vllm.config import set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import initialize_dummy_weights
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager


def _init_distributed(
    world_size: int,
    global_rank: int,
    local_rank: int,
    init_method: str,
    tp_size: int,
) -> None:
    """Bring up the (possibly multi-rank) distributed environment.

    ``DeepseekV4MoE`` queries TP / EP groups at construction time, so the
    process groups must exist before the module is instantiated.

    Rank layout follows vLLM's ``initialize_model_parallel``:
    ``global_rank = dp_rank * tp_size + tp_rank``. The TP group spans the
    ``tp_size`` ranks of one DP replica, the DP group spans the matching ranks
    across replicas, and (with ``--enable-expert-parallel``) the EP group spans
    the whole DP x TP world.

    For DP > 1 we initialize the full unified world ourselves first. That makes
    vLLM's ``init_distributed_environment`` short-circuit (it would otherwise
    rewrite the rank/port for its own DP launch convention and ignore our
    ``init_method``); ``initialize_model_parallel`` then carves TP/DP/EP out of
    the already-initialized world via ``torch.distributed.get_world_size()``.
    """
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            init_method=init_method,
            rank=global_rank,
            world_size=world_size,
            device_id=torch.device("cuda", local_rank),
        )
    tp_rank = global_rank % tp_size
    init_distributed_environment(
        world_size=tp_size,
        rank=tp_rank,
        distributed_init_method=init_method,
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=tp_size)


# ue8m0 scale byte that decodes to 1.0: _ue8m0_uint8_to_float(b) = 2**(b-127),
# so b = 127 gives a unit scale (see DeepseekV4MegaMoEExperts).
_UE8M0_UNIT_SCALE = 127


@torch.no_grad()
def _init_dummy_weights(module: torch.nn.Module, model_config) -> None:
    """Assign random dummy weights, the same way the dummy loader does.

    No checkpoint is loaded, so we only need finite, well-scaled values.
    ``initialize_dummy_weights`` (used by vLLM's ``DummyModelLoader``) handles
    floating-point params -- including sub-16-bit FP8 -- via an fp16
    intermediate, and safely skips integer tensors (it would otherwise leave
    them as constructed zeros).

    The MegaMoE/fp4 experts store weights as *packed uint8* with separate
    *ue8m0 uint8* block scales, which the integer-skip above leaves at zero
    (i.e. all-zero experts). We fill those explicitly so the benchmark exercises
    representative, non-degenerate compute:
      * packed weight bytes -> uniform random bytes,
      * ue8m0 scale bytes   -> a unit scale (127), keeping dequant well-scaled
        and, crucially, avoiding the inf that random scale exponents would
        produce in finalize_weights().
    """
    initialize_dummy_weights(module, model_config)

    for name, param in module.named_parameters():
        if param.dtype not in (torch.uint8, torch.int8):
            continue
        # Block scales are tagged with quant_method="block" on the param.
        if getattr(param, "quant_method", None) == "block" or name.endswith(
            "_weight_scale"
        ):
            param.data.fill_(_UE8M0_UNIT_SCALE)
        else:
            param.data.random_(0, 256)


def make_vllm_config(args):
    """Build the VllmConfig from the model path (no checkpoint is loaded)."""
    # DeepSeek-V4 routes the MoE quant method off the HF config's `expert_dtype`
    # (quant_config.py): "fp4" -> MXFP4/NVFP4 experts, "fp8" -> block-FP8 experts.
    # Overriding it here switches which expert weight precision (and hence which
    # MoE kernel backends) the layer uses, without needing a different checkpoint
    # -- the FusedMoE method allocates the params for that precision and we fill
    # them with dummy values. (No real conversion happens; see
    # deepseek_v4_moe_backends.md for why MXFP4 -> FP8 is numerically lossless.)
    hf_overrides: dict = {}
    if args.expert_dtype != "auto":
        hf_overrides["expert_dtype"] = args.expert_dtype

    engine_args = EngineArgs(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        moe_backend=args.moe_backend,
        all2all_backend=args.all2all_backend,
        hf_overrides=hf_overrides,
        # We never load a checkpoint; build the config only.
        load_format="dummy",
        max_model_len=args.max_model_len,
    )
    return engine_args.create_engine_config()


def build_module(args, vllm_config, device):
    """Construct the DeepseekV4MoE module on ``device``.

    Assumes the caller has already entered ``set_current_vllm_config`` and
    initialized the distributed environment + workspace manager.
    """
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MoE

    dtype = vllm_config.model_config.dtype
    prefix = f"model.layers.{args.layer_idx}.mlp"
    # Construct directly on the device, under the model dtype. The model
    # loader builds modules inside set_default_torch_dtype, which is what
    # makes the expert weights land in the model dtype while layers that
    # opt out (e.g. the float32 router gate) keep their own dtype. We must
    # NOT blanket-cast afterwards.
    with set_default_torch_dtype(dtype), torch.device(device):
        module = DeepseekV4MoE(vllm_config, prefix=prefix)
    _init_dummy_weights(module, vllm_config.model_config)
    # Mirror the model loader's post-load step: this is where the
    # FusedMoE kernel is selected/initialized and weights are repacked.
    process_weights_after_loading(module, vllm_config.model_config, device)
    # MegaMoE stages/quantizes weights lazily; mirror the model loader.
    # finalize_weights() enforces the fp4 backend's hardware requirements
    # (DeepGEMM + SM100/B200); surface that as a clear message rather than
    # a raw traceback on unsupported GPUs.
    if hasattr(module, "finalize_mega_moe_weights"):
        try:
            module.finalize_mega_moe_weights()
        except NotImplementedError as e:
            raise SystemExit(
                f"\nThe deep_gemm_mega_moe (fp4) backend is unavailable here: "
                f"{e}\nIt requires an SM100 (B200) GPU with DeepGEMM. Use the "
                f"default --moe-backend on other hardware."
            ) from e

    return module


def make_inputs(num_tokens, vllm_config, dtype, device):
    hidden_size = vllm_config.model_config.hf_config.hidden_size
    vocab_size = vllm_config.model_config.hf_config.vocab_size
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    # input_ids are only consumed by hash-routing MoE layers, but always
    # passing them is harmless for non-hash layers. vLLM feeds int32 token ids.
    input_ids = torch.randint(
        0, vocab_size, (num_tokens,), dtype=torch.int32, device=device
    )
    return hidden_states, input_ids


def _tokens_for_dp_rank(total_tokens: int, dp_rank: int, dp_size: int) -> int:
    """Split ``total_tokens`` as evenly as possible across ``dp_size`` DP groups.

    The first ``total_tokens % dp_size`` ranks get one extra token. vLLM's
    DP coordination (coordinate_batch_across_dp) handles unequal per-rank
    counts by padding, so a non-divisible batch is fine.
    """
    base, rem = divmod(total_tokens, dp_size)
    return base + (1 if dp_rank < rem else 0)


def _capture_cuda_graph(run_once, warmup: int, device) -> "torch.cuda.CUDAGraph":
    """Warm up, then capture one MoE forward into a CUDA graph.

    The MoE forward launches a long chain of small kernels (gate, topk, align,
    permute, two grouped GEMMs, activation, unpermute, reduce). Replaying a
    captured graph issues them with a single launch, removing the per-kernel CPU
    launch overhead that dominates at small token counts.

    Capture is wrapped in vLLM's ``graph_capture`` context, which puts the TP/PP
    communicators into a capture-safe mode (e.g. the custom TP all-reduce). This
    is required for TP>1: a bare ``torch.cuda.graph`` capture of the custom
    all-reduce hits an illegal memory access. The warmup runs on the same
    capture stream so lazy allocations / autotuning / JIT happen before capture.
    """
    from vllm.distributed.parallel_state import graph_capture

    with graph_capture(device) as ctx:
        ctx.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(ctx.stream):
            for _ in range(max(warmup, 3)):
                run_once()
            graph = torch.cuda.CUDAGraph()
            # Inputs are fixed for the benchmark, so we capture against the
            # existing static tensors and never copy new data in before replay.
            with torch.cuda.graph(graph, stream=ctx.stream):
                run_once()
    return graph


@torch.no_grad()
def benchmark(args, vllm_config, module, hidden_states, input_ids):
    def run_once():
        return module(hidden_states, input_ids)

    # The underlying FusedMoE / MegaMoE kernels read the forward context.
    with set_forward_context(
        attn_metadata=None,
        vllm_config=vllm_config,
        num_tokens=hidden_states.shape[0],
    ):
        if args.cuda_graph:
            # Capture also performs the warmup (on the capture stream).
            try:
                graph = _capture_cuda_graph(
                    run_once, args.warmup, hidden_states.device
                )
            except Exception as e:
                raise SystemExit(
                    f"\nCUDA graph capture failed for this MoE path: "
                    f"{type(e).__name__}: {e}\n"
                    "Not every backend is graph-capturable -- some do host syncs "
                    "or data-dependent launches during the forward. Try "
                    "'--moe-backend triton', or drop --cuda-graph."
                ) from e
            step = graph.replay
        else:
            # Warmup (always outside the profiled region).
            for _ in range(args.warmup):
                run_once()
            step = run_once
        torch.cuda.synchronize()

        # cudaProfilerStart/Stop, scoping the capture to just the measured
        # iterations. Pair with e.g.
        #   nsys profile --capture-range=cudaProfilerApi -c nvtx ...
        # or `ncu --profile-from-start off ...` to profile only this region.
        if args.profile:
            torch.cuda.profiler.start()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        latencies_ms = []
        for _ in range(args.iters):
            start.record()
            step()
            end.record()
            end.synchronize()
            latencies_ms.append(start.elapsed_time(end))

        if args.profile:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()

    return latencies_ms


def _sharding_summary(module, cfg) -> str:
    """Describe how the experts are actually placed on each rank.

    NOTE: ``module.n_local_experts`` is only the model's weight-loading
    bookkeeping (``n_routed_experts // tp_size``) and does NOT reflect the
    real layout. We read it from the experts submodule instead:
      * EP off (TP): all experts are present per rank, the intermediate dim is
        TP-sharded -> e.g. "256 experts (full count), inter 1024 (1/2 of 2048)".
      * EP on: experts are count-sharded, full intermediate dim.
    """
    experts = module.experts
    full = cfg.n_routed_experts
    # FusedMoE exposes local_num_experts; the MegaMoE experts use num_local_experts.
    local = getattr(experts, "local_num_experts", None)
    if local is None:
        local = getattr(experts, "num_local_experts", "?")
    moe_cfg = getattr(experts, "moe_config", None)
    inter_full = cfg.moe_intermediate_size
    inter_local = getattr(moe_cfg, "intermediate_size_per_partition", inter_full)
    count = f"{local} experts (of {full})" if local != full else f"all {full} experts"
    width = (
        f", inter {inter_local} (of {inter_full})" if inter_local != inter_full else ""
    )
    return count + width


def _report(args, vllm_config, module, dtype, world_size, results):
    """Print the (constant) config header once, then one row per token count.

    ``results`` is a list of ``(global_tokens, rank0_tokens, latencies_ms)``.
    """
    cfg = vllm_config.model_config.hf_config
    ep = "on" if args.enable_expert_parallel else "off"
    dp = args.data_parallel_size
    tp = args.tensor_parallel_size
    expert_dtype = getattr(cfg, "expert_dtype", "n/a")
    # Effective expert precision: resolved expert_dtype + the concrete MoE quant
    # method class instantiated (e.g. Fp8MoEMethod, Mxfp4MoEMethod).
    moe_method = type(getattr(module.experts, "quant_method", module.experts)).__name__
    print("=" * 72)
    print("DeepseekV4MoE standalone benchmark")
    print("-" * 72)
    print(f"  model                : {args.model}")
    print(f"  moe_backend          : {vllm_config.kernel_config.moe_backend}")
    print(f"  expert precision     : {expert_dtype} ({moe_method})")
    print(f"  parallelism          : TP={tp} DP={dp} (world={world_size})")
    print(f"  expert_parallel      : {ep}")
    if args.enable_expert_parallel and world_size > 1:
        # Report the resolved backend (vLLM rewrites removed values like naive).
        print(f"  all2all_backend      : {vllm_config.parallel_config.all2all_backend}")
    print(f"  experts/rank         : {_sharding_summary(module, cfg)}")
    print(f"  layer_idx            : {args.layer_idx}")
    print(f"  hidden_size          : {cfg.hidden_size}")
    print(f"  n_routed_experts     : {cfg.n_routed_experts}")
    print(f"  num_experts_per_tok  : {cfg.num_experts_per_tok}")
    print(f"  moe_intermediate_size: {cfg.moe_intermediate_size}")
    print(f"  dtype                : {dtype}")
    print(f"  cuda_graph           : {'on' if args.cuda_graph else 'off'}")
    print(f"  iters                : {args.iters} (warmup {args.warmup})")
    print("-" * 72)

    # Results table (one row per swept token count).
    cols = ("tokens", "mean(ms)", "p50(ms)", "min(ms)", "tok/s")
    print(f"  {cols[0]:>9}  {cols[1]:>9}  {cols[2]:>9}  {cols[3]:>9}  {cols[4]:>14}")
    for global_tokens, _rank0_tokens, latencies_ms in results:
        lat = torch.tensor(latencies_ms)
        mean_ms = lat.mean().item()
        tps = global_tokens / (mean_ms / 1e3)
        print(
            f"  {global_tokens:>9}  {mean_ms:>9.4f}  {lat.median().item():>9.4f}  "
            f"{lat.min().item():>9.4f}  {tps:>14,.0f}"
        )
    if world_size > 1:
        print("-" * 72)
        print("  throughput is global (all DP groups run concurrently);")
        print("  latency is rank 0's; collectives keep ranks in lockstep.")
        if dp > 1:
            print(f"  (DP={dp}: each 'tokens' value is split evenly across DP groups)")
    print("=" * 72)

    if args.output:
        _write_results_jsonl(args, vllm_config, module, dtype, world_size, results)
        print(f"  wrote results -> {args.output}")


def _write_results_jsonl(args, vllm_config, module, dtype, world_size, results):
    """Append one JSON record per swept token count to ``args.output``.

    JSONL keeps multiple runs (different backends / parallelism) accumulating in
    one file that the plotting script aggregates. Called only on rank 0.
    """
    cfg = vllm_config.model_config.hf_config
    moe_method = type(getattr(module.experts, "quant_method", module.experts)).__name__
    base = {
        "model": args.model,
        "moe_backend": args.moe_backend,
        "resolved_moe_backend": vllm_config.kernel_config.moe_backend,
        "expert_dtype": getattr(cfg, "expert_dtype", None),
        "moe_method": moe_method,
        "tp": args.tensor_parallel_size,
        "dp": args.data_parallel_size,
        "world_size": world_size,
        "expert_parallel": bool(args.enable_expert_parallel),
        "all2all_backend": vllm_config.parallel_config.all2all_backend,
        "cuda_graph": bool(args.cuda_graph),
        "dtype": str(dtype),
        "layer_idx": args.layer_idx,
        "iters": args.iters,
        "warmup": args.warmup,
    }
    with open(args.output, "a") as f:
        for global_tokens, rank0_tokens, latencies_ms in results:
            lat = torch.tensor(latencies_ms)
            rec = {
                **base,
                "num_tokens": global_tokens,
                "rank0_tokens": rank0_tokens,
                "mean_ms": lat.mean().item(),
                "p50_ms": lat.median().item(),
                "min_ms": lat.min().item(),
                "tokens_per_s": global_tokens / (lat.mean().item() / 1e3),
            }
            f.write(json.dumps(rec) + "\n")


def _run_worker(local_rank: int, world_size: int, init_method: str, args) -> None:
    """Per-process entry point: build the module on this rank and benchmark it.

    Used directly for single-GPU runs and via ``mp.spawn`` for TP/EP runs.
    """
    rank = local_rank  # single node: global rank == local rank
    tp_size = args.tensor_parallel_size
    dp_rank = rank // tp_size
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    # Seed identically on every rank so the router gate weights and the hash
    # routing table (both deterministic across ranks) are consistent, which the
    # TP/EP collectives rely on for matching routing decisions.
    torch.manual_seed(args.seed)

    vllm_config = make_vllm_config(args)
    vllm_config.parallel_config.rank = rank % tp_size
    vllm_config.parallel_config.data_parallel_rank = dp_rank
    dtype = vllm_config.model_config.dtype

    with set_current_vllm_config(vllm_config):
        _init_distributed(world_size, rank, local_rank, init_method, tp_size)
        # The modular MoE kernels allocate scratch space from a global
        # workspace manager that the GPU model runner normally initializes.
        init_workspace_manager(device)

        module = build_module(args, vllm_config, device)

        # Sweep the requested token counts, reusing the one built module. Each
        # value is the global batch; distribute it evenly across the DP groups
        # (TP ranks within a group replicate their shard). Seed by dp_rank so
        # every DP group gets distinct tokens while the TP ranks in a group stay
        # bit-identical (required for consistent routing/collectives).
        results = []
        for total_tokens in args.num_tokens:
            num_tokens = _tokens_for_dp_rank(
                total_tokens, dp_rank, args.data_parallel_size
            )
            torch.manual_seed(args.seed + dp_rank)
            hidden_states, input_ids = make_inputs(
                num_tokens, vllm_config, dtype, device
            )
            if world_size > 1:
                torch.distributed.barrier()
            latencies_ms = benchmark(
                args, vllm_config, module, hidden_states, input_ids
            )
            results.append((total_tokens, num_tokens, latencies_ms))

        if rank == 0:
            _report(args, vllm_config, module, dtype, world_size, results)

    # Best-effort cleanup of the distributed environment.
    with contextlib.suppress(Exception):
        cleanup_dist_env_and_memory()


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA device.")
    if args.tensor_parallel_size < 1 or args.data_parallel_size < 1:
        raise SystemExit("--tensor-parallel-size / --data-parallel-size must be >= 1")
    world_size = args.tensor_parallel_size * args.data_parallel_size
    num_gpus = torch.cuda.device_count()
    if world_size > num_gpus:
        raise SystemExit(
            f"Requested TP={args.tensor_parallel_size} x DP={args.data_parallel_size} "
            f"= {world_size} ranks but only {num_gpus} CUDA device(s) are visible."
        )
    if args.data_parallel_size > 1 and min(args.num_tokens) < args.data_parallel_size:
        raise SystemExit(
            f"every --num-tokens value must be >= --data-parallel-size "
            f"({args.data_parallel_size}) so each DP rank gets >=1 token; got "
            f"{args.num_tokens}."
        )

    init_method = f"tcp://127.0.0.1:{get_open_port()}"
    if world_size == 1:
        _run_worker(0, 1, init_method, args)
    else:
        # 'spawn' (torch.multiprocessing default) is required for CUDA.
        import torch.multiprocessing as mp

        mp.spawn(
            _run_worker,
            args=(world_size, init_method, args),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Standalone benchmark for the DeepseekV4MoE module."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-V4-Flash",
        help="Model path/repo to source the HF config from (no weights loaded).",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--moe-backend",
        type=str,
        default="auto",
        help="kernel_config.moe_backend, e.g. 'auto' or 'deep_gemm_mega_moe'.",
    )
    parser.add_argument(
        "--expert-dtype",
        type=str,
        default="auto",
        choices=["auto", "fp4", "fp8"],
        help="Override DeepSeek-V4's expert weight precision via hf_overrides. "
        "'auto' keeps the checkpoint's value; 'fp4' = MXFP4/NVFP4 experts; "
        "'fp8' = block-FP8 experts (unlocks the FP8 MoE backends: deep_gemm, "
        "cutlass, flashinfer_*, marlin, triton). Weights are dummy, so this "
        "only changes which kernel runs, not accuracy.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=1,
        help="TP degree. Dense weights are sharded across these ranks.",
    )
    parser.add_argument(
        "--data-parallel-size",
        "-dp",
        type=int,
        default=1,
        help="DP degree. The global --num-tokens batch is split evenly across "
        "the DP groups (each TP replica gets its own shard). Total ranks = "
        "TP x DP, launched one process per rank via mp.spawn.",
    )
    parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        help="Shard experts (EP) across the whole TP x DP world instead of "
        "TP-sharding them. Required by the deep_gemm_mega_moe backend, and the "
        "usual way to scale DP for MoE (tokens dispatched via all-to-all).",
    )
    parser.add_argument(
        "--all2all-backend",
        type=str,
        default="allgather_reducescatter",
        choices=[
            "allgather_reducescatter",
            "deepep_high_throughput",
            "deepep_low_latency",
            "mori",
            "nixl_ep",
            "flashinfer_nvlink_two_sided",
            "flashinfer_nvlink_one_sided",
        ],
        help="EP dispatch/combine backend (only used with "
        "--enable-expert-parallel and >1 EP rank). Default "
        "'allgather_reducescatter' needs no extra deps; the others require "
        "external kernels (DeepEP / nixl / mori / flashinfer-comm). See "
        "deepseek_v4_moe_backends.md.",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=3,
        help="Layer index used to build the prefix; selects hash vs. routed MoE.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[1024],
        metavar="N",
        help="Token count(s) to benchmark; pass several to sweep, e.g. "
        "--num-tokens 64 256 1024 4096. Each value is the global batch (split "
        "across DP groups). The module is built once and reused across values.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16"],
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Call cudaProfilerStart/Stop around the measured iterations "
        "(warmup excluded), for use under nsys/ncu --capture-range=cudaProfilerApi.",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture the MoE forward into a CUDA graph and time graph replays "
        "instead of eager launches. Removes per-kernel CPU launch overhead "
        "(big win at small token counts). Not every backend is graph-capturable; "
        "'--moe-backend triton' is the safest. Errors are reported clearly.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Append results as JSON lines (one record per swept token count) to "
        "PATH, in addition to printing. Accumulate many runs (different backends "
        "/ parallelism) into one file for the plotting script.",
    )
    main(parser.parse_args())
