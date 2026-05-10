# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Standalone benchmark for ``DeepseekV4MultiHeadLatentAttentionWrapper``.

The wrapper is the production attention layer used by DeepSeek-V4. This script
constructs a single layer in isolation (no engine, no scheduler, no real
weights), wires up dummy KV caches and metadata that match what the real
runner would build, then times the wrapper's ``forward(positions, hidden_states)``.

It supports the three layer variants the model uses:

    - ``swa``    (compress_ratio=1)   :: SWA-only, no compressor, no indexer.
    - ``c4a``    (compress_ratio=4)   :: SWA + compressor + Lightning Indexer.
    - ``c128a``  (compress_ratio=128) :: SWA + compressor.

It supports prefill / decode / mixed workloads and a configurable batch size /
context length / prefill query length.

Example:

    .venv/bin/python benchmarks/kernels/benchmark_deepseek_v4_attention.py \\
        --variant c4a --workload decode --batch-size 32 --ctx-len 16384

Note: weights are randomly initialized and ``weight_scale_inv`` tensors are set
to ones, so the *numerical* output is meaningless. Only the *timings* are.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from vllm.config import (
    CacheConfig,
    CompilationConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.models.deepseek_v4 import DeepseekV4Attention
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.workspace import init_workspace_manager


# V4 Flash
# DEFAULT_HF_CONFIG = (
#     "/mnt/lustre/hf-models/hub/models--deepseek-ai--DeepSeek-V4-Flash/"
#     "snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136/config.json"
# )

# V4 Pro
DEFAULT_HF_CONFIG = (
    "/mnt/lustre/hf-models/hub/models--deepseek-ai--DeepSeek-V4-Pro/"
    "snapshots/45040942eb0d1c4e29fa6b92a6195f110e9e7444/config.json"
)


# ---------------------------------------------------------------------------
# VllmConfig construction
# ---------------------------------------------------------------------------


def _load_hf_config(path: str) -> DeepseekV4Config:
    with open(path) as f:
        raw = json.load(f)
    # DeepseekV4Config accepts arbitrary kwargs through PretrainedConfig.
    # Rope: model code reads both ``rope_scaling`` and ``rope_parameters``.
    rope_scaling = raw.get("rope_scaling")
    if rope_scaling is not None and "rope_type" not in rope_scaling:
        rope_scaling = {**rope_scaling, "rope_type": rope_scaling.get("type", "yarn")}
    raw["rope_scaling"] = rope_scaling
    cfg = DeepseekV4Config(**raw)
    # Make sure rope_parameters mirrors rope_scaling so the model code can
    # mutate it in-place (see DeepseekV4Attention.__init__).
    cfg.rope_parameters = dict(rope_scaling) if rope_scaling else {"rope_type": "default"}
    cfg.rope_parameters.setdefault("rope_theta", raw.get("rope_theta", 10000.0))
    return cfg


def _build_vllm_config(
    hf_config: DeepseekV4Config,
    *,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    block_size: int,
    num_gpu_blocks: int,
) -> VllmConfig:
    """Assemble a minimal VllmConfig that exercises the production code paths."""

    # ModelConfig: skip HF download by pointing at a temporary local dir holding
    # only the config we need. The model is never actually loaded.
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="vllm_dsv4_bench_")
    try:
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump(hf_config.to_dict(), f)
        model_config = ModelConfig(
            model=tmp,
            tokenizer=None,
            tokenizer_mode="auto",
            trust_remote_code=True,
            dtype="bfloat16",
            seed=0,
            max_model_len=max_model_len,
            quantization=None,
            enforce_eager=True,
            skip_tokenizer_init=True,
            served_model_name=None,
            limit_mm_per_prompt=None,
            config_format="auto",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Override with our parsed config so the attribute set is exactly what the
    # production model code expects.
    model_config.hf_config = hf_config
    model_config.hf_text_config = hf_config

    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="fp8_ds_mla",
        enable_prefix_caching=False,
    )
    cache_config.num_gpu_blocks = num_gpu_blocks
    cache_config.num_cpu_blocks = 0

    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        enable_chunked_prefill=True,
    )

    parallel_config = ParallelConfig(tensor_parallel_size=1)
    device_config = DeviceConfig()
    load_config = LoadConfig(load_format="dummy")
    compilation_config = CompilationConfig()

    # Match DeepSeek-V4-Pro: block-fp8 with [128,128] scales.
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        load_config=load_config,
        compilation_config=compilation_config,
        quant_config=quant_config,
    )
    return vllm_config


# ---------------------------------------------------------------------------
# Layer initialization
# ---------------------------------------------------------------------------


def _pick_layer_id(hf_config: DeepseekV4Config, variant: str) -> int:
    ratios = list(hf_config.compress_ratios)
    target = {"swa": 0, "c4a": 4, "c128a": 128}[variant]
    for i, r in enumerate(ratios):
        if r == target:
            return i
    raise ValueError(
        f"No layer with compress_ratio={target} in compress_ratios={ratios!r}"
    )


def _randomize_weights(module: nn.Module, dtype: torch.dtype, device: torch.device):
    """Fill all parameters with sane random data.

    For FP8 block-quantized linear layers, we leave ``weight`` (uint8/fp8) as
    is after creation and set ``weight_scale_inv`` to ones so the dequantized
    output stays bounded. Other parameters get small normal noise.
    """
    for name, param in module.named_parameters(recurse=True):
        if param.device != device:
            param.data = param.data.to(device)
        if "weight_scale_inv" in name or name.endswith("weight_scale"):
            param.data.fill_(1.0)
        elif param.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8):
            # FP8 block weight: a small, deterministic spread.
            with torch.no_grad():
                tmp = torch.randn(param.shape, dtype=torch.float32, device=device) * 0.02
                param.data.copy_(tmp.to(param.dtype))
        elif "attn_sink" in name:
            # Leave the -inf padding alone, fill the head slots with zeros so
            # the sink is a no-op (matches a trained sink that never fires).
            with torch.no_grad():
                param.data.zero_()
        elif param.dtype in (torch.float32, torch.float16, torch.bfloat16):
            with torch.no_grad():
                param.data.normal_(0.0, 0.02).to(dtype=param.dtype)
        # else: integer buffers / lookup tables -> leave alone


def _process_weights_after_loading(layer: nn.Module):
    for sub in layer.modules():
        if isinstance(sub, LinearBase) and hasattr(sub, "quant_method"):
            qm = sub.quant_method
            if qm is not None and hasattr(qm, "process_weights_after_loading"):
                try:
                    qm.process_weights_after_loading(sub)
                except Exception as exc:  # noqa: BLE001
                    # Some FP8 paths require specific GPU capabilities; if we
                    # can't run them, fall back to leaving the raw weights.
                    print(
                        f"[warn] process_weights_after_loading failed on "
                        f"{type(sub).__name__}: {exc}"
                    )


def _build_attention_layer(
    vllm_config: VllmConfig,
    layer_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[DeepseekV4Attention, torch.Tensor, list[torch.cuda.Stream] | None]:
    aux_streams = (
        None if current_platform.is_rocm() else [torch.cuda.Stream() for _ in range(3)]
    )
    # -1 marks "invalid token" in the sparse-attention indexer; using zeros or
    # uninitialized memory makes downstream kernels dereference random slot ids.
    topk_indices_buffer = torch.full(
        (
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.model_config.hf_config.index_topk,
        ),
        -1,
        dtype=torch.int32,
        device=device,
    )

    prefix = f"model.layers.{layer_id}"
    # `with torch.device(...)` mirrors the production model loader: default-device
    # tensor allocations (e.g. RoPE tables) go to CUDA so they match buffers we
    # explicitly place on CUDA, avoiding cross-device einsum errors.
    with set_current_vllm_config(vllm_config), torch.device(device):
        layer = DeepseekV4Attention(
            vllm_config=vllm_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_streams,
        ).to(device=device)

    # Cast trainable bf16 params to the requested dtype while leaving
    # quantized buffers alone.
    for p in layer.parameters():
        if p.dtype in (torch.float16, torch.bfloat16, torch.float32):
            p.data = p.data.to(device=device)
        else:
            p.data = p.data.to(device=device)

    _randomize_weights(layer, dtype, device)

    with set_current_vllm_config(vllm_config):
        _process_weights_after_loading(layer)

    return layer, topk_indices_buffer, aux_streams


# ---------------------------------------------------------------------------
# KV cache allocation
# ---------------------------------------------------------------------------


@dataclass
class CacheLayerInfo:
    prefix: str
    layer: Any
    spec: Any
    backend_cls: Any
    block_size: int
    head_size: int
    dtype: torch.dtype


def _collect_cache_layers(
    vllm_config: VllmConfig,
) -> list[CacheLayerInfo]:
    """Walk static_forward_context and return all KV-cache-bearing layers."""
    out: list[CacheLayerInfo] = []
    for prefix, sub in vllm_config.compilation_config.static_forward_context.items():
        spec_fn = getattr(sub, "get_kv_cache_spec", None)
        if spec_fn is None:
            continue
        try:
            spec = spec_fn(vllm_config)
        except Exception:
            continue
        if spec is None:
            continue
        backend = sub.get_attn_backend()
        out.append(
            CacheLayerInfo(
                prefix=prefix,
                layer=sub,
                spec=spec,
                backend_cls=backend,
                block_size=spec.block_size,
                head_size=spec.head_size,
                dtype=spec.dtype,
            )
        )
    return out


def _allocate_kv_caches(
    cache_layers: list[CacheLayerInfo],
    num_blocks: int,
    cache_dtype_str: str,
    device: torch.device,
):
    """Allocate one KV cache per layer.

    When the spec has an alignment requirement (``page_size_padded``), we
    allocate the padded-byte block and expose the cache as a strided view, the
    same way ``GPUModelRunner._reshape_kv_cache_tensors`` does. FlashMLA's
    sparse decode kernel asserts ``stride_kv_block % 576 == 0``.
    """
    from vllm.utils.torch_utils import get_dtype_size

    for info in cache_layers:
        spec = info.spec
        # storage_block_size differs from block_size for compressed MLA caches.
        shape_block_size = getattr(spec, "storage_block_size", info.block_size)
        shape = info.backend_cls.get_kv_cache_shape(
            num_blocks=num_blocks,
            block_size=shape_block_size,
            num_kv_heads=1,
            head_size=info.head_size,
            cache_dtype_str=cache_dtype_str,
        )
        page_size_padded = getattr(spec, "page_size_padded", None)
        if page_size_padded is None:
            info.layer.kv_cache = torch.zeros(shape, dtype=info.dtype, device=device)
            continue

        # Padded path: allocate raw bytes, then expose a strided view that
        # advances by the padded page size per block.
        dtype_size = get_dtype_size(info.dtype)
        assert page_size_padded % dtype_size == 0, (page_size_padded, dtype_size)
        page_stride = page_size_padded // dtype_size
        # Initialize fp8 caches to a deterministic non-zero pattern so dequant
        # yields finite values (zero scales would produce NaN downstream).
        if info.dtype == torch.uint8:
            raw = torch.full(
                (num_blocks * page_stride,), 1, dtype=info.dtype, device=device
            )
        else:
            raw = torch.zeros(num_blocks * page_stride, dtype=info.dtype, device=device)
        strides = list(torch.empty(shape).stride())
        # Stride along the block (first) dim is the padded page size.
        strides[0] = page_stride
        info.layer.kv_cache = torch.as_strided(raw, size=shape, stride=tuple(strides))


# ---------------------------------------------------------------------------
# Workload / metadata builders
# ---------------------------------------------------------------------------


@dataclass
class Workload:
    """Per-request (seq_len, query_len). Decode tokens come first."""

    seq_lens: list[int]
    query_lens: list[int]

    @property
    def total_tokens(self) -> int:
        return sum(self.query_lens)


def _make_workload(workload: str, batch_size: int, ctx_len: int, q_len: int) -> Workload:
    if workload == "decode":
        seqs = [ctx_len] * batch_size
        qs = [1] * batch_size
    elif workload == "prefill":
        # Pure prefill: every request brings ``q_len`` fresh tokens of context.
        seqs = [q_len] * batch_size
        qs = [q_len] * batch_size
    elif workload == "mixed":
        n_dec = max(1, batch_size - 1)
        seqs = [ctx_len] * n_dec + [q_len]
        qs = [1] * n_dec + [q_len]
    else:
        raise ValueError(f"unknown workload: {workload}")
    return Workload(seq_lens=seqs, query_lens=qs)


def _make_common_attn_metadata(
    wl: Workload,
    block_size: int,
    device: torch.device,
    num_gpu_blocks: int = 1 << 30,
) -> CommonAttentionMetadata:
    """Build a common attention metadata tied to ``block_size``.

    Each cache layer in DeepseekV4 has its own block size (SWA=64, indexer
    state caches=4/8, FlashMLA-sparse=256). Production runs put each into a
    separate KV cache group with its own block_table. Here we build one
    CommonAttentionMetadata per (block_size) on demand.
    """
    batch_size = len(wl.seq_lens)
    seq_lens_cpu = torch.tensor(wl.seq_lens, dtype=torch.int32)
    seq_lens = seq_lens_cpu.to(device)
    query_lens = torch.tensor(wl.query_lens, dtype=torch.int32)
    qsl_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
    qsl_cpu[1:] = torch.cumsum(query_lens, dim=0)
    qsl = qsl_cpu.to(device)

    max_blocks = (max(wl.seq_lens) + block_size - 1) // block_size
    # Wrap into the cache so we don't index past num_gpu_blocks. Caches with
    # small ``block_size`` (e.g. compressor state caches with block_size=4)
    # would otherwise overflow when ctx_len * batch_size exceeds num_gpu_blocks.
    block_table = (
        torch.arange(batch_size * max_blocks, dtype=torch.int32, device=device)
        % max(num_gpu_blocks, 1)
    ).view(batch_size, max_blocks)

    # Slot mapping: place each query token at (kv_idx within the request).
    slots: list[int] = []
    for i, (s, q) in enumerate(zip(wl.seq_lens, wl.query_lens)):
        ctx = s - q
        for j in range(q):
            kv_idx = ctx + j
            blk = kv_idx // block_size
            off = kv_idx % block_size
            slots.append(int(block_table[i, blk].item() * block_size + off))
    slot_mapping = torch.tensor(slots, dtype=torch.int64, device=device)

    # Positions: same as kv idx (post-context offset within the request).
    positions: list[int] = []
    for s, q in zip(wl.seq_lens, wl.query_lens):
        ctx = s - q
        for j in range(q):
            positions.append(ctx + j)
    positions_t = torch.tensor(positions, dtype=torch.int64, device=device)

    num_computed_tokens_cpu = torch.tensor(
        [s - q for s, q in zip(wl.seq_lens, wl.query_lens)], dtype=torch.int32
    )

    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=batch_size,
        num_actual_tokens=wl.total_tokens,
        max_query_len=int(query_lens.max().item()),
        max_seq_len=int(seq_lens_cpu.max().item()),
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=True,
        positions=positions_t,
    )


def _build_metadata_dict(
    vllm_config: VllmConfig,
    cache_layers: list[CacheLayerInfo],
    workload: "Workload",
    device: torch.device,
    num_gpu_blocks: int,
) -> tuple[dict[str, Any], dict[int, CommonAttentionMetadata]]:
    """Run every backend's metadata builder and key the results by layer prefix.

    Each cache layer can have its own ``block_size`` (SWA=64, indexer state
    caches=4 or 8, FlashMLA sparse=256). We build a fresh CommonAttentionMetadata
    per distinct block size so the block_table and slot_mapping each backend
    consumes are sized for its own pages.
    """
    builders: dict[tuple[type, int], Any] = {}
    commons: dict[int, CommonAttentionMetadata] = {}

    def _common_for(block_size: int) -> CommonAttentionMetadata:
        if block_size not in commons:
            commons[block_size] = _make_common_attn_metadata(
                workload, block_size, device, num_gpu_blocks=num_gpu_blocks
            )
        return commons[block_size]

    def _get_builder(info: CacheLayerInfo):
        cls = info.backend_cls.get_builder_cls()
        key = (cls, info.block_size)
        if key in builders:
            return builders[key]
        builder = cls(
            kv_cache_spec=info.spec,
            layer_names=[info.prefix],
            vllm_config=vllm_config,
            device=device,
        )
        builders[key] = builder
        return builder

    md: dict[str, Any] = {}
    for info in cache_layers:
        builder = _get_builder(info)
        common = _common_for(info.block_size)
        md[info.prefix] = builder.build(common_prefix_len=0, common_attn_metadata=common)
    return md, commons


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------


def _bench(
    fn,
    *,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    samples.sort()
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p10_ms": samples[int(0.1 * len(samples))],
        "p90_ms": samples[min(len(samples) - 1, int(0.9 * len(samples)))],
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_HF_CONFIG, help="Path to model config.json")
    parser.add_argument(
        "--variant",
        choices=["swa", "c4a", "c128a"],
        default="c128a",
        help="Layer variant: swa (compress_ratio=1), c4a (=4), c128a (=128).",
    )
    parser.add_argument(
        "--workload", choices=["decode", "prefill", "mixed"], default="decode"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--ctx-len",
        type=int,
        default=8192,
        help="Per-request KV-cache context length (decode/mixed).",
    )
    parser.add_argument(
        "--q-len",
        type=int,
        default=4096,
        help="Per-request query length for prefill/mixed.",
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-gpu-blocks", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16"], default="bfloat16"
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph capture; time eager kernel launches instead.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # FP8 quant kernels read torch.get_default_dtype() to set out_dtype.
    torch.set_default_dtype(dtype)

    hf_config = _load_hf_config(args.config)
    if args.max_model_len > hf_config.max_position_embeddings:
        args.max_model_len = hf_config.max_position_embeddings

    # max_num_batched_tokens must accommodate the largest single batch we'll run.
    max_total_tokens = max(
        args.max_num_batched_tokens,
        args.batch_size * max(args.q_len, 1) + args.q_len,
    )

    vllm_config = _build_vllm_config(
        hf_config,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max_total_tokens,
        max_num_seqs=max(args.max_num_seqs, args.batch_size),
        block_size=args.block_size,
        num_gpu_blocks=args.num_gpu_blocks,
    )

    # Single-rank distributed env: required by ColumnParallelLinear /
    # RowParallelLinear. ``initialize_model_parallel`` reads the current
    # vllm_config, so it must be invoked under ``set_current_vllm_config``.
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(delete=False).name
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{temp_file}",
        local_rank=0,
        backend="nccl",
    )
    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
    init_workspace_manager(device)

    layer_id = _pick_layer_id(hf_config, args.variant)
    print(
        f"[setup] variant={args.variant} layer_id={layer_id} "
        f"compress_ratio={hf_config.compress_ratios[layer_id]} "
        f"workload={args.workload} batch={args.batch_size} "
        f"ctx={args.ctx_len} q_len={args.q_len}"
    )

    layer, _, _ = _build_attention_layer(vllm_config, layer_id, device, dtype)

    cache_layers = _collect_cache_layers(vllm_config)
    print(f"[setup] cache layers in static_forward_context: {len(cache_layers)}")
    for info in cache_layers:
        print(
            f"  - {info.prefix}: block_size={info.block_size} "
            f"head_size={info.head_size} dtype={info.dtype} "
            f"backend={info.backend_cls.__name__}"
        )
    _allocate_kv_caches(cache_layers, args.num_gpu_blocks, "fp8_ds_mla", device)

    workload = _make_workload(args.workload, args.batch_size, args.ctx_len, args.q_len)

    with set_current_vllm_config(vllm_config):
        attn_metadata, commons = _build_metadata_dict(
            vllm_config, cache_layers, workload, device, args.num_gpu_blocks
        )

    hidden = torch.randn(
        workload.total_tokens, hf_config.hidden_size, dtype=dtype, device=device
    )
    # Positions are independent of cache block size; reuse from any common.
    positions = next(iter(commons.values())).positions
    assert positions is not None

    # Pre-allocate the output once so CUDA graph capture sees a stable address.
    out_buf = torch.empty_like(hidden)

    def step():
        with set_forward_context(attn_metadata=attn_metadata, vllm_config=vllm_config):
            res = layer.mla_attn(positions, hidden, None)
        out_buf.copy_(res)
        return out_buf

    # Surface wiring errors eagerly + warm up FlashMLASchedMeta planner so
    # `have_initialized=True` before capture (the planner mutates buffers on
    # first call; capturing that mutation breaks replay).
    out = step()
    assert out.shape == hidden.shape, (out.shape, hidden.shape)
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    if args.enforce_eager:
        bench_fn = step
        mode = "eager"
    else:
        graph = torch.cuda.CUDAGraph()
        # `torch.cuda.graph` makes caching allocations within the context use a
        # private memory pool tied to the graph. Aux streams declared on the
        # wrapper are attached automatically when they sync with default.
        with torch.cuda.graph(graph):
            step()
        bench_fn = graph.replay
        mode = "cudagraph"

    stats = _bench(bench_fn, warmup=2, iters=args.iters)
    print()
    print(
        f"[result] variant={args.variant} workload={args.workload} "
        f"batch={args.batch_size} tokens={workload.total_tokens} mode={mode}"
    )
    for k, v in stats.items():
        print(f"  {k:>10}: {v:.4f}")
    tps = workload.total_tokens / (stats["mean_ms"] / 1000.0)
    print(f"  {'tok/s':>10}: {tps:,.1f}")


if __name__ == "__main__":
    main()
