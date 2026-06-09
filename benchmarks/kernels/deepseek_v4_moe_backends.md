# DeepSeek-V4 MoE: Backend & Precision Options

A survey of the MoE kernel backends, expert-parallel all-to-all backends, and
weight-precision options available in this vLLM tree, oriented toward the
`benchmark_deepseek_v4_moe.py` standalone benchmark. For each option it
documents what it does, what it requires (hardware, libraries, precision), and
how you would actually turn it on.

> **Scope & confidence.** Enum values, defaults, and the DeepSeek-V4 dispatch
> logic below were verified directly against the source (paths cited inline).
> The finer per-backend hardware/precision matrices were gathered by a codebase
> survey and should be treated as a strong guide rather than gospel — confirm
> the exact `get_min_capability()` / `is_supported_config()` for a given backend
> before relying on it. All paths are relative to the repo root.
>
> **Reminder for the benchmark itself.** The script loads weights as dummy
> (`load_format="dummy"`), so any precision below measures **kernel
> latency/throughput only, not correctness**. Switching precision changes which
> kernel runs (and whether it runs at all on a given GPU), which is exactly what
> we want to measure.

---

## 0. How the benchmark selects a backend today

The script exposes these relevant knobs (see `benchmark_deepseek_v4_moe.py`):

| Flag | Maps to | Effect |
| --- | --- | --- |
| `--moe-backend` | `kernel_config.moe_backend` (`EngineArgs(moe_backend=...)`) | Picks the MoE execution path; `auto` lets vLLM choose. |
| `--expert-dtype` | `hf_overrides={"expert_dtype": ...}` | Override expert weight precision: `fp4` (MXFP4/NVFP4) or `fp8` (block-FP8). Selects which precision's backend set is available; see §3. |
| `--enable-expert-parallel` | `parallel_config.enable_expert_parallel` | Switches experts from TP-sharded to EP-sharded; required for `deep_gemm_mega_moe`. |
| `--all2all-backend` | `parallel_config.all2all_backend` (`EngineArgs(all2all_backend=...)`) | EP dispatch/combine backend; see §2. Only matters with EP + >1 rank. |
| `--tensor-parallel-size` / `--data-parallel-size` | `parallel_config` | Multi-process TP / DP; see the script docstring. |
| `--dtype` | `model_config.dtype` | Activation/compute dtype (bf16/fp16). Does **not** change *weight* quantization. |
| `--cuda-graph` | `torch.cuda.CUDAGraph` capture/replay | Times graph replays instead of eager launches — removes per-kernel CPU launch overhead (launch-bound at small token counts). Not all backends are graph-capturable (`triton` is safest); capture errors are reported clearly. |

What it does **not** expose yet: arbitrary `--quantization` / general
`hf_overrides` (only `expert_dtype` is wired, via `--expert-dtype`), and the
various `VLLM_*` env vars (set those directly in the environment). Those gaps
are called out per option.

The **weight precision** for DeepSeek-V4 is driven by the checkpoint's
`config.json` (`expert_dtype` + `quantization_config`), resolved in
`vllm/models/deepseek_v4/quant_config.py`. See §3.

---

## 1. MoE kernel backends (`--moe-backend`)

### 1a. The explicit `moe_backend` config values

`MoEBackend` is a `Literal` in `vllm/config/kernel.py` (default `"auto"`):

```
auto, triton, deep_gemm, deep_gemm_mega_moe, cutlass,
flashinfer_trtllm, flashinfer_cutlass, flashinfer_cutedsl, flashinfer_b12x,
marlin, humming, triton_unfused, aiter, emulation
```

Values are normalized as `value.lower().replace("-", "_")`. `auto` defers to
vLLM's per-precision "oracle" (below), which is the recommended starting point.

### 1b. The runtime "oracle" backends (what `auto` chooses between)

vLLM picks a concrete kernel per **weight precision** via selector enums in
`vllm/model_executor/layers/fused_moe/oracle/`. The relevant ones:

- **Unquantized** (`oracle/unquantized.py`): `FlashInfer TRTLLM`, `FlashInfer
  CUTLASS`, `ROCm AITER`, `TRITON`, `BATCHED_TRITON`, plus CPU/XPU/TPU.
- **FP8** (`oracle/fp8.py`): `FLASHINFER_TRTLLM`, `FLASHINFER_CUTLASS`,
  `DEEPGEMM`, `BATCHED_DEEPGEMM`, `MARLIN`, `TRITON`, `BATCHED_TRITON`, `AITER`,
  `VLLM_CUTLASS`, `BATCHED_VLLM_CUTLASS`.
- **NVFP4** (`oracle/nvfp4.py`): `FLASHINFER_TRTLLM`, `FLASHINFER_CUTLASS`,
  `FLASHINFER_CUTEDSL` (+ batched), `FLASHINFER_B12X`, `VLLM_CUTLASS`,
  `MARLIN`, `EMULATION`.
- **MXFP4** (`oracle/mxfp4.py`): `DEEPGEMM_MXFP4`, `FLASHINFER_TRTLLM_MXFP4_*`,
  `FLASHINFER_CUTLASS_MXFP4_*`, `MARLIN`, `BATCHED_MARLIN`, `AITER_MXFP4_*`,
  `TRITON`, `TRITON_UNFUSED`, `HUMMING`, `EMULATION`.
- **WNA16 int4/int8** (`oracle/int_wna16.py`): `MARLIN`, `BATCHED_MARLIN`.
- **W4A8** (`oracle/w4a8.py`): `CUTLASS`.

On our **H200 (SM90)** with the real DeepSeek-V4-Flash config, `auto` selected
**FlashInfer CUTLASS** (unquantized smoke test) and **MARLIN mxfp4** (real fp4
config) — both observed in the benchmark logs.

### 1c. Per-backend caveats

| Backend | GPU | Precision(s) | Needs | Caveats / how to enable |
| --- | --- | --- | --- | --- |
| `triton` | CUDA (broad), ROCm | bf16/fp16, fp8, mxfp4 | Triton (bundled) | Most portable; good baseline. Just `--moe-backend triton`. |
| `triton_unfused` | CUDA | mxfp4 | Triton | Unfused variant for grouped-topk; niche. |
| `flashinfer_cutlass` | SM90/SM100 | bf16/fp16, fp8 (block), nvfp4, mxfp4 | `flashinfer` installed; `VLLM_USE_FLASHINFER_MOE_FP8/FP4/...` to force | Strong on Hopper EP. First-run JIT compile can be slow (we hit this). |
| `flashinfer_trtllm` | SM90/SM100 | bf16/fp16, fp8, nvfp4, mxfp4×{mxfp8,bf16} | `flashinfer`; `VLLM_FLASHINFER_MOE_BACKEND="latency"` selects TRTLLM | Latency-oriented (TensorRT-LLM-GEN). Requires SM 100 currently (https://github.com/vllm-project/vllm/blob/62d6f06e3db276030ae46b2e0b3915674515ca74/vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py#L91) |
| `flashinfer_cutedsl` | CUDA | **nvfp4 only** | `flashinfer` (CuteDSL); `VLLM_FLASHINFER_MOE_BACKEND="masked_gemm"` | FP4-only; pair with an NVFP4 checkpoint. |
| `flashinfer_b12x` | **SM12x** (RTX Pro 6000 / DGX Spark) | nvfp4 | `flashinfer` | Explicit opt-in only; not in `auto`. Wrong GPU here. |
| `cutlass` (`VLLM_CUTLASS`) | CUDA (broad) | fp8 (channel), nvfp4, w4a8 | bundled CUTLASS kernels | Solid fp8/fp4 path without flashinfer. |
| `deep_gemm` | **SM90+** | fp8 block (128×128) | DeepGEMM; `VLLM_MOE_USE_DEEP_GEMM=1` (default on) | Matches the DSV4 linear/block-fp8 layout. |
| `deep_gemm_mega_moe` | **SM100 (B200) only** | fp4 experts + fp8 block | DeepGEMM; `--enable-expert-parallel` | **DSV4 production fp4 path.** Hard-gated to SM100 in `DeepseekV4MegaMoEExperts._check_runtime_supported` (`models/deepseek_v4/nvidia/model.py`) — raises `NotImplementedError` on our H200. Needs hidden/intermediate %128==0. |
| `marlin` / `BATCHED_MARLIN` | SM75+ (≈SM80 for fp4) | fp8, mxfp4, nvfp4, int4/int8 (WNA16) | bundled Marlin | Weight-only-friendly; what `auto` fell back to for fp4 on H200. `VLLM_MXFP4_USE_MARLIN=1`, `VLLM_MARLIN_INPUT_DTYPE`. |
| `humming` | CUDA | mxfp4 (mixed) | Humming kernels; `VLLM_HUMMING_MOE_GEMM_TYPE` | Mixed-precision; niche. |
| `aiter` | **ROCm** (gfx942/950) | bf16/fp16, mxfp4 W4A16/W4A8/W4A4 | AITER; `VLLM_ROCM_USE_AITER_MOE=1` | AMD only — N/A on this NVIDIA box. |
| `emulation` | any | dequant→bf16 | — | Correctness/fallback only; dequantizes then runs a dense GEMM. Useful to run an fp4/nvfp4 config on a GPU lacking the real kernel (slow, but it runs). |

**Relevant env vars** (`vllm/envs.py`): `VLLM_MOE_USE_DEEP_GEMM`,
`VLLM_USE_FLASHINFER_MOE_FP16/FP8/FP4/INT4`, `VLLM_FLASHINFER_MOE_BACKEND`
(`throughput`→CUTLASS / `latency`→TRTLLM / `masked_gemm`→CuteDSL),
`VLLM_MXFP4_USE_MARLIN`, `VLLM_USE_FLASHINFER_MOE_MXFP4_{MXFP8,BF16}`,
`VLLM_MARLIN_INPUT_DTYPE`, `VLLM_HUMMING_MOE_GEMM_TYPE`,
`VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE`.

> To drive these from the benchmark, either pass `--moe-backend <name>` (already
> wired) or set the relevant `VLLM_*` env var before launching. A couple
> (`flashinfer_b12x`, `deep_gemm_mega_moe`) are explicit opt-in and won't be
> chosen by `auto`.

---

## 2. Expert-parallel All-to-All backends (`all2all_backend`)

These govern how tokens are dispatched/combined across EP ranks. Only relevant
when `--enable-expert-parallel` is set **and** there is more than one EP rank
(i.e. `tp_size * dp_size > 1`). `All2AllBackend` is a `Literal` in
`vllm/config/parallel.py` (**default `allgather_reducescatter`**):

```
naive*, pplx*, deepep_high_throughput, deepep_low_latency, mori, nixl_ep,
allgather_reducescatter, flashinfer_all2allv (alias), flashinfer_nvlink_two_sided,
flashinfer_nvlink_one_sided
```

`*` `naive` and `pplx` are **removed** — they log a warning and fall back to
`allgather_reducescatter` (`parallel.py:424`).

### Per-backend caveats

| Backend | Library / hardware | Best for | Caveats |
| --- | --- | --- | --- |
| `allgather_reducescatter` (default) | none (NCCL) | baseline, single-node | Pure PyTorch/NCCL collectives; always available; what our EP smoke tests used. Internode via NCCL. |
| `deepep_high_throughput` | **DeepEP** (`deep_ep`) + NVLink/NVSHMEM, optional RDMA/IBGDA | prefill / throughput | Must build DeepEP (`tools/ep_kernels/`). Hopper/Blackwell. `VLLM_DEEPEP_BUFFER_SIZE_MB`. |
| `deepep_low_latency` | **DeepEP** + RDMA, optional MNNVL | decode / latency | Build DeepEP. Supports FP8 dispatch (`DEEPEP_QUANT_BLOCK_SHAPE`) and NVFP4 dispatch (`VLLM_DEEPEPLL_NVFP4_DISPATCH`). One of two backends supporting *batched DP MoE*. |
| `nixl_ep` | **nixl_ep** + RDMA/TCP | elastic EP | Dynamic rank join/leave; `VLLM_NIXL_EP_MAX_NUM_RANKS`, side-channel host/port env. Supports FP8 dispatch + batched DP MoE. |
| `mori` | **mori** (`has_mori`) | AMD EP | **ROCm only** (gfx942/950) — N/A on NVIDIA. |
| `flashinfer_nvlink_two_sided` (`flashinfer_all2allv`) | `flashinfer.comm.trtllm_alltoall` + NVLink/MNNVL | single-node low-latency | flashinfer build with the comm module. |
| `flashinfer_nvlink_one_sided` | `flashinfer.comm.trtllm_moe_alltoall` + NVLink | single-node high-throughput | Supports nvfp4/mxfp8/bf16 dispatch. |

### EP/DP interaction rules (verified in `parallel.py`)

- `use_sequence_parallel_moe` (sequence-parallel MoE) is enabled for
  `{allgather_reducescatter, deepep_high_throughput, deepep_low_latency, mori,
  nixl_ep}` **when** EP **and** `tp_size>1` **and** `dp_size>1`.
- `use_batched_dp_moe` (batched DP MoE) is supported **only** by
  `{deepep_low_latency, nixl_ep}` with EP and `dp_size>1`.
- The FlashInfer NVLink backends are not in the sequence-parallel set (different
  interaction model; treat as single-node NVLink paths).

### Using `--all2all-backend` in the benchmark

The benchmark now exposes `--all2all-backend` (passed straight to
`EngineArgs(all2all_backend=...)`). It only takes effect with
`--enable-expert-parallel` **and** more than one EP rank (i.e.
`tp_size * dp_size > 1`); with a single rank there is no dispatch to do. The
resolved backend is printed in the report header (`all2all_backend: …`).

Accepted values: `allgather_reducescatter` (default),
`deepep_high_throughput`, `deepep_low_latency`, `mori`, `nixl_ep`,
`flashinfer_nvlink_two_sided`, `flashinfer_nvlink_one_sided`. (`naive`/`pplx`
are intentionally omitted — vLLM has removed them.)

Per-backend instructions:

- **`allgather_reducescatter` (default, no setup)** — works out of the box on
  any NCCL build; the only meaningful choice on a single node without extra
  kernels installed:
  ```bash
  python benchmarks/kernels/benchmark_deepseek_v4_moe.py \
      --model deepseek-ai/DeepSeek-V4-Flash --trust-remote-code \
      -dp 2 -tp 2 --enable-expert-parallel \
      --all2all-backend allgather_reducescatter \
      --num-tokens 2048 --layer-idx 5
  ```

- **`deepep_high_throughput` / `deepep_low_latency`** — require the DeepEP
  kernels (`deep_ep`) built against NVSHMEM. Build them first:
  ```bash
  # one-time, from the repo root (see tools/ep_kernels/README.md):
  bash tools/ep_kernels/install_python_libraries.sh   # builds + installs deep_ep
  # then:
  python benchmarks/kernels/benchmark_deepseek_v4_moe.py \
      --model deepseek-ai/DeepSeek-V4-Flash --trust-remote-code \
      -dp 2 -tp 2 --enable-expert-parallel \
      --all2all-backend deepep_low_latency \
      --num-tokens 2048 --layer-idx 5
  # optional tuning:
  #   VLLM_DEEPEP_BUFFER_SIZE_MB=2048            (a2a buffer size)
  #   VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL=1        (multi-node NVLink for LL)
  #   VLLM_DEEPEPLL_NVFP4_DISPATCH=1             (NVFP4 dispatch, LL only)
  ```
  `deepep_low_latency` additionally enables vLLM's *batched DP MoE* path
  (`use_batched_dp_moe`) and supports FP8 dispatch. Single-node NVLink works;
  multi-node needs RDMA/IBGDA configured (see the install README).

- **`nixl_ep`** — requires the `nixl_ep` library and RDMA/TCP transport; targets
  *elastic* EP (ranks can join/leave). Also supports batched DP MoE + FP8
  dispatch:
  ```bash
  VLLM_NIXL_EP_MAX_NUM_RANKS=32 \
  python benchmarks/kernels/benchmark_deepseek_v4_moe.py \
      --model deepseek-ai/DeepSeek-V4-Flash --trust-remote-code \
      -dp 2 -tp 2 --enable-expert-parallel \
      --all2all-backend nixl_ep --num-tokens 2048 --layer-idx 5
  # side channel host/port: VLLM_NIXL_SIDE_CHANNEL_HOST / _PORT
  ```

- **`flashinfer_nvlink_two_sided` / `flashinfer_nvlink_one_sided`** — require a
  `flashinfer` build that includes the comm module (`flashinfer.comm`), and an
  NVLink/MNNVL fabric. One-sided is the high-throughput variant (supports
  nvfp4/mxfp8/bf16 dispatch); two-sided is the low-latency MNNVL variant:
  ```bash
  python benchmarks/kernels/benchmark_deepseek_v4_moe.py \
      --model deepseek-ai/DeepSeek-V4-Flash --trust-remote-code \
      -dp 2 -tp 2 --enable-expert-parallel \
      --all2all-backend flashinfer_nvlink_one_sided \
      --num-tokens 2048 --layer-idx 5
  ```

- **`mori`** — **ROCm only** (AMD gfx942/gfx950); not usable on NVIDIA. Listed
  for completeness.

> If a backend's library isn't installed, `create_engine_config` /
> communicator setup raises during startup (e.g. a missing-`deep_ep` import).
> On a single node without these kernels built, stick with
> `allgather_reducescatter`. The DeepEP / nixl paths only show their full
> advantage with multi-node RDMA, which is beyond a single-box microbenchmark —
> but they will still run (and can be profiled) intra-node once built.

---

## 3. Weight precision options (and how DeepSeek-V4 chooses)

### 3a. DeepSeek-V4 dispatch (verified in `models/deepseek_v4/quant_config.py`)

`DeepseekV4FP8Config.get_quant_method()` routes the MoE method by **`expert_dtype`**
(read from `config.json`, default `"fp4"`; valid values `("fp4","fp8")`):

```
expert_dtype == "fp4":
    moe_quant_algo == "NVFP4"  -> ModelOptNvFp4FusedMoE   (packed-uint8 NVFP4 weights)
    else                       -> Mxfp4MoEMethod          (packed-uint8 MXFP4 weights)
expert_dtype == "fp8":
    -> Fp8MoEMethod (block-wise float8_e4m3fn, fp32 scales)   # DeepSeek-V4-Flash-Base
```

Linear (dense/attention) layers always use FP8 block quant from the parent
`Fp8Config`. The checkpoint's `scale_fmt` (`ue8m0` vs `float32`) is handled by
`is_scale_e8m0`. So:

- **DeepSeek-V4-Flash** (`expert_dtype: fp4`, default `moe_quant_algo`) →
  **MXFP4** experts (packed uint8, 32-block scales). This is what we ran.
- Setting `moe_quant_algo: "NVFP4"` (via `hf_overrides`) → **NVFP4** experts.
- **DeepSeek-V4-Flash-Base** (`expert_dtype: fp8`) → block-FP8 experts.

### 3b. Precision options relevant to this model

| Precision | Weight dtype | Min GPU | Typical kernels | How to get it |
| --- | --- | --- | --- | --- |
| **bf16/fp16 unquantized** | bf16 | any | FlashInfer CUTLASS/TRTLLM, Triton | Use a non-quantized config, or `hf_overrides` to strip `quantization_config` + set `expert_dtype` to a non-quant path. Largest memory; most portable. **Best apples-to-apples baseline** and what runs everywhere. |
| **FP8 block (e4m3, 128×128)** | `float8_e4m3fn` | SM90+ | DeepGEMM, FlashInfer CUTLASS, Triton, Marlin | Native to DSV4 linear layers; `expert_dtype:"fp8"` for fp8 experts. `VLLM_MOE_USE_DEEP_GEMM`. |
| **MXFP4** (default fp4) | packed uint8 (32-block) | SM80+ (kernel-dependent) | Marlin (SM90), DeepGEMM-MXFP4 / FlashInfer TRTLLM (SM100) | Native DSV4-Flash path. On SM100 prefer `deep_gemm_mega_moe`. |
| **NVFP4** | packed uint8 + e4m3 group scales | SM89/SM100 | FlashInfer TRTLLM/CuteDSL/CUTLASS, Marlin, CUTLASS | `hf_overrides={"moe_quant_algo":"NVFP4"}` (routes to `ModelOptNvFp4FusedMoE`). |
| **int4/int8 (WNA16, AWQ/GPTQ)** | packed int32 | SM75+ | Marlin | Requires an AWQ/GPTQ-quantized checkpoint; not a native DSV4 format. Mostly hypothetical for this model. |

### 3c. Changing precision: pre-quantized vs on-the-fly

- The valid `--quantization` values (registry in
  `vllm/model_executor/layers/quantization/__init__.py`) include: `fp8`,
  `modelopt`, `modelopt_fp4`, `modelopt_mxfp8`, `mxfp4`, `gpt_oss_mxfp4`,
  `awq`/`awq_marlin`, `gptq`/`gptq_marlin`, `auto_gptq`, `compressed-tensors`,
  `bitsandbytes`, `experts_int8`, `quark`, `moe_wna16`, `gguf`, `torchao`,
  `humming`, `online` (+ shorthands `fp8_per_tensor`, `fp8_per_block`,
  `mxfp8`, `int8_per_channel_weight_only`), and `deepseek_v4_fp8`.
- **Most low-bit MoE formats need a pre-quantized checkpoint** (fp8/nvfp4/mxfp4
  weights + scales baked in). vLLM will not, e.g., turn a bf16 checkpoint into
  nvfp4 for free.
- **On-the-fly is possible** for the `online` family
  (`vllm/model_executor/layers/quantization/online/`): `fp8_per_tensor`,
  `fp8_per_block`, `mxfp8`, `int8_per_channel_weight_only` create full-precision
  weights on a meta device and quantize during `process_weights_after_loading`.
  These let you take a bf16 config and benchmark an fp8/int8 MoE kernel without a
  separately-quantized checkpoint.
- For **this benchmark** (dummy weights), the cleanest way to compare precisions
  is to drive `expert_dtype` / `moe_quant_algo` / `quantization` via
  `hf_overrides` or a `--quantization` flag. None of this affects correctness
  here (weights are random) — it only changes *which kernel runs*.

### 3d. Switching expert precision in the benchmark (`--expert-dtype`)

`--expert-dtype {auto,fp4,fp8}` is wired: it sets
`hf_overrides={"expert_dtype": …}` so DeepSeek-V4 routes the experts to that
precision's MoE method. `auto` keeps the checkpoint's value. Because weights are
dummy, the FusedMoE method allocates the params for the chosen precision and we
fill them — no real conversion or checkpoint change is needed (and per §3a/the
numerics note, MXFP4 → FP8 is the lossless direction anyway).

This is the lever for **trying MoE backends that only exist for a given
precision**. Example — flip the real fp4 checkpoint to fp8 and sweep the FP8
backends (verified on H200):

```bash
# Default fp4 (MXFP4) experts -> Mxfp4MoEMethod, auto picks Marlin/etc.
python benchmarks/kernels/benchmark_deepseek_v4_moe.py \
    --model deepseek-ai/DeepSeek-V4-Flash --trust-remote-code --layer-idx 5

# Switch experts to block-FP8 -> Fp8MoEMethod; auto chose TRITON here, out of
# ['TRITON','AITER','FLASHINFER_TRTLLM','FLASHINFER_CUTLASS','DEEPGEMM',
#  'MARLIN','BATCHED_DEEPGEMM','BATCHED_TRITON','XPU','CPU']:
python … --expert-dtype fp8 --layer-idx 5

# Force a specific FP8 backend:
python … --expert-dtype fp8 --moe-backend deep_gemm   --layer-idx 5   # ✅ runs (UE8M0 block-FP8)
python … --expert-dtype fp8 --moe-backend triton      --layer-idx 5   # ✅ runs
python … --expert-dtype fp8 --moe-backend cutlass     --layer-idx 5   # ❌ vLLM: "CUTLASS FP8 MoE backend is disabled for this configuration"
```

The report header prints the resolved precision and the concrete method class,
e.g. `expert precision : fp8 (Fp8MoEMethod)` vs `fp4 (Mxfp4MoEMethod)`, plus the
chosen kernel appears in vLLM's `Using … Fp8 MoE backend out of …` log line.

> If a requested `--moe-backend` isn't valid for the chosen precision/hardware,
> vLLM raises a clear error from its oracle (as with `cutlass` above) rather than
> silently falling back — that's expected. The dummy-weight filler
> (`_init_dummy_weights`) already handles fp8 and packed-uint8 fp4/scales;
> a future precision with a different integer-packed layout may need the same
> treatment. Note `--expert-dtype` only covers `fp4`/`fp8` (DeepSeek-V4's two
> expert paths); NVFP4 still needs a `moe_quant_algo` override (not yet exposed).

---

## 4. What actually runs where (practical matrix)

For the benchmark on this **8×H200 (SM90)** box vs a **B200 (SM100)** box:

| Goal | Command sketch | H200 (SM90) | B200 (SM100) |
| --- | --- | --- | --- |
| Portable baseline | `--moe-backend triton` (bf16/unquant cfg) | ✅ | ✅ |
| Default auto (real fp4 cfg) | `--moe-backend auto` | ✅ (→ Marlin MXFP4) | ✅ (→ DeepGEMM/TRTLLM) |
| FlashInfer EP path | `--moe-backend flashinfer_cutlass --enable-expert-parallel` | ✅ (needs flashinfer; slow first JIT) | ✅ |
| **fp4 MegaMoE (production)** | `--moe-backend deep_gemm_mega_moe --enable-expert-parallel` | ❌ SM100-gated | ✅ |
| NVFP4 experts | `auto` + `hf_overrides moe_quant_algo=NVFP4` | ⚠️ Marlin/CUTLASS only | ✅ FlashInfer/CuteDSL |
| fp8 experts | Flash-Base cfg (`expert_dtype:fp8`) | ✅ DeepGEMM/Triton | ✅ |
| DeepEP / nixl all2all | `--all2all-backend deepep_*` (needs build) | ⚠️ build + ideally multi-node | ⚠️ build + ideally multi-node |
| Force any fp4 kernel anywhere | `--moe-backend emulation` | ✅ (dequant, slow) | ✅ |

Legend: ✅ supported · ⚠️ works with caveats/extra setup · ❌ hardware-gated.

---

## 5. Suggested next steps to broaden the benchmark

1. **`--all2all-backend`** → *done* (passes through to
   `EngineArgs(all2all_backend=...)`, see §2). Possible follow-up: gate the
   DeepEP/nixl/mori/flashinfer choices behind an availability check with a clear
   message (mirror the existing `deep_gemm_mega_moe` SM100 message) instead of
   letting the import error surface raw.
2. **Expert precision** → `--expert-dtype {auto,fp4,fp8}` is *done* (see §3d).
   Follow-ups: expose `moe_quant_algo` (to reach NVFP4 experts), a general
   `--quantization`/`--hf-overrides` passthrough, and the `online` fp8/int8
   precisions for a bf16 config. Extend `_init_dummy_weights` if a new packed
   layout appears.
3. **Env-var documentation** in `--help`: surface the key `VLLM_*` toggles
   (`VLLM_FLASHINFER_MOE_BACKEND`, `VLLM_MXFP4_USE_MARLIN`,
   `VLLM_MOE_USE_DEEP_GEMM`, `VLLM_ALL2ALL_BACKEND`) since several backends are
   only reachable through them.
4. **Report the chosen kernel**: parse/echo the "Using … MoE backend" selection
   so each result row records the concrete kernel, not just `--moe-backend auto`.

---

### Key source references

- MoE backend enum: `vllm/config/kernel.py` (`MoEBackend`)
- MoE oracle selectors: `vllm/model_executor/layers/fused_moe/oracle/{unquantized,fp8,nvfp4,mxfp4,int_wna16,w4a8}.py`
- All2All enum + EP/DP rules: `vllm/config/parallel.py` (`All2AllBackend`, `use_sequence_parallel_moe`, `use_batched_dp_moe`)
- All2All implementations: `vllm/distributed/device_communicators/all2all.py`, `cuda_communicator.py`; `vllm/model_executor/layers/fused_moe/all2all_utils.py` + `prepare_finalize/`
- DeepSeek-V4 precision dispatch: `vllm/models/deepseek_v4/quant_config.py`; fp4 MegaMoE gate: `vllm/models/deepseek_v4/nvidia/model.py` (`DeepseekV4MegaMoEExperts`)
- Env vars: `vllm/envs.py`
- Quantization registry: `vllm/model_executor/layers/quantization/__init__.py`

# MoE Backend Compatibility

| Backend | Weight Dtype | Compatible | Notes |
| :---- | :---- | :---- | :---- |
| Marlin | mxfp4 | ✅ |  |
| Triton (gpt-oss) | mxfp4 | ❌ | Does not support SiLU activation |
| FlashInfer TRTLLM | mxfp4 | ❌ | Requires SM100 |
| Triton | fp8 | ✅ |  |
| FlashInfer Cutlass | fp8 | ✅ |  |
| FlashInfer TRTLLM | fp8 | ❌ | Requires SM100 |
| DeepGEMM | fp8 | ✅ |  |
| MegaMoE FP8 | fp8 | ⚙️ | Yes, but requires [https://github.com/deepseek-ai/DeepGEMM/pull/352](https://github.com/deepseek-ai/DeepGEMM/pull/352)  |


# Example commands

```
nsys profile -o deepep_cg_2 --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
    python benchmarks/kernels/benchmark_deepseek_v4_moe.py \
      --model deepseek-ai/DeepSeek-V4-Pro --trust-remote-code --layer-idx 5 --profile -dp 8 --enable-expert-parallel \
      --all2all-backend deepep_low_latency --num-tokens 16 --cuda-graph

python benchmarks/kernels/benchmark_deepseek_v4_moe.py \
      --model deepseek-ai/DeepSeek-V4-Pro --trust-remote-code \
      --num-tokens 16 --layer-idx 5 --cuda-graph \
      -dp 8 --enable-expert-parallel --max-model-len 32768 --expert-dtype fp4 \
      --moe-backend flashinfer_cutlass
```
