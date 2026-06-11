# In-server CUPTI kernel profiling (`--enable-cupti`)

This branch adds a single flag, `--enable-cupti`, that collects **per-kernel
hardware metrics from inside a running vLLM server** and writes them to per-rank
SQLite DBs you can explore offline. It's built for answering "where does the time
/ HBM traffic / FLOPs go, per kernel, and how does that scale with batch size?"
on a live server under real traffic.

This doc is meant to get a fresh agent up to speed and **validate the feature on a
Hopper (SM90, e.g. H100/H200) machine** — it was developed and tested on Blackwell
(GB200), so the Hopper path is implemented but **unverified**; the gotchas section
calls out exactly what to check.

---

## 1. What it collects (three tiers)

Everything keys off `(kernel_id, num_tokens, metric)`, where `kernel_id =
sha1(demangled_kernel_name + vLLM_callstack)`. Tiers degrade gracefully — if a
tier's prerequisites aren't met it's skipped with a log warning and the cheaper
tiers still work.

| Tier | Metric(s) | Mechanism | Needs |
|---|---|---|---|
| **Duration** (always) | `gpu_dur_ns` | CUPTI **Activity API** (`CONCURRENT_KERNEL`) | just `cupti-python` |
| **HBM bytes** | `dram_bytes_read`, `dram_bytes_write` | CUPTI **Range Profiler** (HW counters) | root + system libcupti preload + `--enforce-eager` |
| **Tensor-op FLOPs** | `tensor_ops_bf16` / `fp8` / `int8` / `fp16` / `tf32` | same Range Profiler session | same as HBM; arch-selected |

The **Duration** tier is portable and needs no privileges. The **HBM/FLOP** tier
uses the CUPTI Range Profiler, which is where all the setup requirements and
gotchas live.

---

## 2. How to run

### Duration only (easy, do this first)
```bash
vllm serve <model> --enable-cupti
# send some traffic, then:
python tools/cupti_report.py ./cupti_metrics
```
No root, no preload, no `--enforce-eager`. If you only see `gpu_dur_ns` columns,
this is the tier you got.

### Full (HBM + FLOPs)
```bash
sudo env LD_PRELOAD=<CUDA>/targets/<arch>-linux/lib/libcupti.so.13 \
     LD_LIBRARY_PATH=<CUDA>/targets/<arch>-linux/lib \
     $(which vllm) serve <model> -tp <N> --enforce-eager --enable-cupti
```
- `<arch>` is `x86_64` on Hopper x86 boxes, `sbsa` on ARM (GB200).
- Use the **system** CUDA's libcupti (e.g. `/usr/local/cuda-13/...`), **not**
  torch's bundled copy (see gotcha #1).
- Optional tuning: `VLLM_CUPTI_DRAM_WINDOW=8` (default 16) shrinks the per-forward
  profiling stall (see §4).

### Flags
- `--enable-cupti` — master switch.
- `--cupti-db-dir DIR` — output dir (default `./cupti_metrics`). Writes
  `cupti_metrics.rank{N}.db`, one per rank.

### Reading results
```bash
python tools/cupti_report.py ./cupti_metrics              # text, all metrics
python tools/cupti_report.py ./cupti_metrics --metric gpu_dur_ns --plot out.png
python tools/cupti_web.py    ./cupti_metrics -o report.html   # web UI (open file)
python tools/cupti_web.py    ./cupti_metrics --serve 8000     # live, re-reads DBs
```
Reader tools need only Python stdlib (`sqlite3`); `matplotlib` is optional. You can
copy the `*.db` files to any machine and read them there.

---

## 3. Architecture / where the code lives

- **`vllm/v1/worker/cupti_profiler.py`** — the orchestrator. Subscribes the
  Activity API (duration) and a DRIVER-API launch callback (captures the vLLM
  callstack + `num_tokens` per launch, and drives the windowed HBM/FLOP capture).
  Owns the SQLite writer (WAL), num_tokens bucketing, the metric selection, and the
  multi-GPU safety logic. Public API: `start_cupti_profiling(...)` /
  `stop_cupti_profiling()`.
- **`vllm/v1/worker/cupti_range_profiler.py`** — self-contained **ctypes** binding
  of the CUPTI Range Profiler (host `cuptiProfilerHost*` + device
  `cuptiRangeProfiler*`), exposing `RangeSession` (persistent session:
  `begin/warmup/reset/start/stop/read/end`) and `num_passes()`.
- **`vllm/v1/worker/gpu_worker.py`** — calls `start_cupti_profiling(...)` at the end
  of `compile_or_warm_up_model()` (after warmup, before serving) so metrics reflect
  steady state and the one-time profiler warmup happens off the serving path.
- **`vllm/config/observability.py`** — `enable_cupti`, `cupti_db_dir` config.
- **`vllm/engine/arg_utils.py`** — the CLI flags.
- **`tools/cupti_report.py`**, **`tools/cupti_web.py`** — offline readers.

### Data flow (HBM/FLOP tier)
1. At startup, a **persistent `RangeSession`** is built and `warmup()`-ed once
   (pays a ~1.5 s one-time `cuptiRangeProfilerStart` cost up front).
2. `model_runner.execute_model` is monkeypatched. Each forward's `num_tokens` is
   **bucketed** (§5) and used as the key for both duration and HBM/FLOP.
3. For each bucket not yet fully swept, the forward is profiled in a **window** of
   `VLLM_CUPTI_DRAM_WINDOW` kernel launches; the launch callback `Start`s/`Stop`s
   the profiler at the window boundaries; the window sweeps across successive
   forwards at that bucket until every kernel is covered (§4).
4. Range values are attributed to kernels by launch order and folded into the
   in-memory aggregate; a background thread snapshots it to SQLite (WAL).

---

## 4. Windowed capture (why per-forward stalls stay small)

Profiling a whole forward with the Range Profiler is enormously slow (kernel-replay
counter collection over thousands of kernels — seconds per forward). Instead we
profile only a **window** of launches per forward and sweep the window across the
many forwards that recur at the same `num_tokens` in real traffic. A bucket of `T`
launches finishes in `ceil(T / window)` forwards. Window size is
`VLLM_CUPTI_DRAM_WINDOW` (default 16); smaller = gentler per-forward stall, more
forwards to complete a sweep.

The persistent session makes per-window `start/stop` ~free (the expensive part is a
one-time warmup). A `cuCtxSynchronize` brackets each window so counter values aren't
polluted by in-flight kernels.

---

## 5. num_tokens bucketing

To keep the table small and let a sweep finish once per "size class" instead of per
exact token count, `total_num_scheduled_tokens` is bucketed before use:
- `n <= 1` → passthrough (incl. the `-1` "no forward context" sentinel),
- `2..1024` → next power of two,
- `> 1024` → nearest thousand, floored at 2000 (so buckets stay strictly
  increasing: `…512, 1024, 2000, 3000, …`).

Both duration and HBM/FLOP use the same bucketed key, so they join in the report.

---

## 6. Multi-GPU behavior

- **Duration tier** is per-process and needs no coordination — it just works under
  `-tp`/`-pp`/`-dp`.
- **HBM/FLOP tier** runs the profiled forward, which contains collectives, so:
  - The metric set is forced **single-pass** (`num_passes == 1`); multi-pass would
    replay kernels and **deadlock** collectives. If it can't stay single-pass, dram
    is disabled.
  - `_dram_enabled` is **agreed across all ranks** (all-reduce MIN) so capture runs
    everywhere or nowhere — a partial enable would deadlock.
  - A **TP-group barrier** aligns ranks before each profiled forward. (Scoped to the
    TP group on purpose; a world barrier would deadlock PP/DP.)
  - The report **sums** per-kernel metrics across ranks for a given key; for TP each
    rank holds a shard, so a per-launch cell is the per-shard average across ranks
    (not the all-GPU total). Point the tool at a single `.db` for one rank.
- PP benefits little (pipeline bubbles persist); TP is the well-supported case.

---

## 7. GOTCHAS — read before debugging

1. **CUPTI version skew (the #1 issue).** Three libcupti versions can be in play:
   torch bundles one (its Range Profiler is broken), `cupti-python` (pip) is another
   (struct sizes mismatch the system lib), and the **system** libcupti is the one we
   use. `cupti_range_profiler.py` loads the system libcupti **by full path** (scans
   `/proc/self/maps` for a libcupti *not* under `site-packages`/`nvidia`), so you
   **must `LD_PRELOAD` the system libcupti**. The ctypes struct layouts were written
   against the **CUDA 13.1** header. Within **13.x** you're fine (CUPTI versions
   structs via `structSize`). **A different major (e.g. 12.x) can mis-offset the
   structs → `INVALID_PARAMETER` or a segfault.**
   **➤ Hopper action:** check the system CUPTI version first —
   `grep CUPTI_API_VERSION <CUDA>/targets/<arch>-linux/include/cupti_version.h`
   (older CUDA layouts: `<CUDA>/extras/CUPTI/include/cupti_version.h`). `1301xx` = 13.1
   (good, matches the struct layouts). A `12xxxx` value means **12.x** → the HBM/FLOP
   tier may break; diff the struct field layouts in `cupti_range_profiler.py` against
   that version's `cupti_profiler_host.h` / `cupti_range_profiler.h` and adjust. The
   headers sit next to `cupti_version.h`. (`ls -l <CUDA>/.../lib/libcupti.so.13` also
   shows the file version, e.g. `libcupti.so.2025.x.x`.)
   The **duration tier is unaffected** (pure cupti-python Activity API).
2. **Admin/root required for HW counters.** Gated by `RmProfilingAdminOnly`. Run as
   root, or set the NVIDIA driver to allow non-admin profiling. Without it the dram
   preflight fails → duration-only. DB files written under `sudo` are root-owned; the
   readers open with `?mode=ro` so that's fine.
3. **`--enforce-eager` required for dram/FLOP.** CUDA-graph replay isn't profilable.
   Without it the tier is skipped.
4. **DCGM / `dcgm-exporter` conflict → `CUPTI_ERROR_HARDWARE_BUSY`.** DCGM (and
   `nsys`/`ncu`, and `nvidia-smi dmon` profiling) hold the profiling counters, which
   the Range Profiler needs exclusively. **Stop/pause `dcgm-exporter` on the box
   before running.** Check with `nvidia-smi --query-compute-apps=...` and
   `pgrep -af dcgm`.
5. **Hopper tensor-op metric names are a different family** and **untested.** Arch is
   auto-detected by compute-capability major: `>=10` (Blackwell) uses the `utc*`
   tensor-core path (`utchmma/utcimma/utcqmma`); `<10` (Hopper SM90, Ampere, Ada)
   uses `hmma_src_{bf16,fp16,tf32}`, `imma_src_int8`, `qmma_src_e4m3`. Invalid names
   self-filter (the greedy selector tries each and keeps only valid + single-pass
   ones), so over-listing is safe — **but the exact Hopper names are unverified.**
   **➤ Hopper action:** confirm against your chip with
   `ncu --query-metrics --chip <chip> | grep ops_path_tensor_op` (chip e.g. `gh100`),
   and check the startup log line `CUPTI: dram/FLOP metrics enabled … : [list]` to see
   which FLOP metrics actually got selected. If FLOP columns are missing but dram is
   present, the `hmma/imma/qmma` names in `cupti_profiler.py:_tensor_op_candidates`
   need fixing for your chip. (HBM `dram__bytes_read/write.sum` are arch-independent.)
6. **Single-pass is load-bearing.** Adding metrics can push the set multi-pass; the
   greedy selector drops anything that would. Don't hand-add metrics without checking
   `num_passes`.
7. **WAL DB reads.** Readers must open `file:...?mode=ro` (they do). Do **not** use
   `immutable=1` — it ignores the live `-wal` and you'll get "no such table".
8. **Library path differs by arch:** `targets/x86_64-linux` (Hopper x86) vs
   `targets/sbsa-linux` (ARM/GB200).
9. **Windowing assumes a stable kernel sequence per bucket** (true for dense models +
   `--enforce-eager`). MoE/speculative decode can vary the sequence; because we key by
   `kernel_id`, drift just causes re-sampling, not corruption.
10. **One-time warmup (~1.5 s).** Logged as `CUPTI: range profiler session warmed up
    in N.Ns (one-time)` during `compile_or_warm_up_model`. Never on the serving path.

---

## 8. Hopper validation checklist

1. **Duration tier:** `vllm serve <model> --enable-cupti`; send a few requests; stop;
   `python tools/cupti_report.py ./cupti_metrics`. Expect `gpu_dur_ns` per kernel with
   real vLLM callstacks. (No root needed.)
2. **Check the system CUPTI version is 13.x** (gotcha #1). If 12.x, expect to adapt
   the Range Profiler struct layouts before the next steps.
3. **Stop `dcgm-exporter`** and any `ncu`/`nsys` (gotcha #4).
4. **Full tier:** add `sudo env LD_PRELOAD=<x86_64 libcupti.so.13> … --enforce-eager`.
   In the startup log confirm:
   - `CUPTI: range profiler session warmed up …` (session built), and
   - `CUPTI: dram/FLOP metrics enabled … : [<metric list>]` — verify HBM metrics are
     present and which (if any) tensor-op FLOP metrics were selected.
5. Send traffic at a couple of distinct batch sizes; confirm the report shows
   `dram_bytes_read/write` (+ `tensor_ops_*` if present) populating across buckets, and
   that a bucket's sweep logs `CUPTI: dram sweep COMPLETE for num_tokens=…`.
6. **Multi-GPU:** `-tp 2`; confirm both `cupti_metrics.rank{0,1}.db` populate and the
   log shows the dram-enable agreement.

If FLOP metrics never appear on Hopper, that's almost certainly the metric-name issue
(gotcha #5) — verify names via `ncu --query-metrics` and patch
`_tensor_op_candidates`. If the Range Profiler errors/segfaults at startup, that's the
version-skew issue (gotcha #1).

---

## 9. Reference: standalone prototypes

Low-level CUPTI prototypes (developed under a separate `microbenchmarks/CUPTI/`
directory, not part of this vLLM branch) can validate the Range Profiler / PM
Sampling mechanics **without a full server** — useful for isolating a version-skew or
metric-name problem. If you have them, run under the same `sudo env LD_PRELOAD=…`:
- `range_profiler_wrapper.py` — minimal per-kernel `dram__bytes` capture.
- `design_b_windowing.py` — the windowed-capture proof.
- `pm_sampling_wrapper.py` — NVLink byte-counter time-series sampling (a separate,
  not-yet-integrated direction for capturing collective/link volume).

These are optional; the vLLM feature above is self-contained.
