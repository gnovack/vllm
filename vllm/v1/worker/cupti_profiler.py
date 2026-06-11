# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUPTI-based per-kernel metric collection for vLLM.

Enabled by ``--enable-cupti``. Uses the CUPTI Activity API to measure the GPU
execution time of every kernel, attributes each launch to the vLLM code that
issued it and the current batch's token count, and aggregates the results in
memory keyed by ``(kernel identity, num_tokens, metric)``. The aggregates are
snapshotted periodically to a per-rank SQLite database for offline / UI use.

Design (see also tools/cupti_report.py which reads these DBs):

  * Kernel identity = (demangled name, vLLM callstack). A stable ``kernel_id``
    hash lets the reader merge ranks and compare runs.

  * We do NOT store raw per-launch events (millions of them). Instead we keep
    running statistics -- count / sum / sum_sq / min / max -- per
    ``(kernel_id, num_tokens, metric)``. This is bounded (a model has a fixed
    set of kernels, num_tokens is bounded), gives mean/std/range for free, and
    is metric-agnostic: adding a future metric is just another ``metric`` value,
    no schema change.

  * Capture spans two threads, joined by CUPTI ``correlation_id``:
      - DRIVER-domain launch callback: synchronous on the launching thread, so
        it can read the Python callstack and the (thread-visible, live) vLLM
        forward context for num_tokens. Stashes both by correlation_id.
      - Activity records: delivered async on a CUPTI thread with the true GPU
        duration; we pop the stashed context by correlation_id and fold the
        sample into the in-memory aggregate.

  * A flush thread periodically drains CUPTI buffers and writes a snapshot of
    the (cumulative) in-memory aggregate to SQLite via upsert-overwrite, which
    is idempotent and crash-tolerant. The in-memory dict is the source of truth.

Limitations: kernels replayed from a CUDA graph go through cuGraphLaunch (no
per-launch callback), so they aggregate under an empty callstack / num_tokens=-1.
"""

import ctypes
import hashlib
import os
import sqlite3
import sys
import threading
import time
from collections import OrderedDict

from vllm.logger import init_logger

logger = init_logger(__name__)

_SCHEMA_VERSION = 1

# ---- lifecycle / CUPTI state ----------------------------------------------
_lock = threading.Lock()
_active = False
_cupti = None
_demangle = None
_kernel_kinds: set = set()
_subscriber = None

# vLLM forward-context accessors, resolved in start_cupti_profiling().
_get_fwd_ctx = None
_fwd_ctx_available = None

# ---- per-launch context captured on the launching thread ------------------
# correlation_id -> (vLLM callstack str, num_tokens). Bounded FIFO.
_pending: "OrderedDict[int, tuple]" = OrderedDict()
_pending_lock = threading.Lock()
_MAX_PENDING = 50000
_STACK_MAX_FRAMES = 12
_vllm_dir = ""
_driver_launch_cbids: set = set()

# ---- dram-bytes via Range Profiler (one-shot per num_tokens) ---------------
# The Range Profiler reads HW counters (works on cuBLAS GEMMs, unlike SASS) but
# conflicts with the Activity API, so we PAUSE activity for each capture forward,
# profile it once (autorange), and attribute each range to a kernel by launch
# order (the callback records (name, stack) in order while _dram_capturing).
_dram_enabled = False
_dram_capturing = False
_dram_order: list = []          # ordered (short_name, full_name, stack) per WINDOWED launch
_dram_captured: set = set()     # num_tokens whose kernel sweep is COMPLETE
# Windowed capture: instead of profiling a whole forward in one (huge) stall, we
# profile only a WINDOW of _DRAM_WINDOW launches each forward and sweep the window
# across recurring forwards at the same num_tokens (real traffic repeats sizes).
# _dram_cursor[num_tokens] = next launch index to profile; _win is per-forward
# scratch the launch callback drives (start at cursor, stop at cursor+W).
_DRAM_WINDOW = max(1, int(os.environ.get("VLLM_CUPTI_DRAM_WINDOW", "16")))
_dram_cursor: dict = {}
_win: dict = {"cursor": 0, "W": 0, "count": 0, "started": False,
              "stopped": False, "err": None}
_cu = None                      # cuda.bindings.driver, set at startup (callback hot path)
# (stored_name, cupti_metric_name). Base memory counters are always collected;
# tensor-op (FLOP) counters are added per-GPU-arch at startup (_select_dram_metrics).
_DRAM_BASE_PAIRS = [
    ("dram_bytes_read", "dram__bytes_read.sum"),
    ("dram_bytes_write", "dram__bytes_write.sum"),
]
_dram_metric_pairs = list(_DRAM_BASE_PAIRS)            # finalized in start
_dram_metrics = [c for _, c in _dram_metric_pairs]     # cupti names for Range Profiler
_orig_execute_model = None
_model_runner = None
_ctx_int = 0
_device = 0
_dram_session = None            # persistent cupti_range_profiler.RangeSession
# num_tokens of the forward currently running (set by _wrapped_execute_model),
# used to key BOTH duration and dram so they share the same num_tokens and join
# in the unified view. -1 when no forward is in progress.
_cur_forward_num_tokens = -1


def pause_activity() -> None:
    """Stop CONCURRENT_KERNEL activity tracing (it conflicts with the Range
    Profiler; counts come back 0 while activity is enabled). Flush first."""
    if not _active:
        return
    try:
        _cupti.activity_flush_all(1)
        _cupti.activity_disable(_cupti.ActivityKind.CONCURRENT_KERNEL)
    except Exception as e:
        logger.warning("CUPTI: pause_activity failed: %s", e)


def resume_activity() -> None:
    if not _active:
        return
    try:
        _cupti.activity_enable(_cupti.ActivityKind.CONCURRENT_KERNEL)
    except Exception as e:
        logger.warning("CUPTI: resume_activity failed: %s", e)

# ---- in-memory aggregate (source of truth) --------------------------------
# kernel_id -> (short_name, full_name, stack)
_registry: dict = {}
# (kernel_id, num_tokens, metric) -> [count, sum, sum_sq, min, max]
_agg: dict = {}
_agg_lock = threading.Lock()

# ---- flush thread / sqlite -------------------------------------------------
_flush_stop = threading.Event()
_flush_thread: "threading.Thread | None" = None
_FLUSH_INTERVAL_S = 5.0
_BUFFER_SIZE = 1 * 1024 * 1024
_db_path = ""
_rank = 0
_model_name = ""
_run_id = ""


# ---------------------------------------------------------------------------
# Name handling
# ---------------------------------------------------------------------------
def _make_demangler():
    """Return a function that demangles C++ symbol names via libstdc++."""
    try:
        libcxx = ctypes.CDLL("libstdc++.so.6")
        cxa = libcxx.__cxa_demangle
        cxa.restype = ctypes.c_void_p
        cxa.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_int),
        ]
        free = ctypes.CDLL("libc.so.6").free
        free.argtypes = [ctypes.c_void_p]

        def demangle(name: str) -> str:
            if not name:
                return name
            status = ctypes.c_int()
            res = cxa(name.encode(), None, None, ctypes.byref(status))
            if status.value == 0 and res:
                out = ctypes.cast(res, ctypes.c_char_p).value.decode()
                free(res)
                return out
            return name

        return demangle
    except Exception:
        return lambda name: name


def _short_kernel_name(demangled: str, max_len: int = 200) -> str:
    """Readable, length-capped form of a demangled kernel name (no C++ parsing)."""
    if not demangled:
        return "<unknown>"
    s = demangled.strip()
    if s.startswith("void "):
        s = s[5:]
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _kernel_id(full_name: str, stack: str) -> str:
    """Stable identity hash for a kernel = demangled name + vLLM callstack."""
    h = hashlib.sha1()
    h.update(full_name.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(stack.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Per-launch context capture (launching thread, via driver callback)
# ---------------------------------------------------------------------------
def _capture_vllm_stack() -> str:
    """vLLM-only call chain (innermost first) of the launching Python stack."""
    frames = []
    try:
        f = sys._getframe(2)  # skip this fn + _launch_callback
    except ValueError:
        return ""
    while f is not None and len(frames) < _STACK_MAX_FRAMES:
        code = f.f_code
        filename = code.co_filename
        # Keep vLLM frames, but exclude our own profiler modules (cupti_*.py).
        # During a dram capture the call chain passes through
        # cupti_range_profiler.py(profile_once) etc., which are under the vLLM
        # dir; including them would make the dram stack differ from the duration
        # stack and break the kernel_id join.
        if (filename.startswith(_vllm_dir)
                and not os.path.basename(filename).startswith("cupti_")):
            rel = filename[len(_vllm_dir) + 1 :]
            frames.append(f"{rel}:{f.f_lineno}({code.co_name})")
        f = f.f_back
    return " <- ".join(frames)


def _get_num_tokens():
    """Padded batch token count from the live vLLM forward context, or None."""
    if _fwd_ctx_available is None:
        return None
    try:
        if not _fwd_ctx_available():
            return None
        bd = getattr(_get_fwd_ctx(), "batch_descriptor", None)
        if bd is not None:
            return bd.num_tokens
    except Exception:
        return None
    return None


def _launch_callback(user_data, domain, callback_id, cbdata):
    """DRIVER-domain launch callback (API_ENTER): stash callstack + num_tokens by
    correlation_id for duration attribution; while a dram capture forward is
    running, also record (name, stack) in launch order for range attribution."""
    if domain != _cupti.CallbackDomain.DRIVER_API:
        return
    if callback_id not in _driver_launch_cbids:
        return
    if cbdata.callback_site != _cupti.ApiCallbackSite.API_ENTER:
        return
    stack = _capture_vllm_stack()
    if _dram_capturing:
        # Windowed capture: count every launch (so we learn the forward's total),
        # but only Start the profiler at the window's first launch, record the
        # launches inside the window, and Stop at the launch just past it. Start/
        # Stop sync first to keep the window's ranges clean (no spillover from
        # in-flight kernels). Safe to call from here (verified: no re-entrancy
        # deadlock). Errors stop the window rather than killing the forward.
        i = _win["count"]
        _win["count"] = i + 1
        try:
            if (_win["started"] and not _win["stopped"]
                    and i == _win["cursor"] + _win["W"]):
                _cu.cuCtxSynchronize()
                _dram_session.stop()
                _win["stopped"] = True
            if (i == _win["cursor"] and not _win["started"]
                    and not _win["stopped"] and _win["err"] is None):
                _cu.cuCtxSynchronize()
                _dram_session.start()
                _win["started"] = True
            if _win["started"] and not _win["stopped"]:
                full = _demangle(cbdata.symbol_name)
                _dram_order.append((_short_kernel_name(full), full, stack))
        except Exception as e:
            _win["err"] = repr(e)
            _win["stopped"] = True
        return  # activity is paused during capture -> no duration record to pair
    with _pending_lock:
        _pending[cbdata.correlation_id] = (stack, _cur_forward_num_tokens)
        if len(_pending) > _MAX_PENDING:
            _pending.popitem(last=False)


# ---------------------------------------------------------------------------
# Activity records -> in-memory aggregate (CUPTI / flush thread)
# ---------------------------------------------------------------------------
def _buffer_requested():
    return _BUFFER_SIZE, 0


def _record_metrics(kernel_id, num_tokens, metrics: dict):
    """Fold a per-launch sample into the running aggregate (caller holds lock)."""
    for metric, value in metrics.items():
        key = (kernel_id, num_tokens, metric)
        st = _agg.get(key)
        if st is None:
            _agg[key] = [1, value, value * value, value, value]
        else:
            st[0] += 1
            st[1] += value
            st[2] += value * value
            if value < st[3]:
                st[3] = value
            if value > st[4]:
                st[4] = value


def _buffer_completed(activities: list):
    """Fold a batch of completed kernel activity records into the aggregate."""
    for activity in activities:
        if activity.kind not in _kernel_kinds:
            continue
        try:
            duration = activity.end - activity.start
            full_name = _demangle(activity.name)
            with _pending_lock:
                stack, num_tokens = _pending.pop(
                    activity.correlation_id, (None, None))
        except Exception:
            continue

        stack = stack or ""
        num_tokens = int(num_tokens) if num_tokens is not None else -1
        kid = _kernel_id(full_name, stack)

        # Metric-agnostic: extend this dict to add future metrics.
        metrics = {"gpu_dur_ns": duration}

        with _agg_lock:
            if kid not in _registry:
                _registry[kid] = (_short_kernel_name(full_name), full_name, stack)
            _record_metrics(kid, num_tokens, metrics)


# ---------------------------------------------------------------------------
# dram bytes via the Range Profiler (one-shot per num_tokens)
# ---------------------------------------------------------------------------
def _dram_preflight(enforce_eager) -> bool:
    if not enforce_eager:
        logger.warning("CUPTI: dram metrics need --enforce-eager (graph replay "
                       "isn't profilable); collecting duration only.")
        return False
    if os.geteuid() != 0:
        logger.warning("CUPTI: dram metrics need admin (HW counters are gated by "
                       "RmProfilingAdminOnly); run as root. Duration only.")
        return False
    try:
        with open(f"/proc/{os.getpid()}/maps") as f:
            sys_cupti = any("libcupti" in ln and "site-packages" not in ln
                            and "/nvidia/" not in ln for ln in f)
        if not sys_cupti:
            logger.warning("CUPTI: dram metrics need the system libcupti preloaded "
                           "(LD_PRELOAD=.../libcupti.so.13); duration only.")
            return False
    except Exception:
        pass
    return True


def _tensor_op_candidates(cc_major: int):
    """Per-GPU-arch tensor-op (math op count) metrics, as (stored_name, cupti).
    Blackwell (sm_100+) uses the UTC* tensor-core path; Hopper (sm_90) uses the
    warpgroup *gmma* counters (hgmma/igmma) with hmma/imma as fallbacks for
    Ampere/Ada. Invalid names for a given chip are filtered out at startup, so
    over-listing is harmless."""
    # Order = priority: a single HW pass only fits dram + a couple of distinct
    # tensor-core pipes, and the greedy selector (_select_dram_metrics) keeps the
    # ones listed first. For vLLM inference the dominant GEMM dtypes are bf16 and
    # fp8 (quantized), so they come before fp16/tf32, which are rarely the main
    # path in serving.
    if cc_major >= 10:  # Blackwell: UTC* tensor-core path
        base = "sm__ops_path_tensor_op_utc"
        return [
            ("tensor_ops_bf16", base + "hmma_src_bf16_dst_fp32_sparsity_off.sum"),
            ("tensor_ops_fp8",
             base + "qmma_src_fp4_fp6_fp8_dst_fp32_sparsity_off.sum"),
            ("tensor_ops_int8", base + "imma_src_int8_sparsity_off.sum"),
            ("tensor_ops_fp16", base + "hmma_src_fp16_dst_fp32_sparsity_off.sum"),
            ("tensor_ops_tf32", base + "hmma_src_tf32_dst_fp32_sparsity_off.sum"),
        ]
    # Hopper (sm_90) / Ampere / Ada. Hopper's dominant GEMM path is the
    # warpgroup MMA (WGMMA), which increments the *gmma* counters
    # (hgmma/igmma) rather than the per-warp hmma/imma counters; cuBLAS and
    # cutlass Hopper kernels use WGMMA, so list the warpgroup variants FIRST
    # (the greedy single-pass selector keeps the earliest that fit). FP8 on
    # Hopper is counted by sm__ops_path_tensor_src_fp8 -- there is no
    # qmma_src_e4m3 op-path metric on gh100 (verified via
    # `ncu --query-metrics --chip gh100`). The per-warp hmma/imma variants are
    # kept as lower-priority, distinctly-named fallback columns: Ampere/Ada use
    # them, and on Hopper they catch any non-WGMMA kernels. Invalid names for a
    # given chip self-filter at startup, so over-listing is safe.
    op = "sm__ops_path_tensor_op_"
    return [
        ("tensor_ops_bf16", op + "hgmma_src_bf16_dst_fp32_sparsity_off.sum"),
        ("tensor_ops_fp8", "sm__ops_path_tensor_src_fp8_sparsity_off.sum"),
        ("tensor_ops_int8", op + "igmma_src_int8_sparsity_off.sum"),
        ("tensor_ops_fp16", op + "hgmma_src_fp16_sparsity_off.sum"),
        ("tensor_ops_tf32", op + "hgmma_src_tf32_dst_fp32_sparsity_off.sum"),
        # Per-warp MMA fallbacks (Ampere/Ada; non-WGMMA Hopper kernels):
        ("tensor_ops_bf16_mma", op + "hmma_src_bf16_dst_fp32_sparsity_off.sum"),
        ("tensor_ops_int8_mma", op + "imma_src_int8_sparsity_off.sum"),
        ("tensor_ops_fp16_mma", op + "hmma_src_fp16_dst_fp32_sparsity_off.sum"),
        ("tensor_ops_tf32_mma", op + "hmma_src_tf32_dst_fp32_sparsity_off.sum"),
    ]


def _select_dram_metrics(ctx_int, device, np) -> bool:
    """Finalize _dram_metric_pairs / _dram_metrics = base memory counters + the
    arch-appropriate tensor-op counters that are (a) valid on this chip and
    (b) keep the whole set SINGLE-PASS. Returns False if even the base memory
    counters aren't single-pass (then dram is disabled)."""
    global _dram_metric_pairs, _dram_metrics
    from cuda.bindings import driver as cu
    from vllm.v1.worker import cupti_range_profiler as rp

    def passes(pairs):
        return rp.num_passes([c for _, c in pairs], ctx_int, device=device, np=np)

    try:
        if passes(_DRAM_BASE_PAIRS) != 1:
            logger.warning("CUPTI: base dram counters not single-pass; disabling dram.")
            return False
    except Exception as e:
        logger.warning("CUPTI: dram base-metric check failed: %s; disabling dram.", e)
        return False

    err, major = cu.cuDeviceGetAttribute(
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
    cc_major = int(major) if int(err) == 0 else 0

    accepted = list(_DRAM_BASE_PAIRS)
    for stored, cupti in _tensor_op_candidates(cc_major):
        try:
            if passes(accepted + [(stored, cupti)]) == 1:  # valid + still 1 pass
                accepted.append((stored, cupti))
        except Exception:
            pass  # metric name invalid on this chip -> skip
    _dram_metric_pairs = accepted
    _dram_metrics = [c for _, c in accepted]
    return True


def _dram_capture(num_tokens, run_forward):
    """Profile ONE WINDOW of this forward's launches on the persistent Range
    Profiler session, sweeping the window across successive forwards at this
    num_tokens until every kernel is covered. The launch callback drives
    start/stop at the window boundaries; here we set up the window, run the
    forward exactly once, then attribute the window's ranges to its kernels and
    advance the cursor. Marks num_tokens done once the cursor passes the forward's
    last launch."""
    global _dram_capturing
    from cuda.bindings import driver as cu

    cursor = _dram_cursor.get(num_tokens, 0)
    _win.update(cursor=cursor, W=_DRAM_WINDOW, count=0, started=False,
                stopped=False, err=None)

    pause_activity()      # Activity API conflicts with an ACTIVE profiler window
    _dram_order.clear()
    per_range = []
    try:
        _dram_session.reset()
    except Exception as e:
        logger.warning("CUPTI: dram reset failed at num_tokens=%d: %s; "
                       "skipping window.", num_tokens, e)
        resume_activity()
        return run_forward()

    _dram_capturing = True
    try:
        result = run_forward()          # exactly once; exceptions propagate below
    finally:
        try:
            if _win["started"] and not _win["stopped"]:   # window ran to fwd end
                cu.cuCtxSynchronize()
                _dram_session.stop()
                _win["stopped"] = True
            if _win["started"] and _win["err"] is None:
                per_range = _dram_session.read()
        except Exception as e:
            logger.warning("CUPTI: dram stop/read failed at num_tokens=%d: %s",
                           num_tokens, e)
        _dram_capturing = False
        resume_activity()
    if _win["err"] is not None:
        logger.warning("CUPTI: dram window callback error at num_tokens=%d: %s",
                       num_tokens, _win["err"])

    total = _win["count"]   # total launches this forward (callback counts all)
    order = list(_dram_order)
    if per_range and len(order) != len(per_range):
        logger.warning("CUPTI: dram %d ranges vs %d windowed launches at "
                       "num_tokens=%d; attribution may be approximate.",
                       len(per_range), len(order), num_tokens)
    n = min(len(order), len(per_range))
    with _agg_lock:
        for i in range(n):
            short, full, stack = order[i]
            kid = _kernel_id(full, stack)
            if kid not in _registry:
                _registry[kid] = (short, full, stack)
            _record_metrics(kid, num_tokens, {
                stored: per_range[i].get(cupti, 0.0)
                for (stored, cupti) in _dram_metric_pairs
            })

    nxt = cursor + _DRAM_WINDOW
    if not _win["started"] or nxt >= total:
        _dram_captured.add(num_tokens)       # swept to the end -> done
        _dram_cursor.pop(num_tokens, None)
        logger.info("CUPTI: dram sweep COMPLETE for num_tokens=%d "
                    "(%d launches total)", num_tokens, total)
    else:
        _dram_cursor[num_tokens] = nxt
        logger.info("CUPTI: dram window [%d,%d) of ~%d for num_tokens=%d "
                    "(%d kernels)", cursor, nxt, total, num_tokens, n)
    return result


def _agree_dram_enabled(local_enabled: bool) -> bool:
    """Collectively AND the dram-enabled flag across ALL ranks so dram capture
    runs everywhere or nowhere. A partial enable would make one rank enter the
    slow profiled forward while peers run the fast path -> the forward's
    collectives deadlock. Best-effort: single-process returns the local value;
    on any collective error we disable dram (the safe default for >1 rank)."""
    try:
        import torch

        if not (torch.distributed.is_available()
                and torch.distributed.is_initialized()):
            return local_enabled  # single process / no distributed
        from vllm.distributed.parallel_state import get_world_group

        grp = get_world_group()
        if grp.world_size <= 1:
            return local_enabled
        t = torch.tensor([1 if local_enabled else 0], dtype=torch.int32)
        torch.distributed.all_reduce(
            t, op=torch.distributed.ReduceOp.MIN, group=grp.cpu_group)
        agreed = bool(t.item() == 1)
        if agreed != local_enabled:
            logger.info("CUPTI: dram-enable agreed across ranks: %s -> %s "
                        "(runs everywhere or nowhere)", local_enabled, agreed)
        return agreed
    except Exception as e:
        logger.warning("CUPTI: dram-enable agreement failed (%s); disabling dram "
                       "to avoid a partial-enable deadlock.", e)
        return False


def _dram_barrier() -> None:
    """Barrier the tensor-parallel group before a profiled forward. Scoped to the
    TP group on purpose: TP ranks share scheduler_output (same num_tokens, same
    capture decision) so they arrive together, while PP/DP ranks live in other
    groups and would deadlock if coupled here. Uses the group's CPU (gloo) barrier
    (its NCCL barrier mishandles the current device). Best-effort."""
    try:
        from vllm.distributed.parallel_state import get_tp_group

        grp = get_tp_group()
        if grp.world_size > 1:
            grp.barrier()
    except Exception:
        pass


def _bucket_num_tokens(n: int) -> int:
    """Bucket the raw token count so the metric table stays small (and a dram
    sweep only has to finish once per bucket, not per exact value): powers of two
    up to 1024, then the nearest thousand above that. Applied to BOTH duration and
    dram so they keep a common key and still join. (n<=1 passes through, incl. the
    -1 'no forward context' sentinel.)"""
    if n <= 1:
        return n
    if n <= 1024:
        return 1 << (n - 1).bit_length()        # smallest power of two >= n
    return max(2000, ((n + 500) // 1000) * 1000)  # nearest thousand, never < 2000


def _wrapped_execute_model(scheduler_output, *args, **kwargs):
    """Wraps every forward so duration AND dram key off the same (bucketed)
    num_tokens. dram-sweeps each bucket once when dram is enabled."""
    global _cur_forward_num_tokens
    raw = getattr(scheduler_output, "total_num_scheduled_tokens", 0)
    num_tokens = _bucket_num_tokens(raw)
    _cur_forward_num_tokens = num_tokens
    try:
        if _dram_enabled and num_tokens > 0 and num_tokens not in _dram_captured:
            # Align TP ranks before the (slow) profiled forward so Range Profiler
            # session setup overhead is symmetric and no rank is left blocked at a
            # collective inside the forward. TP ranks share scheduler_output, so
            # they reach this together; the barrier is scoped to the TP group only
            # (PP/DP ranks are in other groups and must NOT be coupled here).
            _dram_barrier()
            return _dram_capture(
                num_tokens,
                lambda: _orig_execute_model(scheduler_output, *args, **kwargs))
        return _orig_execute_model(scheduler_output, *args, **kwargs)
    finally:
        _cur_forward_num_tokens = -1


# ---------------------------------------------------------------------------
# SQLite snapshotting (flush thread owns the connection)
# ---------------------------------------------------------------------------
def _open_db() -> sqlite3.Connection:
    """Create/clear the per-rank DB and write run metadata. Flush-thread only."""
    os.makedirs(os.path.dirname(_db_path), exist_ok=True)
    conn = sqlite3.connect(_db_path)
    # WAL so a reader can read while the server is still writing. The reader must
    # open with mode=ro (NOT immutable) so it sees the live -wal contents.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    # Fresh data per server run.
    conn.executescript(
        """
        DROP TABLE IF EXISTS meta;
        DROP TABLE IF EXISTS kernels;
        DROP TABLE IF EXISTS kernel_metrics;
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE kernels (
            kernel_id TEXT PRIMARY KEY,
            name      TEXT NOT NULL,
            full_name TEXT NOT NULL,
            stack     TEXT NOT NULL
        );
        CREATE TABLE kernel_metrics (
            kernel_id  TEXT    NOT NULL,
            num_tokens INTEGER NOT NULL,
            metric     TEXT    NOT NULL,
            count      INTEGER NOT NULL,
            sum        REAL    NOT NULL,
            sum_sq     REAL    NOT NULL,
            min        REAL    NOT NULL,
            max        REAL    NOT NULL,
            PRIMARY KEY (kernel_id, num_tokens, metric)
        );
        """
    )
    conn.executemany(
        "INSERT INTO meta(key, value) VALUES(?, ?)",
        [
            ("schema_version", str(_SCHEMA_VERSION)),
            ("run_id", _run_id),
            ("rank", str(_rank)),
            ("pid", str(os.getpid())),
            ("model", _model_name),
            ("started_at", str(time.time())),
        ],
    )
    conn.commit()
    return conn


def _snapshot(conn: sqlite3.Connection) -> None:
    """Write the current cumulative aggregate to SQLite (upsert-overwrite)."""
    with _agg_lock:
        registry = [(kid, n, f, s) for kid, (n, f, s) in _registry.items()]
        rows = [
            (kid, ntok, metric, st[0], st[1], st[2], st[3], st[4])
            for (kid, ntok, metric), st in _agg.items()
        ]
    if not rows:
        return
    conn.executemany(
        "INSERT INTO kernels(kernel_id, name, full_name, stack) VALUES(?, ?, ?, ?) "
        "ON CONFLICT(kernel_id) DO NOTHING",
        registry,
    )
    conn.executemany(
        """
        INSERT INTO kernel_metrics
            (kernel_id, num_tokens, metric, count, sum, sum_sq, min, max)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(kernel_id, num_tokens, metric) DO UPDATE SET
            count=excluded.count, sum=excluded.sum, sum_sq=excluded.sum_sq,
            min=excluded.min, max=excluded.max
        """,
        rows,
    )
    conn.commit()


def _flush_loop():
    """Periodically drain CUPTI buffers and snapshot the aggregate to SQLite."""
    try:
        conn = _open_db()
    except Exception as e:
        logger.warning("CUPTI: failed to open metrics DB %s: %s", _db_path, e)
        return
    try:
        while not _flush_stop.wait(_FLUSH_INTERVAL_S):
            try:
                _cupti.activity_flush_all(1)  # drain -> _buffer_completed
                _snapshot(conn)
            except Exception as e:
                logger.warning("CUPTI: snapshot failed: %s", e)
        # Final drain + snapshot on shutdown.
        try:
            _cupti.activity_flush_all(1)
            _snapshot(conn)
        except Exception:
            pass
    finally:
        with _agg_lock:
            n_kernels, n_rows = len(_registry), len(_agg)
        logger.info(
            "CUPTI: wrote %d kernels / %d metric rows to %s",
            n_kernels,
            n_rows,
            _db_path,
        )
        conn.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def start_cupti_profiling(
    db_dir: "str | None" = None, rank: int = 0, model_name: str = "",
    model_runner=None, enforce_eager: bool = False,
) -> None:
    """Enable CUPTI kernel metric collection (idempotent, best-effort).

    Always collects per-kernel GPU duration (Activity API). Additionally collects
    per-kernel dram bytes (Range Profiler, one-shot per num_tokens) when the
    prerequisites are met (admin + system libcupti preloaded + --enforce-eager);
    otherwise dram is skipped with a warning and duration still works.

    Must be called from the GPU worker process. Failures are logged and swallowed
    so they never bring down the worker.
    """
    global _active, _cupti, _demangle, _kernel_kinds, _subscriber
    global _get_fwd_ctx, _fwd_ctx_available, _vllm_dir, _driver_launch_cbids
    global _flush_thread, _db_path, _rank, _model_name, _run_id
    global _dram_enabled, _orig_execute_model, _model_runner, _ctx_int, _device
    global _dram_session, _cu

    with _lock:
        if _active:
            return

        try:
            from cupti import cupti
        except ImportError:
            logger.warning(
                "--enable-cupti was set but the 'cupti-python' package is not "
                "installed; kernel tracing is disabled."
            )
            return

        try:
            import vllm

            _cupti = cupti
            _demangle = _make_demangler()
            _vllm_dir = os.path.dirname(os.path.abspath(vllm.__file__))
            _rank = rank
            _model_name = model_name or ""
            _run_id = time.strftime("%Y%m%d-%H%M%S")
            db_dir = db_dir or os.path.join(os.getcwd(), "cupti_metrics")
            _db_path = os.path.join(db_dir, f"cupti_metrics.rank{rank}.db")

            try:
                from vllm.forward_context import (
                    get_forward_context,
                    is_forward_context_available,
                )

                _get_fwd_ctx = get_forward_context
                _fwd_ctx_available = is_forward_context_available
            except Exception:
                _get_fwd_ctx = None
                _fwd_ctx_available = None

            _kernel_kinds = {cupti.ActivityKind.CONCURRENT_KERNEL}
            cupti.activity_register_callbacks(_buffer_requested, _buffer_completed)
            cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)

            # cupti-python renamed this enum across releases: 13.3 exposes
            # ``Driver_api_trace_cbid`` while 12.x / 13.0 expose the lowercase
            # ``driver_api_trace_cbid``. Accept whichever this install provides
            # (the matching version is dictated by the torch CUDA build, since
            # cupti-python must match the already-loaded libcupti soname).
            drv = getattr(cupti, "Driver_api_trace_cbid", None) or \
                cupti.driver_api_trace_cbid
            _driver_launch_cbids = {
                int(drv.cuLaunchKernel),
                int(drv.cuLaunchKernel_ptsz),
                int(drv.cuLaunchKernelEx),
                int(drv.cuLaunchKernelEx_ptsz),
            }
            _subscriber = cupti.subscribe(_launch_callback, None)
            cupti.enable_domain(1, _subscriber, cupti.CallbackDomain.DRIVER_API)

            # dram/FLOP bytes via Range Profiler -- opportunistic, swept in small
            # windows across recurring forwards per num_tokens (needs admin +
            # system libcupti + --enforce-eager).
            if model_runner is not None and _dram_preflight(enforce_eager):
                try:
                    from cuda.bindings import driver as _cu
                    import numpy as _np

                    _, ctx = _cu.cuCtxGetCurrent()
                    _ctx_int = int(ctx)
                    _device = 0
                    # Finalize the metric set: base memory counters + whatever
                    # tensor-op (FLOP) counters this GPU's arch supports, keeping
                    # the set single-pass (see _select_dram_metrics). Multi-pass
                    # would make the Range Profiler replay kernels, re-running the
                    # (stateful) forward -> corruption, so we only ever keep a
                    # single-pass set.
                    if _select_dram_metrics(_ctx_int, _device, _np):
                        _dram_enabled = True
                        logger.info("CUPTI: dram/FLOP metrics enabled (Range "
                                    "Profiler, single-pass, one-shot per "
                                    "num_tokens): %s", _dram_metrics)
                    else:
                        _dram_enabled = False
                        logger.warning("CUPTI: base dram metric set %s is not "
                                       "single-pass; disabling dram/FLOP capture.",
                                       _dram_metrics)
                except Exception as e:
                    logger.warning("CUPTI: dram setup failed: %s", e)
                    _dram_enabled = False

            # Multi-GPU safety: dram capture must run on EVERY rank or none. A
            # partial enable (e.g. one node missing the libcupti preload) would
            # send one rank into the slow profiled forward while peers take the
            # fast path -> their in-forward collectives deadlock / time out.
            # Agree on the flag across all ranks (every rank reaches this, dram
            # attempted or not). Single-pass is already enforced above, so the
            # profiled forward runs its collectives exactly once on every rank.
            _dram_enabled = _agree_dram_enabled(_dram_enabled)

            # Build the PERSISTENT Range Profiler session and warm it up now
            # (pays the one-time ~1.5s first-Start cost here, during warmup,
            # never during serving). Reused across all per-num_tokens captures.
            if _dram_enabled:
                try:
                    import numpy as _np
                    from vllm.v1.worker import cupti_range_profiler as _rp

                    _dram_session = _rp.RangeSession(
                        _dram_metrics, _ctx_int, device=_device, np=_np).begin()
                    t_warm = time.time()
                    _dram_session.warmup()
                    logger.info("CUPTI: range profiler session warmed up in "
                                "%.1fs (one-time)", time.time() - t_warm)
                except Exception as e:
                    logger.warning("CUPTI: range session init failed: %s; "
                                   "disabling dram.", e)
                    if _dram_session is not None:
                        try:
                            _dram_session.end()
                        except Exception:
                            pass
                    _dram_session = None
                    _dram_enabled = False

            # Wrap execute_model whenever a runner is available so BOTH duration
            # and dram key off the same num_tokens (total_num_scheduled_tokens),
            # which makes them join in the unified view. (Also drives the dram
            # one-shot capture when enabled.)
            if model_runner is not None:
                _orig_execute_model = model_runner.execute_model
                _model_runner = model_runner
                model_runner.execute_model = _wrapped_execute_model

            _flush_stop.clear()
            _flush_thread = threading.Thread(
                target=_flush_loop, name="cupti-flush", daemon=True
            )
            _flush_thread.start()

            _active = True
            logger.info("CUPTI kernel metrics enabled; db=%s", _db_path)
        except Exception as e:
            logger.warning("Failed to enable CUPTI profiling: %s", e)
            _active = False


def stop_cupti_profiling() -> None:
    """Disable CUPTI tracing, flush a final snapshot, and release resources."""
    global _active, _dram_session

    with _lock:
        if not _active:
            return
        try:
            # Drain outstanding activity records into the aggregate first.
            _cupti.activity_flush_all(1)
            # Stop the flush thread (it writes a final snapshot and closes the DB).
            _flush_stop.set()
            if _flush_thread is not None:
                _flush_thread.join(timeout=_FLUSH_INTERVAL_S + 5.0)
            if _orig_execute_model is not None and _model_runner is not None:
                _model_runner.execute_model = _orig_execute_model
            if _subscriber is not None:
                _cupti.unsubscribe(_subscriber)
            _cupti.activity_disable(_cupti.ActivityKind.CONCURRENT_KERNEL)
            if _dram_session is not None:
                _dram_session.end()       # Disable device obj + host deinit
        except Exception as e:
            logger.warning("Failed to disable CUPTI profiling: %s", e)
        finally:
            _dram_session = None
            _active = False
            with _pending_lock:
                _pending.clear()
