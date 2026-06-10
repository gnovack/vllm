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
        if filename.startswith(_vllm_dir) and filename != __file__:
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
    """DRIVER-domain launch callback: stash callstack + num_tokens by corr id."""
    if domain != _cupti.CallbackDomain.DRIVER_API:
        return
    if callback_id not in _driver_launch_cbids:
        return
    if cbdata.callback_site != _cupti.ApiCallbackSite.API_ENTER:
        return
    stack = _capture_vllm_stack()
    num_tokens = _get_num_tokens()
    with _pending_lock:
        _pending[cbdata.correlation_id] = (stack, num_tokens)
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
                stack, num_tokens = _pending.pop(activity.correlation_id, (None, None))
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
# SQLite snapshotting (flush thread owns the connection)
# ---------------------------------------------------------------------------
def _open_db() -> sqlite3.Connection:
    """Create/clear the per-rank DB and write run metadata. Flush-thread only."""
    os.makedirs(os.path.dirname(_db_path), exist_ok=True)
    conn = sqlite3.connect(_db_path)
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
    db_dir: "str | None" = None, rank: int = 0, model_name: str = ""
) -> None:
    """Enable CUPTI kernel metric collection (idempotent, best-effort).

    Must be called from the GPU worker process. Failures (missing cupti-python,
    enable errors) are logged and swallowed so they never bring down the worker.
    """
    global _active, _cupti, _demangle, _kernel_kinds, _subscriber
    global _get_fwd_ctx, _fwd_ctx_available, _vllm_dir, _driver_launch_cbids
    global _flush_thread, _db_path, _rank, _model_name, _run_id

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

            drv = cupti.Driver_api_trace_cbid
            _driver_launch_cbids = {
                int(drv.cuLaunchKernel),
                int(drv.cuLaunchKernel_ptsz),
                int(drv.cuLaunchKernelEx),
                int(drv.cuLaunchKernelEx_ptsz),
            }
            _subscriber = cupti.subscribe(_launch_callback, None)
            cupti.enable_domain(1, _subscriber, cupti.CallbackDomain.DRIVER_API)

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
    global _active

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
            if _subscriber is not None:
                _cupti.unsubscribe(_subscriber)
            _cupti.activity_disable(_cupti.ActivityKind.CONCURRENT_KERNEL)
        except Exception as e:
            logger.warning("Failed to disable CUPTI profiling: %s", e)
        finally:
            _active = False
            with _pending_lock:
                _pending.clear()
