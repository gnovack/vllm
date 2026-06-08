# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight CUPTI-based kernel tracing.

Enabled by ``--enable-cupti``. When started (from the GPU worker's
``init_device``), this uses the CUPTI **Activity API** to record every CUDA
kernel that executes on the device and logs, for each one, the GPU start
timestamp, the GPU execution duration, and the demangled kernel name.

Why the Activity API (rather than launch callbacks):
  * It reports the *GPU execution* time of each kernel, not just the CPU-side
    launch overhead.
  * It captures kernels uniformly regardless of how they were launched -- the
    runtime API (``cudaLaunchKernel``, e.g. vLLM's hand-written ``<<<>>>``
    kernels), the driver API (``cuLaunchKernel``/``Ex``, e.g. cuBLAS/cutlass
    GEMMs and cuDNN attention), AND kernels replayed from a captured CUDA graph
    (``cuGraphLaunch``). Launch callbacks miss the driver-only and graph cases.

Callstack attribution:
  The Activity API runs on a background thread, so it cannot see the launching
  thread's Python stack. To attribute each kernel to the vLLM code that launched
  it, we ALSO subscribe a DRIVER-domain launch callback (cuLaunchKernel/Ex),
  which fires synchronously on the launching thread. There we walk the Python
  stack, keep only frames inside the vLLM package, and stash them keyed by the
  CUPTI ``correlation_id``. The kernel activity record carries the same
  correlation_id, so we join the two. (The driver domain is the universal key:
  a runtime ``cudaLaunchKernel`` shares its correlation_id with the driver
  launch it funnels into, and driver-only kernels -- cuBLAS/cutlass GEMMs,
  attention -- match too.)

  The same launch callback also records the current batch's token count from the
  vLLM forward context (``batch_descriptor.num_tokens``). This is only readable
  there: the forward context is a global that is live during the launch but torn
  down by the time the (async) activity record is delivered.

Notes / current limitations:
  * Activity records are delivered asynchronously on a CUPTI background thread,
    in batches, when a buffer fills or is flushed. A periodic flush thread keeps
    console output reasonably prompt; output is still batched, not per-launch.
  * Logging every kernel is very high volume in a running vLLM server. This is
    intentional for the initial "log to console" milestone; a later iteration
    will aggregate/persist instead.
  * Kernels replayed from a captured CUDA graph go through ``cuGraphLaunch`` and
    fire no per-launch callback, so they are logged with an empty callstack.
"""

import ctypes
import os
import sys
import threading
from collections import OrderedDict

from vllm.logger import init_logger

logger = init_logger(__name__)

# Module-level state, set up once by start_cupti_profiling().
_lock = threading.Lock()
_active = False
_cupti = None
_demangle = None
_kernel_kinds: set = set()
_subscriber = None

# vLLM forward-context accessors, resolved in start_cupti_profiling().
_get_fwd_ctx = None
_fwd_ctx_available = None

# correlation_id -> (vLLM callstack str, num_tokens), populated by the launch
# callback (launching thread, where the forward context is live) and consumed by
# the activity handler (CUPTI background thread).
_pending: "OrderedDict[int, tuple]" = OrderedDict()
_stacks_lock = threading.Lock()
_MAX_PENDING_STACKS = 50000  # bound memory if some launches never get an activity
_STACK_MAX_FRAMES = 12       # coarse-grained: keep at most this many vLLM frames

# Set in start_cupti_profiling(): absolute path of the vllm package directory.
_vllm_dir = ""
_driver_launch_cbids: set = set()

# Periodic forced-flush thread, so records reach the console promptly.
_flush_stop = threading.Event()
_flush_thread: threading.Thread | None = None
_FLUSH_INTERVAL_S = 2.0

# CUPTI activity buffer sizing (smaller buffer => more frequent auto-flush).
_BUFFER_SIZE = 1 * 1024 * 1024


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
        # Fall back to raw (mangled) names if libstdc++ is unavailable.
        return lambda name: name


def _short_kernel_name(demangled: str, max_len: int = 200) -> str:
    """Return a readable, length-capped form of a demangled kernel name.

    Deliberately simple and robust: we do NOT parse C++ structure. Angle-bracket
    / parenthesis depth tracking is fragile (operators, lambdas, and non-type
    template params all use ``<``/``>``/``(``/``)`` non-nestingly) and, when it
    miscounts, can emit garbage such as a stray parameter-list fragment like
    "BFloat16 const*, float, int, int)". Instead we just drop a leading "void "
    return type and truncate. The identifying kernel name is always at the front,
    so an expert can still read it; plain library names (cutlass / cuDNN / fmha)
    fit under the cap unchanged.
    """
    if not demangled:
        return "<unknown>"
    s = demangled.strip()
    if s.startswith("void "):
        s = s[5:]
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _capture_vllm_stack() -> str:
    """Walk the current Python stack and return the vLLM-only call chain.

    Innermost (closest to the launch) first. Frames outside the vLLM package
    (torch, the CUPTI shim, stdlib, etc.) are skipped, which both filters to the
    code the user cares about and keeps the output compact. Runs on the
    launching thread, called from the launch callback.
    """
    frames = []
    try:
        # 0: this function, 1: _launch_callback -> 2: the launching frame.
        f = sys._getframe(2)
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
    """Number of tokens in the current forward batch, or None.

    Reads the vLLM forward context, which is a module global set on this
    (launching) thread during the model forward pass. ``batch_descriptor`` is
    populated by the model runner for every forward; ``num_tokens`` is the padded
    batch token count. Returns None when no forward is in progress (e.g. weight
    loading, NCCL setup, warmup).
    """
    if _fwd_ctx_available is None:
        return None
    try:
        if not _fwd_ctx_available():
            return None
        ctx = _get_fwd_ctx()
        bd = getattr(ctx, "batch_descriptor", None)
        if bd is not None:
            return bd.num_tokens
    except Exception:
        return None
    return None


def _launch_callback(user_data, domain, callback_id, cbdata):
    """DRIVER-domain launch callback: stash callstack + token count by corr id.

    Synchronous on the launching thread, so both the Python stack and the vLLM
    forward context here belong to the code that issued the launch.
    """
    if domain != _cupti.CallbackDomain.DRIVER_API:
        return
    if callback_id not in _driver_launch_cbids:
        return
    if cbdata.callback_site != _cupti.ApiCallbackSite.API_ENTER:
        return
    stack = _capture_vllm_stack()
    num_tokens = _get_num_tokens()
    with _stacks_lock:
        _pending[cbdata.correlation_id] = (stack, num_tokens)
        if len(_pending) > _MAX_PENDING_STACKS:
            _pending.popitem(last=False)  # evict oldest


def _buffer_requested():
    """CUPTI asks for an activity buffer: (size_bytes, max_num_records)."""
    return _BUFFER_SIZE, 0


def _buffer_completed(activities: list):
    """CUPTI returns a batch of completed activity records (background thread)."""
    for activity in activities:
        if activity.kind not in _kernel_kinds:
            continue
        try:
            start = activity.start
            duration = activity.end - start
            name = _short_kernel_name(_demangle(activity.name))
            with _stacks_lock:
                stack, num_tokens = _pending.pop(activity.correlation_id, (None, None))
        except Exception:
            continue
        logger.info(
            "[cupti] kernel start_ns=%d gpu_dur_ns=%d num_tokens=%s name=%s vllm_stack=%s",
            start,
            duration,
            num_tokens if num_tokens is not None else "?",
            name,
            stack if stack else "<none>",
        )


def _periodic_flush():
    while not _flush_stop.wait(_FLUSH_INTERVAL_S):
        try:
            # Forced flush so partially filled buffers reach the console.
            _cupti.activity_flush_all(1)
        except Exception:
            pass


def start_cupti_profiling() -> None:
    """Enable CUPTI kernel activity tracing (idempotent, best-effort).

    Must be called from the GPU worker process. Failures (missing cupti-python,
    enable errors) are logged and swallowed so they never bring down the worker.
    """
    global _active, _cupti, _demangle, _kernel_kinds, _flush_thread
    global _subscriber, _vllm_dir, _driver_launch_cbids
    global _get_fwd_ctx, _fwd_ctx_available

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

            # Forward-context accessors for the per-launch batch token count.
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
            # CONCURRENT_KERNEL covers normal + graph kernels; include KERNEL as
            # a fallback name for the serialized-kernel record kind.
            _kernel_kinds = {cupti.ActivityKind.CONCURRENT_KERNEL}
            if hasattr(cupti.ActivityKind, "KERNEL"):
                _kernel_kinds.add(cupti.ActivityKind.KERNEL)

            cupti.activity_register_callbacks(_buffer_requested, _buffer_completed)
            cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)

            # DRIVER-domain launch callback for per-kernel vLLM callstacks.
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
                target=_periodic_flush, name="cupti-flush", daemon=True
            )
            _flush_thread.start()

            _active = True
            logger.info("CUPTI kernel activity tracing enabled.")
        except Exception as e:
            logger.warning("Failed to enable CUPTI profiling: %s", e)
            _active = False


def stop_cupti_profiling() -> None:
    """Disable CUPTI tracing and flush any remaining records (best-effort)."""
    global _active

    with _lock:
        if not _active:
            return
        try:
            _flush_stop.set()
            if _flush_thread is not None:
                _flush_thread.join(timeout=_FLUSH_INTERVAL_S + 1.0)
            if _subscriber is not None:
                _cupti.unsubscribe(_subscriber)
            _cupti.activity_disable(_cupti.ActivityKind.CONCURRENT_KERNEL)
            _cupti.activity_flush_all(1)
        except Exception as e:
            logger.warning("Failed to disable CUPTI profiling: %s", e)
        finally:
            _active = False
            with _stacks_lock:
                _pending.clear()
