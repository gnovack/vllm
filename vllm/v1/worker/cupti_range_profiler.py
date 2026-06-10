# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-contained ctypes binding of the CUPTI Range Profiler (autorange +
kernel-replay) for per-kernel `dram__bytes` (HW counters).

Used by cupti_profiler.py to collect HBM read/write bytes per kernel -- which,
unlike SASS instruction patching, works on cuBLAS/cutlass GEMM kernels too.

Both the host-side (cuptiProfilerHost*) and device-side (cuptiRangeProfiler*)
APIs are bound here from the CUPTI 13.1 header layout. We do NOT use cupti-python
for the host side: it's v13.3 and its struct sizes mismatch the (system) libcupti
13.1 -> ERROR_INVALID_PARAMETER. So everything goes through this one libcupti.

Requirements (checked by the caller): admin/root (HW perf counters are gated by
RmProfilingAdminOnly), and the system libcupti preloaded.
"""

import ctypes as C
import os


def _system_libcupti_path() -> str:
    """Path of the SYSTEM libcupti mapped in this process (the LD_PRELOAD'd one),
    NOT torch's bundled libcupti.so.13 (cu130 wheel = CUPTI 13.0.0, whose Range
    Profiler returns CUPTI_ERROR_UNKNOWN). CDLL('libcupti.so.13') by soname can
    resolve to torch's copy, so we load the system one by full path."""
    try:
        with open(f"/proc/{os.getpid()}/maps") as f:
            for ln in f:
                if "libcupti.so" in ln:
                    p = ln.split()[-1]
                    if "site-packages" not in p and "/nvidia/" not in p:
                        return p
    except Exception:
        pass
    return "libcupti.so.13"


_lib = C.CDLL(_system_libcupti_path())
_lib.cuptiGetResultString.argtypes = [C.c_int, C.POINTER(C.c_char_p)]
_lib.cuptiGetResultString.restype = C.c_int

AUTO_RANGE = 1            # CUpti_AutoRange
KERNEL_REPLAY = 2        # CUpti_KernelReplay
PROFILER_TYPE_RANGE = 0  # CUPTI_PROFILER_TYPE_RANGE_PROFILER
_P = C.c_void_p
_MAX_RANGES = 4096


def _f(*fields):
    return [("structSize", C.c_size_t), ("pPriv", C.c_void_p)] + list(fields)


def _ssize(cls, last):
    types = {n: t for n, t, *_ in cls._fields_}
    return getattr(cls, last).offset + C.sizeof(types[last])


def _check(rc, fn):
    if rc != 0:
        s = C.c_char_p()
        _lib.cuptiGetResultString(rc, C.byref(s))
        raise RuntimeError(f"{fn}: CUPTI error {rc} ({s.value and s.value.decode()})")


def _call(name, params):
    fn = getattr(_lib, name)
    fn.argtypes = [C.c_void_p]
    fn.restype = C.c_int
    _check(fn(C.byref(params)), name)


# ---- profiler init / chip name ----
class _ProfInit(C.Structure):
    _fields_ = _f()


class _ChipName(C.Structure):
    _fields_ = _f(("deviceIndex", C.c_size_t), ("pChipName", C.c_char_p))


# ---- host ----
class _GetCtrAvail(C.Structure):
    _fields_ = _f(("ctx", _P), ("counterAvailabilityImageSize", C.c_size_t),
                  ("pCounterAvailabilityImage", _P), ("bAllowDeviceLevelCounters", C.c_bool))


class _HInit(C.Structure):
    _fields_ = _f(("profilerType", C.c_int), ("pChipName", C.c_char_p),
                  ("pCounterAvailabilityImage", _P), ("pHostObject", _P))


class _HDeinit(C.Structure):
    _fields_ = _f(("pHostObject", _P))


class _HConfigAdd(C.Structure):
    _fields_ = _f(("pHostObject", _P), ("ppMetricNames", _P), ("numMetrics", C.c_size_t))


class _HCfgSize(C.Structure):
    _fields_ = _f(("pHostObject", _P), ("configImageSize", C.c_size_t))


class _HCfgImage(C.Structure):
    _fields_ = _f(("pHostObject", _P), ("configImageSize", C.c_size_t), ("pConfigImage", _P))


class _HEval(C.Structure):
    _fields_ = _f(("pHostObject", _P), ("pCounterDataImage", _P),
                  ("counterDataImageSize", C.c_size_t), ("rangeIndex", C.c_size_t),
                  ("ppMetricNames", _P), ("numMetrics", C.c_size_t), ("pMetricValues", _P))


class _HNumPasses(C.Structure):
    _fields_ = _f(("configImageSize", C.c_size_t), ("pConfigImage", _P),
                  ("numOfPasses", C.c_size_t))


# ---- device (range profiler) ----
class _Enable(C.Structure):
    _fields_ = _f(("ctx", _P), ("pRangeProfilerObject", _P))


class _Disable(C.Structure):
    _fields_ = _f(("pRangeProfilerObject", _P))


class _Start(C.Structure):
    _fields_ = _f(("pRangeProfilerObject", _P))


class _Stop(C.Structure):
    _fields_ = _f(("pRangeProfilerObject", _P), ("passIndex", C.c_size_t),
                  ("targetNestingLevel", C.c_size_t), ("isAllPassSubmitted", C.c_uint8))


class _SetConfig(C.Structure):
    _fields_ = _f(("pRangeProfilerObject", _P), ("configSize", C.c_size_t), ("pConfig", _P),
                  ("counterDataImageSize", C.c_size_t), ("pCounterDataImage", _P),
                  ("range", C.c_int), ("replayMode", C.c_int), ("maxRangesPerPass", C.c_size_t),
                  ("numNestingLevels", C.c_uint16), ("minNestingLevel", C.c_uint16),
                  ("passIndex", C.c_size_t), ("targetNestingLevel", C.c_uint16))


class _GetCDSize(C.Structure):
    _fields_ = _f(("pRangeProfilerObject", _P), ("pMetricNames", _P), ("numMetrics", C.c_size_t),
                  ("maxNumOfRanges", C.c_size_t), ("maxNumRangeTreeNodes", C.c_uint32),
                  ("counterDataSize", C.c_size_t))


class _CDInit(C.Structure):
    _fields_ = _f(("pRangeProfilerObject", _P), ("counterDataSize", C.c_size_t), ("pCounterData", _P))


class _Decode(C.Structure):
    _fields_ = _f(("pRangeProfilerObject", _P), ("numOfRangeDropped", C.c_size_t))


class _CDInfo(C.Structure):
    _fields_ = _f(("pCounterDataImage", _P), ("counterDataImageSize", C.c_size_t),
                  ("numTotalRanges", C.c_size_t))


_profiler_initialized = False


def _ensure_init():
    global _profiler_initialized
    if not _profiler_initialized:
        p = _ProfInit()
        p.structSize = _ssize(_ProfInit, "pPriv")
        _call("cuptiProfilerInitialize", p)
        _profiler_initialized = True


def chip_name(device=0):
    _ensure_init()
    g = _ChipName()
    g.structSize = _ssize(_ChipName, "pChipName")
    g.deviceIndex = device
    _call("cuptiDeviceGetChipName", g)
    return g.pChipName.decode()


def _build_config_image(metrics, ctx_int, device, np):
    """Returns (host_object, config_image, names_ptr_array, names_p). Caller must
    cuptiProfilerHostDeinitialize the host object when done."""
    chip = chip_name(device).encode()
    name_ptrs = (C.c_char_p * len(metrics))(*[m.encode() for m in metrics])
    names_p = C.cast(name_ptrs, _P)
    ga = _GetCtrAvail()
    ga.structSize = _ssize(_GetCtrAvail, "bAllowDeviceLevelCounters")
    ga.ctx = ctx_int
    _call("cuptiProfilerGetCounterAvailability", ga)
    avail = np.zeros(ga.counterAvailabilityImageSize, np.uint8)
    ga.pCounterAvailabilityImage = avail.ctypes.data
    _call("cuptiProfilerGetCounterAvailability", ga)
    hi = _HInit()
    hi.structSize = _ssize(_HInit, "pHostObject")
    hi.profilerType = PROFILER_TYPE_RANGE
    hi.pChipName = chip
    hi.pCounterAvailabilityImage = avail.ctypes.data
    _call("cuptiProfilerHostInitialize", hi)
    host = hi.pHostObject
    ca = _HConfigAdd()
    ca.structSize = _ssize(_HConfigAdd, "numMetrics")
    ca.pHostObject = host
    ca.ppMetricNames = names_p
    ca.numMetrics = len(metrics)
    _call("cuptiProfilerHostConfigAddMetrics", ca)
    cs = _HCfgSize()
    cs.structSize = _ssize(_HCfgSize, "configImageSize")
    cs.pHostObject = host
    _call("cuptiProfilerHostGetConfigImageSize", cs)
    config_image = np.zeros(cs.configImageSize, np.uint8)
    cig = _HCfgImage()
    cig.structSize = _ssize(_HCfgImage, "pConfigImage")
    cig.pHostObject = host
    cig.configImageSize = config_image.nbytes
    cig.pConfigImage = config_image.ctypes.data
    _call("cuptiProfilerHostGetConfigImage", cig)
    return host, config_image, name_ptrs, names_p


def _host_deinit(host):
    hd = _HDeinit()
    hd.structSize = _ssize(_HDeinit, "pHostObject")
    hd.pHostObject = host
    _call("cuptiProfilerHostDeinitialize", hd)


def num_passes(metrics, ctx_int, device=0, np=None) -> int:
    """Number of GPU passes a metric SET needs. ==1 means single-pass (safe: no
    kernel replay). >1 means multi-pass (would re-run kernels -> unsafe for
    stateful vLLM forwards). Note: passes are a property of the whole SET, not
    individual metrics -- adding a metric can push a set to multi-pass."""
    if np is None:
        import numpy as np
    host, config_image, _keep, _ = _build_config_image(metrics, ctx_int, device, np)
    try:
        gp = _HNumPasses()
        gp.structSize = _ssize(_HNumPasses, "numOfPasses")
        gp.configImageSize = config_image.nbytes
        gp.pConfigImage = config_image.ctypes.data
        _call("cuptiProfilerHostGetNumOfPasses", gp)
        return int(gp.numOfPasses)
    finally:
        _host_deinit(host)


def profile_once(metrics, run_forward, ctx_int, device=0, np=None):
    """Profile a single execution of run_forward() under autorange + kernel
    replay. Returns (per_range_values, forward_result):
      per_range_values: list indexed by range (= kernel launch order),
                        each a {metric: float}.
      forward_result:   whatever run_forward() returned.
    run_forward() is invoked exactly ONCE (single-pass metrics only; re-running
    would corrupt stateful kernels)."""
    if np is None:
        import numpy as np
    chip = chip_name(device).encode()
    name_ptrs = (C.c_char_p * len(metrics))(*[m.encode() for m in metrics])
    names_p = C.cast(name_ptrs, _P)

    ga = _GetCtrAvail()
    ga.structSize = _ssize(_GetCtrAvail, "bAllowDeviceLevelCounters")
    ga.ctx = ctx_int
    _call("cuptiProfilerGetCounterAvailability", ga)
    avail = np.zeros(ga.counterAvailabilityImageSize, np.uint8)
    ga.pCounterAvailabilityImage = avail.ctypes.data
    _call("cuptiProfilerGetCounterAvailability", ga)

    hi = _HInit()
    hi.structSize = _ssize(_HInit, "pHostObject")
    hi.profilerType = PROFILER_TYPE_RANGE
    hi.pChipName = chip
    hi.pCounterAvailabilityImage = avail.ctypes.data
    _call("cuptiProfilerHostInitialize", hi)
    host = hi.pHostObject
    try:
        ca = _HConfigAdd()
        ca.structSize = _ssize(_HConfigAdd, "numMetrics")
        ca.pHostObject = host
        ca.ppMetricNames = names_p
        ca.numMetrics = len(metrics)
        _call("cuptiProfilerHostConfigAddMetrics", ca)
        cs = _HCfgSize()
        cs.structSize = _ssize(_HCfgSize, "configImageSize")
        cs.pHostObject = host
        _call("cuptiProfilerHostGetConfigImageSize", cs)
        config_image = np.zeros(cs.configImageSize, np.uint8)
        cig = _HCfgImage()
        cig.structSize = _ssize(_HCfgImage, "pConfigImage")
        cig.pHostObject = host
        cig.configImageSize = config_image.nbytes
        cig.pConfigImage = config_image.ctypes.data
        _call("cuptiProfilerHostGetConfigImage", cig)

        en = _Enable()
        en.structSize = _ssize(_Enable, "pRangeProfilerObject")
        en.ctx = ctx_int
        _call("cuptiRangeProfilerEnable", en)
        obj = en.pRangeProfilerObject
        try:
            gs = _GetCDSize()
            gs.structSize = _ssize(_GetCDSize, "counterDataSize")
            gs.pRangeProfilerObject = obj
            gs.pMetricNames = names_p
            gs.numMetrics = len(metrics)
            gs.maxNumOfRanges = _MAX_RANGES
            gs.maxNumRangeTreeNodes = _MAX_RANGES
            _call("cuptiRangeProfilerGetCounterDataSize", gs)
            counter_data = np.zeros(gs.counterDataSize, np.uint8)
            ci = _CDInit()
            ci.structSize = _ssize(_CDInit, "pCounterData")
            ci.pRangeProfilerObject = obj
            ci.counterDataSize = counter_data.nbytes
            ci.pCounterData = counter_data.ctypes.data
            _call("cuptiRangeProfilerCounterDataImageInitialize", ci)

            sc = _SetConfig()
            sc.structSize = _ssize(_SetConfig, "targetNestingLevel")
            sc.pRangeProfilerObject = obj
            sc.configSize = config_image.nbytes
            sc.pConfig = config_image.ctypes.data
            sc.counterDataImageSize = counter_data.nbytes
            sc.pCounterDataImage = counter_data.ctypes.data
            sc.range = AUTO_RANGE
            sc.replayMode = KERNEL_REPLAY
            sc.maxRangesPerPass = _MAX_RANGES
            sc.numNestingLevels = 1
            sc.minNestingLevel = 1
            sc.passIndex = 0
            sc.targetNestingLevel = 1
            _call("cuptiRangeProfilerSetConfig", sc)

            st = _Start()
            st.structSize = _ssize(_Start, "pRangeProfilerObject")
            st.pRangeProfilerObject = obj
            _call("cuptiRangeProfilerStart", st)
            result = run_forward()  # exactly once
            sp = _Stop()
            sp.structSize = _ssize(_Stop, "isAllPassSubmitted")
            sp.pRangeProfilerObject = obj
            _call("cuptiRangeProfilerStop", sp)

            dd = _Decode()
            dd.structSize = _ssize(_Decode, "numOfRangeDropped")
            dd.pRangeProfilerObject = obj
            _call("cuptiRangeProfilerDecodeData", dd)
        finally:
            dis = _Disable()
            dis.structSize = _ssize(_Disable, "pRangeProfilerObject")
            dis.pRangeProfilerObject = obj
            _call("cuptiRangeProfilerDisable", dis)

        gi = _CDInfo()
        gi.structSize = _ssize(_CDInfo, "numTotalRanges")
        gi.pCounterDataImage = counter_data.ctypes.data
        gi.counterDataImageSize = counter_data.nbytes
        _call("cuptiRangeProfilerGetCounterDataInfo", gi)

        per_range = []
        for i in range(gi.numTotalRanges):
            vals = np.zeros(len(metrics), np.float64)
            ev = _HEval()
            ev.structSize = _ssize(_HEval, "pMetricValues")
            ev.pHostObject = host
            ev.pCounterDataImage = counter_data.ctypes.data
            ev.counterDataImageSize = counter_data.nbytes
            ev.rangeIndex = i
            ev.ppMetricNames = names_p
            ev.numMetrics = len(metrics)
            ev.pMetricValues = vals.ctypes.data
            _call("cuptiProfilerHostEvaluateToGpuValues", ev)
            per_range.append({m: float(v) for m, v in zip(metrics, vals)})
        return per_range, result
    finally:
        hd = _HDeinit()
        hd.structSize = _ssize(_HDeinit, "pHostObject")
        hd.pHostObject = host
        _call("cuptiProfilerHostDeinitialize", hd)
