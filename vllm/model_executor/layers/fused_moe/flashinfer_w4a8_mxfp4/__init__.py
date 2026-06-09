# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A8 MXFP4 MoE kernel for Hopper (SM90), vendored from FlashInfer PR #3516.

FP8-e4m3 activation x MXFP4 weight grouped GEMMs with a fused gather/scatter +
SwiGLU epilogue and a Triton reduction pass. The kernel files
(``w4a8_mxfp4_moe.py``, ``w4a8_mxfp4_grouped_gemm_sm90.py``,
``moe_reduce_triton.py``) are copied near-verbatim from upstream; only their
imports of FlashInfer-internal helpers are redirected to ``_shims`` so they run
standalone. See ``_shims.py`` for details.

Public surface used by vLLM:
    - ``w4a8_mxfp4_moe`` (full MoE forward)
    - ``interleave_w4a8_fc1_gate_up`` (one-time load-time weight prep)
"""

from functools import cache


@cache
def has_flashinfer_w4a8_mxfp4() -> bool:
    """True if the vendored W4A8 MXFP4 kernel and its CuTe-DSL/Triton
    dependencies import successfully on this machine."""
    try:
        from .w4a8_mxfp4_moe import (  # noqa: F401
            interleave_w4a8_fc1_gate_up,
            w4a8_mxfp4_moe,
        )

        return True
    except Exception:
        return False


__all__ = ["has_flashinfer_w4a8_mxfp4"]
