# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op shims for the FlashInfer-internal helpers the vendored W4A8 MXFP4
kernel files reference.

The kernel sources are copied near-verbatim from FlashInfer PR #3516
(``flashinfer/fused_moe/cute_dsl/``) so they stay easy to re-sync. Upstream they
import a couple of non-essential utilities (an API-logging decorator and a trace
template) from elsewhere in the ``flashinfer`` package. We don't vendor those
subsystems, so this module provides inert stand-ins and the vendored files import
from here instead.
"""


def flashinfer_api(func=None, *, trace=None, **kwargs):
    """Identity decorator standing in for ``flashinfer.api_logging.flashinfer_api``.

    Supports both usages found in the kernel sources::

        @flashinfer_api
        def f(...): ...

        @flashinfer_api(trace=some_trace)
        def g(...): ...
    """
    if func is not None and callable(func):
        return func

    def _wrap(f):
        return f

    return _wrap


# Stand-in for ``flashinfer.trace.templates.moe.w4a8_mxfp4_moe_trace``. Only ever
# passed as the ``trace=`` argument to the (no-op) ``flashinfer_api`` above.
w4a8_mxfp4_moe_trace = None
