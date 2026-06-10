#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read CUPTI metric SQLite DBs produced by --enable-cupti and report, per
kernel, how a metric varies with num_tokens.

Each kernel is identified by (name + vLLM callstack). Aggregates from multiple
per-rank DB files are merged (count/sum/sum_sq add; min/max combine).

Examples:
  python tools/cupti_report.py ./cupti_metrics
  python tools/cupti_report.py ./cupti_metrics --metric gpu_dur_ns --top 15
  python tools/cupti_report.py ./cupti_metrics --filter rms_norm
  python tools/cupti_report.py ./cupti_metrics --plot out.png
"""
import argparse
import glob
import math
import os
import sqlite3
import sys


def _db_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "*.db")))
    if not files:
        sys.exit(f"No .db files found under {path!r}")
    return files


def load(path: str):
    """Merge all rank DBs. Returns (kernels, agg).

    kernels: kernel_id -> (name, stack)
    agg:     (kernel_id, num_tokens, metric) -> [count, sum, sum_sq, min, max]
    """
    kernels: dict = {}
    agg: dict = {}
    for f in _db_files(path):
        conn = sqlite3.connect(f)
        try:
            for kid, name, stack in conn.execute(
                "SELECT kernel_id, name, stack FROM kernels"
            ):
                kernels.setdefault(kid, (name, stack))
            for kid, ntok, metric, count, s, ssq, mn, mx in conn.execute(
                "SELECT kernel_id, num_tokens, metric, count, sum, sum_sq, min, max "
                "FROM kernel_metrics"
            ):
                key = (kid, ntok, metric)
                cur = agg.get(key)
                if cur is None:
                    agg[key] = [count, s, ssq, mn, mx]
                else:
                    cur[0] += count
                    cur[1] += s
                    cur[2] += ssq
                    cur[3] = min(cur[3], mn)
                    cur[4] = max(cur[4], mx)
        finally:
            conn.close()
    return kernels, agg


def _mean_std(count, s, ssq):
    if count <= 0:
        return 0.0, 0.0
    mean = s / count
    var = max(ssq / count - mean * mean, 0.0)
    return mean, math.sqrt(var)


def report(kernels, agg, metric: str, top: int, name_filter: str | None):
    # Per-kernel rows for this metric, plus total time for ranking.
    per_kernel: dict = {}  # kid -> {num_tokens: [count, sum, sum_sq, min, max]}
    totals: dict = {}      # kid -> total sum
    for (kid, ntok, m), st in agg.items():
        if m != metric:
            continue
        per_kernel.setdefault(kid, {})[ntok] = st
        totals[kid] = totals.get(kid, 0.0) + st[1]

    ranked = sorted(per_kernel, key=lambda k: totals.get(k, 0.0), reverse=True)
    if name_filter:
        ranked = [k for k in ranked if name_filter in kernels.get(k, ("", ""))[0]]
    if top:
        ranked = ranked[:top]

    if not ranked:
        print(f"No data for metric {metric!r}"
              + (f" matching {name_filter!r}" if name_filter else ""))
        return

    for kid in ranked:
        name, stack = kernels.get(kid, ("<unknown>", ""))
        print("=" * 100)
        print(f"{name}")
        print(f"  id={kid}  total_{metric}={totals[kid]:,.0f}")
        if stack:
            print(f"  stack: {stack}")
        print(f"  {'num_tokens':>10} {'count':>8} {'mean':>12} "
              f"{'min':>10} {'max':>10} {'std':>10}")
        for ntok in sorted(per_kernel[kid]):
            count, s, ssq, mn, mx = per_kernel[kid][ntok]
            mean, std = _mean_std(count, s, ssq)
            label = "no-ctx" if ntok < 0 else str(ntok)
            print(f"  {label:>10} {count:>8} {mean:>12.1f} "
                  f"{mn:>10.0f} {mx:>10.0f} {std:>10.1f}")
    print("=" * 100)


def plot(kernels, agg, metric: str, top: int, name_filter, out_png: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not installed; install it or drop --plot")

    per_kernel: dict = {}
    totals: dict = {}
    for (kid, ntok, m), st in agg.items():
        if m != metric or ntok < 0:
            continue
        per_kernel.setdefault(kid, {})[ntok] = st
        totals[kid] = totals.get(kid, 0.0) + st[1]
    ranked = sorted(per_kernel, key=lambda k: totals.get(k, 0.0), reverse=True)
    if name_filter:
        ranked = [k for k in ranked if name_filter in kernels.get(k, ("", ""))[0]]
    ranked = ranked[: top or 10]

    plt.figure(figsize=(10, 6))
    for kid in ranked:
        xs = sorted(per_kernel[kid])
        ys = [per_kernel[kid][x][1] / per_kernel[kid][x][0] for x in xs]
        plt.plot(xs, ys, marker="o", label=kernels.get(kid, (kid,))[0][:40])
    plt.xlabel("num_tokens")
    plt.ylabel(f"mean {metric}")
    plt.title("Kernel metric vs num_tokens")
    plt.legend(fontsize=7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="DB file or directory of per-rank *.db files")
    ap.add_argument("--metric", default="gpu_dur_ns")
    ap.add_argument("--top", type=int, default=20, help="show top-N kernels by total")
    ap.add_argument("--filter", dest="name_filter", default=None,
                    help="only kernels whose name contains this substring")
    ap.add_argument("--plot", metavar="OUT.png", default=None)
    args = ap.parse_args()

    kernels, agg = load(args.path)
    report(kernels, agg, args.metric, args.top, args.name_filter)
    if args.plot:
        plot(kernels, agg, args.metric, args.top, args.name_filter, args.plot)


if __name__ == "__main__":
    main()
