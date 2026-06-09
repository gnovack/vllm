# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bar chart of MoE-backend latency vs token count from the sweep JSONL.

Reads the JSONL written by ``benchmark_deepseek_v4_moe.py --output`` (one record
per backend x token count) and draws a grouped bar chart: x-axis is the token
count, one bar group per backend, bar height is the chosen latency metric
(median ``p50_ms`` by default). Missing (backend, token) cells are simply
omitted. Y-axis is log-scaled since latency spans a wide range over the sweep.

Example:
    .venv/bin/python benchmarks/kernels/plot_deepseek_v4_moe.py \
        --input benchmarks/kernels/sweep_results/results_dp8ep.jsonl \
        --output benchmarks/kernels/sweep_results/latency_dp8ep.png \
        --title "DeepseekV4MoE median latency (DP/EP=8)"
"""

import argparse
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="results_*.jsonl from the sweep")
    ap.add_argument("--output", required=True, help="output PNG path")
    ap.add_argument("--title", default="DeepseekV4MoE latency by MoE backend")
    ap.add_argument(
        "--metric",
        default="p50_ms",
        choices=["p50_ms", "mean_ms", "min_ms"],
        help="which latency stat to plot (default p50_ms = median)",
    )
    args = ap.parse_args()

    rows = _load(args.input)
    if not rows:
        raise SystemExit(f"no records in {args.input}")

    def label(r):
        return f"{r['moe_backend']} ({r['expert_dtype']})"

    # data[backend][num_tokens] = metric
    data: dict[str, dict[int, float]] = defaultdict(dict)
    tokens: set[int] = set()
    for r in rows:
        data[label(r)][r["num_tokens"]] = r[args.metric]
        tokens.add(r["num_tokens"])
    token_list = sorted(tokens)
    backends = sorted(data)

    x = np.arange(len(token_list))
    width = 0.8 / max(len(backends), 1)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.7 * len(token_list)), 5.5))
    for i, b in enumerate(backends):
        vals = [data[b].get(t, np.nan) for t in token_list]
        offset = x + (i - (len(backends) - 1) / 2) * width
        bars = ax.bar(offset, vals, width, label=b)
        ax.bar_label(bars, fmt="%.2f", fontsize=6, rotation=90, padding=2)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in token_list])
    ax.set_xlabel("num_tokens (global batch)")
    ax.set_ylabel(f"{args.metric} latency (ms, log scale)")
    ax.set_title(args.title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}  ({len(backends)} backends, {len(token_list)} sizes)")


if __name__ == "__main__":
    main()
