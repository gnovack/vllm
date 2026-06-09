#!/usr/bin/env bash
# Sweep the compatible DeepseekV4MoE MoE backends x a token-count range for two
# 8-GPU parallelism configs:
#   * dp8ep : --data-parallel-size 8 --enable-expert-parallel
#   * tp8   : --tensor-parallel-size 8   (no EP)
# Each run benchmarks all token counts and appends JSONL to
# $OUTDIR/results_<mode>.jsonl. Failures are logged and skipped (the chart then
# shows whatever succeeded). See deepseek_v4_moe_backends.md for the backend
# compatibility table this list is drawn from.
#
# Env overrides: MODEL, OUTDIR, TOKENS, LAYER, WARMUP, ITERS, PY.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

PY="${PY:-.venv/bin/python}"
BENCH="benchmarks/kernels/benchmark_deepseek_v4_moe.py"
MODEL="${MODEL:-./.dsv4_real_cfg}"        # dir with a real DeepSeek-V4 config.json
OUTDIR="${OUTDIR:-benchmarks/kernels/sweep_results}"
TOKENS="${TOKENS:-16 64 256 1024 4096 8192}"
LAYER="${LAYER:-5}"                        # routed layer (>= num_hash_layers)
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-30}"
# Set CUDA_GRAPH=1 to time CUDA-graph replays instead of eager launches.
# (Not every backend is graph-capturable; failures are logged and skipped.)
GRAPH_FLAG=""
[[ "${CUDA_GRAPH:-0}" == "1" ]] && GRAPH_FLAG="--cuda-graph"
mkdir -p "$OUTDIR"

# Compatible backends (deepseek_v4_moe_backends.md): label|expert_dtype|moe_backend|needs_ep
BACKENDS=(
  "marlin_mxfp4|fp4|marlin|0"
  "triton_fp8|fp8|triton|0"
  "flashinfer_cutlass_fp8|fp8|flashinfer_cutlass|0"
  "deep_gemm_fp8|fp8|deep_gemm|0"
  "mega_fp8|fp8|deep_gemm_mega_moe|1"
)

run_mode() {
  local mode="$1"; local parallel="$2"
  local out="$OUTDIR/results_${mode}.jsonl"
  : > "$out"
  echo "########## MODE=$mode ($parallel) ##########"
  for spec in "${BACKENDS[@]}"; do
    IFS='|' read -r label edt mb need_ep <<< "$spec"
    if [[ "$mode" == "tp8" && "$need_ep" == "1" ]]; then
      echo "[skip] $label (needs expert parallel, not run in $mode)"
      continue
    fi
    local log="$OUTDIR/log_${mode}_${label}.log"
    echo "=== [$mode] $label : expert_dtype=$edt moe_backend=$mb ==="
    # shellcheck disable=SC2086
    if $PY -u "$BENCH" --model "$MODEL" --trust-remote-code \
        --num-tokens $TOKENS --layer-idx "$LAYER" \
        --warmup "$WARMUP" --iters "$ITERS" --max-model-len 16384 \
        --expert-dtype "$edt" --moe-backend "$mb" $parallel $GRAPH_FLAG \
        --output "$out" > "$log" 2>&1; then
      echo "  ok -> $out"
    else
      echo "  FAILED (exit $?); last lines of $log:"
      tail -4 "$log" | sed 's/^/    /'
    fi
  done
  echo "wrote $out"
}

run_mode "dp8ep" "--data-parallel-size 8 --enable-expert-parallel"
run_mode "tp8"   "--tensor-parallel-size 8"
echo "DONE. Results: $OUTDIR/results_dp8ep.jsonl , $OUTDIR/results_tp8.jsonl"
