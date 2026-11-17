#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
out="$ROOT/results/rtx5070/vllm027"
mkdir -p "$out"
uv sync --project "$ROOT/benchmarks/vllm027" --locked
source "$ROOT/benchmarks/vllm027/env.sh"
uv pip check --python "$VIRTUAL_ENV/bin/python"
cd "$ROOT/benchmarks/vllm027"
python -u compare_gemm.py --mode stock --phase checks 2>&1 | tee "$out/gemm-stock-checks.log"
python -u compare_gemm.py --mode invariant --phase checks 2>&1 | tee "$out/gemm-invariant-checks.log"
python -u compare_attention.py --phase checks 2>&1 | tee "$out/attention-checks.log"
python -u compare_gemm.py --mode invariant --phase bench 2>&1 | tee "$out/gemm-invariant-bench.log"
python -u compare_attention.py --phase bench 2>&1 | tee "$out/attention-bench.log"
python report.py
