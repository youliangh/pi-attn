#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PI_ATTN_PROFILE=rtx5070
source "$ROOT/env.sh"
mkdir -p "$ROOT/results/rtx5070"
phase=${1:-all}
case "$phase" in all|kernels|rust|e2e|bench) ;; *) echo "Usage: $0 [all|kernels|rust|e2e|bench]" >&2; exit 2 ;; esac
python "$ROOT/scripts/check_environment.py" --out "$ROOT/results/rtx5070/environment.json"
if [[ $phase == all || $phase == kernels ]]; then
  cd "$ROOT/kernels"
  for test in test_invariance test_directed test_matrix test_gemm test_mutants; do
    extra=()
    [[ $test != test_gemm ]] || extra=(--single-gpu)
    python -u "$test.py" "${extra[@]}" 2>&1 | tee "$ROOT/results/rtx5070/$test.log"
  done
fi
if [[ $phase == all || $phase == rust ]]; then
  cd "$ROOT/llm_infer_rust"
  cargo build --locked --release 2>&1 | tee "$ROOT/results/rtx5070/build.log"
  for mode in cpu cuda; do
    SGLANG_BUILD_PROFILE=release bash scripts/run-rust-tests.sh "$mode" 2>&1 | tee "$ROOT/results/rtx5070/rust-$mode.log"
  done
fi
if [[ $phase == all || $phase == e2e ]]; then
  python -u "$ROOT/scripts/validate_serving.py" --mode correctness
fi
if [[ $phase == all || $phase == bench ]]; then
  python -u "$ROOT/scripts/benchmark_rtx5070.py" 2>&1 | tee "$ROOT/results/rtx5070/kernel-benchmark.log"
  python -u "$ROOT/scripts/validate_serving.py" --mode benchmark
fi
