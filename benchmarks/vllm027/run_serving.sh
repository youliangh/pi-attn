#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
uv sync --project "$ROOT/benchmarks/vllm027" --locked
source "$ROOT/benchmarks/vllm027/env.sh"
uv pip check --python "$VIRTUAL_ENV/bin/python"
exec python -u "$ROOT/benchmarks/vllm027/run_serving.py" "$@"
