# Source after (or instead of) the root env.sh. Keep vLLM in its own uv workspace.
export ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export VENV_DIR="$ROOT/benchmarks/vllm027/.venv"
source "$ROOT/scripts/env_rtx5070.sh"
export TORCH_EXTENSIONS_DIR="$ROOT/env/cache/vllm027/torch_extensions"
export TRITON_CACHE_DIR="$ROOT/env/cache/vllm027/triton"
export VLLM_CACHE_ROOT="$ROOT/env/cache/vllm027/vllm"
export FLASHINFER_WORKSPACE_BASE="$ROOT/env/cache/vllm027/flashinfer"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"
