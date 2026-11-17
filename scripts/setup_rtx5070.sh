#!/usr/bin/env bash
# Project-local toolkit and uv environment; use the already installed Rust toolchain.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
download_models=true
case "${1:-}" in
  --no-models) download_models=false; shift ;;
esac
[[ $# == 0 ]] || { echo "Usage: $0 [--no-models]" >&2; exit 2; }
export PI_ATTN_PROFILE=rtx5070
source "$ROOT/scripts/env_rtx5070.sh"
command -v uv >/dev/null
rustc --version
cargo --version
mkdir -p "$ROOT/env/downloads" "$ROOT/results/rtx5070"
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  installer="$ROOT/env/downloads/cuda_13.0.2_580.95.05_linux.run"
  curl -fL --retry 5 -C - -o "$installer" \
    "${CUDA_URL:-https://developer.download.nvidia.com/compute/cuda}/13.0.2/local_installers/$(basename "$installer")"
  install_tmp=$(mktemp -d "$TMPDIR/cuda-install.XXXXXX")
  sh "$installer" --silent --toolkit --toolkitpath="$CUDA_HOME" \
    --no-opengl-libs --no-man-page --no-drm --tmpdir="$install_tmp"
  rmdir "$install_tmp"
fi
cd "$ROOT"
UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV" uv sync --locked --all-packages --extra cuda --group dev
echo rtx5070 > "$ROOT/env/profile"
source "$ROOT/env.sh"
uv pip check --python "$VIRTUAL_ENV/bin/python"
python "$ROOT/scripts/check_environment.py" --out "$ROOT/results/rtx5070/environment.json"
if "$download_models"; then
  python "$ROOT/scripts/download_models.py"
fi
