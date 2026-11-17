# Sourced by ../env.sh; ROOT is the checkout directory.
export PI_ATTN_PROFILE=rtx5070
export VIRTUAL_ENV="${VENV_DIR:-$ROOT/.venv}"
export VENV_DIR="$VIRTUAL_ENV"
export CUDA_HOME="${PI_ATTN_CUDA_HOME:-$ROOT/env/cuda-13.0}"
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$HOME/.cargo/bin:$PATH"
export LIBTORCH_USE_PYTORCH=1
export PI_ATTN_ARCH=120 FLASHINFER_CUDA_ARCH=120 TORCH_CUDA_ARCH_LIST=12.0
export CC="${CC:-gcc}" CXX="${CXX:-g++}"
export NVCC_CCBIN="$CXX" CUDAHOSTCXX="$CXX"
export UV_PYTHON_INSTALL_DIR="$ROOT/env/python" UV_CACHE_DIR="$ROOT/env/uv-cache"
export HF_HOME="$ROOT/env/hf-home"
export XDG_CACHE_HOME="$ROOT/env/cache/rtx5070" XDG_CONFIG_HOME="$ROOT/env/config/rtx5070"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" CUDA_CACHE_PATH="$XDG_CACHE_HOME/nv"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export TMPDIR="$ROOT/env/tmp"
export MAX_JOBS="${MAX_JOBS:-2}" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"
if [[ -x "$VIRTUAL_ENV/bin/python" ]] && "$VIRTUAL_ENV/bin/python" -c 'import importlib.util; assert importlib.util.find_spec("torch")' 2>/dev/null; then
  TORCH_LIB=$("$VIRTUAL_ENV/bin/python" -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("torch").origin).parent / "lib")')
  export TORCH_LIB
  export LD_LIBRARY_PATH="$TORCH_LIB:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
