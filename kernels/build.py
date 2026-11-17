"""JIT-build the PI-Attn torch extension. Arch list via PI_ATTN_ARCH (default: the local GPU)."""
import os

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))


def cuda_arch():
    """An explicit target also works when a build runs with the GPU hidden."""
    arch = os.environ.get("PI_ATTN_ARCH")
    if arch is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}"
    if not arch.isdigit():
        raise ValueError(f"PI_ATTN_ARCH must be a numeric SM target, got {arch!r}")
    return arch


def build(verbose=False):
    arch = cuda_arch()
    return load(
        name=f"pi_attn_ext_sm{arch}",
        sources=[os.path.join(HERE, "pi_attn_torch.cu")],
        extra_include_paths=[HERE],
        extra_cuda_cflags=["-O3", "-std=c++17", f"-gencode=arch=compute_{arch},code=sm_{arch}",
                           "--expt-relaxed-constexpr", "-lineinfo"],
        build_directory=os.environ.get("PI_ATTN_BUILD_DIR") or None,
        verbose=verbose,
    )
