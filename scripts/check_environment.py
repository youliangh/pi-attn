"""Fail early unless the complete RTX 5070 build/runtime environment is usable."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess

import flashinfer
import torch


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert torch.cuda.is_available(), "CUDA is not available (WSL GPU access may require leaving the sandbox)"
    assert torch.cuda.get_device_capability() == (12, 0), "rtx5070 profile expects SM120"
    assert torch.cuda.is_bf16_supported(), "BF16 is required"
    assert torch.version.cuda == "13.0", f"expected cu130, got CUDA {torch.version.cuda}"
    nvcc = Path(os.environ["CUDA_HOME"]) / "bin/nvcc"
    assert "sm_120" in command(str(nvcc), "--list-gpu-code"), "nvcc cannot compile SM120"
    data = Path(flashinfer.__file__).parent / "data"
    for header in (data / "include/flashinfer/attention/decode.cuh", data / "cutlass/include/cutlass/cutlass.h"):
        assert header.is_file(), f"missing build header: {header}"
    x = torch.ones(32, 64, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(x @ x.T, torch.full((32, 32), 64, device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(0)
    report = {
        "gpu": props.name, "capability": list(torch.cuda.get_device_capability()),
        "memory_bytes": props.total_memory, "shared_memory_per_block_optin": props.shared_memory_per_block_optin,
        "multiprocessors": props.multi_processor_count,
        "driver": command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        "python": command("python", "--version"), "uv": command("uv", "--version"),
        "rustc": command("rustc", "--version"), "cargo": command("cargo", "--version"),
        "nvcc": command(str(nvcc), "--version"), "cxx": command(os.environ.get("CXX", "g++"), "--version"),
        "packages": {name: metadata.version(name) for name in ("torch", "triton", "flashinfer-python", "transformers")},
        "git_commit": command("git", "-C", str(root), "rev-parse", "HEAD"),
        "lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "arch": {name: os.environ.get(name) for name in ("PI_ATTN_ARCH", "FLASHINFER_CUDA_ARCH", "TORCH_CUDA_ARCH_LIST")},
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
