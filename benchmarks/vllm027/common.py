"""Helpers for direct, unmodified vLLM 0.27.0 / PI kernel measurements."""
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/rtx5070/vllm027"
sys.path.insert(0, str(ROOT / "kernels"))


def initialize(invariant=True):
    assert importlib.metadata.version("vllm") == "0.27.0"
    os.environ["VLLM_BATCH_INVARIANT"] = "1" if invariant else "0"
    from vllm.model_executor.layers import batch_invariant as bi
    if invariant:
        bi.init_batch_invariance()
        assert bi._batch_invariant_MODE
    assert torch.cuda.get_device_capability() == (12, 0)
    torch.manual_seed(0)
    return bi


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(bi):
    from vllm.platforms import current_platform
    p = torch.cuda.get_device_properties(0)
    return {
        "gpu": p.name, "capability": list(torch.cuda.get_device_capability()),
        "packages": {n: importlib.metadata.version(n) for n in ("vllm", "torch", "triton", "flashinfer-python")},
        "lock_sha256": digest(Path(__file__).with_name("uv.lock")),
        "batch_invariant_source_sha256": digest(inspect.getfile(bi)),
        "batch_invariant_enabled": bi._batch_invariant_MODE,
        "sm80_dispatch": current_platform.is_device_capability_family(80),
        "blas_library": str(torch.backends.cuda.preferred_blas_library()),
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cublaslt_workspace": os.environ.get("CUBLASLT_WORKSPACE_SIZE"),
        "bf16_reduced_precision": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    }


def save(name, report):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(path)


def compare_bits(got, expected):
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    assert torch.isfinite(got).all() and torch.isfinite(expected).all()
    integer = torch.int16 if got.element_size() == 2 else torch.int32
    different = got.contiguous().view(integer) != expected.contiguous().view(integer)
    return int(different.sum()), got.numel()


def pi_native():
    import flashinfer
    from build import cuda_arch
    from torch.utils.cpp_extension import load
    return load(
        name=f"pi_native_vllm027_sm{cuda_arch()}",
        sources=[str(ROOT / "kernels/pi_vllm_torch.cu")],
        extra_include_paths=[str(ROOT / "llm_infer_rust/src/kernels/pi"),
                             str(Path(flashinfer.__file__).parent / "data/cutlass/include")],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                          f"-gencode=arch=compute_{cuda_arch()},code=sm_{cuda_arch()}"],
    )


def time_graph(fn, layers=1, repeats=20):
    """Return median CUDA time per call, outside planning and Python launches."""
    calls = ((10 + layers - 1) // layers) * layers
    fn(0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(calls):
            fn(i % layers)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b) * 1000 / calls)
    return statistics.median(samples)


def benchmark(functions, layers):
    samples = {name: [] for name in functions}
    for repeat in range(3):
        names = list(functions)
        if repeat % 2:
            names.reverse()
        for name in names:
            samples[name].append(time_graph(functions[name], layers))
    return {"samples_us": samples,
            "median_us": {name: statistics.median(values) for name, values in samples.items()}}
