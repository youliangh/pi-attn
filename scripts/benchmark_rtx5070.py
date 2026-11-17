"""Same-device CUDA-graph latency: cold weights/KV, three alternating rounds."""
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels"))
import torch
import bench_attention as attention
import test_gemm as gemm


def main():
    torch.manual_seed(0)
    attention.warm_gpu()
    rows = []
    out = ROOT / "results/rtx5070/kernel-benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def save(complete=False):
        out.write_text(json.dumps({"device": torch.cuda.get_device_name(), "complete": complete, "rows": rows}, indent=2) + "\n")

    for shape, (n, k) in gemm.SHAPES.items():
        if not shape.startswith(("0.6B", "1.7B")):
            continue
        count = max(1, -(-(160 << 20) // (n * k * 2)))
        weights = [torch.randn(n, k, dtype=torch.bfloat16, device="cuda") for _ in range(count)]
        for m in (1, 8, 32, 128, 512):
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            # Select by device execution time; every candidate is validated separately.
            timings = {candidate: attention.timeit(lambda i, candidate=candidate: gemm.ext.linear(x, weights[i], candidate, False), count)
                       for candidate in gemm.candidates(m, n, k)}
            candidate = min(timings, key=timings.get)
            samples = {"pi": [], "cublas": []}
            torch.cuda.reset_peak_memory_stats()
            for repeat in range(3):
                for mode in (("pi", "cublas") if repeat % 2 == 0 else ("cublas", "pi")):
                    fn = (lambda i: gemm.ext.linear(x, weights[i], candidate, False)) if mode == "pi" else (lambda i: x @ weights[i].T)
                    samples[mode].append(attention.timeit(fn, count))
            rows.append({"kind": "gemm", "shape": shape, "m": m, "n": n, "k": k, "kernel": candidate,
                         "samples_us": samples, "median_us": {name: statistics.median(v) for name, v in samples.items()},
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
            print(json.dumps(rows[-1]), flush=True)
            save()
            del x
        del weights
        torch.cuda.empty_cache()
    for kind, cases in (("decode", [(1, 1024), (4, 2048), (1, 8192)]),
                        ("prefill", [(1, 512, 0), (1, 2048, 0), (4, 512, 0), (1, 128, 4096)])):
        for case in cases:
            torch.cuda.reset_peak_memory_stats()
            samples = [(attention.bench_decode(*case, 16, 8, pi_first=bool(repeat % 2)) if kind == "decode"
                        else attention.bench_prefill(*case, 16, 8, pi_first=bool(repeat % 2))) for repeat in range(3)]
            for result in samples:
                assert isinstance(result["fi"], (int, float)), f"FlashInfer baseline failed: {result}"
            rows.append({"kind": kind, "shape": case, "samples_us": samples,
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
            print(json.dumps(rows[-1]), flush=True)
            save()
    save(complete=True)


if __name__ == "__main__":
    main()
