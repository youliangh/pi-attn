"""Needle-row batch invariance and cold-weight latency, on one physical GPU."""
import argparse
import math

from common import benchmark, compare_bits, initialize, manifest, pi_native, save, torch

BATCHES = (1, 2, 3, 7, 8, 16, 31, 32, 33, 64, 65, 128, 129, 256, 257, 512)


def check_shape(name, n, k, functions, candidates, seed):
    torch.manual_seed(seed)
    w = (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
    probe = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
    refs = {key: fn(probe, w).clone() for key, fn in functions.items()}
    counts = {key: {"comparisons": 0, "different_elements": 0, "elements": 0, "failures": []} for key in functions}
    if candidates:
        import test_gemm as gemm
        fp32_ref = gemm.ext.linear(probe, w, 0, True).clone()
        for key in ("pi_all_candidates", "pi_all_candidates_fp32"):
            counts[key] = {"comparisons": 0, "different_elements": 0, "elements": 0, "failures": []}

    def check(key, got, pos, m, variant, ref):
        diff, size = compare_bits(got[pos], ref.expand(len(pos), -1))
        row = counts[key]
        row["comparisons"] += 1
        row["different_elements"] += diff
        row["elements"] += size
        if diff:
            row["failures"].append({"m": m, "variant": variant, "different_elements": diff})

    for m in BATCHES:
        pos = sorted({0, m // 2, m - 1})
        # Change unrelated rows twice; identical probes move with a permutation.
        for variant in (0, 1):
            x = torch.randn(m, k + 16, device="cuda", dtype=torch.bfloat16)[:, :k]
            x.mul_(1.0 if variant == 0 else 7.0)
            x[pos] = probe
            for key, fn in functions.items():
                check(key, fn(x, w), pos, m, variant, refs[key])
            if candidates and variant == 0:
                import test_gemm as gemm
                for kernel in gemm.candidates(m, n, k):
                    got = gemm.ext.linear(x, w, kernel, False)
                    assert got is not None
                    check("pi_all_candidates", got, pos, m, kernel, refs["pi_native"])
                    got = gemm.ext.linear(x, w, kernel, True)
                    assert got is not None
                    check("pi_all_candidates_fp32", got, pos, m, kernel, fp32_ref)
                del got
            del x

    # Independent FP64 reference over full K, only output columns are tiled.
    accuracy = {key: {"max_abs_error": 0.0, "squared_error": 0.0} for key in refs}
    ref_norm = 0.0
    for start in range(0, n, 4096):
        y64 = probe.double() @ w[start:start + 4096].double().T
        ref_norm += float(y64.square().sum())
        for key, y in refs.items():
            delta = y[:, start:start + 4096].double() - y64
            accuracy[key]["max_abs_error"] = max(accuracy[key]["max_abs_error"], float(delta.abs().max()))
            accuracy[key]["squared_error"] += float(delta.square().sum())
    for value in accuracy.values():
        value["relative_l2"] = math.sqrt(value.pop("squared_error") / ref_norm)
        assert value["relative_l2"] < 0.01, value
    return {"shape": name, "n": n, "k": k, "seed": seed, "checks": counts, "accuracy_fp64": accuracy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stock", "invariant"], default="invariant")
    ap.add_argument("--phase", choices=["checks", "bench"], default="checks")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    bi = initialize(a.mode == "invariant")
    import test_gemm as gemm
    shapes = {key: value for key, value in gemm.SHAPES.items() if key.startswith(("0.6B", "1.7B"))}
    if a.smoke:
        shapes = {"0.6B qkv": shapes["0.6B qkv"]}
    functions = {"stock_cublas": lambda x, w: x @ w.T}
    native = None
    if a.mode == "invariant":
        native = pi_native()
        functions = {"pi_native": native.linear,
                     "vllm_active_cublaslt": lambda x, w: x @ w.T,
                     "vllm_triton_explicit": lambda x, w: bi.matmul_persistent(x, w.T)}
    measured_batches = list(BATCHES) if a.phase == "checks" else ([1, 128] if a.smoke else [1, 8, 32, 128, 512])
    report = {"environment": manifest(bi), "batches": measured_batches, "rows": [], "complete": False}
    output = f"gemm-{a.mode}-{a.phase}{'-smoke' if a.smoke else ''}.json"
    for name, (n, k) in shapes.items():
        if a.phase == "checks":
            for seed in ((0,) if a.smoke else (0, 1, 7)):
                row = check_shape(name, n, k, functions, a.mode == "invariant", seed)
                report["rows"].append(row)
                print(name, seed, {key: val["different_elements"] for key, val in row["checks"].items()}, flush=True)
                save(output, report)
        else:
            torch.manual_seed(0)
            layers = max(1, math.ceil((160 << 20) / (n * k * 2)))
            weights = [(torch.randn(n, k, device="cuda") * 0.02).bfloat16() for _ in range(layers)]
            for m in ((1, 128) if a.smoke else (1, 8, 32, 128, 512)):
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                fns = {key: (lambda i, fn=fn: fn(x, weights[i])) for key, fn in functions.items()}
                row = {"shape": name, "m": m, "n": n, "k": k, **benchmark(fns, layers)}
                if native:
                    row["pi_kernel"] = native.tuned_kernel(n, k, native.bucket(m))
                report["rows"].append(row)
                print(row, flush=True)
                save(output, report)
            del weights, x, fns
        torch.cuda.empty_cache()
    report["complete"] = True
    save(output, report)
    # A baseline failure is an observed result. A PI failure must fail the run.
    if a.phase == "checks" and a.mode == "invariant":
        assert all(row["checks"][key]["different_elements"] == 0 for row in report["rows"]
                   for key in ("pi_native", "pi_all_candidates", "pi_all_candidates_fp32")), "PI batch invariance failed"


if __name__ == "__main__":
    main()
