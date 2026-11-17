"""Same-request bitwise checks under batching, caches, graph and config changes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import random
import sys

from serving_utils import ROOT, OUT, server, save, difference, assert_pi_routing


def sampling(i, sampled):
    return {"temperature": 0.8, "top_p": 0.95, "top_k": 50, "seed": 7000 + i} if sampled else {"temperature": 0}


def prompts_for(model):
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT / "detcheck"))
    from detcheck import make_prompts
    tok = AutoTokenizer.from_pretrained(ROOT / "models" / model)
    return make_prompts(tok, random.Random(19), 16, [24, 300, 1100, 1500])


def run(model, mode, config):
    refpath = OUT / model / mode / "references.json"
    prompts = prompts_for(model)
    rng = random.Random(29)
    report = {"model": model, "mode": mode, "configuration": config, "complete": False, "rows": []}
    with server(model, mode, config) as (c, folder):
        def record(scenario, i, got, ref, **details):
            report["rows"].append({"scenario": scenario, "request": i, **details,
                                   **difference(got, ref), "output": got})

        if config == "base":
            refs = {"prompts": prompts, "greedy": [], "sampled": []}
            for kind, sampled in (("greedy", False), ("sampled", True)):
                c.flush()
                refs[kind] = [c.generate(p, 64, sampling(i, sampled)) for i, p in enumerate(prompts)]
            save(refpath, refs)
            for kind, sampled in (("greedy", False), ("sampled", True)):
                for concurrency in (1, 2, 3, 4, 7, 8, 9, 16):
                    c.flush()
                    order = list(range(16))
                    rng.shuffle(order)
                    with ThreadPoolExecutor(concurrency) as pool:
                        target = [(i, pool.submit(c.generate, prompts[i], 64, sampling(i, sampled))) for i in order]
                        # Additional arrivals differ in input and output lengths.
                        noise = [pool.submit(c.generate, [rng.randrange(1000, 150000) for _ in range(n)], t,
                                             {"temperature": 0.8, "seed": 99900 + j})
                                 for j, (n, t) in enumerate(((16, 8), (300, 64), (1500, 96)))]
                        for i, f in target:
                            record("batch", i, f.result(), refs[kind][i], sampling=kind, concurrency=concurrency)
                        for f in noise:
                            f.result()
                    save(folder / "checks.json", report)
                for i in range(4):
                    for state in ("cold", "warm", "partial"):
                        if state != "warm":
                            c.flush()
                        if state == "partial":
                            c.generate(prompts[i][:len(prompts[i]) // 2], 1)
                        record("cache_" + state, i, c.generate(prompts[i], 64, sampling(i, sampled)), refs[kind][i], sampling=kind)
                with ThreadPoolExecutor(16) as pool:
                    fs = [pool.submit(c.generate, prompts[0], 64, sampling(0, sampled)) for _ in range(20)]
                    for i, f in enumerate(fs):
                        record("repeat", i, f.result(), refs[kind][0], sampling=kind)
            # A genuinely shared prefix, with different suffixes.
            shared = [rng.randrange(1000, 150000) for _ in range(384)]
            shared_prompts = [shared + [rng.randrange(1000, 150000) for _ in range(128)] for _ in range(16)]
            shared_refs = [c.generate(p) for p in shared_prompts]
            c.flush()
            c.generate(shared + [1000], 1)
            with ThreadPoolExecutor(16) as pool:
                got = list(pool.map(c.generate, reversed(shared_prompts)))
            for i, g in zip(reversed(range(16)), got):
                record("shared_prefix", i, g, shared_refs[i])
            for i in range(4):
                for cut in (1, 16, 32):
                    c.flush()
                    original = refs["greedy"][i]
                    tail = {key: val[cut:] for key, val in original.items()}
                    record("greedy_reprefill", i, c.generate(prompts[i] + original["token_ids"][:cut], 64-cut), tail, cut=cut)
                c.flush()
                original = refs["sampled"][i]
                record("sampled_teacher_force", i, c.teacher_force(prompts[i], original["token_ids"]), original)
        else:
            refs = json.loads(refpath.read_text())
            assert refs["prompts"] == prompts
            for kind, sampled in (("greedy", False), ("sampled", True)):
                c.flush()
                with ThreadPoolExecutor(16) as pool:
                    fs = [(i, pool.submit(c.generate, prompts[i], 64, sampling(i, sampled))) for i in reversed(range(16))]
                    for i, f in fs:
                        record("cross_configuration", i, f.result(), refs[kind][i], sampling=kind)
        if mode == "pi":
            status = c.rpc("pi_status")
            assert_pi_routing(status, model)
            report["pi_runtime"] = status
        report["complete"] = True
        report["token_mismatches"] = sum(x["token_mismatches"] for x in report["rows"])
        report["logprob_mismatches"] = sum(x["logprob_mismatches"] for x in report["rows"])
        save(folder / "checks.json", report)
        print(model, mode, config, len(report["rows"]), report["token_mismatches"], report["logprob_mismatches"], flush=True)
        if mode == "pi":
            assert report["token_mismatches"] == report["logprob_mismatches"] == 0, "PI determinism failed; raw outputs saved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen3-0.6B", "Qwen3-1.7B"])
    ap.add_argument("--modes", nargs="+", default=["native", "batch-invariant", "pi"])
    ap.add_argument("--configs", nargs="+", default=["base", "eager", "nochunk", "chunk256", "maxseq1", "page32", "restart", "coldrestart"])
    a = ap.parse_args()
    for model in a.models:
        for mode in a.modes:
            for config in a.configs:
                run(model, mode, config)


if __name__ == "__main__":
    main()
