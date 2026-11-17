"""Controlled request queues plus a short real-model GPU kernel trace."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import uuid

from serve import environment, ROOT
from serving_utils import OUT, save, difference, assert_pi_routing, expected_shapes


def pack(result):
    o = result.outputs[0]
    values = [row[token].logprob for row, token in zip(o.logprobs, o.token_ids)]
    assert len(values) == len(o.token_ids) == 32
    assert all(math.isfinite(x) for x in values)
    return {"token_ids": o.token_ids, "logprobs": values,
            "logprob_bits": [struct.unpack("<I", struct.pack("<f", x))[0] for x in values]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    os.environ.update(environment("pi", a.model, "base", audit=True))
    from vllm import LLM, SamplingParams
    folder = OUT / a.model / "pi" / "offline"
    trace_dir = ROOT / "env/cache/vllm027/traces" / a.model / uuid.uuid4().hex
    trace_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=str(ROOT / "models" / a.model), dtype="bfloat16", max_model_len=2048,
              max_num_seqs=16, max_num_batched_tokens=512, block_size=16,
              kv_cache_memory_bytes=1879048192, gpu_memory_utilization=0.85, seed=0,
              enable_prefix_caching=True, enable_chunked_prefill=True, async_scheduling=True,
              attention_backend="CUSTOM", worker_cls="pi_attn_vllm.worker.PIWorker",
              compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16],
                                  "cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["all"],
                                  "pass_config": {"enable_qk_norm_rope_fusion": False}},
              profiler_config={"profiler": "torch", "torch_profiler_dir": str(trace_dir),
                               "torch_profiler_with_stack": False, "torch_profiler_use_gzip": True,
                               "torch_profiler_record_shapes": True})
    rng = random.Random(771)
    prompts = [[rng.randrange(1000, 150000) for _ in range(24)] for _ in range(16)]
    rows = []
    references = {}
    llm.collective_rpc("pi_reset_audit")
    try:
        for sampled in (False, True):
            params = [SamplingParams(temperature=0.8 if sampled else 0, top_p=0.95 if sampled else 1,
                                     top_k=50 if sampled else -1, seed=2000+i, max_tokens=32, ignore_eos=True, logprobs=0)
                      for i in range(16)]
            refs = [pack(llm.generate([{"prompt_token_ids": p}], [params[i]], use_tqdm=False)[0]) for i, p in enumerate(prompts)]
            references["sampled" if sampled else "greedy"] = refs
            for batch in (1, 2, 3, 4, 7, 8, 9, 16):
                assert llm.reset_prefix_cache()
                for order in (list(range(batch)), list(reversed(range(batch)))):
                    results = llm.generate([{"prompt_token_ids": prompts[i]} for i in order],
                                           [params[i] for i in order], use_tqdm=False)
                    for i, out in zip(order, results):
                        got = pack(out)
                        d = difference(got, refs[i])
                        rows.append({"batch": batch, "sampled": sampled, "request": i, **d, "output": got})
                        assert d["token_mismatches"] == d["logprob_mismatches"] == 0, rows[-1]
            print(a.model, sampled, "controlled batches passed", flush=True)
        runtime = llm.collective_rpc("pi_status")[0]
        assert_pi_routing(runtime, a.model)
        # Separate trace from the timed performance runs. Fresh prompts ensure
        # real prefill; multiple generated tokens exercise captured decode.
        assert llm.reset_prefix_cache()
        llm.start_profile(profile_prefix="pi_serving")
        llm.generate([{"prompt_token_ids": p} for p in prompts[:4]],
                     SamplingParams(temperature=0, max_tokens=4, ignore_eos=True), use_tqdm=False)
        llm.stop_profile()
        traces = set(trace_dir.rglob("*.json.gz"))
        assert traces, "profiler produced no fresh trace"
        counts = {}
        linear_shapes = set()
        for path in traces:
            with gzip.open(path, "rt") as f:
                data = json.load(f)
            for event in data.get("traceEvents", []):
                if event.get("name") == "pi_attn::linear":
                    dims = event.get("args", {}).get("Input Dims", [])
                    if len(dims) >= 2 and len(dims[1]) == 2:
                        linear_shapes.add(tuple(dims[1]))
                if event.get("cat") == "kernel":
                    name = event["name"]
                    counts[name] = counts.get(name, 0) + 1
        for expected in ("prefill_kernel", "decode_kernel"):
            assert any(expected in name for name in counts), (expected, list(counts)[:20])
        assert any("pi_gemm" in name or "gemm_kernel" in name or "cutlass::Kernel" in name for name in counts), list(counts)[:20]
        assert linear_shapes == expected_shapes(a.model), linear_shapes
        save(folder / "trace-kernels.json", {"kernel_counts": counts,
            "request_linear_weight_shapes": sorted(linear_shapes), "files": [
            {"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(traces)]})
        save(folder / "checks.json", {"complete": True, "prompts": prompts, "references": references,
                                      "rows": rows, "pi_runtime": runtime})
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
