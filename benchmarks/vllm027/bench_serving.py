"""Streaming service measurements with token-ID based timing and raw samples."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import random
import statistics
import time

import requests
from serving_utils import server, save

WORKLOADS = [(64, 64, 1, 0), (64, 128, 16, 0), (512, 128, 4, 0),
             (1500, 128, 4, 0), (512, 128, 4, 384)]


def streaming_one(client, prompt, tokens, seed, sampled, epoch=None):
    sp = {"temperature": 0.8, "top_p": 0.95, "top_k": 50} if sampled else {"temperature": 0}
    payload = {"model": "default", "prompt": prompt, "max_tokens": tokens, "ignore_eos": True,
               "seed": seed, **sp, "return_token_ids": True, "stream": True,
               "stream_options": {"include_usage": True}}
    start = time.perf_counter()
    events, ids, usage = [], [], None
    with requests.post(client.url + "/v1/completions", json=payload, stream=True, timeout=1200) as r:
        r.raise_for_status()
        # None reads each HTTP chunk immediately instead of waiting for a
        # fixed-size buffer to fill across multiple token events.
        for line in r.iter_lines(chunk_size=None):
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            item = json.loads(line[5:])
            if item.get("usage"):
                usage = item["usage"]
            for choice in item.get("choices", []):
                delta = choice.get("token_ids") or []
                if delta:
                    events.append({"seconds": time.perf_counter() - start, "tokens": len(delta)})
                    ids.extend(delta)
    elapsed = time.perf_counter() - start
    assert usage and usage["completion_tokens"] == tokens == len(ids), (usage, len(ids), tokens)
    assert usage["prompt_tokens"] == len(prompt), usage
    assert events
    # Only single-token events define observable per-token intervals. Larger
    # chunks are explicitly retained, not split into fictional timestamps.
    itl = [(b["seconds"]-a["seconds"])*1000 for a, b in zip(events, events[1:]) if a["tokens"] == b["tokens"] == 1]
    return {"prompt_token_ids": prompt, "seed": seed, "sampling": sp,
            "started_seconds": start - epoch if epoch is not None else 0,
            "finished_seconds": start + elapsed - epoch if epoch is not None else elapsed,
            "token_ids": ids, "events": events, "usage": usage, "ttft_ms": events[0]["seconds"] * 1000,
            "tpot_ms": (events[-1]["seconds"]-events[0]["seconds"]) * 1000 / max(tokens-1, 1),
            "e2e_ms": elapsed*1000, "observable_itl_ms": itl}


def percentile(values, p):
    return sorted(values)[math.ceil(len(values)*p)-1]


def check_streaming(client, folder, mode):
    """Verify the timed API path (streaming, no logprobs) against strict output."""
    prompt = [9707, 11, 358, 2776, 264, 1657, 13]
    rows = []
    for sampled in (False, True):
        sp = {"temperature": .8, "top_p": .95, "top_k": 50} if sampled else {"temperature": 0}
        client.flush()
        ref = client.generate(prompt, 32, {**sp, "seed": 123})
        for state in ("cold", "warm"):
            if state == "cold":
                client.flush()
            got = streaming_one(client, prompt, 32, 123, sampled)
            different = sum(x != y for x, y in zip(ref["token_ids"], got["token_ids"]))
            rows.append({"sampled": sampled, "cache": state, "token_mismatches": different,
                         "reference": ref, "streaming": got})
    save(folder / "streaming-checks.json", {"complete": True, "rows": rows})
    if mode == "pi":
        assert all(r["token_mismatches"] == 0 for r in rows), "PI streaming path changed token IDs"


def measure(client, workload, sampled, repeat, num=64, mode=None):
    in_len, out_len, conc, shared = workload
    rng = random.Random(f"vllm027:{repeat}:{workload}:{sampled}")
    prefix = [rng.randrange(1000, 150000) for _ in range(shared)]
    prompts = [prefix + [rng.randrange(1000, 150000) for _ in range(in_len-shared)] for _ in range(num+conc)]
    # Warm the same shape and all decode graph buckets before the timed run.
    with ThreadPoolExecutor(conc) as pool:
        list(pool.map(lambda p: streaming_one(client, p, 16, 13, sampled), prompts[-conc:]))
    client.flush()
    if shared:
        client.generate(prefix + [1000], 1)
    tuned_before = client.rpc("pi_status")["tuned_buckets"] if mode == "pi" else None
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    with ThreadPoolExecutor(conc) as pool:
        rows = list(pool.map(lambda pair: streaming_one(client, pair[1], out_len, pair[0], sampled, started), enumerate(prompts[:num])))
    elapsed = time.perf_counter()-started
    finished_utc = datetime.now(timezone.utc).isoformat()
    tuned_after = client.rpc("pi_status")["tuned_buckets"] if mode == "pi" else None
    assert tuned_before == tuned_after, "PI performed tuning during the timed workload"
    totals = sum(r["usage"]["completion_tokens"] for r in rows)
    summary = {"output_tokens_per_second": totals/elapsed, "requests_per_second": num/elapsed, "wall_seconds": elapsed}
    for key in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = [r[key] for r in rows]
        summary.update({key + "_mean": statistics.mean(values), key + "_p50": statistics.median(values),
                        key + "_p95": percentile(values, 0.95), key + "_p99": percentile(values, 0.99)})
    intervals = [t for row in rows for t in row["observable_itl_ms"]]
    summary["observable_itl_ms_mean"] = statistics.mean(intervals) if intervals else None
    for label, quantile in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        summary["observable_itl_ms_" + label] = percentile(intervals, quantile) if intervals else None
    summary["multi_token_chunks"] = sum(e["tokens"] > 1 for row in rows for e in row["events"])
    return {"input_tokens": in_len, "output_tokens": out_len, "concurrency": conc, "shared_prefix_tokens": shared,
            "started_utc": started_utc, "finished_utc": finished_utc,
            "pi_tuning_before": tuned_before, "pi_tuning_after": tuned_after,
            "sampled": sampled, "repeat": repeat, "requests": rows, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen3-0.6B", "Qwen3-1.7B"])
    ap.add_argument("--modes", nargs="+", default=["native", "batch-invariant", "pi"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--num", type=int, default=64)
    a = ap.parse_args()
    for model in a.models:
        for repeat in range(a.repeats):
            order = a.modes[repeat % len(a.modes):] + a.modes[:repeat % len(a.modes)]
            for mode in order:
                with server(model, mode, tag=f"bench-{repeat}", audit=False) as (c, folder):
                    check_streaming(c, folder, mode)
                    report = {"model": model, "mode": mode, "repeat": repeat, "complete": False, "rows": []}
                    for workload, sampled in [(w, False) for w in WORKLOADS] + [((512, 128, 4, 0), True)]:
                        row = measure(c, workload, sampled, repeat, a.num, mode)
                        report["rows"].append(row)
                        save(folder / "benchmark.json", report)
                        print(model, mode, repeat, workload, sampled, row["summary"], flush=True)
                    report["complete"] = True
                    save(folder / "benchmark.json", report)


if __name__ == "__main__":
    main()
