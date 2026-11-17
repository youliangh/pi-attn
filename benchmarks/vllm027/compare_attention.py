"""Direct paged-attention batch, path, and CUDA-graph checks on SM120."""
import argparse
import inspect
import math
import random
from dataclasses import dataclass

from common import benchmark, compare_bits, digest, initialize, manifest, save, torch

PAGE, D = 16, 128
CASES = [(1, 1), (7, 31), (16, 33), (31, 65), (1, 513), (33, 1025), (65, 2049), (17, 8193)]


@dataclass
class Request:
    q: torch.Tensor
    pages: list
    kv_len: int


def inputs(cases, hq, hkv, seed):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    n = 1 + sum(math.ceil(kv / PAGE) for _, kv in cases)
    k = torch.full((n, PAGE, hkv, D), float("nan"), dtype=torch.bfloat16, device="cuda")
    v = torch.full_like(k, float("nan"))
    free = list(range(1, n))
    rng.shuffle(free)
    reqs = []
    for qlen, kvlen in cases:
        count = math.ceil(kvlen / PAGE)
        pages, free = free[:count], free[count:]
        slots = (torch.tensor(pages, device="cuda")[:, None] * PAGE + torch.arange(PAGE, device="cuda")).flatten()[:kvlen]
        k.view(-1, hkv, D)[slots] = torch.randn(kvlen, hkv, D, device="cuda").bfloat16()
        v.view(-1, hkv, D)[slots] = torch.randn(kvlen, hkv, D, device="cuda").bfloat16()
        q = (torch.randn(qlen, hq, D, device="cuda") * (3 if seed != 7 else 6)).bfloat16()
        reqs.append(Request(q, pages, kvlen))
    return k, v, reqs


class Batch:
    def __init__(self, reqs, hq):
        self.reqs = reqs
        self.q = torch.cat([r.q for r in reqs]).contiguous()
        self.qlens = [r.q.shape[0] for r in reqs]
        self.maxq, self.maxkv = max(self.qlens), max(r.kv_len for r in reqs)
        qo, pi, indices = [0], [0], []
        table = []
        maxpages = math.ceil(self.maxkv / PAGE)
        for r in reqs:
            qo.append(qo[-1] + r.q.shape[0])
            pages = r.pages[:math.ceil(r.kv_len / PAGE)]
            indices += pages
            pi.append(len(indices))
            table.append(pages + [0] * (maxpages - len(pages)))
        make = lambda x: torch.tensor(x, dtype=torch.int32, device="cuda")
        self.qo, self.pi, self.indices = make(qo), make(pi), make(indices)
        self.lens, self.table = make([r.kv_len for r in reqs]), make(table)
        self.out = torch.empty_like(self.q)
        rows = len(reqs) * math.ceil(self.maxkv / 512)
        self.ws_o = torch.empty(rows, hq, D, dtype=torch.float32, device="cuda")
        self.ws_ml = torch.empty(rows, hq, 8, dtype=torch.float32, device="cuda")
        self.counters = torch.zeros(rows, hq, 4, dtype=torch.int32, device="cuda")

    def copy_metadata_and_queries(self, other):
        assert self.maxq == other.maxq and self.maxkv == other.maxkv
        for name in ("q", "qo", "pi", "indices", "lens", "table"):
            getattr(self, name).copy_(getattr(other, name))
        self.qlens = other.qlens


def adapters():
    from build import build
    from vllm.v1.attention.ops import triton_unified_attention as triton_attn
    from vllm.v1.attention.backends import fa_utils
    ext = build()
    assert ext.GROUP_KEYS == 512 and ext.DECODE_WARPS == 4
    assert triton_attn.is_batch_invariant
    version = fa_utils.get_flash_attn_version()

    def pi(b, k, v, decode=False, splits=0, variant=0):
        if decode:
            assert b.maxq == 1
            return ext.decode(b.q, k, v, b.pi, b.indices, b.lens, b.ws_o, b.ws_ml,
                              b.counters, b.maxkv, 1 / math.sqrt(D))
        return ext.prefill(b.q, k, v, b.qo, b.pi, b.indices, b.lens,
                           b.maxq, b.maxkv, 1 / math.sqrt(D), splits, variant)

    def triton(b, k, v, decode=False):
        triton_attn.unified_attention(
            q=b.q, k=k, v=v, out=b.out, cu_seqlens_q=b.qo, max_seqlen_q=b.maxq,
            seqused_k=b.lens, max_seqlen_k=b.maxkv, softmax_scale=1 / math.sqrt(D),
            causal=True, window_size=(-1, -1), block_table=b.table, softcap=0,
            q_descale=None, k_descale=None, v_descale=None,
        )
        return b.out

    def flash(b, k, v, decode=False):
        return fa_utils.flash_attn_varlen_func(
            q=b.q, k=k, v=v, cu_seqlens_q=b.qo, max_seqlen_q=b.maxq,
            seqused_k=b.lens, max_seqlen_k=b.maxkv, softmax_scale=1 / math.sqrt(D),
            causal=True, block_table=b.table, fa_version=version, num_splits=1,
        )

    sources = {"triton_attention_sha256": digest(inspect.getfile(triton_attn)),
               "flash_attention_version": version,
               "triton_batch_invariant": triton_attn.is_batch_invariant,
               "flash_attention_num_splits": 1}
    return ext, {"pi": pi, "vllm_triton": triton, "vllm_flash_attention": flash}, sources


def check_geometry(hq, hkv, seed, ext, functions):
    k, v, reqs = inputs(CASES, hq, hkv, seed)
    stats = {name: {} for name in functions}

    def check(name, kind, got, ref):
        diff, size = compare_bits(got, ref)
        row = stats[name].setdefault(kind, {"comparisons": 0, "different_elements": 0, "elements": 0, "failed_comparisons": 0})
        row["comparisons"] += 1
        row["different_elements"] += diff
        row["elements"] += size
        row["failed_comparisons"] += bool(diff)

    refs = {}
    decode_refs = {}
    for name, fn in functions.items():
        refs[name] = [fn(Batch([r], hq), k, v).clone() for r in reqs]
        decode_refs[name] = []
        for index, r in enumerate(reqs):
            last = Request(r.q[-1:], r.pages, r.kv_len)
            y = fn(Batch([last], hq), k, v, decode=True).clone()
            decode_refs[name].append(y)
            check(name, "decode_vs_prefill", y, refs[name][index][-1:])
    rng = random.Random(seed)
    # Every request is compared in differently sized and ordered mixed batches.
    for count in (1, 2, 3, 4, 8, 16, 32):
        for target in range(len(reqs)):
            order = [rng.randrange(len(reqs)) for _ in range(count)]
            pos = (target * 7 + seed) % count
            order[pos] = target
            for perm in (order, list(reversed(order))):
                b = Batch([reqs[i] for i in perm], hq)
                for name, fn in functions.items():
                    got = fn(b, k, v)
                    for index, part in zip(perm, got.split(b.qlens)):
                        check(name, "batch_prefill", part, refs[name][index])
                dreqs = [Request(reqs[i].q[-1:], reqs[i].pages, reqs[i].kv_len) for i in perm]
                db = Batch(dreqs, hq)
                for name, fn in functions.items():
                    got = fn(db, k, v, decode=True)
                    for index, part in zip(perm, got.split(1)):
                        check(name, "batch_decode", part, decode_refs[name][index])
                assert b.counters.count_nonzero() == 0 and db.counters.count_nonzero() == 0

    for index, req in enumerate(reqs):
        cuts = sorted({0, 1, req.q.shape[0] // 2, req.q.shape[0]})
        for name, fn in functions.items():
            chunks = []
            for start, stop in zip(cuts, cuts[1:]):
                piece = Request(req.q[start:stop].contiguous(), req.pages, req.kv_len - req.q.shape[0] + stop)
                chunks.append(fn(Batch([piece], hq), k, v).clone())
            check(name, "chunked_prefill", torch.cat(chunks), refs[name][index])
        for variant in (1, 2, 3, 4):
            if ext.prefill_smem_bytes(variant) > torch.cuda.get_device_properties(0).shared_memory_per_block_optin:
                continue
            got = functions["pi"](Batch([req], hq), k, v, splits=3, variant=variant)
            check("pi", "split_and_variant", got, refs["pi"][index])

    # Reuse one graph with changed row order AND changed page/length metadata.
    for name, fn in functions.items():
        b = Batch(reqs, hq)
        fn(b, k, v)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fn(b, k, v)
        for order in (list(range(len(reqs))), list(reversed(range(len(reqs))))):
            b.copy_metadata_and_queries(Batch([reqs[i] for i in order], hq))
            graph.replay()
            for index, part in zip(order, captured.split(b.qlens)):
                check(name, "graph_replay", part, refs[name][index])
        del graph, captured, b

    # Last query of every case: complete-K FP64 softmax for each GQA head.
    accuracy = {name: {"squared_error": 0.0, "max_abs_error": 0.0} for name in functions}
    norm = 0.0
    for index, req in enumerate(reqs):
        key = k[req.pages].flatten(0, 1)[:req.kv_len]
        value = v[req.pages].flatten(0, 1)[:req.kv_len]
        for head in range(hq):
            kh = head // (hq // hkv)
            score = req.q[-1, head].double() @ key[:, kh].double().T / math.sqrt(D)
            reference = score.softmax(-1) @ value[:, kh].double()
            norm += float(reference.square().sum())
            for name in functions:
                delta = refs[name][index][-1, head].double() - reference
                accuracy[name]["squared_error"] += float(delta.square().sum())
                accuracy[name]["max_abs_error"] = max(accuracy[name]["max_abs_error"], float(delta.abs().max()))
    for row in accuracy.values():
        row["relative_l2"] = math.sqrt(row.pop("squared_error") / norm)
        assert row["relative_l2"] < 0.02, row
    return {"hq": hq, "hkv": hkv, "seed": seed, "checks": stats, "accuracy_fp64": accuracy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["checks", "bench"], default="checks")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    bi = initialize()
    ext, functions, sources = adapters()
    report = {"environment": manifest(bi), "sources": sources, "page_size": PAGE,
              "cases_q_kv": CASES, "rows": [], "complete": False}
    output = f"attention-{a.phase}{'-smoke' if a.smoke else ''}.json"
    if a.phase == "checks":
        for hq, hkv in ((16, 8),) if a.smoke else ((16, 8), (32, 8)):
            for seed in (0,) if a.smoke else (0, 1, 7):
                row = check_geometry(hq, hkv, seed, ext, functions)
                report["rows"].append(row)
                print(row, flush=True)
                save(output, report)
                torch.cuda.empty_cache()
    else:
        for kind, batch, qlen, prefix in [
            ("decode", 1, 1, 1023), ("decode", 4, 1, 2047), ("decode", 1, 1, 8191),
            ("prefill", 1, 512, 0), ("prefill", 1, 2048, 0),
            ("prefill", 4, 512, 0), ("prefill", 1, 128, 4096),
        ]:
            k, v, reqs = inputs([(qlen, qlen + prefix)] * batch, 16, 8, 0)
            b = Batch(reqs, 16)
            layers = min(28, max(1, math.ceil((160 << 20) / (2 * k.numel() * k.element_size()))))
            caches = [(k, v)] + [(torch.randn_like(k), torch.randn_like(v)) for _ in range(layers - 1)]
            fns = {name: (lambda i, fn=fn: fn(b, *caches[i], decode=kind == "decode")) for name, fn in functions.items()}
            row = {"kind": kind, "batch": batch, "q_len": qlen, "prefix": prefix, "kv_len": qlen + prefix,
                   **benchmark(fns, layers)}
            report["rows"].append(row)
            print(row, flush=True)
            save(output, report)
            del b, k, v, caches, fns
            torch.cuda.empty_cache()
    report["complete"] = True
    save(output, report)
    if a.phase == "checks":
        assert all(check["different_elements"] == 0 for row in report["rows"] for check in row["checks"]["pi"].values()), "PI attention invariance failed"


if __name__ == "__main__":
    main()
