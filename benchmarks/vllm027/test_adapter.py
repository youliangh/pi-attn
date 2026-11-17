"""GPU contract checks for the new dense-table/output-buffer adapter."""
import json
import math
from pathlib import Path

import torch
import compare_attention as ca
from common import compare_bits, initialize, save
from pi_attn_vllm import gemm_module
from pi_attn_vllm.backend import Implementation


def main():
    initialize()
    from build import build
    reference = build()
    ext = gemm_module().ext()
    rows = []
    with torch.inference_mode():
        for page in (16, 32):
            ca.PAGE = page
            for hq in (16, 32):
                for seed in (0, 7):
                    k, v, reqs = ca.inputs(ca.CASES, hq, 8, seed)
                    for order in (list(range(8)), list(reversed(range(8))), [0, 1, 3, 6, 7] * 3):
                        b = ca.Batch([reqs[i] for i in order], hq)
                        groups = math.ceil(b.maxkv / 512)
                        ws = (torch.empty(b.q.shape[0], hq, groups, 128, device="cuda"),
                              torch.empty(b.q.shape[0], hq, groups, 2, device="cuda"))
                        ref = reference.prefill(b.q, k, v, b.qo, b.pi, b.indices, b.lens,
                                                b.maxq, b.maxkv, 1 / math.sqrt(128), 0, 0)
                        for splits in (0, 1, 3):
                            ext.prefill_out(b.q, k, v, b.qo, b.table, b.lens, b.out, *ws,
                                            b.maxq, b.maxkv, 1 / math.sqrt(128), splits)
                            assert compare_bits(b.out, ref)[0] == 0
                        # Decode includes zero-length graph padding with NaN cache page 0.
                        dq = [ca.Request(reqs[i].q[-1:], reqs[i].pages, reqs[i].kv_len) for i in order]
                        db = ca.Batch(dq, hq)
                        refd = reference.decode(db.q, k, v, db.pi, db.indices, db.lens,
                                                db.ws_o, db.ws_ml, db.counters, db.maxkv, 1 / math.sqrt(128))
                        graph = torch.cuda.CUDAGraph()
                        ext.decode_out(db.q, k, v, db.table, db.lens, db.out, db.ws_o, db.ws_ml, db.counters, 1 / math.sqrt(128))
                        torch.cuda.synchronize()
                        with torch.cuda.graph(graph):
                            ext.decode_out(db.q, k, v, db.table, db.lens, db.out, db.ws_o, db.ws_ml, db.counters, 1 / math.sqrt(128))
                        for perm in (torch.arange(len(order), device="cuda"), torch.arange(len(order)-1, -1, -1, device="cuda")):
                            q0, t0, l0 = db.q.clone(), db.table.clone(), db.lens.clone()
                            db.q.copy_(q0[perm]); db.table.copy_(t0[perm]); db.lens.copy_(l0[perm])
                            graph.replay()
                            assert compare_bits(db.out, refd[perm])[0] == 0
                            db.q.copy_(q0); db.table.copy_(t0); db.lens.copy_(l0)
                        db.lens[-1] = 0
                        db.table[-1].zero_()
                        graph.replay()
                        assert compare_bits(db.out[:-1], refd[:-1])[0] == 0
                        assert not db.counters.count_nonzero()
                        rows.append({"page": page, "hq": hq, "seed": seed, "batch": len(order), "bit_mismatches": 0})
                        del graph, ws
                    # vLLM KV writer must preserve NHD values and skip padding slots.
                    impl = Implementation(hq, 128, 1 / math.sqrt(128), 8)
                    cache = torch.full((2, 4, page, 8, 128), float("nan"), device="cuda", dtype=torch.bfloat16)
                    key = torch.randn(5, 8, 128, device="cuda", dtype=torch.bfloat16)
                    value = torch.randn_like(key)
                    slots = torch.tensor([page, page+1, 3*page, 3*page+1, -1], device="cuda", dtype=torch.int64)
                    layer = type("Scales", (), {"_k_scale": torch.ones((), device="cuda"), "_v_scale": torch.ones((), device="cuda")})()
                    impl.do_kv_cache_update(layer, key, value, cache, slots)
                    assert compare_bits(cache[0].view(-1, 8, 128)[slots[:4]], key[:4])[0] == 0
                    assert compare_bits(cache[1].view(-1, 8, 128)[slots[:4]], value[:4])[0] == 0
                    assert cache[:, 0].isnan().all()
                    print(page, hq, seed, "adapter checks passed", flush=True)
    save("serving/adapter-checks.json", {"complete": True, "rows": rows})


if __name__ == "__main__":
    main()
