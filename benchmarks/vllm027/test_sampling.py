"""PI sampling checks around vLLM's B=8 dispatch boundary and tied logits."""
import os
os.environ.update(PI_VLLM_ENABLE="1", VLLM_BATCH_INVARIANT="1", PI_VLLM_STRICT="1")

import torch
from common import initialize, save
from pi_attn_vllm import register


def main():
    initialize()
    register()
    from vllm.v1.sample.ops import topk_topp_sampler as s
    sampler = s.TopKTopPSampler("raw_logprobs", False)
    rows = []
    for vocab in (1024, 151936):
        for case in ("random", "ties", "threshold"):
            torch.manual_seed(7)
            probe = torch.randn(1, vocab, device="cuda")
            if case == "ties":
                probe.round_()
            if case == "threshold":
                probe.fill_(-10)
                probe[0, :100] = torch.linspace(0.0, 0.001, 100, device="cuda")
            for topk, topp in ((50, 0.95), (1, 1.0), (vocab, 0.5), (vocab, 0.999)):
                def evaluate(x, pos):
                    k = torch.full((x.shape[0],), topk, device="cuda", dtype=torch.int32)
                    p = torch.full((x.shape[0],), topp, device="cuda")
                    masked = s.apply_top_k_top_p(x.clone(), k, p)
                    gen = {i: torch.Generator(device="cuda").manual_seed(71 if i in pos else i) for i in range(x.shape[0])}
                    result, _ = sampler.forward_native(x.clone(), gen, k, p)
                    return masked, result
                ref, sampled_ref = evaluate(probe, [0])
                for b in (1, 2, 3, 4, 7, 8, 9, 16):
                    positions = sorted({0, b // 2, b - 1})
                    x = torch.randn(b, vocab, device="cuda") * 6
                    x[positions] = probe
                    masked, sampled = evaluate(x, positions)
                    assert torch.equal(masked[positions].view(torch.int32), ref.expand(len(positions), -1).contiguous().view(torch.int32))
                    assert (sampled[positions] == sampled_ref).all()
                    rows.append({"vocab": vocab, "case": case, "top_k": topk, "top_p": topp, "batch": b, "mismatches": 0})
            print(vocab, case, "passed", flush=True)
    save("serving/sampling-checks.json", {"complete": True, "rows": rows})


if __name__ == "__main__":
    main()
