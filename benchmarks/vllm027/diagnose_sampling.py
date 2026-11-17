"""Official sampler negative control at the observed B=8 request boundary."""
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ["VLLM_PLUGINS"] = ""
os.environ.pop("PI_VLLM_ENABLE", None)

import torch
from common import initialize, save


def main():
    initialize()
    from vllm.v1.sample.ops import topk_topp_sampler as s
    rows = []
    for case in ("random", "bf16_logits", "ties", "threshold"):
        torch.manual_seed(7)
        probe = torch.randn(1, 151936, device="cuda")
        if case == "bf16_logits":
            probe = probe.bfloat16().float()
        if case == "ties":
            probe.round_()
        if case == "threshold":
            probe.fill_(-10)
            probe[0, :100] = torch.linspace(0, .001, 100, device="cuda")
        def masked(fn, x):
            return fn(x.clone(), torch.full((len(x),), 50, device="cuda", dtype=torch.int32),
                      torch.full((len(x),), .95, device="cuda"))
        ref = masked(s.apply_top_k_top_p, probe)
        for batch in (1, 2, 7, 8, 9, 16):
            positions = sorted({0, batch // 2, batch - 1})
            x = torch.randn(batch, probe.shape[1], device="cuda") * 6
            x[positions] = probe
            original = masked(s.apply_top_k_top_p, x)[positions]
            fixed = masked(s.apply_top_k_top_p_pytorch, x)[positions]
            target = ref.expand(len(positions), -1).contiguous()
            bad = (original.view(torch.int32) != target.view(torch.int32)).nonzero()
            assert torch.equal(fixed.view(torch.int32), target.view(torch.int32))
            rows.append({"case": case, "batch": batch, "probe_positions": positions,
                         "official_filter_bit_mismatches": len(bad), "fixed_filter_bit_mismatches": 0,
                         "first_mismatch_positions": bad[:8].tolist()})
    save("serving/sampling-dispatch-diagnostic.json", {"complete": True, "seed": 7,
         "top_k": 50, "top_p": .95, "vocab_size": 151936, "rows": rows})
    print(rows, flush=True)


if __name__ == "__main__":
    main()
