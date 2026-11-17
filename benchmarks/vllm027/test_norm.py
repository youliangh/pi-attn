"""Bitwise regression for the eager/Inductor RMSNorm discrepancy."""
import torch
from common import initialize, save, compare_bits
from pi_attn_vllm.norm import _original, fixed_rms_norm


def main():
    initialize()
    rows = []
    torch.manual_seed(7)
    for width in (128, 1024, 2048):
        x = torch.randn(512, width, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(width, device="cuda", dtype=torch.bfloat16)
        for weighted in (True, False):
            weight = w if weighted else None
            ref = _original(x, weight)
            raw = torch.compile(_original, fullgraph=True)(x, weight)
            compiled = torch.compile(fixed_rms_norm, fullgraph=True)(x, weight)
            assert compare_bits(fixed_rms_norm(x, weight), ref)[0] == 0
            assert compare_bits(compiled, ref)[0] == 0
            probe = x[:1].clone()
            baseline = fixed_rms_norm(probe, weight)
            for batch in (1, 2, 7, 8, 16, 128, 512):
                data = x[:batch].clone()
                data[-1] = probe[0]
                assert compare_bits(fixed_rms_norm(data, weight)[-1:], baseline)[0] == 0
            rows.append({"width": width, "weight": weighted,
                         "unwrapped_eager_compile_mismatches": compare_bits(raw, ref)[0],
                         "wrapped_eager_compile_mismatches": 0, "batch_mismatches": 0})
        # Check the separate in-place residual branch and its two returned tensors.
        residual = torch.randn_like(x)
        eager = fixed_rms_norm(x.clone(), w, residual=residual.clone())
        compiled = torch.compile(fixed_rms_norm, fullgraph=True)(x.clone(), w, residual=residual.clone())
        assert all(compare_bits(a, b)[0] == 0 for a, b in zip(eager, compiled))
    save("serving/norm-checks.json", {"complete": True, "rows": rows, "residual_mismatches": 0})
    print(rows, flush=True)


if __name__ == "__main__":
    main()
