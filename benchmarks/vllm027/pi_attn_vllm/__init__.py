"""Opt-in PI Attention/GEMM plugin for the pinned vLLM 0.27.0 workspace."""
import importlib.metadata
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
_registered = False


def gemm_module():
    path = str(ROOT / "kernels")
    if path not in sys.path:
        sys.path.insert(0, path)
    import pi_vllm
    return pi_vllm


def register():
    global _registered
    if _registered or os.environ.get("PI_VLLM_ENABLE") != "1":
        return
    assert importlib.metadata.version("vllm") == "0.27.0"
    assert os.environ.get("VLLM_BATCH_INVARIANT") == "1"
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    register_backend(AttentionBackendEnum.CUSTOM, "pi_attn_vllm.backend.PIAttentionBackend")
    gemm_module().install()
    from . import norm
    norm.install()
    # The default sampler changes algorithms at B=8. Keep vLLM's native small
    # batch program for every B, while preserving per-request generator state.
    from vllm.v1.sample.ops import topk_topp_sampler as sampling
    def fixed_top_k_top_p(logits, k, p):
        if k is None and p is None:
            return logits
        return sampling.apply_top_k_top_p_pytorch(logits, k, p)
    sampling.apply_top_k_top_p = fixed_top_k_top_p
    _registered = True
