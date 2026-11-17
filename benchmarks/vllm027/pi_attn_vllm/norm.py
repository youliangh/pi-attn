"""Keep the official BI RMSNorm program identical in eager and compiled mode."""
import torch
from vllm.model_executor.layers import batch_invariant as bi

_original = bi.rms_norm_batch_invariant


@torch.library.custom_op("pi_attn::rms_norm", mutates_args=())
def rms_norm(input: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    # Without this boundary Inductor recompiles the Triton program with different
    # floating-point optimization options, changing even BF16 output bits.
    return _original(input, weight, eps)


@rms_norm.register_fake
def _(input, weight, eps):
    return torch.empty(input.shape, device=input.device, dtype=input.dtype)


def fixed_rms_norm(input, weight, eps=1e-6, residual=None):
    if residual is not None:
        # The official fused CUDA op is already opaque and mutates both inputs.
        return _original(input, weight, eps, residual)
    return rms_norm(input, weight, eps)


def install():
    from vllm.model_executor.layers import layernorm
    bi.rms_norm_batch_invariant = fixed_rms_norm
    layernorm.rms_norm_batch_invariant = fixed_rms_norm
