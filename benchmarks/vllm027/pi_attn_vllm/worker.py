"""Worker-side extension initialization and auditable model routing."""
import time
import torch
from vllm.v1.worker.gpu_worker import Worker
from . import gemm_module


class PIWorker(Worker):
    def init_device(self):
        super().init_device()
        c = self.vllm_config
        assert torch.cuda.get_device_capability() == (12, 0)
        assert c.parallel_config.world_size == 1 and c.model_config.dtype == torch.bfloat16
        assert c.quant_config is None and c.lora_config is None and c.speculative_config is None
        assert not c.model_config.using_transformers_backend()
        assert c.model_config.hf_config.model_type == "qwen3"
        assert c.compilation_config.custom_ops == ["all"]
        assert c.compilation_config.pass_config.enable_qk_norm_rope_fusion is False
        gemm_module().ext()

    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        # Mixed prefill has many more M buckets than decode graph captures.
        # Tune every reachable bucket before requests, using one real weight per
        # shape; the native tuner keys by shape, not the individual layer.
        started = time.monotonic()
        ext = gemm_module().ext()
        representatives = {}
        sc = self.vllm_config.scheduler_config
        # Qwen3 may alias lm_head to embed_tokens; keep alias names so the
        # generation projection is warmed even with tied embeddings.
        for name, module in self.model_runner.get_model().named_modules(remove_duplicate=False):
            short = name.rsplit(".", 1)[-1]
            if short not in {"qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head"}:
                continue
            weight = module.weight
            assert weight.is_cuda and weight.dtype == torch.bfloat16 and weight.is_contiguous()
            limit = sc.max_num_seqs if short == "lm_head" else sc.max_num_batched_tokens
            key = tuple(weight.shape)
            if key not in representatives or limit > representatives[key][1]:
                representatives[key] = (weight, limit)
        assert len(representatives) == 5, representatives.keys()
        with torch.inference_mode():
            for weight, limit in representatives.values():
                for bucket in sorted({ext.bucket(m) for m in range(1, limit + 1)}):
                    ext.linear(torch.zeros(bucket, weight.shape[1], dtype=weight.dtype, device=weight.device), weight)
        torch.cuda.synchronize()
        self._pi_warmup_seconds = time.monotonic() - started

    def pi_status(self):
        from .backend import OBSERVED
        g = gemm_module()
        return {"gemm_shapes": g.shapes(), "gemm_calls": dict(g.STATS),
                "heuristic_calls_during_capture": g.ext().heuristic_calls(),
                "tuned_buckets": g.ext().tuning_status(), "warmup_seconds": self._pi_warmup_seconds,
                "attention_batches": sorted(OBSERVED), "backend": "PIAttentionBackend",
                "model": self.vllm_config.model_config.model}

    def pi_reset_audit(self):
        from .backend import OBSERVED
        OBSERVED.clear()
        gemm_module().STATS.clear()
        return True
