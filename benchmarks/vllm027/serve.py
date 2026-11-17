"""Launch one of the three pinned vLLM serving configurations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]


def command(mode, model, port=18098, config="base"):
    assert mode in ("native", "batch-invariant", "pi")
    assert config in ("base", "eager", "nochunk", "chunk256", "maxseq1", "page32", "restart", "coldrestart")
    comp = {"cudagraph_capture_sizes": [1, 2, 4, 8, 16], "cudagraph_mode": "FULL_AND_PIECEWISE"}
    if mode == "pi":
        comp.update(custom_ops=["all"], pass_config={"enable_qk_norm_rope_fusion": False})
    args = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(ROOT / "models" / model),
            "--served-model-name", "default", "--host", "127.0.0.1", "--port", str(port),
            "--dtype", "bfloat16", "--max-model-len", "2048", "--max-num-seqs", "1" if config == "maxseq1" else "16",
            "--max-num-batched-tokens", "2048" if config == "nochunk" else "256" if config == "chunk256" else "512",
            "--block-size", "32" if config == "page32" else "16", "--kv-cache-memory-bytes", str(1879048192),
            "--gpu-memory-utilization", "0.85", "--seed", "0", "--enable-prefix-caching", "--async-scheduling",
            "--no-enable-chunked-prefill" if config == "nochunk" else "--enable-chunked-prefill",
            "--compilation-config", json.dumps(comp), "--logprobs-mode", "raw_logprobs", "--no-enable-log-requests"]
    if mode == "pi":
        args += ["--attention-backend", "CUSTOM", "--worker-cls", "pi_attn_vllm.worker.PIWorker"]
    if config == "eager":
        args += ["--enforce-eager"]
    return args


def environment(mode, model, config="base", audit=False):
    env = dict(os.environ)
    for key in ("CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE", "PI_VLLM_ENABLE", "PI_VLLM_STRICT"):
        env.pop(key, None)
    env.update(VLLM_BATCH_INVARIANT="0" if mode == "native" else "1", VLLM_SERVER_DEV_MODE="1",
               VLLM_PLUGINS="pi_attn" if mode == "pi" else "", PI_VLLM_AUDIT="1" if audit else "0",
               VLLM_WORKER_MULTIPROC_METHOD="spawn", VLLM_USE_V2_MODEL_RUNNER="0")
    # MRv2 requires UVA, which vLLM disables on WSL. Use the supported V1
    # model runner for all three modes; vLLM version remains exactly 0.27.0.
    if mode == "pi":
        env.update(PI_VLLM_ENABLE="1", PI_VLLM_STRICT="1")
    # Source changes invalidate PI's graph cache, including opaque custom-op
    # implementations which Inductor cannot inspect. Restarts share this cache.
    revision = "official"
    if mode == "pi":
        sources = sorted((ROOT / "benchmarks/vllm027/pi_attn_vllm").glob("*.py"))
        sources += [ROOT / "kernels/pi_vllm.py", ROOT / "kernels/pi_vllm_torch.cu"]
        revision = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()[:12]
    namespace = "base" if config == "restart" else config
    if config == "coldrestart":
        namespace += "-" + uuid.uuid4().hex
    cache = ROOT / "env/cache/vllm027/serving" / mode / model / revision / namespace
    for name, part in (("VLLM_CACHE_ROOT", "vllm"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton")):
        env[name] = str(cache / part)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["native", "batch-invariant", "pi"], required=True)
    ap.add_argument("--model", choices=["Qwen3-0.6B", "Qwen3-1.7B"], required=True)
    ap.add_argument("--port", type=int, default=18098)
    ap.add_argument("--config", default="base")
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()
    args = command(a.mode, a.model, a.port, a.config)
    print(json.dumps({"mode": a.mode, "model": a.model, "command": args}), flush=True)
    os.execve(sys.executable, args, environment(a.mode, a.model, a.config, a.audit))


if __name__ == "__main__":
    main()
