"""Owned local server lifecycle and strict output checks for serving experiments."""
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import signal
import socket
import struct
import subprocess
import threading
import time

import requests
from serve import ROOT, command, environment

OUT = ROOT / "results/rtx5070/vllm027/serving"


@lru_cache
def model_fingerprint(model):
    expected = json.loads((ROOT / "results/rtx5070/models.json").read_text())[model]
    actual = {}
    for name, digest in expected["sha256"].items():
        with (ROOT / "models" / model / name).open("rb") as f:
            actual[name] = hashlib.file_digest(f, "sha256").hexdigest()
        assert actual[name] == digest, f"model file changed: {model}/{name}"
    return {"repo": expected["repo"], "revision": expected["revision"], "sha256": actual}


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


class Client:
    def __init__(self, port):
        self.url = f"http://127.0.0.1:{port}"

    def rpc(self, method, *args):
        r = requests.post(self.url + "/collective_rpc", json={"method": method, "args": list(args), "timeout": 120}, timeout=150)
        r.raise_for_status()
        return r.json()["results"][0]

    def flush(self):
        for _ in range(20):
            r = requests.post(self.url + "/reset_prefix_cache", timeout=60)
            r.raise_for_status()
            if r.json().get("success") is True:
                return
            time.sleep(0.1)
        raise RuntimeError("vLLM did not release/reset its prefix cache")

    def generate(self, ids, tokens=64, sampling=None, **extra):
        payload = dict(model="default", prompt=ids, max_tokens=tokens, ignore_eos=True,
                       return_token_ids=True, logprobs=0, **(sampling or {"temperature": 0}))
        payload.update(extra)
        r = requests.post(self.url + "/v1/completions", json=payload, timeout=1200)
        r.raise_for_status()
        data = r.json()
        c = data["choices"][0]
        ids = c["token_ids"]
        lp = c["logprobs"]["token_logprobs"]
        assert len(ids) == tokens and len(lp) == tokens, data
        assert all(x is not None and math.isfinite(x) and x != -9999.0 for x in lp), lp
        bits = [struct.unpack("<I", struct.pack("<f", x))[0] for x in lp]
        return {"token_ids": ids, "logprob_bits": bits, "logprobs": lp}

    def teacher_force(self, prompt, generated):
        r = requests.post(self.url + "/v1/completions", json={
            "model": "default", "prompt": prompt + generated, "max_tokens": 1, "temperature": 0,
            "ignore_eos": True, "return_token_ids": True, "prompt_logprobs": 0, "logprobs": 0}, timeout=1200)
        r.raise_for_status()
        raw = r.json()["choices"][0]["prompt_logprobs"]
        values = [raw[len(prompt) + i][str(token)]["logprob"] for i, token in enumerate(generated)]
        assert all(x is not None and math.isfinite(x) and x != -9999.0 for x in values), values
        return {"token_ids": generated, "logprobs": values,
                "logprob_bits": [struct.unpack("<I", struct.pack("<f", x))[0] for x in values]}


def difference(a, b):
    assert len(a["token_ids"]) == len(b["token_ids"])
    bad_ids = [i for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"])) if x != y]
    bad_lp = [i for i, (x, y) in enumerate(zip(a["logprob_bits"], b["logprob_bits"])) if x != y]
    return {"token_mismatches": len(bad_ids), "logprob_mismatches": len(bad_lp),
            "first_token_difference": next(iter(bad_ids), None), "first_logprob_difference": next(iter(bad_lp), None)}


@contextmanager
def server(model, mode, config="base", tag=None, port=18098, audit=True):
    folder = OUT / model / mode / (tag or config)
    folder.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"port {port} is already in use")
    args, env = command(mode, model, port, config), environment(mode, model, config, audit)
    sources = list((ROOT / "benchmarks/vllm027").glob("*.py")) + list((ROOT / "benchmarks/vllm027/pi_attn_vllm").glob("*.py"))
    sources += [ROOT / "kernels/pi_vllm.py", ROOT / "kernels/pi_vllm_torch.cu", ROOT / "benchmarks/vllm027/uv.lock"]
    sources += sorted((ROOT / "llm_infer_rust/src/kernels/pi").glob("*.cu*"))
    meta = {"model": model, "mode": mode, "configuration": config, "command": args,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "weights": model_fingerprint(model), "platform": platform.platform(), "python": platform.python_version(),
            "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"], text=True).strip(),
            "environment": {k: v for k, v in env.items() if k.startswith(("PI_", "VLLM_", "TRITON_CACHE", "TORCHINDUCTOR_CACHE"))},
            "packages": {n: importlib.metadata.version(n) for n in ("vllm", "torch", "triton", "flashinfer-python")},
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    save(folder / "manifest.json", meta)
    stop = threading.Event()
    samples = []

    def monitor():
        while not stop.is_set():
            try:
                value = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
                samples.append(int(value.splitlines()[0]))
            except (subprocess.SubprocessError, ValueError):
                pass
            stop.wait(1)

    with (folder / "server.log").open("w") as log:
        p = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        client = Client(port)
        started = time.monotonic()
        try:
            while time.monotonic() - started < 1200:
                if p.poll() is not None:
                    raise RuntimeError(f"server exited {p.returncode}: {folder / 'server.log'}")
                try:
                    if requests.get(client.url + "/health", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError(f"server startup timed out: {folder}")
            meta["startup_seconds"] = time.monotonic() - started
            if mode == "pi":
                client.rpc("pi_reset_audit")
            save(folder / "manifest.json", meta)
            print(f"READY {model} {mode} {config} ({meta['startup_seconds']:.1f}s)", flush=True)
            yield client, folder
            if mode == "pi":
                save(folder / "pi-runtime.json", client.rpc("pi_status"))
        finally:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait(timeout=10)
            # Worker children share the owned process group. Kill lingering
            # descendants only; never use a system-wide vLLM/process stop script.
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stop.set()
            thread.join(timeout=5)
            save(folder / "memory.json", {"whole_gpu_peak_mib": max(samples, default=0), "samples_mib": samples})


def expected_shapes(model):
    cfg = json.loads((ROOT / "models" / model / "config.json").read_text())
    h, inter = cfg["hidden_size"], cfg["intermediate_size"]
    qkv = (cfg["num_attention_heads"] + 2 * cfg["num_key_value_heads"]) * cfg["head_dim"]
    return {(qkv, h), (h, cfg["num_attention_heads"] * cfg["head_dim"]), (2 * inter, h), (h, inter), (cfg["vocab_size"], h)}


def assert_pi_routing(status, model):
    assert {tuple(x) for x in status["gemm_shapes"]} == expected_shapes(model), status
    assert all(k == "pi" or n == 0 for k, n in status["gemm_calls"].items()), status
    assert status["heuristic_calls_during_capture"] == 0, status
    assert status["attention_batches"], status
