"""Single-device serving checks; every server is owned and cleaned up by this process."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/rtx5070"


def run(args, log, cwd=ROOT, env=None):
    print("Running:", " ".join(map(str, args)), flush=True)
    with log.open("w") as stream:
        subprocess.run(list(map(str, args)), cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


@contextmanager
def server(model, tag, port, extra):
    folder = OUT / model
    folder.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"port {port} is already in use")
    args = ["bash", ROOT / "scripts/serve.sh", str(port), ROOT / "models" / model,
            "--dtype", "bfloat16", "--max-running-req", "4", "--max-seq-len", "2048",
            "--max-total-tokens", "8192", "--chunked-prefill-size", "512", "--cuda-graph-bs", "4",
            "--prefill-cuda-graph-max-tokens", "512", "--log-dir", folder / "logs", *extra]
    peak = [0]
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                memory = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
                peak[0] = max(peak[0], int(memory.splitlines()[0]))
            except (subprocess.SubprocessError, ValueError):
                pass
            stop.wait(1)

    with (folder / f"{tag}.server.log").open("w") as log:
        process = subprocess.Popen(list(map(str, args)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        try:
            deadline = time.monotonic() + 900
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}; see {folder / (tag + '.server.log')}")
                try:
                    response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                    if response.ok:
                        break
                except requests.RequestException:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(f"server startup timed out: {tag}")
                time.sleep(1)
            yield folder
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            stop.set()
            monitor_thread.join(timeout=5)
            (folder / f"{tag}.memory.json").write_text(json.dumps({"whole_device_peak_mib": peak[0]}) + "\n")


def correctness(model, port):
    configs = [("base", []), ("nochunk", ["--chunked-prefill-size", "0"]),
               ("eager", ["--cuda-graph-bs", "0", "--prefill-cuda-graph-max-tokens", "0"]),
               ("maxreq1", ["--max-running-req", "1"]), ("page32", ["--page-size", "32"]),
               ("restart", [])]
    for name, extra in configs:
        with server(model, name, port, ["--deterministic", *extra]) as folder:
            command = [sys.executable, ROOT / "detcheck/detcheck.py", "--url", f"http://127.0.0.1:{port}",
                       "--tokenizer", ROOT / "models" / model, "--logprobs", "--prompts", "8",
                       "--lengths", "24", "300", "1100", "1500", "--tokens", "64", "--concurrency", "1", "4",
                       "--shared", "4", "--repeat", "20", "--flush", "flood:12288", "--flush-chunk-tokens", "1536",
                       "--out", folder / f"{name}.json"]
            command += ["--save", folder / "references.json"] if name == "base" else ["--checks", "none", "--compare", folder / "references.json"]
            run(command, folder / f"{name}.log")
            if name == "base":
                env = dict(os.environ, SGLANG_E2E_BASE_URL=f"http://127.0.0.1:{port}/v1", SGLANG_E2E_DETERMINISTIC="1")
                run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
                    folder / "http-tests.log", cwd=ROOT / "llm_infer_rust", env=env)


def benchmark(model, port):
    for repeat in range(3):
        modes = [("flashinfer", ["--attention-backend", "flashinfer"]), ("pi", ["--deterministic"])]
        if repeat % 2:
            modes.reverse()
        for mode, extra in modes:
            tag = f"bench-{mode}-{repeat}"
            with server(model, tag, port, extra) as folder:
                run([sys.executable, ROOT / "eval/loadgen.py", "--url", f"http://127.0.0.1:{port}",
                     "--num", "16", "--repeats", "1", "--workloads", "64:64:1", "512:128:4", "1500:128:4", "512:128:4:384",
                     "--out", folder / f"{tag}.json"], folder / f"{tag}.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["correctness", "benchmark"], required=True)
    ap.add_argument("--models", nargs="+", default=["Qwen3-0.6B", "Qwen3-1.7B"])
    ap.add_argument("--port", type=int, default=18097)
    a = ap.parse_args()
    for model in a.models:
        (correctness if a.mode == "correctness" else benchmark)(model, a.port)


if __name__ == "__main__":
    main()
