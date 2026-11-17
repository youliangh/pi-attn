"""Sequential correctness/performance runner for the three local service modes."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def run(script, *args):
    subprocess.run([sys.executable, str(HERE / script), *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["check", "bench", "all", "report"], nargs="?", default="all")
    ap.add_argument("--models", nargs="+", default=["Qwen3-0.6B", "Qwen3-1.7B"])
    a = ap.parse_args()
    if a.phase in ("check", "all"):
        for test in ("test_adapter.py", "test_sampling.py", "test_norm.py", "diagnose_sampling.py"):
            run(test)
        # The PI strict checks must pass before spending time on baselines.
        run("check_serving.py", "--models", *a.models, "--modes", "pi")
        for model in a.models:
            run("check_offline.py", "--model", model)
        run("check_serving.py", "--models", *a.models, "--modes", "native", "batch-invariant")
    if a.phase in ("bench", "all"):
        run("bench_serving.py", "--models", *a.models)
    if a.phase in ("report", "all"):
        run("report_serving.py", "--models", *a.models)


if __name__ == "__main__":
    main()
