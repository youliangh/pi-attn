"""Download only the two dense BF16 Qwen3 models used by the single-GPU checks."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_REVISIONS = {
    "Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=["Qwen3-0.6B", "Qwen3-1.7B"],
                        default=["Qwen3-0.6B", "Qwen3-1.7B"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reports = {}
    for name in args.models:
        dest = root / "models" / name
        repo = "Qwen/" + name
        revision = MODEL_REVISIONS[name]
        snapshot_download(repo, revision=revision, local_dir=dest,
                          allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"])
        hashes = {}
        for path in sorted(dest.iterdir()):
            if path.is_file():
                with path.open("rb") as stream:
                    hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        reports[name] = {"repo": repo, "revision": revision, "sha256": hashes}
        print(f"Downloaded and hashed {repo}@{revision}", flush=True)
    out = root / "results/rtx5070/models.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        reports = json.loads(out.read_text()) | reports
    out.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
