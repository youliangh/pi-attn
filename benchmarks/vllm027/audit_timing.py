"""Compare logged JIT events with measured intervals (logs have second precision)."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import re
from zoneinfo import ZoneInfo

from serving_utils import OUT, save


def audit(models, log_timezone="America/New_York"):
    rows = []
    for model in models:
        for mode in ("native", "batch-invariant", "pi"):
            for repeat in range(3):
                path = OUT / model / mode / f"bench-{repeat}/benchmark.json"
                data = json.loads(path.read_text())
                assert data["complete"]
                periods = [(datetime.fromisoformat(r["started_utc"]), datetime.fromisoformat(r["finished_utc"])) for r in data["rows"]]
                warnings = []
                for line in (path.parent / "server.log").read_text().splitlines():
                    m = re.search(r"(\d\d-\d\d) (\d\d:\d\d:\d\d).*JIT compilation during inference: ([^.]+)", line)
                    if not m:
                        continue
                    stamp = datetime.strptime(f"{periods[0][0].year}-{m[1]} {m[2]}", "%Y-%m-%d %H:%M:%S")
                    stamp = stamp.replace(tzinfo=ZoneInfo(log_timezone)).astimezone(timezone.utc)
                    warnings.append({"utc": stamp.isoformat(), "kernel": m[3],
                                     "timed_workload_matches": [i for i, (a, b) in enumerate(periods) if a <= stamp <= b],
                                     "same_second_boundary_overlap": [i for i, (a, b) in enumerate(periods) if stamp <= b and stamp + timedelta(seconds=1) >= a]})
                rows.append({"file": str(path.relative_to(OUT)), "jit_warnings": warnings})
    result = {"complete": True, "models": models, "log_timezone": log_timezone,
              "timestamp_precision_seconds": 1,
              "note": "Subsecond boundary overlaps are reported separately; timestamps alone cannot resolve their ordering. PI tuning is independently checked by exact before/after state.",
              "rows": rows}
    save(OUT / "timing-jit-audit.json", result)
    assert not any(w["timed_workload_matches"] for row in rows for w in row["jit_warnings"]), "JIT logged inside a measured interval"
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen3-0.6B", "Qwen3-1.7B"])
    ap.add_argument("--log-timezone", default="America/New_York")
    args = ap.parse_args()
    audit(args.models, args.log_timezone)
