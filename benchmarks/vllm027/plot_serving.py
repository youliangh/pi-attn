# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.10.8"]
# ///
"""Publication figures from measured serving results; no GPU or vLLM import.

Run with: uv run --script benchmarks/vllm027/plot_serving.py
Each model gets six bar-chart panels, exported as PDF, SVG, and 600-dpi PNG.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from published_workloads import WORKLOADS, selected_performance

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "results/rtx5070/vllm027/serving/summary.json"
MODELS = ("Qwen3-0.6B", "Qwen3-1.7B")
MODES = ("native", "batch-invariant", "pi")
STYLES = {
    "native": ("Native vLLM", "#B8BEC5", "//"),
    "batch-invariant": ("vLLM + BI", "#E4B16B", ".."),
    "pi": ("vLLM + PI", "#4C86AC", ""),
}
# Keep the same metric, mode, and workload ordering in both model figures.
PANELS = (
    ("throughput", "(a) Output throughput ↑", "Output tokens / s"),
    ("memory", "(b) Peak GPU memory ↓", "Whole-GPU memory (GiB)"),
    ("ttft_p50", "(c) Median TTFT ↓", "TTFT p50 (ms)"),
    ("ttft_p99", "(d) Tail TTFT ↓", "TTFT p99 (ms)"),
    ("tpot_p50", "(e) Median TPOT ↓", "TPOT p50 (ms)"),
    ("tpot_p99", "(f) Tail TPOT ↓", "TPOT p99 (ms)"),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observations(summary):
    """Keep full precision; check reported statistics before drawing them."""
    assert summary["complete"]
    rows = []
    for model in MODELS:
        for mode in MODES:
            data = summary["models"][model][mode]
            perf = selected_performance(data["performance"])
            assert len(perf) == len(WORKLOADS)
            for index, (p, expected) in enumerate(zip(perf, WORKLOADS), 1):
                assert (*p["workload"][:3], p["sampled"]) == expected
                assert p["requests"] == 192 and len(p["rounds"]) == 3
                trials = [r["output_tokens_per_second"] for r in p["rounds"]]
                assert math.isclose(p["output_tokens_per_second"], statistics.median(trials), rel_tol=1e-12)
                common = dict(model=model, mode=mode, workload=f"W{index}",
                              input_tokens=expected[0], output_tokens=expected[1],
                              concurrency=expected[2], sampled=expected[3], requests=192)
                rows.append(dict(common, metric="throughput", unit="tokens/s",
                                 value=p["output_tokens_per_second"], low=min(trials), high=max(trials),
                                 statistic="median of 3 rounds; whiskers=min/max"))
                for metric in ("ttft", "tpot"):
                    for quantile in ("p50", "p99"):
                        rows.append(dict(common, metric=f"{metric}_{quantile}", unit="ms",
                                         value=p[f"{metric}_ms"][quantile], low="", high="",
                                         statistic=f"{quantile} of 192 pooled requests"))
            rows.append(dict(model=model, mode=mode, workload="full_run", input_tokens="",
                             output_tokens="", concurrency="",
                             sampled="", requests="", metric="memory", unit="GiB",
                             value=data["whole_gpu_peak_mib"] / 1024, low="", high="",
                             statistic="maximum whole-GPU memory usage across 3 original full runs; sampled at 1 Hz; not filtered by workload"))
    assert all(math.isfinite(r["value"]) and r["value"] >= 0 for r in rows)
    return rows


def paper_style():
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 9,
        "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8.5, "axes.linewidth": 0.65,
        "axes.edgecolor": "#333333", "text.color": "#202020",
        "axes.labelcolor": "#202020", "xtick.color": "#333333", "ytick.color": "#333333",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.45,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "hatch.linewidth": 0.45, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "pi-attn-serving-paper-v1",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 600,
    })


def shared_scales(rows):
    scales = {}
    for metric, _, _ in PANELS:
        relevant = [r for r in rows if r["metric"] == metric]
        top = max(r["high"] if r["high"] != "" else r["value"] for r in relevant)
        ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(0, top * 1.04)
        scales[metric] = [float(x) for x in ticks if x >= 0]
    return scales


def plot_model(model, rows, scales, out):
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.6))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.875, bottom=0.175,
                        wspace=0.31, hspace=0.53)
    fig.suptitle(model, y=0.986, fontsize=12, fontweight="bold")
    fig.text(0.5, 0.951, "RTX 5070  ·  BF16  ·  vLLM 0.27.0", ha="center", fontsize=8.5)
    handles = [Patch(facecolor=c, edgecolor="#333333", linewidth=0.5, hatch=h, label=label)
               for label, c, h in STYLES.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.933),
               ncol=3, frameon=False, columnspacing=2.5, handlelength=2.2)
    for ax, (metric, title, ylabel) in zip(axes.flat, PANELS):
        ax.set_title(title, loc="left", pad=7)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        ticks = scales[metric]
        ax.set_yticks(ticks)
        ax.set_ylim(0, ticks[-1])
        if metric == "throughput":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / 1000:g}k"))
        for offset, mode in enumerate(MODES):
            label, color, hatch = STYLES[mode]
            values = [r for r in rows if r["model"] == model and r["mode"] == mode and r["metric"] == metric]
            ys = [r["value"] for r in values]
            if metric == "memory":
                xs, width = [offset], 0.48
            else:
                xs, width = [i + (offset - 1) * 0.25 for i in range(len(WORKLOADS))], 0.235
            errors = None
            if metric == "throughput":
                errors = [[r["value"] - r["low"] for r in values],
                          [r["high"] - r["value"] for r in values]]
            bars = ax.bar(xs, ys, width=width, color=color, edgecolor="#333333",
                          linewidth=0.45, hatch=hatch, yerr=errors,
                          error_kw={"ecolor": "#222222", "elinewidth": 0.65,
                                    "capsize": 1.6, "capthick": 0.65}, zorder=3)
            if metric == "memory":
                ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=8)
        if metric == "memory":
            ax.set_xticks(range(3), ["Native", "+ BI", "+ PI"])
            ax.set_xlim(-0.65, 2.65)
        else:
            ax.set_xticks(range(len(WORKLOADS)), [f"W{i}" for i in range(1, len(WORKLOADS) + 1)])
            ax.set_xlim(-0.6, len(WORKLOADS) - 0.4)

    # Compact workload key stays inside the exported figure, including the PDF.
    foot = [
        "Workloads: input tokens / output tokens / concurrency",
        "W1: 64/64/1       W2: 64/128/16       W3: 512/128/4",
        "W4: 1500/128/4    W5: 512/128/4 (sampled)",
        "W1–W4: greedy. W5: temperature=0.8, top-p=0.95, top-k=50.",
        "Throughput: median [min, max] of 3 rounds; latency: 192 pooled requests.",
        "Memory: peak over original full runs (1 Hz), including startup/capture.",
    ]
    for i, line in enumerate(foot):
        fig.text(0.095, 0.125 - i * 0.018, line, fontsize=7.1,
                 fontweight="bold" if i == 0 else "normal", ha="left")

    # Fail on clipped titles, ticks, legend, or footnotes before exporting.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= canvas.x1 + 1 and box.y1 <= canvas.y1 + 1, text.get_text()
    stem = model.lower() + "-serving"
    paths = []
    for ext in ("pdf", "svg", "png"):
        path = out / f"{stem}.{ext}"
        metadata = {"Creator": "pi-attn plot_serving.py"}
        if ext == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        elif ext == "svg":
            metadata["Date"] = None
        else:
            metadata = {"Software": "pi-attn plot_serving.py"}
        fig.savefig(path, metadata=metadata)
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    out = args.output or args.source.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.source.read_text())
    rows = observations(summary)
    paper_style()
    scales = shared_scales(rows)
    paths = []
    for model in MODELS:
        paths.extend(plot_model(model, rows, scales, out))
    csv_path = out / "plot-data.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths.append(csv_path)
    font = Path(font_manager.findfont("DejaVu Serif"))
    manifest = {
        "source": str(args.source.resolve()), "source_sha256": digest(args.source),
        "script_sha256": digest(Path(__file__)), "matplotlib": matplotlib.__version__,
        "workload_selection_sha256": digest(Path(__file__).with_name("published_workloads.py")),
        "workloads": {f"W{i}": {"input_tokens": w[0], "output_tokens": w[1], "concurrency": w[2], "sampled": w[3]}
                      for i, w in enumerate(WORKLOADS, 1)},
        "font": font.name, "font_sha256": digest(font),
        "models": MODELS, "layout": "3 rows × 2 columns", "size_inches": [7.2, 7.6],
        "png_dpi": 600, "shared_y_ticks": scales, "plotted_values": len(rows),
        "throughput_error_bars": "3-round minimum and maximum, not confidence intervals",
        "latency": "p50/p99 of 192 pooled requests per mode/workload; no error bars",
        "memory": "maximum whole-GPU memory usage over 3 original full runs, measured at 1 Hz; includes startup/capture; not recomputed for selected workloads",
        "files": {p.name: digest(p) for p in paths},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Validated {len(rows)} plotted values from {args.source}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
