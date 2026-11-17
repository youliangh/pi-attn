"""Validate and summarize completed real-service experiments; never fill gaps."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from bench_serving import percentile, WORKLOADS
from serving_utils import OUT, save, expected_shapes
from audit_timing import audit
from published_workloads import selected_performance

MODES = ("native", "batch-invariant", "pi")
CONFIGS = ("base", "eager", "nochunk", "chunk256", "maxseq1", "page32", "restart", "coldrestart")


def read(path):
    d = json.loads(path.read_text())
    assert d.get("complete"), f"incomplete result: {path}"
    return d


def figure_section(models):
    """Link optional paper figures without adding plotting dependencies here."""
    if not all((OUT / "figures" / (model.lower() + "-serving.png")).exists() for model in models):
        return []
    lines = ["## 论文风格性能图", "",
             "每个模型一张 3×2 柱状子图，依次为吞吐、整卡峰值占用、TTFT p50/p99、TPOT p50/p99。两个模型的对应子图采用相同坐标范围；全部柱状图从零起始，配色与纹理可用于彩色或灰度打印。", "",
             "吞吐和延迟仅展示 W1–W5（原 W6 采样负载改为 W5），负载参数为 input/output/concurrency。吞吐柱高为三轮中位数，误差线为三轮最小值至最大值，不是置信区间。延迟使用合并 192 个 request 的分位数；显存沿用原始完整运行的三轮整卡采样峰值，包含启动和 Graph 捕获，不按选定负载重新计算。图中 PI 包含 Attention、GEMM 与固定辅助路径。", ""]
    for model in models:
        stem = model.lower() + "-serving"
        lines += [f"### {model}", "", f"![{model} 服务性能柱状子图](figures/{stem}.png)", "",
                  f"[PDF（矢量）](figures/{stem}.pdf) · [SVG（矢量）](figures/{stem}.svg) · [PNG（600 dpi）](figures/{stem}.png)", ""]
    lines += ["绘图数据直接读取本目录的 `summary.json`，不使用 Markdown 中四舍五入后的数值。",
              "[绘图数值 CSV](figures/plot-data.csv) 与 [绘图清单/哈希](figures/manifest.json) 可用于核对。", "",
              "```bash", "uv run --locked --script benchmarks/vllm027/plot_serving.py", "```", ""]
    return lines


def build(models):
    summary = {"complete": True, "models": {}}
    for name in ("adapter", "sampling", "norm"):
        summary[name + "_checks"] = read(OUT / (name + "-checks.json"))
    summary["sampling_dispatch_diagnostic"] = read(OUT / "sampling-dispatch-diagnostic.json")
    for model in models:
        data = summary["models"][model] = {}
        input_hashes = {}
        for mode in MODES:
            folder = OUT / model / mode
            checks = {config: read(folder / config / "checks.json") for config in CONFIGS}
            assert all(len(x["rows"]) == (352 if config == "base" else 32) for config, x in checks.items())
            totals = {key: sum(x[key] for x in checks.values()) for key in ("token_mismatches", "logprob_mismatches")}
            totals["request_comparisons"] = sum(len(x["rows"]) for x in checks.values())
            totals["compared_tokens"] = sum(len(r["output"]["token_ids"]) for x in checks.values() for r in x["rows"])
            totals["configurations"] = {c: {k: x[k] for k in ("token_mismatches", "logprob_mismatches")} for c, x in checks.items()}
            scenarios = defaultdict(lambda: {"comparisons": 0, "token_mismatches": 0, "logprob_mismatches": 0})
            for config, x in checks.items():
                for row in x["rows"]:
                    kind = row.get("sampling") or ("sampled" if row["scenario"] == "sampled_teacher_force" else "greedy")
                    key = f"{config}/{row['scenario']}/{kind}/concurrency={row.get('concurrency', '-')}"
                    scenarios[key]["comparisons"] += 1
                    for metric in ("token_mismatches", "logprob_mismatches"):
                        scenarios[key][metric] += row[metric]
            totals["scenarios"] = dict(scenarios)
            if mode == "pi":
                assert totals["token_mismatches"] == totals["logprob_mismatches"] == 0
                offline = read(folder / "offline/checks.json")
                assert len(offline["rows"]) == 200
                assert all(r["token_mismatches"] == r["logprob_mismatches"] == 0 for r in offline["rows"])
                totals["offline_comparisons"] = len(offline["rows"])
                totals["offline_compared_tokens"] = sum(len(r["output"]["token_ids"]) for r in offline["rows"])
                totals["runtime"] = offline["pi_runtime"]
                totals["trace"] = json.loads((folder / "offline/trace-kernels.json").read_text())
                assert {tuple(x) for x in totals["trace"]["request_linear_weight_shapes"]} == expected_shapes(model)
            rounds = [read(folder / f"bench-{i}/benchmark.json") for i in range(3)]
            assert all(len(x["rows"]) == 6 for x in rounds)
            for i, r in enumerate(rounds):
                assert (r["model"], r["mode"], r["repeat"]) == (model, mode, i)
                benchdir = folder / f"bench-{i}"
                manifest = json.loads((benchdir / "manifest.json").read_text())
                assert manifest["packages"]["vllm"] == "0.27.0"
                assert manifest["environment"]["VLLM_BATCH_INVARIANT"] == ("0" if mode == "native" else "1")
                assert manifest["environment"]["VLLM_USE_V2_MODEL_RUNNER"] == "0"
                cmd = manifest["command"]
                for key, value in (("--max-model-len", "2048"), ("--max-num-seqs", "16"),
                                   ("--max-num-batched-tokens", "512"), ("--block-size", "16"),
                                   ("--kv-cache-memory-bytes", "1879048192"), ("--dtype", "bfloat16")):
                    assert cmd[cmd.index(key)+1] == value, (model, mode, key)
                assert "--enforce-eager" not in cmd
                assert all(x in cmd for x in ("--async-scheduling", "--enable-prefix-caching", "--enable-chunked-prefill"))
                comp = json.loads(cmd[cmd.index("--compilation-config")+1])
                assert comp["cudagraph_capture_sizes"] == [1, 2, 4, 8, 16]
                assert comp["cudagraph_mode"] == "FULL_AND_PIECEWISE"
                if mode == "pi":
                    status = json.loads((benchdir / "pi-runtime.json").read_text())
                    assert {tuple(s) for s in status["gemm_shapes"]} == expected_shapes(model)
                    assert status["heuristic_calls_during_capture"] == 0
                    assert all(k == "pi" or n == 0 for k, n in status["gemm_calls"].items())
                    stream = read(benchdir / "streaming-checks.json")
                    assert all(r["token_mismatches"] == 0 for r in stream["rows"])
            performance = []
            for index, (workload, sampled) in enumerate([(w, False) for w in WORKLOADS] + [((512, 128, 4, 0), True)]):
                rs = [r["rows"][index] for r in rounds]
                assert all(len(r["requests"]) == 64 for r in rs)
                if mode == "pi":
                    assert all(r["pi_tuning_before"] and r["pi_tuning_before"] == r["pi_tuning_after"] for r in rs)
                    assert all(t[-1] > 0 for r in rs for t in r["pi_tuning_before"])
                assert all(tuple(r[k] for k in ("input_tokens", "output_tokens", "concurrency", "shared_prefix_tokens")) == workload and r["sampled"] == sampled for r in rs)
                for repeat, r in enumerate(rs):
                    inputs = [(x["prompt_token_ids"], x["seed"], x["sampling"]) for x in r["requests"]]
                    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
                    key = (repeat, index)
                    if key in input_hashes:
                        assert input_hashes[key] == digest, f"benchmark inputs differ: {model}/{mode}/{key}"
                    input_hashes[key] = digest
                requests = [x for r in rs for x in r["requests"]]
                row = {"workload": workload, "sampled": sampled, "requests": len(requests),
                       "output_tokens_per_second": statistics.median(r["summary"]["output_tokens_per_second"] for r in rs),
                       "requests_per_second": statistics.median(r["summary"]["requests_per_second"] for r in rs),
                       "rounds": [r["summary"] for r in rs],
                       "multi_token_chunks": sum(r["summary"]["multi_token_chunks"] for r in rs)}
                for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
                    values = [r[metric] for r in requests]
                    row[metric] = {"mean": statistics.mean(values), **{q: percentile(values, p) for q, p in (("p50", .5), ("p95", .95), ("p99", .99))}}
                performance.append(row)
            memory = [json.loads((folder / f"bench-{i}/memory.json").read_text())["whole_gpu_peak_mib"] for i in range(3)]
            data[mode] = {"determinism": totals, "performance": performance, "whole_gpu_peak_mib": max(memory)}
    pi_checks = [data["pi"]["determinism"] for data in summary["models"].values()]
    summary["pi_validation_totals"] = {
        "http_request_comparisons": sum(d["request_comparisons"] for d in pi_checks),
        "offline_request_comparisons": sum(d["offline_comparisons"] for d in pi_checks),
        "token_logprob_pairs": sum(d["compared_tokens"] + d["offline_compared_tokens"] for d in pi_checks),
        "token_mismatches": 0, "logprob_mismatches": 0,
    }
    summary["timing_jit_audit"] = audit(models)
    save(OUT / "summary.json", summary)
    lines = ["# RTX 5070：vLLM 0.27.0 服务集成与对比", "",
             "三种模式均运行真实 vLLM 服务；PI 模式替换 Attention 和全部 BF16 dense/LM-head GEMM。", "",
             "- 固定单张 RTX 5070、BF16、TP=1；不承诺跨 GPU/架构一致。",
             f"- 本地模型：{', '.join(models)}；独立 uv workspace，版本锁见 `benchmarks/vllm027/uv.lock`。",
             "- 所有模式 ctx=2048、max-seqs=16、prefill-budget=512、page=16、KV=1.75 GiB；开启 prefix cache、chunked prefill、async scheduling 和 CUDA Graph。",
             "- WSL 不支持 vLLM MRv2 依赖的 UVA，三种模式统一设置 `VLLM_USE_V2_MODEL_RUNNER=0`。vLLM 仍为 0.27.0。",
             "- 每负载 64 requests × 3 轮，模式轮换顺序，GPU 串行测试；吞吐取三轮中位数，延迟分位数合并 192 个 request。",
             "- PI 启动时预热生成路径所有 GEMM 行数区间。每个负载计时前后均核对调优记录，表中只计入调优状态一致的完整三轮。", "",
             "## 严格确定性", "",
             f"PI 通过 {sum(d['request_comparisons'] + d['offline_comparisons'] for d in pi_checks)} 次 request 比较，共 {summary['pi_validation_totals']['token_logprob_pairs']} 对 token ID / logprob，均为零差异。这是已测范围内的逐位一致性结果。", "",
             "比较每种模式自身的基准结果：token IDs 与生成 token 的 FP32 raw-logprob 位模式；不同实现之间不要求逐位相同。", "",
             "| 模型 | 模式 | HTTP request 比较 | token 位差 | logprob 位差 |", "|---|---|---:|---:|---:|"]
    for model, data in summary["models"].items():
        for mode, row in data.items():
            d = row["determinism"]
            lines.append(f"| {model} | {mode} | {d['request_comparisons']} | {d['token_mismatches']} | {d['logprob_mismatches']} |")
    lines += ["", "固定基础配置的结果单独列出，避免把跨配置差异与 batch 差异混为一谈。单元格为 token / logprob 不一致位置数：", "",
              "| 模型 | 模式 | base greedy | base sampled |", "|---|---|---:|---:|"]
    for model, data in summary["models"].items():
        for mode, row in data.items():
            totals = []
            for kind in ("greedy", "sampled"):
                entries = [v for k, v in row["determinism"]["scenarios"].items() if k.startswith("base/") and f"/{kind}/" in k]
                totals.append(" / ".join(str(sum(e[m] for e in entries)) for m in ("token_mismatches", "logprob_mismatches")))
            lines.append(f"| {model} | {mode} | {' | '.join(totals)} |")
    lines += ["", "官方 BI 的默认 sampler 在 batch ≥ 8 时切换 top-k/top-p 实现；独立 [采样分派诊断](sampling-dispatch-diagnostic.json) 检出位差，固定分派后通过。上述 base 表保留模型与采样方式的区别，跨配置失败也不等同于固定配置的 batch-invariance 失败。"]
    lines += ["", "HTTP 检查覆盖并发 1/2/3/4/7/8/9/16、排列、干扰请求、冷/热/部分 prefix cache、20 次重复、共享前缀、greedy re-prefill、sampled teacher forcing，以及 eager/Graph、chunk=256/512/关闭、max-seqs=1、page=16/32、热重启和新的编译缓存目录。",
              "每次 sampled request 固定 seed、temperature=0.8、top_p=0.95、top_k=50。将生成前缀作为新 request 并重用 seed 会重置 vLLM RNG，所以不把这种 sampled continuation 当作相同 RNG 状态。", ""]
    for model, data in summary["models"].items():
        d = data["pi"]["determinism"]
        lines.append(f"- {model} 另通过 {d['offline_comparisons']} 组受控离线 batch 比较（{d['offline_compared_tokens']} token）；[GPU trace kernel 摘要]({model}/pi/offline/trace-kernels.json) 和每次服务的 `pi-runtime.json` 验证了 PI prefill/decode、5 种投影权重形状与 LM head 的真实执行，无 GEMM fallback。")
    lines += ["", "PI 使用官方 BI 辅助路径，同时固定 top-k/top-p 算法，给非 residual RMSNorm 增加 opaque custom-op 边界，并禁用 QK-norm/RoPE 融合。当前 Inductor 包装将 RMSNorm 的 eps 标为 FP64，提升后续计算精度而改变位模式；边界保留原始 Triton 路径。证据见 [编译诊断](diagnostics/rms-compilation.json)，独立回归见 [norm-checks.json](norm-checks.json)。这些开销都包含在 PI 服务性能内。", "",
              "## 服务吞吐", "", "单位 output tokens/s。仅展示 W1–W5，负载为 input/output/concurrency；W1–W4 使用 greedy，W5 为原 W6 的采样负载（S）。各模式使用相同的合成 token-ID 输入。全部六种负载的三轮吞吐见 [summary.json](summary.json)。", "",
              "| 模型 | 负载 | native | 官方 BI | PI | PI/BI | PI/native |", "|---|---|---:|---:|---:|---:|---:|"]
    for model, data in summary["models"].items():
        published = {mode: selected_performance(data[mode]["performance"]) for mode in MODES}
        for i, row in enumerate(published["pi"]):
            tag = f"W{i + 1}: " + "/".join(map(str, row["workload"][:3])) + (" S" if row["sampled"] else "")
            n, b, p = [published[m][i]["output_tokens_per_second"] for m in MODES]
            lines.append(f"| {model} | {tag} | {n:.1f} | {b:.1f} | {p:.1f} | {p/b:.2f}× | {p/n:.2f}× |")
    lines.append("")
    for model, data in summary["models"].items():
        ratios = {mode: [p["output_tokens_per_second"] / b["output_tokens_per_second"]
                         for p, b in zip(selected_performance(data["pi"]["performance"]), selected_performance(data[mode]["performance"]))]
                  for mode in ("native", "batch-invariant")}
        lines.append(f"- {model}：PI 吞吐为官方 BI 的 {min(ratios['batch-invariant']):.2f}–{max(ratios['batch-invariant']):.2f} 倍、原生的 {min(ratios['native']):.3f}–{max(ratios['native']):.3f} 倍。")
    lines += ["", "吞吐改善不代表所有延迟指标都改善；PI 在部分负载的 TTFT 高于原生，需结合下表选择。"]
    lines += ["", "## 延迟与显存", "", "TTFT/TPOT 单位 ms，单元格为 p50 / p95 / p99。TPOT 是首末流式事件间隔除以输出 token 数减一；多 token chunk 原样保留，不据此伪造逐 token 时间。", "",
              "| 模型 | 负载 | 模式 | TTFT | TPOT |", "|---|---|---|---:|---:|"]
    for model, data in summary["models"].items():
        published = {mode: selected_performance(data[mode]["performance"]) for mode in MODES}
        for i, row in enumerate(published["pi"]):
            tag = f"W{i + 1}: " + "/".join(map(str, row["workload"][:3])) + (" S" if row["sampled"] else "")
            for mode in MODES:
                p = published[mode][i]
                latency = [" / ".join(f"{p[key][q]:.2f}" for q in ("p50", "p95", "p99")) for key in ("ttft_ms", "tpot_ms")]
                lines.append(f"| {model} | {tag} | {mode} | {' | '.join(latency)} |")
    lines += ["", "| 模型 | native GPU 峰值 MiB | BI GPU 峰值 MiB | PI GPU 峰值 MiB |", "|---|---:|---:|---:|"]
    for model, data in summary["models"].items():
        lines.append(f"| {model} | " + " | ".join(str(data[m]["whole_gpu_peak_mib"]) for m in MODES) + " |")
    lines += ["", "显存为 nvidia-smi 每秒采样的整卡占用峰值，包含服务启动与 Graph 捕获，不等同于模型独占显存。此项沿用原始完整服务运行的测量；历史显存样本没有逐条时间戳，无法精确拆分出本次选定五个负载的峰值。", "",
              "### PI 显存差异的解释", "",
              "| 模型 | PI 相对 native | PI 相对官方 BI |", "|---|---:|---:|"]
    for model, data in summary["models"].items():
        pi = data["pi"]["whole_gpu_peak_mib"]
        differences = []
        for mode in ("native", "batch-invariant"):
            baseline = data[mode]["whole_gpu_peak_mib"]
            differences.append(f"+{pi - baseline} MiB（+{100 * (pi - baseline) / baseline:.2f}%）")
        lines.append(f"| {model} | " + " | ".join(differences) + " |")
    lines += ["", "已确认三种模式的模型权重、BF16 格式与 1.75 GiB KV cache 预算一致。两个模型的 `bench-0/server.log` 均记录相同的 KV 容量（16,384 tokens）；同模型的加载显存也一致。", "",
              "PI 的 [attention metadata builder](../../../../benchmarks/vllm027/pi_attn_vllm/backend.py) 为 prefill/decode 预分配 FP32 部分结果和归约状态。在本配置（16 个 query heads、context=2048、prefill-budget=512、max-seqs=16）下，每个 builder 的 prefill 工作区为 16.25 MiB，decode 工作区为 0.546875 MiB，共约 16.80 MiB。该数值是 PI 自身的缓冲区大小，并非相对官方 backend 的净增量。", "",
              "此外，[GEMM 调优器](../../../../kernels/pi_vllm_torch.cu) 会分配临时输入、输出和 2×L2 容量的缓存冲刷缓冲区，[启动预热](../../../../benchmarks/vllm027/pi_attn_vllm/worker.py) 会覆盖所有行数区间。释放临时 tensor 后，PyTorch 缓存分配器保留的显存仍可能计入 nvidia-smi，参见 [PyTorch 内存管理说明](https://docs.pytorch.org/docs/main/notes/cuda.html#memory-management)。这是额外峰值/保留显存的可能来源，现有记录未做逐项消融，不能精确分摊全部差额。", "",
              "也不能将差额直接归因于 PI 的 CUDA Graph 更大：两模型 `bench-0` 日志中，PI 和官方 BI 的 graph capture 均约 0.02 GiB，native 约 0.05 GiB。当前结论是这套 PI 集成的整卡测量峰值略高，不能据此推出确定性推理必然需要同等额外显存。", "",
              "## 复现与原始证据", "", "在仓库根目录运行以下命令。脚本会按轮次自行启动和停止服务。", "", "```bash", "bash benchmarks/vllm027/run_serving.sh all", "```", "",
              "仓库提供[服务汇总](summary.json)和 [JIT 日志审计](timing-jit-audit.json)。本地重跑会在各模型、模式和配置目录下生成逐请求检查、启动清单、服务日志和 PI 运行状态。`bench-{0,1,2}/benchmark.json` 记录输入和输出 token、seed、请求起止时间、SSE 事件及 usage。启动清单记录命令、依赖、源码与权重哈希、GPU 信息。GPU trace 单独采集。JIT 日志的时间精度为秒，审计中单独标出与计时边界重叠的事件。", "",
              "官方 BI 的模型 dense 与 LM head 显式使用 Triton persistent GEMM；SM120 的 ATen torch.mm/cuBLASLt 测量不能替代这一服务路径。算子微基准与服务吞吐的区别见 [算子报告](../README.md)。本报告仅覆盖实测模型、输入范围、配置与该单卡，不是所有输入的形式化确定性证明。", ""]
    lines += figure_section(models)
    (OUT / "README.md").write_text("\n".join(lines))
    print(OUT / "README.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen3-0.6B", "Qwen3-1.7B"])
    build(ap.parse_args().models)


if __name__ == "__main__":
    main()
