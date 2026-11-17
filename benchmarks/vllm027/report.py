"""Validate finished artifacts and regenerate the comparison report (CPU only)."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results/rtx5070/vllm027"
LABELS = {
    "pi_native": "PI 原生 GEMM（自动调优）",
    "pi_all_candidates": "PI 全部候选，BF16 输出",
    "pi_all_candidates_fp32": "PI 全部候选，FP32 输出",
    "vllm_active_cublaslt": "BI 设置下 ATen/torch.mm cuBLASLt",
    "vllm_triton_explicit": "vLLM BI 线性层的 Triton GEMM（直接调用）",
    "stock_cublas": "普通 cuBLAS 负向对照",
    "pi": "PI Attention",
    "vllm_triton": "vLLM Triton Attention",
    "vllm_flash_attention": "vLLM FlashAttention 2",
}


def read(name, count):
    d = json.loads((OUT / f"{name}.json").read_text())
    assert d["complete"] and len(d["rows"]) == count, name
    assert d["environment"]["packages"]["vllm"] == "0.27.0"
    assert d["environment"]["capability"] == [12, 0]
    return d


def total_gemm(d):
    sums = defaultdict(lambda: defaultdict(int))
    for row in d["rows"]:
        for mode, c in row["checks"].items():
            for key in ("comparisons", "elements", "different_elements"):
                sums[mode][key] += c[key]
            sums[mode]["failed_comparisons"] += len(c["failures"])
            sums[mode]["failed_shape_seed_cases"] += bool(c["different_elements"])
    return {mode: dict(row) for mode, row in sums.items()}


def main():
    stock = read("gemm-stock-checks", 30)
    gemm = read("gemm-invariant-checks", 30)
    attn = read("attention-checks", 6)
    gb = read("gemm-invariant-bench", 50)
    ab = read("attention-bench", 7)
    assert stock["environment"]["batch_invariant_enabled"] is False
    for d in (gemm, attn, gb, ab):
        e = d["environment"]
        assert e["batch_invariant_enabled"] and not e["sm80_dispatch"]
        assert e["cublas_workspace"] == ":16:8" and e["cublaslt_workspace"] == "1"
        assert "Cublaslt" in e["blas_library"]
        assert not e["bf16_reduced_precision"]
    assert stock["batches"] == gemm["batches"] and 8 in gemm["batches"]
    assert gb["batches"] == sorted({r["m"] for r in gb["rows"]})
    assert attn["sources"]["flash_attention_version"] == 2
    assert attn["sources"]["triton_batch_invariant"]
    assert attn["sources"]["flash_attention_num_splits"] == 1
    gt = total_gemm(gemm) | total_gemm(stock)
    at = defaultdict(lambda: defaultdict(int))
    for row in attn["rows"]:
        for mode, checks in row["checks"].items():
            for c in checks.values():
                for key in ("comparisons", "elements", "different_elements", "failed_comparisons"):
                    at[mode][key] += c[key]
    at = {mode: dict(row) for mode, row in at.items()}
    assert gt["stock_cublas"]["different_elements"] > 0, "negative control did not detect batch variance"
    assert all(gt[k]["different_elements"] == 0 for k in ("pi_native", "pi_all_candidates", "pi_all_candidates_fp32"))
    assert at["pi"]["different_elements"] == 0
    baselines = ("vllm_active_cublaslt", "vllm_triton_explicit")
    ratios = {mode: statistics.geometric_mean(r["median_us"]["pi_native"] / r["median_us"][mode] for r in gb["rows"]) for mode in baselines}
    source_files = list(HERE.glob("*.py")) + list(HERE.glob("*.sh"))
    source_files += [HERE / "pyproject.toml", HERE / "uv.lock"]
    source_files += list((ROOT / "llm_infer_rust/src/kernels/pi").glob("*"))
    source_files += [ROOT / "kernels" / name for name in ("build.py", "test_gemm.py", "gemm_torch.cu", "pi_attn_torch.cu", "pi_vllm_torch.cu")]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files if p.is_file()}
    previous = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    result = {"gemm_checks": gt, "attention_checks": at, "gemm_pi_over_baseline_geomean": ratios,
              "environment": gemm["environment"], "report_generation_source_sha256": hashes,
              "measurement_source_sha256": previous.get("measurement_source_sha256", previous.get("source_sha256", {})),
              "path_correction": "vllm_active_cublaslt is the ATen microbenchmark, not the vLLM serving linear path"}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = ["# RTX 5070：PI 与 vLLM 0.27.0 batch-invariant 算子对照", "",
             "本机同一张 RTX 5070（SM120），BF16 输入；结果记录于 2026-11-11。",
             "PI 的 GEMM 与 Attention 在本轮矩阵中均逐位一致。BI 设置下 ATen 的 cuBLASLt 路径、",
             "显式 Triton GEMM、FlashAttention 2 和 Triton Attention 的具体检查结果见下表。", "",
             "## 实际比较路径", "",
             "更正：vLLM 0.27.0 在 SM120 上启用 batch-invariant 模式后，普通 `torch.mm` 使用 cuBLASLt，",
             "设置 `CUBLAS_WORKSPACE_CONFIG=:16:8`、`CUBLASLT_WORKSPACE_SIZE=1`，并关闭 BF16 降精度 reduction。",
             "但模型服务的 BF16 dense 线性层和 LM head 会显式调用 `linear_batch_invariant → matmul_persistent`，实际是 Triton。",
             "此前把 ATen 路径标为服务实际 GEMM 的结论不正确；保留原始数值和 JSON key，在此更正归属。",
             "证据：[线性层入口](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/model_executor/layers/linear.py)、",
             "[LM head 入口](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/model_executor/layers/vocab_parallel_embedding.py)、",
             "[BI 实现](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/model_executor/layers/batch_invariant.py)。", "",
             "Attention 测量官方 FlashAttention 2（`num_splits=1`）及 Triton unified attention（batch-invariant 模式禁用 3D split 路径）。",
             "已安装的相关源码逐文件匹配官方 v0.27.0 tag，wheel 哈希匹配独立 uv.lock；见 [来源记录](upstream-sources.json)。", "",
             "| 算子 | PI 的固定计算顺序 | vLLM 0.27.0 的受测路径 |", "|---|---|---|",
             "| Attention | 绝对位置的 32-key tile、512-key group；组内及组间按顺序归约，允许分组并行后按序合并 | FA2 固定 `num_splits=1`；Triton 固定使用 2D 路径，避免随 batch 切换 split 归约 |",
             "| GEMM | 每个输出元素的 16-wide MMA 累加及分段顺序仅由权重 N/K 决定；M 可改变 tile 和调优选择 | 服务 BI 线性层调用固定 128×128×64 tile 的 Triton；普通 ATen 使用受限工作区的 cuBLASLt |", "",
             "PI 的计算契约见[实现说明](../../../llm_infer_rust/docs/deterministic-mode.md)。",
             "vLLM 的 Attention 分派见 [FlashAttention backend](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/v1/attention/backends/flash_attn.py)",
             "与 [Triton unified attention](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/v1/attention/ops/triton_unified_attention.py)。", "",
             "## 逐位检查", "",
             "GEMM：Qwen3-0.6B/1.7B 的 10 个投影形状，3 个种子；batch 大小为 `" + ", ".join(map(str, gemm["batches"])) + "`。",
             "固定探针行置于首/中/尾，改变其他行数据和幅度，并使用带 padding 的行 stride。",
             "每种实现与自身单行参考比较；PI 另外枚举全部可用候选，检查 BF16 和 FP32 输出。", "",
             "| GEMM 路径 | 比较次数 | 不一致比较 | 不一致元素累计数 |", "|---|---:|---:|---:|"]
    for mode, row in gt.items():
        lines.append(f"| {LABELS[mode]} | {row['comparisons']:,} | {row['failed_comparisons']:,} | {row['different_elements']:,} |")
    lines += ["", f"负向对照在 30 个形状/种子组合中的 {gt['stock_cublas']['failed_shape_seed_cases']} 个检出了 batch 位差。", "",
              "Attention：heads=16/8 与 32/8，3 个种子，batch=1/2/3/4/8/16/32，KV 长度覆盖 1–8193。",
              "验证混合长度和排列、独立 decode batch、decode/prefill、分块 prefill、CUDA Graph 重放时修改 query/页表/长度元数据；",
              "PI 另验证可用 kernel variant 与 split。未用 `allclose` 替代逐位比较，非有限输出会直接失败。", "",
              "| Attention 路径 | 比较次数 | 不一致比较 | 不一致元素累计数 |", "|---|---:|---:|---:|"]
    for mode, row in at.items():
        lines.append(f"| {LABELS[mode]} | {row['comparisons']:,} | {row['failed_comparisons']:,} | {row['different_elements']:,} |")
    lines += ["", "FP64 参考保留完整 reduction 维度：GEMM 检查探针行的所有输出列，Attention 检查各 case 最后一行的所有 heads。",
              "精度误差与各类不变性分别记录在 [GEMM JSON](gemm-invariant-checks.json) 和 [Attention JSON](attention-checks.json)。",
              "不同实现使用不同浮点程序；它们之间不要求逐位相同。", "", "## GEMM 性能", "",
              "50 组形状/batch；PI 使用原生自动调优入口。权重环至少 160 MiB，CUDA Graph 排除 Python launch 和调优耗时。",
              "每项三轮，交替实现顺序；每轮取 20 次 graph replay 的中位数，再取三轮中位数。下表为延迟比，越小越快。", "",
              "| M | PI / ATen cuBLASLt | PI / BI 线性层 Triton（直接调用） |", "|---:|---:|---:|"]
    for m in sorted({r["m"] for r in gb["rows"]}):
        rows = [r for r in gb["rows"] if r["m"] == m]
        vals = [statistics.geometric_mean(r["median_us"]["pi_native"] / r["median_us"][b] for r in rows) for b in baselines]
        lines.append(f"| {m} | {vals[0]:.3f} | {vals[1]:.3f} |")
    lines += [f"| 全部 50 项几何平均 | {ratios[baselines[0]]:.3f} | {ratios[baselines[1]]:.3f} |", "",
              "逐形状延迟、调优选择和原始三轮数据见 [GEMM 基准](gemm-invariant-bench.json)。", "",
              "## Attention 性能", "", "同一组 Q/K/V、NHD 分页布局、page=16、head_dim=128、heads=16/8。",
              "KV 环目标 160 MiB、最多 28 组；三轮 CUDA Graph 计时，方法同上。时间单位 µs。", "",
              "| 路径 | B / Q / KV | PI | vLLM FA2 | vLLM Triton | PI/FA2 | PI/Triton |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in ab["rows"]:
        p, f, t = (r["median_us"][k] for k in ("pi", "vllm_flash_attention", "vllm_triton"))
        lines.append(f"| {r['kind']} | {r['batch']} / {r['q_len']} / {r['kv_len']} | {p:.2f} | {f:.2f} | {t:.2f} | {p/f:.3f} | {p/t:.3f} |")
    lines += ["", "完整数据见 [Attention 基准](attention-bench.json)。本轮 PI 的三个 decode 场景均快于上述两条 vLLM batch-invariant 路径；",
              "无前缀 prefill 相对 FA2 较慢，带 4096-token prefix 的短 prefill 更快。不能概括为所有场景都更快。", "",
              "## 复现与边界", "", "在仓库根目录执行：", "", "```bash", "bash benchmarks/vllm027/run.sh", "```", "",
              "独立 uv workspace 位于 `benchmarks/vllm027/`，没有修改原 Rust 环境的依赖。",
              "本轮使用 vLLM 0.27.0、PyTorch 2.13.0+cu130、Triton 3.7.1、FlashInfer 0.6.16.post3。",
              "版本、原测量源码与报告生成时的源码哈希分别保存在 [summary.json](summary.json)。安装说明与测试实现见",
              "[benchmark 目录](../../../benchmarks/vllm027/README.md)。", "",
              "本报告是算子级验证和计时；不推导整个 vLLM 服务的吞吐。",
              "不验证跨 GPU 或跨构建相等；未锁定时钟，也未做 SM120 深度调优。",
              "测试证明这里列出的输入、布局、形状和调度变化通过，不是对任意输入的穷举证明。", ""]
    (OUT / "README.md").write_text("\n".join(lines))
    print("Validated all completed artifacts; wrote summary.json and README.md")


if __name__ == "__main__":
    main()
