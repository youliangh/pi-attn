# vLLM 0.27.0：PI 服务集成与算子对照

此目录是独立 uv workspace，针对同一张 RTX 5070 / SM120 做 BF16 算子比较。
使用项目已安装的 CUDA 13.0 toolkit；不替换根目录 `.venv` 或 Rust 服务的依赖。

## 实际服务

手动启动单个服务：

```bash
uv sync --project benchmarks/vllm027 --locked
bash benchmarks/vllm027/serve.sh --mode pi --model Qwen3-0.6B
# mode: native / batch-invariant / pi；model: Qwen3-0.6B / Qwen3-1.7B
# 默认只监听 127.0.0.1:18098，OpenAI model 名为 default
```

要复现实验，先释放 18098 端口，再运行以下命令。脚本会逐轮启动三个模式的服务。

```bash
bash benchmarks/vllm027/run_serving.sh all
```

也可依次传入 `check`、`bench`、`report` 分步执行。最后一步需要前两步在本机生成的逐请求结果。

`native` 保持官方优化并关闭 BI；`batch-invariant` 开启官方 BI；`pi` 开启 BI 并接入
PI Attention 与所有 BF16 projection/LM-head GEMM。插件只在 PI 模式启用，未修改 site-packages。
PI 还固定 top-k/top-p 分派、用 opaque custom op 保持 RMSNorm 的 eager/编译程序一致，
关闭 QK-norm/RoPE 融合。PI 的计时包含这些固定路径的开销。
模型加载后会预热生成路径的全部 GEMM 行数区间，包括混合 prefill 的非二次幂区间；
性能脚本逐负载核对计时前后的调优记录，计时中出现新调优即失败。
在当前版本中，Inductor 将 RMSNorm 的 `eps` 参数标为 FP64，进而提升部分计算的精度，
与原始 Triton 调用的 FP32 路径产生位差；证据见
[编译诊断](../../results/rtx5070/vllm027/serving/diagnostics/rms-compilation.json)。
官方采样分派的独立反例见
[采样诊断](../../results/rtx5070/vllm027/serving/sampling-dispatch-diagnostic.json)。

三种模式使用相同模型、KV 预算和调度参数。在 WSL 上统一选择 vLLM 的 V1 model runner
（`VLLM_USE_V2_MODEL_RUNNER=0`），因为 MRv2 要求的 UVA 在该平台不可用。
已发布的[服务报告](../../results/rtx5070/vllm027/serving/README.md)和机器可读汇总位于
`results/rtx5070/vllm027/serving/`。在本机运行时，各次启动的 `manifest.json` 记录执行命令、环境、
版本与源码哈希，逐请求数据记录输入/输出 token IDs、seed、SSE 到达时间与 usage。

`check_serving.py` 比较 token IDs 和 raw logprob 的 FP32 位模式，覆盖 batch、顺序、prefix cache、
分块、分页、eager/Graph 和重启；`check_offline.py` 使用受控 batch，并单独采集真实模型 GPU trace。
`test_adapter.py` 检查分页适配、输出 buffer、KV 写入和 Graph 重放；`test_sampling.py`、
`test_norm.py` 检查辅助路径。PI 出现任何位差即失败，官方模式的位差则作为对照结果保留。

PI 集成限定单卡 SM120、原生 Qwen3、BF16、head_dim=128、page=16/32、常规 causal attention。
TP、量化、LoRA、speculative decoding、滑窗等未支持配置会明确拒绝。开发 RPC 仅用于本机审计；
默认监听 loopback。模型自己的采样语义保持不变：同 seed 的新 request 会重新初始化 RNG，
因此 sampled continuation 使用 teacher forcing 验证概率，不把重启 RNG 当作延续 RNG。

## 论文图

```bash
uv run --locked --script benchmarks/vllm027/plot_serving.py
```

绘图脚本通过 uv 的独立脚本环境和 `plot_serving.py.lock` 管理依赖，读取已完成的服务
`summary.json`，无需 GPU，也不改变推理环境。两个模型分别输出 3×2 柱状子图：
吞吐、整卡峰值占用、TTFT p50/p99、TPOT p50/p99。产物位于服务报告的 `figures/`，
包含嵌入字体的 PDF、可编辑 SVG、600 dpi PNG、完整精度 CSV 与来源哈希。
吞吐误差线表示三轮范围，延迟柱表示合并 request 的分位数；对应坐标轴在两张图间保持一致。
吞吐和延迟展示 W1–W4 与原 W6（现 W5），标签仅列 input/output/concurrency；
选择逻辑见 `published_workloads.py`。显存沿用原始完整运行峰值，服务汇总保留了全部六种负载的三轮数据。

## 直接算子对照

```bash
# 仓库根目录；首次同步会下载 vLLM 0.27.0 及其依赖
bash benchmarks/vllm027/run.sh
```

所有数据写到 `results/rtx5070/vllm027/`。完整结果见
[生成报告](../../results/rtx5070/vllm027/README.md)。只重新汇总已有数据可运行：

```bash
python3 benchmarks/vllm027/report.py
```

`compare_gemm.py` 检查固定探针行在不同 batch 大小、位置、干扰行和 row stride 下的位模式。
除了 PI 原生自动调优入口，还枚举 `gemm_dispatch.cuh` 的可用候选，分别检查 BF16 和 FP32 输出。
`--mode stock` 独立进程运行普通 cuBLAS 负向对照；`--mode invariant` 调用官方
`init_batch_invariance()` 后运行 PI、ATen `torch.mm` 路径及显式 Triton persistent GEMM。

SM120 的 ATen 不走 SM80 的 Triton override，而是使用受限工作区的 cuBLASLt。
但 vLLM 服务的 BF16 dense 层与 LM head 会显式调用 BI Triton GEMM，绕过该 ATen 分派。
旧报告把 cuBLASLt 标为服务实际路径的结论已更正；直接 kernel 计时仍不能视为服务吞吐。
见 [BI 实现](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/model_executor/layers/batch_invariant.py)
与[线性层入口](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/model_executor/layers/linear.py)。

`compare_attention.py` 直接调用 PI 分页 kernel、官方 FlashAttention 2 包装函数和 Triton unified attention。
batch-invariant 模式下，FA2 使用 `num_splits=1`，Triton 使用 2D 路径。
检查混合长度/排列、decode/prefill、分块、PI split/variant，以及 CUDA Graph 重放时更换 Q 和分页元数据。

比较使用整数位模式，出现 NaN/Inf 即失败。每个实现与自己的单独执行结果比较；
不同实现之间的数值偏差通过独立 FP64 参考检查。所有原始计数、失败样例和精度误差保存在 JSON 中。
vLLM 对照失败会保留为测量结果；PI 的不变性失败会让脚本返回失败。报告生成也要求负向对照检出位差。

性能测试排除 JIT/自动调优和 Python launch 开销，使用 CUDA Graph，轮转权重/KV，三轮交替顺序。
这是 kernel 级比较，不包含 vLLM 服务端模型加载、调度和通信。
