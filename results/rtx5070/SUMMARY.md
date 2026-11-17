# RTX 5070 单卡迁移记录

记录日期：2026-11-11。

本目录记录本机实测，独立于仓库原有 A800 / RTX 4090 D 论文数据。
目标是固定物理 GPU、权重、精度、构建和采样 seed 下的逐位确定性推理。
不承诺同架构其他 GPU、跨架构、不同编译器或不同依赖版本间逐位相同。

## 仓库分析与迁移边界

| 模块 | 作用与本次处理 |
|---|---|
| `paper/`、根 README、`results/` | 描述数值契约和历史实验；原始性能结论不作为 5070 的实测结果 |
| `llm_infer_rust/src/kernels/pi/` | 正式 CUDA 实现；`kernels/` 中相应源文件链接到这里，避免维护两份算子 |
| `kernels/` | PyTorch binding、FP64 参考、分割/批次不变性、mutation control、微基准；适配 12 GB 显存 |
| `llm_infer_rust/` | Rust/libtorch Qwen3 引擎；图捕获、分块 prefill、radix cache、异步 metadata 和 CPU/GPU overlap 保留 |
| `detcheck/`、`eval/` | 黑盒 token/logprob 检查与服务负载；新增本机入口，复用现有检查和负载生成器 |
| `gdn/` | 独立的 Gated DeltaNet / 旧 vLLM 0.17.1 集成；不在本次已选范围内 |
| `archive/`、历史优化记录 | 研究与历史实验，不是当前 serving 的构建依赖 |

保留 attention 的 32-key tile、512-key fold，以及 GEMM 的
`mma.m16n8k16` 与按权重形状确定的分段累加顺序。SM120 可以编译、执行这些实现，
因此本次没有另建一套数学行为不同的 Blackwell 算子。

主要硬件差异是 RTX 5070 的 opt-in shared memory 仅 99 KiB。
120 KiB 的 GEMM 候选（67、72）不适用；正式 dispatcher 在 launch 前检查资源，
正确返回不支持，并且仅缓存成功的资源查询/属性设置。自动调优从可执行候选中选择。
测试侧独立计算 CUTLASS tile 资源要求，验证拒绝行为与可执行候选的逐位一致性。

## 环境与复现

实测配置见 [environment.json](environment.json)，模型 revision 与 SHA-256 见
[models.json](models.json)。已测 release 二进制和源文件哈希见
[build-manifest.json](build-manifest.json)；`cuobjdump` 确认二进制包含 SM120 native image。
本机为 WSL2 / Ubuntu 24.04、RTX 5070 12 GB、SM120、48 SM，
驱动 616.64。使用已有 Rust 1.99.1 与 GCC 13.3；Python 3.12.13、PyTorch 2.13.0+cu130、
FlashInfer 0.6.18、Triton 3.7.1，通过根目录 uv workspace 与 `uv.lock` 管理。
CUDA 13.0.2 toolkit 安装在项目 `env/cuda-13.0/`，不使用本机旧 nvcc 12.0，也不安装 Linux GPU 驱动。

```bash
bash setup_env.sh --profile rtx5070
source env.sh
(cd llm_infer_rust && cargo build --locked --release)
bash scripts/validate_rtx5070.sh kernels
bash scripts/validate_rtx5070.sh rust
bash scripts/validate_rtx5070.sh e2e
bash scripts/validate_rtx5070.sh bench
```

服务验证使用 BF16、context 2048、KV pool 8192 tokens、max-running-req 4、
chunked prefill 512、decode graph 最大 batch 4、prefill graph 最大 512 tokens。
这是一组适合本机显存的已测配置，不表示所有 context/batch 组合均可容纳。
模型仅下载 Qwen3-0.6B 和 1.7B，revision 已在下载脚本中固定。
首次迁移测量未包含 vLLM。追加的 **vLLM 0.27.0 算子级对照**使用独立 uv 环境，
涵盖 PI 与官方 GEMM/Attention 的 batch-invariance、FP64 精度及性能检查，见
[对照报告](vllm027/README.md)。根目录环境与 Rust 服务仍使用上表依赖。
随后完成的 vLLM 插件与三模式实际服务比较另见 [vLLM 服务报告](vllm027/serving/README.md)，
使用独立 uv 环境和相同的本地模型；本页以下服务表仍是原始 Rust 引擎测量。

## 算子与原生引擎验证

| 检查 | 结果与证据 |
|---|---|
| attention 基础不变性 | 0 failures；[test_invariance.log](test_invariance.log) |
| attention 定向边界 | 0 failures；[test_directed.log](test_directed.log) |
| attention 几何/尺度矩阵，含 32768-token 场景 | 0 failures；[test_matrix.log](test_matrix.log) |
| GEMM 全部 27 个投影形状、候选、批次与 FP64 参考 | 0 failures；[test_gemm.log](test_gemm.log) |
| 负向对照 | 3 个 attention mutant 全被捕获；cuBLAS 的 27/27 形状随 M 变化；[test_mutants.log](test_mutants.log) |
| Rust CPU 单元与集成 | 162 passed，0 failed；[rust-cpu.log](rust-cpu.log) |
| Rust CUDA 单元与集成 | 31 passed，0 failed；[rust-cuda.log](rust-cuda.log) |

CPU 与 CUDA 测试分别执行，CUDA 测试入口检查 native 算子确实被编入，不能靠跳过 GPU 用例通过。
原仓库说明中曾失败的 FlashInfer segmented-prefill 用例，在本环境也通过。
`benchmark_sampling_cuda` 是原有手动性能实验，未作为正确性测试运行。

FP64 attention 参考只切分互相独立的 query 行/heads，GEMM 参考只切分输出列，
保留完整 reduction 维度与原精度阈值。forced-split attention 的批次工作区限制在 2 GiB，
长序列的原有边界与路径检查保留。`test_gemm.py --single-gpu` 排除 TP 专用检查。

## 服务逐位验证

两个模型的全部六种配置均通过 token ID **及 fp32 logprob 位模式**比较。
base 覆盖并发 1/4、冷/热/部分 prefix cache、共享前缀、decode/re-prefill、
固定 seed 采样、采样续写及 20 次重复请求；其他配置分别比较已保存的
8 个 greedy 和 8 个 seeded-sampling 参考输出，每个输出 64 tokens。
主提示由长度 24/300/1100/1500 的语料片段加短题干组成。

| 配置 | 相对 base 的变化 | 0.6B 不一致数 | 1.7B 不一致数 |
|---|---|---:|---:|
| base | 全部默认优化开启 | 0 | 0 |
| nochunk | chunked-prefill-size=0 | 0 | 0 |
| eager | decode 与 prefill CUDA Graph 均关闭 | 0 | 0 |
| maxreq1 | max-running-req=1 | 0 | 0 |
| page32 | page-size=32 | 0 | 0 |
| restart | 同配置重新启动、重新自动调优 | 0 | 0 |

证据：[0.6B base](Qwen3-0.6B/base.json)、[1.7B base](Qwen3-1.7B/base.json)，
以及对应模型目录下 `nochunk/eager/maxreq1/page32/restart.json`。
每个模型的 HTTP 协议、流式、并发、thinking、工具调用和确定性测试均 **16/16 通过**：
[0.6B HTTP](Qwen3-0.6B/http-tests.log)、[1.7B HTTP](Qwen3-1.7B/http-tests.log)。

`*.memory.json` 为每秒采样得到的整卡峰值，包含桌面/其他进程占用，
并非精确的进程分配峰值。base 分别为 5178 / 8180 MiB，
关闭 chunking 后分别为 8208 / 9961 MiB；这些已测配置均未发生 OOM。

## 性能测量

原始数据保存在 [kernel-benchmark.json](kernel-benchmark.json) 和各模型 `bench-*.json`。
基准独占本任务的 GPU 工作阶段，不与上述测试并行执行。每项三轮，交替 PI/基线顺序，
报告三轮中位数；未锁定 GPU 时钟。这里只建立本机基线，未做 SM120 深度调优。
算子 JSON 的 `peak_allocated_bytes` 仅统计 PyTorch allocator，不包含 CUDA 原生分配。

### GEMM

10 个 Qwen3-0.6B/1.7B 投影形状 × M=1/8/32/128/512，共 50 项。
PI 在已验证的可用候选中按 GPU 时间选择最快者，和 `torch.matmul` 的 cuBLAS 路径比较。
权重环至少 160 MiB；CUDA Graph 计时排除 Python launch 开销。
下表为各模型 5 个投影形状的延迟比几何平均，**PI/cuBLAS 越小越快**。
微基准选出的最优候选不一定与 serving 启动自动调优所选候选相同。

| M | 0.6B | 1.7B |
|---:|---:|---:|
| 1 | 1.000 | 1.005 |
| 8 | 0.858 | 0.952 |
| 32 | 0.982 | 0.936 |
| 128 | 0.926 | 0.968 |
| 512 | 0.978 | 0.997 |

全部 50 项延迟比几何平均 **0.959**，单项范围 **0.708–1.088**。
这不是端到端吞吐比。

### Attention

BF16，head_dim=128，16 query heads / 8 KV heads，page=16。
沿用 `bench_attention.py` 的多层 KV 环与 CUDA Graph 计时；表中对照为默认 FlashInfer，
其他 FlashInfer 模式也保存在原始 JSON 中。时间单位 µs，比值越小越快。

| 路径 | batch / Q / KV 或 prefix | PI | FlashInfer | PI/FI |
|---|---|---:|---:|---:|
| decode | B=1, KV=1024 | 14.0 | 10.1 | 1.386 |
| decode | B=4, KV=2048 | 55.7 | 56.2 | 0.991 |
| decode | B=1, KV=8192 | 56.6 | 56.6 | 1.000 |
| prefill | B=1, Q=512, prefix=0 | 44.2 | 57.8 | 0.765 |
| prefill | B=1, Q=2048, prefix=0 | 403.0 | 407.3 | 0.989 |
| prefill | B=4, Q=512, prefix=0 | 114.6 | 112.2 | 1.021 |
| prefill | B=1, Q=128, prefix=4096 | 104.7 | 89.4 | 1.171 |

### 完整服务

同一 Rust 引擎下比较 `--deterministic` 与 FlashInfer/cuBLAS 基线。
两个模型 × 两种模式 × 三轮 × 四种负载，共 **48 组**，每组 16 个请求，
全部输出 token 数符合预期。每个模式每轮重启服务，初始化与图捕获不计入请求耗时。
汇总见 [serving-benchmark-summary.json](serving-benchmark-summary.json)，原始逐轮数据见模型目录。

负载列为 **输入 tokens / 输出 tokens / 并发 / 共享前缀 tokens**。
成对数值均为 **PI / FlashInfer**；吞吐比越大越快，TTFT/TPOT 单位 ms。

| 模型 | 负载 | 吞吐 tok/s：PI / FI | 吞吐比 | TTFT：PI / FI | TPOT：PI / FI |
|---|---|---:|---:|---:|---:|
| 0.6B | 64 / 64 / 1 / 0 | 348.5 / 349.6 | 0.997 | 5.61 / 5.67 | 2.82 / 2.81 |
| 0.6B | 512 / 128 / 4 / 0 | 1033.7 / 960.4 | 1.076 | 43.60 / 44.24 | 3.55 / 3.84 |
| 0.6B | 1500 / 128 / 4 / 0 | 713.2 / 682.1 | 1.046 | 112.18 / 111.89 | 4.75 / 5.01 |
| 0.6B | 512 / 128 / 4 / 384 | 1235.2 / 1164.3 | 1.061 | 19.30 / 20.64 | 3.11 / 3.30 |
| 1.7B | 64 / 64 / 1 / 0 | 145.2 / 148.6 | 0.977 | 10.21 / 10.09 | 6.83 / 6.67 |
| 1.7B | 512 / 128 / 4 / 0 | 478.3 / 449.7 | 1.064 | 97.56 / 99.29 | 7.65 / 8.17 |
| 1.7B | 1500 / 128 / 4 / 0 | 357.9 / 339.2 | 1.055 | 234.99 / 241.86 | 9.38 / 9.94 |
| 1.7B | 512 / 128 / 4 / 384 | 540.4 / 515.8 | 1.048 | 44.36 / 45.13 | 7.10 / 7.45 |

四种负载的吞吐比几何平均：**0.6B 为 1.045，1.7B 为 1.035**。
基准过程中整卡显存采样峰值：0.6B PI/FI 为 **5132/4984 MiB**，
1.7B 为 **8172/8070 MiB**。单请求吞吐略低，不能由几何平均推出每种场景都更快。
这是每轮 16 请求的短基准，时钟与系统状态会造成波动；原始三轮数值全部保留。
以上服务吞吐仅比较本机 Rust 引擎的 PI/FI 模式。vLLM 0.27.0 的
[算子级测量](vllm027/README.md)与[实际服务比较](vllm027/serving/README.md)分别报告；
未比较其他 GPU，也未声称支持任意 batch/context 配置。
