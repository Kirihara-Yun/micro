# MoE Baseline 复现矩阵（总报告）

> 日期：2026-11-22。汇总 A100 WorkBuddy 任务单两条 baseline 的当前复现状态。
> 分层结论：区分「代码能运行 / 系统机制成立 / 论文图表数值可复现」三层。

## 状态总览

| 任务单条目 | 状态 | 证据 |
|---|---|---|
| §0 环境盘点 | ✅ 已复现 | 见 §1（本报告） |
| §1.1 对齐 artifact（四方法×多模型 CSV） | ⚠️ 部分复现 | base-8/64/128 四方法已跑；base-256、large-128 权重未转（磁盘受限） |
| §1.2 稳定性（3 轮重复） | ✅ 已复现 | base-8 有 35 采样；base-64 已补 3 轮独立重复（`2026-11-22_pregated_base64_3rounds.md`） |
| §1.3 显存异常排查 | ⚠️ 部分复现 | arena 定位已闭环；nvidia-smi 各阶段 + 4.4GB↔3.37GB 差异待查 |
| §1.4 机制计时（BLK/CMP/NMB/MEM） | ✅ 已复现 | `2026-11-22_pregated_profiling_mechanism.md` |
| §1.5 质量（SQuAD EM/F1） | ❌ 受资源阻塞 | 需确认 checkpoint 是否含预门控微调权重 |
| §1.6 拓展（base-128/256、large-128） | ⚠️ 部分复现 | base-128 已转并跑四方法；base-256/large-128 磁盘受限（剩 132GB） |
| §2 KTransformers 端到端推理 | ✅ 已复现 | Mixtral-8x7B CPU+GPU 端到端跑通（`2026-11-22_ktransformers_mixtral_e2e.md`），decode ~8.9 tok/s |
| §2 KTransformers 机制验证 | ✅ 已复现 | 8 专家每层 1 GPU + 7 CPU，GPU 显存仅 16GB（全模型 bf16 ~94GB） |
| §2 论文 AMX 路径 | ❌ 受资源阻塞 | Xeon 8360Y 无 AMX；仅验证 AVX-512 路径功能复现 |
| §3 复现矩阵 | ✅ 本报告 | — |

## §1 环境盘点

- GPU：1× NVIDIA A100 80GB PCIe，驱动 560.35.03（CUDA 12.6），当前空闲。
- CPU：Intel Xeon Platinum 8360Y @ 2.40GHz，72 核，1 TiB 内存，4 NUMA node。
- 容器：`pregated`（`e2af132bfeea`，`nvcr.io/nvidia/pytorch:22.09-py3`，CUDA 11.8），宿主机 `/home/qinzy/baselines/Pregated_MoE` ↔ 容器 `/workspace/FasterTransformer`，`/data` 双向挂载。
- 权重：`/data/ft/switch-base-8`（4.4GB）、`/data/ft/switch-base-64`（29GB，已转）；`/data/qinzy/models/` 有 DeepSeek-V2-Lite、Mixtral-8x7B。

## §2 Pre-Gated MoE：四方法结果

> 显存取值：`MEM usage` 的峰值字节（`cudaMemGetInfo` 的 `total−free`），不套 `eval_all.py` 的减 arena 公式（该公式在 GPU-only 下得负值，见 profiling 报告 §5）。

### switch-base-8（权重 4.4GB）

| 方法 | BLK (ms) | 吞吐 (tok/s) | 峰值显存 | max_active | cache hit |
|---|---|---|---|---|---|
| GPU-only | 0.911 | 153 | 3.36 GB | 0 | 0 |
| Pre-gated | 1.081 | 129 | 23.36 GB | 1 | 1.0 |
| DeepSpeed | 1.749 | 85 | 23.36 GB | 1 | 1.0 |
| SE-MoE | 6.368 | 25 | 23.35 GB | 1 | 1.0 |

### switch-base-64（权重 29GB）

| 方法 | BLK (ms) | 吞吐 (tok/s) | 峰值显存 | max_active | cache hit |
|---|---|---|---|---|---|
| GPU-only | 0.914 | 151 | 16.05 GB | 0 | 0 |
| Pre-gated | 1.088 | 120 | 26.05 GB | 1 | 1.0 |
| DeepSpeed | 1.753 | 81 | 26.05 GB | 1 | 1.0 |
| SE-MoE | 50.543 | 3 | 26.05 GB | 1 | 1.0 |

### switch-base-128（权重 56GB）

| 方法 | BLK (ms) | 吞吐 (tok/s) | 峰值显存 | max_active | cache hit |
|---|---|---|---|---|---|
| GPU-only | 0.919 | 150 | 30.55 GB | 0 | 0 |
| Pre-gated | 1.096 | 114 | **29.12 GB** | 1 | 1.0 |
| DeepSpeed | 1.764 | 78 | 29.12 GB | 1 | 1.0 |
| SE-MoE | 101.026 | 2 | 29.12 GB | 1 | 1.0 |

原始 CSV：`experiments/switch-base-8_results.csv`、`experiments/switch-base-64_results.csv`、`experiments/switch-base-128_results.csv`；解析脚本 `scripts/parse_pregated_logs.py`；跑法 `scripts/run_pregated_methods.sh`、`scripts/run_pregated_3rounds.sh`。

### 观察

- GPU-only 吞吐几乎不随专家数变化（153→151→150），因为 topk=1 且全常驻；BLK 稳定在 ~0.91ms。
- Pre-gated 吞吐损失：base-8 −16%（129/153）、base-64 −21%（120/151）、base-128 −24%（114/150），随专家数略增。
- **显存（关键）**：GPU-only 随专家数线性增长（3.37→16.05→30.55GB）；Pre-gated 显存 = 20GB arena + 模型非专家部分 + 1 个激活专家，增长缓慢（23.37→26.05→29.12GB）。**在 base-128 上 Pre-gated（29.12GB）首次低于 GPU-only（30.55GB）**，与论文图 12「大模型时 Pre-gated 显存低于 GPU-only」的方向一致。
- SE-MoE（`fetch_all=1`）随专家数暴跌：base-8 25 → base-64 3 → base-128 2 tok/s（每次全量重载 `num_layer×total_experts` 专家），验证其仅适用于特定场景。

## §3 与论文图 10–12 的对比

- **吞吐**：论文称 Switch-Base 各配置 Pre-gated ≈ GPU-only 的 **81%**。实测 base-8 = 84%、base-64 = 79%、base-128 = 76%，方向一致；但我们的模型是 `google/switch-base-*`（未任务微调）、`beam_width=4`、`perf_benchmark.py`，与论文的任务微调模型 + `eval_all.py` 协议不同，**数值不可直接等同**。
- **显存**：论文图 12 称 Pre-gated 峰值显存约为 GPU-only 的 0.65 倍。本机小模型（base-8）是 6.94 倍（20GB arena 反成主导），但**随专家数增长 Pre-gated 显存优势显现：base-128 时 Pre-gated 29.12GB 首次低于 GPU-only 30.55GB（0.95 倍）**，方向与论文一致。差异来自「20GB arena 仅 Pre-gated 分配」（见 profiling 报告 §4）——论文机器（AMD EPYC 7V12）与本机（Xeon 8360Y）环境不同，显存统计方式也可能不同，绝对值本就不可直接对比。
- **待核查**：转换目录 4.4GB 与 GPU-only 峰值 3.37GB 的差异（未定位），以及 MEM/CMP 的重叠需 profiler timeline 实测（profiling 报告 §3 标注为源码推断）。

## §4 阻塞项与下一步

1. **Pre-Gated 权重**：base-128 已转并跑四方法；**base-256（~116GB）与 large-128（~200GB+）受 `/data` 磁盘剩余 132GB 限制，无法转换**，标资源阻塞。
2. **稳定性**：✅ base-8 35 采样、base-64 3 轮独立重复均已完成。
3. **质量（SQuAD）**：**受资源阻塞**——artifact 仓库（`6b0a666`）无 SQuAD 微调 checkpoint / 评估脚本，README 仅做性能评估；论文质量评估用的是未发布的 SQuAD 微调模型。
4. **KTransformers**：✅ Mixtral-8x7B CPU+GPU 端到端已跑通（`2026-11-22_ktransformers_mixtral_e2e.md`）；遗留 bug：prompt > 4096 token 的 CPU prefill 崩溃（`LlamafileMoEWrapper.submit_write_weight_scale_to_buffer` 缺失）。
5. **显存**：20GB arena 分配位置已定位（profiling 报告 §4）；模型加载前/后、峰值、卸载后各阶段 nvidia-smi 已记录（switch-base-8 报告）。

## §5 结论层级

- **代码能运行**：✅ Pre-Gated 四方法在 base-8/64/128 三模型跑通；KTransformers Mixtral 端到端跑通。
- **系统机制成立**：✅ Pre-gated 的预取/缓存/单专家激活机制经源码+日志验证（profiling 报告）；KTransformers 的 CPU expert offload 经 GPU 显存（16GB vs 94GB）+ `kt_num_gpu_experts=1` 验证。
- **论文图表数值可复现**：⚠️ 吞吐方向一致（Pre-gated ≈ 76–84% GPU-only）但配置不同；显存方向在 base-128 上已与论文一致（Pre-gated < GPU-only），但绝对值因机器环境不可直接对比（§3）。
