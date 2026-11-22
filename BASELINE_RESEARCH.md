# MoE baseline 研究起点

更新：2026-11-22。本文区分仓库 README 已记录的本机冒烟结果与论文结果；两者的实验配置尚未对齐。

## 论文与实现

| Baseline | 论文 | 本仓库记录的代码版本 |
|---|---|---|
| Pre-Gated MoE | [Hwang et al., ISCA 2024](https://arxiv.org/abs/2308.12066)；[作者公开的正式稿与 artifact 说明](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf) | [Pregated_MoE](https://github.com/rangghiwang/Pregated_MoE) `6b0a666` |
| KTransformers | [Chen et al., SOSP 2025](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)；[项目仓库](https://github.com/kvcache-ai/ktransformers) | `ca198c5` |

## 研究上的关键差别

- **Pre-Gated MoE**：提前确定下一 MoE 层的专家，在当前层执行期间把被选中的专家权重从 CPU 内存迁移到 GPU，目标是隐藏 PCIe 传输延迟并减少显存占用。论文的系统比较包含 GPU-only、按需迁移和预取等设计。论文在 A100 80GB、CPU/GPU PCIe Gen4、batch 1 上评估 Switch Transformer；artifact 建议至少 40GB 显存、128GB 主存。其公开评估脚本输出 block latency、throughput 和 peak GPU memory。
- **KTransformers**：让专家权重常驻 CPU 内存并在 CPU 上计算，GPU 执行其他部分；核心包括 AMX/AVX-512 专家 kernel、量化布局、异步 CPU-GPU 调度和 NUMA 优化。因此它与 Pre-Gated MoE 的异构路径不同，不能只比较一个 tokens/s 数字就归因算法优劣。论文性能数据基于其特定 CPU、GPU、模型及并发配置。
- README 里的 Pre-Gated `154`/`130 tokens/s` 只有 4 次迭代，是冒烟数据，不构成论文性能复现；KTransformers 目前只验证 `kt_kernel` 导入及 `avx512_vbmi` 检测，尚无模型推理或 CUDA 后端数据。当前 Xeon 8360Y 不具备 AMX，无法直接复现论文的 AMX 路径。

## 下一步验证顺序

1. 固定两个上游 commit、模型权重版本、运行命令、软件版本和硬件拓扑，保存原始日志。
2. 在 A100 上扩充 Pre-Gated MoE 的 Switch-Base-8 运行次数，分别测 GPU-only / Pre-gated 的端到端吞吐、MoE block 延迟及显存峰值；测量热身后稳定区间，并检查输出一致性或任务质量。
3. KTransformers 先选择其当前版本支持、且远端已有权重的 MoE 模型，完成 CPU+GPU 端到端推理；记录 kernel 路径、量化格式和 CPU/GPU 分工。若目标是复现论文 AMX 性能，还需 AMX CPU，单有 A100 不够。
4. 只有在模型、精度、batch、输入输出长度、硬件与计时口径可比时，才做跨系统性能表；否则分别报告各自复现程度。

## A100 远端 WorkBuddy 交接

工作文档见 [A100_WORKBUDDY.md](A100_WORKBUDDY.md)。远端执行时请将日志和配置回传到本仓库的实验记录目录或提供可读取路径；不要提交权重、编译产物和大体积日志。
