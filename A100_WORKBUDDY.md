# WorkBuddy：两条 MoE baseline 一次性复现任务单

更新：2026-11-22。请自主执行本单中当前硬件和可获得模型允许的全部实验，不用逐项等待新指令。遇到不可获得的硬件、权重或不兼容软件时保留证据，继续其他独立任务。最终推送报告到 `micro`。已有结果见 `experiments/`，论文差异见 `papers/READING_NOTES.md`。

## 最终交付

在 `experiments/` 提交总报告、各实验 CSV/小体积日志及复现命令。总报告逐项标记 **已复现 / 部分复现 / 受资源阻塞 / 结果不一致**，分清代码能运行、系统机制成立、论文图表数值可复现这三层结论。不要以 `import`、`kt doctor`、单次生成或论文原值代替测量。不要提交权重、编译产物、GB 级日志或密钥；大日志给出远端绝对路径及 SHA256。完成后推送 `main`，回报提交 SHA。

## 0. 环境与资源盘点

1. `git pull --ff-only`，读取 README、两份现有实验报告和论文笔记。保留上游 Pre-Gated `6b0a666`、KTransformers `ca198c5` 的 SHA；若另测新版，单独标记，绝不混用版本。
2. 记录 A100 型号、驱动、CUDA toolkit/runtime、PCIe 链路，CPU 型号/指令集/NUMA、内存、可用磁盘、容器和 Python 包。检查 GPU 是否被其他服务占用；计时期间保证独占，并在结束后恢复原服务状态。存档开始和结束时的 `nvidia-smi`。
3. 盘点 Switch Base 8/64/128/256、Switch Large 128、DeepSeek-V2-Lite、Mixtral 8x7B，以及 SOSP 论文中的 DeepSeek-V3-0324、DeepSeek-V2.5-1210、Qwen2-57B-A14B。记录路径、revision、格式、体积和完整性。缺少的模型可按上游脚本获取；先估算空间和时间。无法取得的列为资源阻塞。

## 1. Pre-Gated MoE：机制、论文图 10–12、质量

原文：[ISCA 2024 正式稿](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf)；[artifact 仓库](https://github.com/ranggihwang/Pregated_MoE)。论文比较 GPU-only、Pre-gated、MoE-OnDemand（代码中 DeepSpeed）、MoE-Prefetch（代码中 SE-MoE）。现有 Switch-Base-8 的 GPU-only / Pre-gated 重复计时已完成，可复用原始结果，但以下缺口仍要补。

1. **对齐 artifact**：检查 `scripts/eval_all.py` 当前模型、方法、batch、metric 列表和实际命令。用其默认协议（batch=1）运行所有当前可用模型与四种方法，收集 `block_lats.csv`、`throughputs.csv`、`peak_mems.csv`。GPU-only 的预期 OOM 单独记录。若全量矩阵超出资源，先完成 Base-8 四方法，再扩大专家数；保证每步结果可提交。
2. **稳定性**：每组记录热身/正式迭代、median/mean/std、轮间波动；同方法至少独立重复 3 轮。把论文图 10–12 的可比点及比值并排，区分吞吐、MoE block latency、显存的口径。远端 CPU 与论文不同，绝对值未必一致。
3. **显存异常优先排查**：现有 Base-8 报告 GPU-only 3.37GB、Pre-gated 23.37GB；论文图 12 的 Base-8 Pre-gated 约为 GPU-only 的 0.65。分别记录模型加载前、加载后、热身后、运行峰值、卸载后的 `nvidia-smi` 和分配器统计（若适用）；查明 20GB arena 的分配位置。核对 4.4GB 转换目录与 3.37GB GPU 峰值的差异、`beam_width=4` 与论文脚本配置差异。没有证据前不定性为论文矛盾或 arena 根因。
4. **机制计时**：查明 BLK/CMP/NMB/MEM 的代码定义、CUDA event 起止点及重叠关系；解释 `MEM≈1.03ms`、`BLK≈1.08ms` 与 `cache hit rate=1` 如何同时出现。用代码位置、时间线或 profiler 证据说明预门控与搬运确实发生，记录专家激活数。
5. **质量与模型配置**：确认当前 `google/switch-base-8` 是否包含论文的预门控微调权重。若作者发布了 checkpoint/脚本及数据集，按原配置复现至少 Base-8 的 SQuAD EM/F1；条件允许再做 XSum、CB Web QA。无权重时尝试论文公开训练流程，明确时间/资源。预训练权重的生成样例不能当论文准确率复现。
6. **拓展配置**：有资源时优先 Base-64/128 和 Large-128，以回答容量收益；Base-256 属显存图拓展。小模型 GPU-only 可运行时，仍需比较 Pre-gated 与两个 CPU offload baseline，不能写“只有 GPU 放不下时才有收益”。

## 2. KTransformers：端到端、机制、论文可比性

原文：[SOSP 2025 论文](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)。当前 `kt-kernel 0.7.2` CUDA 构建及导入已完成；这只证明后端可加载。已有构建参数：`CUDA_HOME=/usr/local/cuda-12.6`、`CUDAHOSTCXX=/usr/bin/g++-12`、`CPUINFER_USE_CUDA=1`、`CPUINFER_CUDA_ARCHS=80`。

1. **补顶层推理框架**：根据 `ca198c5` 对应文档和 submodule 版本安装 **kvcache-ai 的 `sglang-kt` fork**，不要误用普通 `sglang`。[官方当前安装说明](https://www.ktransformers.net/en/docs/getting-started/installation)可参考，但先核对与 pinned 源码兼容性。记录 SGLang fork SHA/版本、依赖树和启动命令。安装失败要定位并尝试修复实际错误，不在首个异常处停下。
2. **先跑真实模型**：从现有 DeepSeek-V2-Lite 或 Mixtral 权重中选与 pinned 版本兼容且完整的一种；按对应模型文档准备正确的 GGUF/其他格式与量化。启动 CPU+GPU 服务，发送固定输入，保存输出、日志、专家 CPU/GPU 放置、实际 GPU kernel/CPU variant、显存/主存和 NUMA 配置。若两者均不兼容，选上游支持且可取得的最小 MoE 模型，记录依据。
3. **可重复性能**：区分 prefill 与 decode，固定 batch=1，至少测试短/中/长 prompt（如 32/512/2048 tokens）与固定生成长度；每条件热身后至少 3 次，报告 tokens/s、first-token latency、peak GPU/CPU memory。记录量化、精度、CPU 线程、GPU 专家数及实际输入输出 token 数，保存请求和原始数据。
4. **机制验证**：用日志、profiler 或内核调用证据验证 routed experts 确实在 CPU 上计算、其他模块在 GPU、CPU/GPU 存在异步重叠。仅检测到 GPU 不算。若实现支持，做最小消融（关闭关键调度或改变 GPU 专家数）并解释变化；不支持则注明。
5. **论文数值边界**：SOSP 使用双路 Xeon 8452Y（AMX、DDR5）、A100 40GB/RTX 4080、DeepSeek-V3-0324/DeepSeek-V2.5-1210/Qwen2-57B-A14B，并比较 Fiddler 和改造过的 llama.cpp。当前 Xeon 8360Y 无 AMX，先完成现有硬件功能复现。仅在 AMX CPU、论文同款模型和 baseline 代码可取得时，按论文 §6 做 prefill/decode 数值与消融复现；否则将 AMX kernel、Expert Deferral 质量、Fiddler/llama.cpp 对比逐项标资源阻塞，写清所需资源，不能称全论文复现。

## 3. 汇总与收尾

- 用矩阵列出每个关键主张、对应实验、状态、证据链接、与论文差异及剩余阻塞。分清 Pre-gated 相对 GPU-only 与相对其他 offload 方法，以及 KTransformers 当前功能与 SOSP 旧版论文指标。
- 修订旧报告中被新证据推翻的解释，保留原始测量值并注明修订日期，不默默覆写历史数据。
- 确保 CSV 和报告不依赖 `/tmp` 才能复核，敏感路径/凭据不入库。运行 `git diff --check` 后提交推送；若推送冲突先拉取整合，回报 commit SHA 和全部仍受阻项。
