# 两篇 MoE baseline 论文研读笔记

日期：2026-11-22。两篇作者公开的论文已下载到本地 `papers/`；PDF 和抽取的纯文本仅供本地阅读，不提交仓库。远端按下列原文链接获取。

| 论文 | 原文 | 本地 PDF SHA256 |
|---|---|---|
| Pre-gated MoE, ISCA 2024 | [Microsoft Research 正式稿与 artifact](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf) | `C4BBEC50D7496CDE7F455D36AB3B89EBDCC8A8A351258CD5918AB0056A3AD7C8` |
| KTransformers, SOSP 2025 | [清华大学作者 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) | `D3C5AF7A9E118AEF23F0FC2B95A61BFE098979B56680A1291A7B41B848E328F3` |

## Pre-gated MoE：论文主张和当前证据

论文的比较对象是 GPU-only、MoE-OnDemand、MoE-Prefetch 与 Pre-gated。Pre-gated 提前选择下一层专家，让迁移和当前 MoE 计算重叠。论文采用 A100 80GB、AMD EPYC 7V12、PCIe Gen4，主要考察 batch=1；其模型用于性能测试前经过 SQuAD 微调。Switch-Base 各配置中，Pre-gated 平均达到 GPU-only 吞吐的 81%，相比两个**其他 CPU offload 方案**更快。Switch-Large-128 则因约 105.6GB 模型无法放进 GPU，GPU-only OOM。参见论文第 V、VI-A 节及图 11。

远端 [Switch-Base-8 报告](../experiments/2026-11-22_pregated_switch-base-8.md) 的 153/129 tok/s、0.911/1.080ms 延迟重复性良好，但与论文仍非同一实验：报告使用 `google/switch-base-8` 权重、`perf_benchmark.py`、`beam_width=4`，论文性能评估使用任务微调模型及 `eval_all.py`。论文把 GPU-only 视为能容纳模型时的性能上界；小模型 Pre-gated 吞吐较低与此相容。不过“收益仅在大模型 GPU 装不下时体现”过强：即使 GPU-only 可运行，论文仍考察 Pre-gated 相对其他 CPU offload 方法的延迟与显存收益。

**待核查的显存冲突：**论文图 12 中 Switch-Base-8 的 Pre-gated 峰值显存约为 GPU-only 的 0.65 倍，远端报告却是 23.37GB 对 3.37GB（约 6.94 倍）。这不是论文结果的直接复现。`~20GB arena` 是报告的解释，但目前缺分配器证据和同口径统计；还需解释为什么转换权重目录 4.4GB，而报告 GPU-only 峰值只有 3.37GB。优先核查 profiler 的内存定义、统计时点、模型加载方式、额外预分配及 `beam_width` 的影响。报告中的 `MEM≈1.03ms` 与 `BLK≈1.08ms` 也不应简单相加；需要确认事件的起止点与重叠程度。

## KTransformers：论文主张和当前证据

论文让 routed experts 常驻 CPU 内存并由 CPU 执行，GPU 执行其他部分；优化包括 AMX 布局与 kernel、低算术强度时的 AVX-512 路径、异步 CPU-GPU 调度、NUMA 感知和 Expert Deferral。其评估机器为双路 Xeon 8452Y（DDR5、AMX），GPU 包含 A100 40GB 和 RTX 4080；模型是 DeepSeek-V3-0324、DeepSeek-V2.5-1210、Qwen2-57B-A14B，分别测 prefill/decode，batch=1。参见论文第 3–6 节。SOSP 论文对应的是当时系统及模型，不能把现在 `kt-kernel 0.7.2` 的导入成功视为论文性能复现。

远端 [CUDA 后端报告](../experiments/2026-11-22_ktransformers_cuda_backend.md) 已证实构建和加载，但尚无 SGLang 驱动的端到端推理。Xeon 8360Y 无 AMX，无法直接验证论文的 AMX 性能主张。下一步可先以已有 DeepSeek-V2-Lite 或 Mixtral 权重验证现代版本的 CPU+GPU 功能；之后若研究论文数值，再准备相同模型、量化、AMX CPU 与计时口径。

## 研究问题

1. Pre-gated 的预选门控是否采用经微调的权重？当前 `google/switch-base-8` 转换路径与论文质量评估配置差在哪？
2. 23.37GB 峰值显存由哪些分配组成？与论文图 12 的统计口径能否对齐？
3. `cache hit rate=1` 的定义是什么？是否意味着传输全部被命中，或只是指标名称不同？与每层 `MEM≈1.03ms` 如何同时成立？
4. KTransformers 当前版本的 CPU/GPU 放置、专家量化与调度路径，和 SOSP 论文实现相差多少？
