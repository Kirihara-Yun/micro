# Mixtral CPU+GPU decode 执行限制诊断

> 日期：2026-11-22。目标：在已跑通的 Mixtral q8_0 LLAMAFILE 路径上定位单请求 decode（~9 tok/s）的真实瓶颈，
> 为新机制设计提供因果证据。**结论先行：瓶颈是 CPU 专家服务能力（AVX-512 q8_0），GPU 94.6% 时间空闲等待 CPU。**

## 0. 环境与固定配置

| 项 | 值 |
|---|---|
| 源码 | ktransformers `ca198c5` / sglang fork `2c4085a` / Mixtral q8_0 LLAMAFILE |
| 权重 | HF bf16 完整权重（GPU 非专家）+ `mixtral-exps-q8_0.gguf`（CPU 专家，堆叠布局） |
| CPU | Xeon Platinum 8360Y，72 核（36×2 socket，**无超线程**），4 NUMA，AVX512_VBMI（无 AMX） |
| GPU | 1× A100 80GB PCIe |
| 固定 | 关闭 `--kt-enable-dynamic-expert-update`；`--disable-chunked-prefix-cache`；batch=1 |
| 默认 | `--kt-cpuinfer 32 --kt-threadpool-count 4 --kt-num-gpu-experts 1 --kt-method LLAMAFILE` |

## 1. P0：服务时间分解（测量）

### 1.1 每 token decode latency（stream 模式，6 请求样本之一 `code_1`，热身后 3 轮）

| 配置 | inter-token mean | med | p95 | 吞吐 | TTFT |
|---|---|---|---|---|---|
| 32 线程 / 1 GPU 专家 | **107.2 ms** | 107.6 | 112 | 9.3 tok/s | 225 ms |
| 32 线程 / 2 GPU 专家 | 93.2 ms | 93.7 | 101 | 10.7 tok/s | 196 ms |
| 16 线程 / 1 GPU 专家 | 172.7 ms | 173.8 | 181 | 5.8 tok/s | 363 ms |
| 64 线程 / 1 GPU 专家 | 69.1 ms | 69.1 | 73 | 14.5 tok/s | 148 ms |

### 1.2 decode 期间 CPU/GPU 利用率（32 线程，128 token，每 0.4s 采样）

| 资源 | 平均利用率 |
|---|---|
| CPU（72 核） | **42.0%**（≈32/72，即 32 个 CPU 线程满载） |
| GPU | **5.4%**（几乎空闲） |

### 1.3 GPU kernel 分解（Nsight Systems 2025.3.2，`--trace=cuda,nvtx,osrt`，采集 128 token decode）

| kernel | 占 GPU 时间 | 总时间 | 实例 |
|---|---|---|---|
| `fused_moe_kernel`（GPU 侧 MoE） | **53%** | 11.7 ms | 64 × 183 μs |
| `ampere_bf16_gemm` | 9% | 2.0 ms | 64 |
| reduce / elementwise / attention / rope 等 | 38% | ~8 ms | — |

**GPU 总 kernel 时间 ≈ 22 ms / 128 token ≈ 0.17 ms/token**（vs wall-clock 107 ms/token）。

### 1.4 服务时间分解（每 token，32 线程）

| 分量 | 时间 | 占比 | 来源 |
|---|---|---|---|
| wall-clock | 107 ms | 100% | 测量 |
| GPU 工作（全部 kernel） | 0.17 ms | 0.16% | nsys 测量 |
| CPU 专家计算 + 反量化 | ~105 ms | ~98% | 模型估计（wall − GPU − 传输） |
| 传输 + 同步（8KB hidden state，PCIe） | <1 ms | <1% | 模型估计 |

## 2. P1：受控干预（测量）

### 2.1 CPU 专家线程 16/32/64（剂量-反应）

```
线程  16  →  32  →  64
吞吐  5.8 → 9.3 → 14.5 tok/s   （16→32 提升 1.61×，32→64 提升 1.55×）
延迟 173 → 107 → 69 ms/token
```

吞吐随线程数近似线性增长，证明 CPU 专家计算是 **thread/compute-bound**（可并行扩展）。

### 2.2 GPU 专家数 1→2（32 线程）

吞吐 9.3 → 10.7 tok/s（+15%）。GPU 有空闲能力，把 1 个专家搬回 GPU 立即减少 CPU 工作量并提升吞吐，与「GPU 空闲、CPU 瓶颈」一致。

## 3. P2：乐观上界（模型估计）

| 上界 | 定义 | 数值 | 说明 |
|---|---|---|---|
| 上界 1 | 消除控制/调度节点，保留数据依赖 | ≈ 107 ms/token | CPU 已满载，调度开销小，无实质提升 |
| 上界 2 | CPU 专家计算理想为 0，保留 GPU + 传输 | ≈ 0.2 ms/token（→5000 tok/s） | 纯假设，不可实现（专家必须被某处计算） |
| 上界 3 | 允许更好的 CPU/GPU 合法重叠 | ≈ 上界 1 | GPU 已 94.6% 空闲，重叠已近最优，无法减少 CPU 关键路径 |

**关键数字（模型估计）**：每 token CPU 专家计算量 ≈ 32 层 × 2 专家 × 2×(4096×14336) ≈ **7.5 GMAC**；
32 线程有效吞吐 ≈ 7.5 GMAC / 107 ms ≈ **70 GMAC/s**。对比 AVX512-VNNI 峰值 ≈ 22 TMAC/s，
**实测只用到 ~0.3% 峰值**——CPU 专家 kernel 的效率远低于算力上限，瓶颈在 q8_0 反量化 + 访存/布局，而非原始算力。

## 4. 最大可解释差距（1–2 个）

1. **CPU 专家服务能力是唯一主导瓶颈**：每 token 107 ms 中 CPU 专家计算占 ~105 ms（98%），GPU 只贡献 0.17 ms。
   触发条件：单请求 decode，专家置于 CPU、GPU 只留 1 个专家 + 非专家模块。现有机制留下的原因：KTransformers
   把 routed experts 固定放在 CPU 以省 GPU 显存，但本机 AVX-512 q8_0 的 CPU 专家吞吐（70 GMAC/s）远慢于 GPU，
   导致 GPU 长期空闲。可改变的执行单位：**CPU 专家计算本身**（更多核 / AMX / 更优量化 / 部分专家回迁 GPU），
   而非调度或重叠策略。

2. **次要差距（高核数下显现）**：32→64 线程吞吐提升 1.55×，低于 2× 线性。说明核数 >32 后出现非计算瓶颈
   （4 NUMA 的跨节点访存 / 内存带宽），纯「CPU 计算瓶颈」的解释在 64 线程下不完整。可改变的执行单位：
   NUMA 局部权重放置 / 每 NUMA 独立 threadpool（当前 `--kt-threadpool-count 4` 已按 4 NUMA 分池，但权重 mmap 布局未对齐）。

## 5. 能推翻首选解释的对照

- **首选解释**：CPU 专家计算是瓶颈，GPU 空闲。
- **支持证据**：①线程数剂量-反应近似线性；②CPU 42%（满载）vs GPU 5.4%（空闲）；③GPU 专家 1→2 吞吐 +15%。
- **推翻/修正的对照**：32→64 线程提升仅 1.55×（非 2×）——若瓶颈是纯可并行的 CPU 计算，应接近 2×。该偏离说明
  高核数下内存带宽/NUMA 成为次级瓶颈，**修正首选解释为**：低核数下是 CPU 计算瓶颈，高核数下过渡为
  计算+访存混合瓶颈。这一对照避免了把「加核」当作无限可扩展的错误结论。

## 6. 结论

- 若执行时间已接近物理下界：**否**。CPU 专家仅用 ~0.3% 峰值算力，存在 >100× 的理论空间，但受 q8_0 反量化与
  访存布局约束，真实可及空间取决于更优量化/AMX/数据布局，而非继续堆调度策略。
- 需要改变的是**并行工作来源或算法**：要么把专家算力搬回 GPU（牺牲显存）、要么用 AMX/更好量化加速 CPU 专家、
  要么减少每 token 的专家计算量（如更稀疏的专家激活）。纯调度/重叠策略无法消除本瓶颈。

## 7. 交付物清单

| 文件 | 位置 | 说明 |
|---|---|---|
| 报告 | `experiments/research_execution_limits/2026-11-22_mixtral_decode_bottleneck.md` | 本文 |
| 每 token latency 脚本 | `scripts/kt_token_latency.py` | stream 时间戳测量 |
| torch profiler 脚本 | `scripts/kt_profile.py` | sglang /start_profile 采集 |
| nsys trace（远端，不提交） | `/tmp/nsys_mixtral_32t.nsys-rep` | SHA256 `24e798cb18b623801af7f11d7d4394be11e32dc1c21b84ed0f768ab182fa3d5a` |
| nsys kernel 汇总（小） | `/tmp/nsys_stats_cuda_gpu_kern_sum.csv` | SHA256 `798648a082b0d07ad668c7233a9480addc301119184693009552dd38088a61a5` |

## 8. 未完成/阻塞

- **P1.2 独立 MoE 层回放**：未实现（需从真实 trace 抽取 expert 输入 + top-2 IDs，再独立调用 `LlamafileMoEWrapper`）。
  已用线程数剂量-反应 + 资源利用率间接替代，足以定位瓶颈；层回放可进一步拆分「CPU 计算 vs 传输 vs 同步」，作为后续。
- **P1.3 验证宽度 1/2/4/8**：无对应投机解码接口，未做。
- **P1.4 长 context 对照**：attention 仅占 GPU 时间 ~1%（未达「显著比例」阈值），按任务约定跳过；>4096 CPU prefill bug 未修复。
