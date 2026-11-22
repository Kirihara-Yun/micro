# Pre-Gated MoE：profiling 机制与解析脚本核查

> 日期：2026-11-22
> 范围：`ranggihwang/Pregated_MoE` @ `6b0a666` 的 profiling 事件（BLK/CMP/NMB/MEM）、cache hit rate、20GB arena 分配位置，以及 `scripts/eval_all.py` / `scripts/eval_ubench.py` 解析脚本的正确性。
> 方法：源码静态核查 + 仓库 `baselines/Pregated_MoE/logs/` 下 4 份真机日志交叉验证。
> 结论层级：事件定义/统计方式 = 源码确认；`MEM≈1.03ms / BLK≈1.08ms / cache hit rate=1` 并存 = 源码机制解释（未做 profiler timeline 实测，见 §3 标注）。

## 0. 结论摘要

1. BLK/CMP/NMB/MEM 四类事件的定义与 CUDA event 起止点已查明（§1）。
2. cache hit rate 的统计方式已查明（§2）。
3. `MEM≈1.03ms`、`BLK≈1.08ms`、`cache hit rate=1` 三者并存成立，机制已解释（§3）。
4. 20GB arena 分配位置已定位：`FetcherContext` 构造 → `GroupedMemoryArena::initIfUninit` → `MemoryArena` 构造 → `cudaMallocPitch`（§4）。GPU-only 不创建 fetcher，故无 arena，解释了 3.37GB vs 23.37GB 的差异。
5. 解析脚本发现 **3 处 bug**，其中 `eval_all.py` 的 `peak_mem` 在 GPU-only 下为 **−20.87GB（负值）**，直接影响 `peak_mems.csv` 的正确性（§5）。

---

## 1. profiling 事件定义与统计方式

事件枚举与名称一一对应（`src/fastertransformer/utils/profiling.h:9` 与 `profiling.cc:21`）：

| 枚举 | 名称 | CSV 前缀 |
|---|---|---|
| `NON_MOE_BLOCK_START/END` | `NMB_S` / `NMB_E` | NMB |
| `BLOCK_START/END` | `BLK_S` / `BLK_E` | BLK |
| `MEM_START/END` | `MEM_S` / `MEM_E` | MEM |
| `COMP_START/END` | `CMP_S` / `CMP_E` | CMP |

插入点与语义：

| 事件 | 插入位置 | 语义 | 记录 stream |
|---|---|---|---|
| **BLK** | `models/t5/T5Decoder.cc:420` 起、`:670` 止 | decoder **MoE 层**整层区间（`use_moe==true`） | 主 stream |
| **NMB** | 同上 | decoder **非 MoE 层**区间（`use_moe==false`），与 BLK 互斥 | 主 stream |
| **MEM** | `utils/fetcher.cc:84` 起（`fetch()` 发起异步加载前）、`:131` 止（`sync()` 等待 `futures_` 完成并 `cudaStreamSynchronize` 后） | 一次专家权重搬运/拷贝的等待窗口 | fetcher 独立 stream |
| **CMP** | `kernels/moe_kernels.cu:878` 起、`:925` 止 | MoE GEMM（fc1 + fc2）计算区间 | 主 stream |

统计方式（`profiling.cc:56-75`）：`report()` 按 `i += 2` 把 `*_S/*_E` 配对，`cudaEventElapsedTime` 求每对时长，再 `AverageMeter` 取平均，输出 `NMB AVG / BLK AVG / MEM AVG / CMP AVG`（`substr(0,3)` 得前缀）。

真机日志佐证（`logs/switch-base-8_GPU-only_1_0_0_0.log:197`）：

```
Total events num: (NMB)1536 (BLK)1536 (MEM)0 (CMP)1536
```

`1536 = 256 token × 6 MoE 层`（`seq_len=256`，`moe_layer_index=[1,3,5,7,9,11]` 共 6 个 MoE 层）。GPU-only 下 `MEM 0`，印证其无 fetch/sync。

---

## 2. cache hit rate 机制

- 入口：`profiling.h:38` `cacheHit(bool hit)` → `AverageMeter<double>::update(hit)`。
- **唯一调用点**：`utils/arena.cc:61-64`，`MemoryArena::allocate()` 里 `cache_->GetOrPut(tag, nullptr)` 返回的 `repl.second`（该专家 tag 是否已在缓存）。
- `GroupedMemoryArena::allocate`（`arena.h:131-143`）最终走同一 `MemoryArena::allocate`，故每次 `fetch()` 里对每个 active expert 记一次 hit/miss。
- 输出：`profiling.cc:77` `Average cache hit rate: cache_hit_rate_.getAvg()`（0~1 均值；无采样时 `AverageMeter` 空值返回 0，`meter.h:9`）。
- Pre-gated 真机日志：首轮 `0.973958`，随后稳定 `1`——模型仅 8 专家、18MB/专家，全部可驻留 20GB arena，故除首轮外全命中。

---

## 3. 关键现象解释：MEM≈1.03ms、BLK≈1.08ms、cache hit rate=1 如何并存

**核心机制：cache 命中 ≠ 无搬运开销。** `MemoryArena::allocate`（`arena.cc:57-88`）里：

```cpp
auto repl = cache_->GetOrPut(tag, nullptr);      // repl.second = 是否命中
if (profiling) cacheHit(repl.second);
auto future = pool_->push([=](int) {
    if (!use_cache || (repl.first != nullptr && !repl.second && src != nullptr)) {
        cudaMemcpyAsync(repl.first, src, ..., cudaMemcpyHostToDevice, stream_); // 仅 miss 时 H2D
    }
    post_callback(repl.first, stream_);          // 无论命中与否都执行
});
```

- **命中（`repl.second=true`）只省掉 H2D（CPU→GPU），但 `post_callback`（`arena.h:135-141` 的 `cudaMemcpyAsync(dst, cached_ptr, ..., DeviceToDevice)`，把 arena 缓存权重拷到 working buffer）+ `sync()` 里的 `future.wait()` + `cudaStreamSynchronize` **仍然执行**。
- 因此 `cache hit rate=1` 与 `MEM≈1.03ms` 并存是正常的：MEM 计时的是「预取窗口」的墙钟时间，而非仅 H2D 传输时间。

**为什么 MEM≈1.03ms 而非拷贝本身的几十 μs**（switch-base-8 单专家 fp32 权重 18MB，D2D 理论上 <100μs）：
- 拷贝任务经 `ctpl::thread_pool(1)`（`arena.cc:54`，**单线程**）串行派发，存在线程唤醒/调度延迟；
- `sync()` 的 `future.wait()` 阻塞 + `cudaStreamSynchronize` 的同步开销；
- 二者叠加在小模型上成为 MEM 主体。（此条为源码推断，未用 profiler timeline 实测，标注为待验证。）

**MEM 与 CMP 的跨层重叠**（PREFETCH 模式，`moe_kernels.cu:824-843`）：
- 层 L 在 `fetch(permuted_experts_, true)` 里记录 MEM_START，异步**预取下一层 L+1** 专家；
- 层 L 随即进入 `COMP_START → GEMM → COMP_END`；
- 层 L+1 开始时 `sync()` 等待该预取完成并记录 MEM_END。
- 故 MEM 事件对跨两层，与当前层 CMP **并行重叠**；这解释了 `BLK(1.08) ≈ MEM(1.03)` 而 CMP(0.78) 单独并不等于 BLK−MEM。此重叠关系与「pre-gated 用预取隐藏传输延迟」的论文意图一致。

---

## 4. 20GB arena 分配位置与显存差异

分配链路（源码确认）：

1. `FfnLayer<T>::initFetcherContext`（`layers/FfnLayer.cc:505`）：`mode != FetchType::GPU_ONLY` 才 `make_shared<FetcherContext<T>>(...)`。
2. `FetcherContext` 构造函数（`utils/fetcher.cc:219/226`）：`GroupedMemoryArena::instance().initIfUninit(arena_size, {...}, stream)`。
3. `GroupedMemoryArena::initIfUninit → init → make_unique<MemoryArena<char>>(size, chunk_size, stream)`（`arena.h:115-129`）。
4. `MemoryArena` 构造函数（`arena.cc:30/43`）：`chunk_num_ = size_ / chunk_size_`，然后 `ptr_ = mallocBuffer(chunk_size_, chunk_num_)`。
5. `mallocBuffer`（`arena.h:78-85`）：`cudaMallocPitch(&ptr, &pitch_size, width*sizeof(T), height)` —— **立即分配约 `arena_size`（20GB）显存**。

**因此：**
- GPU-only：`encoder_fetcher_mode=0, decoder_fetcher_mode=0` → 无 FetcherContext → 不分配 arena → 峰值显存仅模型权重+激活 = **3.37GB**（真机日志 `MEM usage: 3322937344`）。
- Pre-gated：`decoder_fetcher_mode=2` → 创建 fetcher → 分配 20GB arena → 峰值显存 = 3.37 + 20 = **23.37GB**（真机日志 `MEM usage: 23279435776 23354933248`，约 21.7GiB）。

这与现有报告 `2026-11-22_pregated_switch-base-8.md` 记录的 3.37/23.37GB 一致，差异**是 20GB 固定 arena 的设计开销，而非运行异常**——在 switch-base-8（专家总量仅 4.4GB）这种小模型上，arena 反成为显存主导，这正是论文适用前提「模型大到 GPU 装不下才受益」的反面例证。

---

## 5. 解析脚本 bug（3 处）

### Bug 1（影响 `peak_mems.csv` 正确性）：GPU-only 下 `peak_mem = −20.87GB`

`eval_all.py` 的 `parse_output`（第 26-32 行）用 `re.search(r"MEM usage: (\d+) (\d+)")` 把 `MEM usage:` 行前两个数当「encoder/decoder 峰值显存」。但 GPU-only 下 `memory_usages_` 只有 **1 个采样**（仅 `T5Decoding.cc:74` 的 `recordMemoryUsage`），正则失配，`peak_mem_decoder` 保持默认 `0`。

随后第 143 行公式在 GPU-only 下符号反转：

```python
peak_mem = peak_mem_decoder - arena_size - size_per_expert * (2 * total_experts - used_buffer)
# GPU-only: used_buffer = num_layer * total_experts
```

代入 switch-base-8（`num_layer=6, total_experts=8, size_per_expert=18874368, arena_size=21474836480`）：

```
peak_mem = 0 − 21474836480 − 18874368 × (16 − 48)
         = 0 − 21474836480 + 603979776
         = −20870856704 字节 = −20.87 GB（十进制）
```

与观测值精确一致。**影响**：用 `eval_all.py` 生成的 `peak_mems.csv` 里 GPU-only 行是负值，不可用于论文/报告。

### Bug 2：`peak_mem_encoder` 语义错误（非 GPU-only 也错）

`Profiling::reset()` 在每个 `FetcherContext` 构造时清空 `memory_usages_`（`fetcher.cc:230`）。encoder、decoder 各构造一次 fetcher，encoder 的采样被 decoder 构造时清掉。真机 Pre-gated 日志的 `MEM usage` 只有 **2** 个数字（`23279435776 23354933248`），第 1 个其实是 **decoder 分配后**的显存，被 `parse_output` 误当 `peak_mem_encoder`。

### Bug 3：`eval_ubench.py` 的 `block_lat` 正则失配

`eval_ubench.py:13` 用 `re.search(r"Block avg lats: ([\d\.]+) ms")`，但真机输出是 `BLK AVG: 0.915554 ms`（`profiling.cc:74`），永远匹配不上 → `block_lat` 恒为 0。

---

## 6. 对现有报告的影响与建议

- 现有 `2026-11-22_pregated_switch-base-8.md` 的「峰值显存 3.37/23.37GB」是**直接读日志 `MEM usage` 原始字节值**，未经过 `eval_all.py` 公式，**正确**，不受 Bug 1 影响。
- 但任务单 §1.1 要求收集的 `peak_mems.csv` 走 `eval_all.py::profile_config` 的公式，GPU-only 行会产出负值。**修复前勿用该 CSV 的 `peak_mem` 列**。
- 建议修复（最小侵入，仅改 Python 解析/计算，不动 C++）：
  1. `parse_output` 的 `MEM usage` 解析改为「取该行最后一个值作为已用显存峰值」，不再假设恰好两个数、也不称其为 encoder/decoder；
  2. 移除或修正 `peak_mem` 的 `− arena_size − size_per_expert*(2*total_experts − used_buffer)` 推导，GPU-only 直接报告日志 `MEM usage` 原始值；
  3. `eval_ubench.py` 的 `block_lat` 正则改为 `BLK AVG: ([\d\.]+) ms`。
- 待补：用 `nvidia-smi` 或 profiler timeline 实测确认 §3 的 MEM/CMP 重叠关系与线程池调度开销（本报告为源码推断）。

## 原始证据

- 真机日志：`baselines/Pregated_MoE/logs/switch-base-8_{GPU-only,Pre-gated,DeepSpeed,SE-MoE}_1_0_0_0.log`（本机，4 份，共 <100KB）。
- 关键源码：`src/fastertransformer/utils/{profiling.cc,profiling.h,fetcher.cc,arena.cc,arena.h,meter.h}`、`kernels/moe_kernels.cu`、`models/t5/T5Decoder.cc`、`layers/FfnLayer.cc`。
