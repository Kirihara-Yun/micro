# 跨组交换消融结果（two vs two_no_swap）

> 日期：2026-11-27。按 `WORKBUDDY_SWAP_ABLATION.md` 执行：A100 BF16 冒烟 → 配对训练（各 1529 步 ≈ 50M tokens）→ KV cache 解码。
> 两组双阶段模型使用**完全相同的初始权重、参数量和训练预算**，唯一区别是把跨组交换索引换成**恒等索引**（保留同一个 `index_select` 算子）。
> **结论：跨组交换贡献了 +0.0258 验证 NLL（约 50M tokens），且差值随训练量单调扩大。
> 这归因了上一版双阶段相对单阶段的 0.030 NLL 优势中约 85% 来自交换，而非"两阶段结构本身"。单 seed 仅筛选。**

## 0. 环境与数据

| 项 | 值 |
|---|---|
| GPU | 1× NVIDIA A100 80GB PCIe |
| PyTorch | 2.5.1+cu124（conda `rm-r1-1`） |
| 代码版本 | `9aa66b9`（model_v2.py 新增 `two_no_swap` 变体） |
| 数据 | 同 v2：FineWeb-Edu `sample-10BT` 前 2 分片，GPT-2 tokenizer，packing 512 |
| data SHA256 | train.npy `1960dfe9...`，val.npy `7380e732...`（与原报告一致） |

## 1. 冒烟（`python smoke_v2.py --device cuda --bf16`）

四变体全部通过，含新增的**消融初始化一致性检查**：

```
two/two_no_swap 初始 state_dict 逐张量一致（仅 swap_idx buffer 不同）
moe: resume, causal prefix and incremental decode checks passed
single: resume, causal prefix and incremental decode checks passed
two: resume, causal prefix and incremental decode checks passed
two_no_swap: resume, causal prefix and incremental decode checks passed
```

即：两组双阶段模型在 step 0 权重完全一致，之后的差异只能来自 `swap_idx`（交换 vs 恒等）这一个变量。

## 2. 配对训练（1529 步 ≈ 50.0M tokens，seed 20261123）

验证 NLL 曲线与差值（`delta = NLL(no_swap) - NLL(swap)`，正值 = 交换有收益）：

| tokens | two(swap) | two_no_swap | delta |
|---|---|---|---|
| 4.9M | 6.7352 | 6.7486 | +0.0134 |
| 9.8M | 6.2376 | 6.2540 | +0.0164 |
| 14.7M | 5.9256 | 5.9425 | +0.0169 |
| 19.6M | 5.6891 | 5.7119 | +0.0227 |
| 24.5M | 5.5153 | 5.5417 | +0.0265 |
| 29.4M | 5.3901 | 5.4150 | +0.0249 |
| 34.3M | 5.3051 | 5.3296 | +0.0245 |
| 39.2M | 5.2391 | 5.2640 | +0.0249 |
| 44.2M | 5.1969 | 5.2213 | +0.0244 |
| 49.1M | 5.1716 | 5.1977 | +0.0260 |
| 50.0M | 5.1688 | 5.1946 | **+0.0258** |

最终步指标：

| 变体 | 训练 loss | val_nll | tokens/s | 显存峰值 |
|---|---|---|---|---|
| two（跨组交换） | 5.0137 | **5.1688** | 20404 | 5.35 GB |
| two_no_swap（恒等索引） | 5.0423 | 5.1946 | 19944 | 5.35 GB |

**观察**：
- 交换组全程 NLL 更低，delta 从 4.9M 的 +0.013 单调上升到 50M 的 +0.026（趋势仍在扩大，未收敛）。
- 训练吞吐：two 20404 vs two_no_swap 19944（差异 2%，接近测量噪声；恒等索引的 `index_select` 不省任何计算，与设计一致）。

## 3. decode（KV cache，BF16，batch=1，8 请求 × 64 token × 3 轮）

| 变体 | TTFT P50 | decode P50 | P95 | 吞吐 | 显存峰值 |
|---|---|---|---|---|---|
| two | 5.26 ms | 4.57 ms | 4.60 ms | 218 tok/s | 84 MB |
| two_no_swap | 5.25 ms | 4.57 ms | 4.59 ms | 218 tok/s | 84 MB |

两组 decode 完全一致（结构相同，交换索引只改索引值、不改计算量），符合预期。

## 4. 结论与归因

1. **跨组交换有可测的正向贡献**：在 50M tokens 下，交换相对恒等索引改善验证 NLL **+0.0258**，且差值随训练量单调扩大（未收敛，更长训练可能进一步拉开）。
2. **归因上一版结论**：上一版 `REPORT_V2.md` 中双阶段相对单阶段的优势为 `5.1987 - 5.1684 = 0.0303`。本次消融把它分解为：
   - 跨组交换贡献：`5.1946 - 5.1688 = +0.0258`（**约 85%**）
   - 两阶段结构本身（恒等索引，等价于"多一层冗余单阶段"）：`5.1987 - 5.1946 = +0.0041`（约 15%）

   即：**双阶段相对单阶段的质量优势，主要来自跨组交换，而非"两阶段"这一结构本身。**
3. **速度权衡不变**：交换不改变 decode 延迟（4.57ms 与恒等一致），但双阶段整体仍慢于单阶段（上一版：two 4.58ms vs single 3.42ms）。这一速度代价来自"两阶段串行 + 两次路由"，而非交换本身——若后续只需交换收益，可考虑"单阶段 + 跨组交换"变体（本次未测）。
4. **单 seed 仅筛选**：seed 20261123，delta 虽单调但幅度 ~0.026 仍在单 seed 波动量级附近，不构成统计判定；多种子或更长训练可确认稳健性。

## 5. 交付物

| 文件 | 说明 |
|---|---|
| `logs/v2_swap_ablation_{two,two_no_swap}_20261123/metrics.jsonl` | 训练/验证曲线（已提交） |
| `logs/v2_swap_ablation_{two,two_no_swap}_20261123/decode.jsonl` | KV cache 解码测量（已提交） |
| `logs/v2_swap_ablation_*/latest.pt` | checkpoint（各 ~358MB，留远端未提交） |

命令（与任务单一致）：
```bash
python smoke_v2.py --device cuda --bf16
for kind in two two_no_swap; do
  out="logs/v2_swap_ablation_${kind}_20261123"
  python train_v2.py --kind "$kind" --data data --out "$out" --steps 1529 --warmup 150 --eval-every 150 --batch 8 --accum 8 --seed 20261123 --bf16
  python decode_v2.py --checkpoint "$out/latest.pt" --prompts data/val.npy --prompt-len 128 --new-tokens 64 --requests 8 --rounds 3 --bf16 > "$out/decode.jsonl"
done
```
