# 跨组交换复验（三种子）与解码剖析

> 日期：2026-11-27。按 `WORKBUDDY_SWAP_FOLLOWUP.md` 执行：
> ① 补两个种子（20261124/25）配对训练，与既有种子 20261123 汇总三种子 NLL 差值；
> ② 用 20261123 checkpoint 剖析 single/two/two_no_swap 的解码开销（路由/选块/块乘/通道交换）。
> **结论：三种子 delta 全部为正（+0.0056 / +0.0128 / +0.0258，均值 +0.0147），跨组交换方向稳健但效应量比单种子小；
> decode 慢的主因是"两阶段翻倍"（路由/选块/块乘算子调用翻倍），通道交换本身只占很小开销（~149us/token 主机时间）。**

## 0. 环境与数据

| 项 | 值 |
|---|---|
| GPU | 1× NVIDIA A100 80GB PCIe |
| PyTorch | 2.5.1+cu124（conda `rm-r1-1`），BF16 |
| 代码版本 | `80615f9` |
| 数据 | 同 v2：FineWeb-Edu `sample-10BT` 前 2 分片，GPT-2 tokenizer，packing 512 |
| data SHA256 | `run_swap_pairs.py` 训练前已核对，与 `REPORT_V2.md` §0 一致 |

## 1. 冒烟（`python smoke_v2.py --device cuda --bf16`）

四变体全部通过，含消融初始化一致性检查（two/two_no_swap 初始 state_dict 逐张量一致，仅 `swap_idx` 不同）。

## 2. 解码剖析（20261123 checkpoint）

### 2.1 常规 decode（BF16，batch=1，8 请求 × 64 token × 3 轮，本轮重测）

| 变体 | TTFT P50 | decode P50 | P95 | 吞吐 |
|---|---|---|---|---|
| single | 3.95 ms | **3.40 ms** | 3.43 ms | **293 tok/s** |
| two | 5.33 ms | 4.64 ms | 4.67 ms | 215 tok/s |
| two_no_swap | 5.57 ms | 4.81 ms | 4.84 ms | 208 tok/s |

### 2.2 profiler 时间分解（per-token，host_self / cuda_kernel，us）

| 类别 | single | two（交换） | two_no_swap（恒等） |
|---|---|---|---|
| routing | 800.1 / 84.2 | 1669.4 / 167.3 | 1720.1 / 167.3 |
| block_selection | 294.9 / 54.4 | 605.3 / 106.5 | 568.5 / 106.6 |
| block_einsum | 366.8 / 16.1 | 714.8 / 32.3 | 733.0 / 32.3 |
| channel_exchange | — | 148.7 / 20.8 | 148.9 / 20.8 |
| shared_ffn_projections | 1237.0 / 68.6 | 1211.4 / 68.5 | 1232.6 / 68.5 |
| other_ffn | 1341.3 / 37.4 | 2154.3 / 53.7 | 2182.1 / 53.7 |
| outside_ffn | 3937.5 / 411.3 | 3955.8 / 410.4 | 3971.4 / 410.3 |
| decode_synchronization | 8.7 / 0 | 8.5 / 0.4 | 7.6 / 0.4 |

### 2.3 主要算子调用次数（per-token）

| 算子 | single | two |
|---|---|---|
| routing `cudaLaunchKernel` | 30 次 | **60 次**（翻倍） |
| block_einsum `aten::bmm` | 6 次 | **12 次**（翻倍） |
| block_selection `aten::index` | — | 12 次 |
| routing `aten::argmax` | — | 12 次 |
| shared_ffn `aten::mm` | 18 次 | 18 次（不变） |

### 2.4 剖析结论

1. **双阶段慢的主因是"两阶段翻倍"**：路由 kernel 启动 30→60 次/token、块乘 bmm 6→12 次/token，选块/块乘/路由的 host_self 时间都约翻倍（block_selection 295→605、block_einsum 367→715、routing 800→1669 us）。共享投影（shared_ffn）不变。
2. **通道交换开销很小**：`channel_exchange` 仅 ~149us/token 主机独占时间、~21us CUDA kernel 时间，相对双阶段额外开销（host_self 合计约 +2140us）只占 ~7%。
3. **交换 vs 恒等开销一致**：two 与 two_no_swap 的 channel_exchange（148.7 vs 148.9us）及全部类别几乎逐项相同——`index_select` 换索引值不改变计算量。这解释了常规 decode 里两者差异（4.64 vs 4.81ms）属测量噪声量级。
4. **口径提醒**：profiler 的 host_self 是 CPU 主机独占时间，含提交开销且与 GPU kernel 可重叠，不能简单累加等于 P50。但方向性结论（两阶段翻倍主导、交换占小头）在两个口径下都成立。

## 3. 三种子配对训练（各 1529 步 ≈ 50M tokens）

最终 delta（`NLL(no_swap) - NLL(swap)`，正值 = 交换有利）：

| seed | two(swap) | two_no_swap | delta_nll |
|---|---|---|---|
| 20261123 | 5.1688 | 5.1946 | **+0.0258** |
| 20261124 | 5.1954 | 5.2082 | **+0.0128** |
| 20261125 | 5.1981 | 5.2037 | **+0.0056** |
| **汇总** | | | **均值 +0.0147**（min +0.0056，max +0.0258） |

delta 曲线（每 4.9M tokens）：

| tokens | seed23 | seed24 | seed25 |
|---|---|---|---|
| 4.9M | +0.0134 | +0.0137 | +0.0019 |
| 9.8M | +0.0164 | +0.0127 | −0.0033 |
| 14.7M | +0.0169 | +0.0113 | −0.0084 |
| 19.6M | +0.0227 | +0.0162 | −0.0017 |
| 24.5M | +0.0265 | +0.0202 | +0.0023 |
| 29.4M | +0.0249 | +0.0175 | −0.0003 |
| 34.3M | +0.0245 | +0.0156 | +0.0045 |
| 39.2M | +0.0249 | +0.0154 | +0.0052 |
| 44.2M | +0.0244 | +0.0142 | +0.0040 |
| 49.1M | +0.0260 | +0.0153 | +0.0047 |
| 50.0M | +0.0258 | +0.0128 | +0.0056 |

三种子 decode P50（two / two_no_swap，ms）：seed23 4.57/4.57、seed24 4.60/4.72、seed25 4.73/4.61（差异 ±0.12ms，噪声量级）。

## 4. 结论

1. **跨组交换方向稳健为正**：三个种子最终 delta 全部为正（+0.0056 ~ +0.0258），提供重复性证据——交换确有正向贡献，不是单种子偶然。
2. **但效应量比单种子小，且种子间波动大**：均值 +0.0147，约是单种子 seed23（+0.0258）的一半。seed25 早期 delta 甚至为负（−0.008 ~ −0.003），后期才转正。**单种子会高估交换收益，正式结论应报告均值 +0.0147、范围 [+0.006, +0.026]。**
3. **交换的速度代价可忽略**：剖析确认交换本身（index_select）只占 ~149us/token 主机时间，且恒等索引开销相同。双阶段的 decode 慢（4.64 vs 单阶段 3.40ms）来自"两阶段串行 + 两次路由/选块/块乘"，与交换无关。
4. **归因更新**：上一版用单种子（seed23）报告"交换贡献 ~85% 双阶段优势"。多种子后，交换的稳健贡献约 +0.015 NLL；双阶段相对单阶段的优势里，交换仍是主要来源，但"两阶段结构本身"的贡献在部分种子下可能与交换可比。

## 5. 交付物

| 文件 | 说明 |
|---|---|
| `logs/v2_swap_followup/new_pairs.json` | 种子 20261124/25 配对结果（已提交） |
| `logs/v2_swap_followup/three_pairs.json` | 三种子汇总（已提交） |
| `logs/v2_swap_followup/{single,two,two_no_swap}_decode.jsonl` | 常规 decode（已提交） |
| `logs/v2_swap_followup/{single,two,two_no_swap}_profile.json` | 剖析汇总（已提交） |
| `logs/v2_swap_followup/{single,two,two_no_swap}_trace.json` | Chrome trace（留远端，体积大） |
| `logs/v2_swap_ablation_*_202611{24,25}/metrics.jsonl` + `decode.jsonl` | 新种子训练/解码曲线（已提交） |
| checkpoint（`latest.pt`） | 留远端未提交 |

命令（与任务单一致）：
```bash
python smoke_v2.py --device cuda --bf16
python run_swap_pairs.py > logs/v2_swap_followup/new_pairs.json
python run_swap_pairs.py --summary-only --seeds 20261123 20261124 20261125 > logs/v2_swap_followup/three_pairs.json
# 解码剖析用 20261123 checkpoint（single/two/two_no_swap）
```
