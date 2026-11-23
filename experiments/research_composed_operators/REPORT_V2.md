# 新版因果训练 + KV cache 解码（v2）运行结果

> 日期：2026-11-23。按 `WORKBUDDY_V2.md` 执行：A100 BF16 冒烟 → 校准 → 正式训练（50M tokens）→ KV cache 解码测量。
> 使用修正后的 `model_v2.py`（因果 mask + 位置编码 + 真实 top-1 + 近单位初始化 + KV cache）。
> **结论先行：冒烟全部通过。修正后质量顺序反转——双阶段（two）验证 NLL 最好（5.168），
> 领先单阶段 0.030、标准 MoE 0.062；但双阶段 decode 最慢（4.58ms vs single 3.42ms）。**
> 双阶段的质量优势需等预算无交换消融才能归因于跨组交换（见 WORKBUDDY_V2 第 35 行）。

## 0. 环境与数据

| 项 | 值 |
|---|---|
| GPU | 1× NVIDIA A100 80GB PCIe |
| PyTorch | 2.5.1+cu124（conda `rm-r1-1`） |
| 代码版本 | `bec7f7b`（model_v2.py / train_v2.py / decode_v2.py / smoke_v2.py） |
| 数据 | FineWeb-Edu `sample-10BT` 前 2 分片，GPT-2 tokenizer（vocab 50257），packing 到 512 |

### data 文件 SHA256
```
1960dfe9dedd11709434bcc99c076fef300df32ff1fd15c4a3561e56047865e6  data/train.npy  (228339840 bytes)
7380e73206904faa0c9fb8f242bb9b298c770c7445c0fd4593dc5fb679c66a9b  data/val.npy    (11663488 bytes)
```

## 1. 冒烟（`python smoke_v2.py --device cuda --bf16`）

三种变体 **全部通过**，输出：
```
moe: resume, causal prefix and incremental decode checks passed
single: resume, causal prefix and incremental decode checks passed
two: resume, causal prefix and incremental decode checks passed
```

验证项：
- 断点续跑（`--stop-after 2` 后 `--resume`）与连续训练权重逐张量一致（CUDA 容差 1e-6）✅
- 修改后缀（`ids[:, 4:] += 12`）不影响前缀 logits（因果 mask 生效）✅
- 分段 KV cache（(0,3),(3,4),(4,7),(7,8)）与完整前向 logits 在 BF16 容差（atol 3e-3/rtol 1e-2）内一致 ✅
- MoE 未选专家隔离（改 `down[1]` 输出不变）+ batch=1 路径一致性 ✅

## 2. 校准（各 200 步，warmup 50，batch 8×accum 8）

| 变体 | tokens/s | 显存峰值 | 200 步 val_nll |
|---|---|---|---|
| moe | 22732 | 3.75 GB | 6.718 |
| single | 23686 | 4.34 GB | 6.637 |
| two | 20130 | 5.35 GB | 6.599 |

## 3. 正式训练（1529 步 ≈ 50.0M tokens，warmup 150，eval 每 150 步）

验证 NLL 曲线（target_tokens 为有效预测目标数）：

| tokens | moe | single | two |
|---|---|---|---|
| 4.9M | 6.9011 | 6.7701 | **6.7326** |
| 9.8M | 6.3446 | 6.2607 | **6.2397** |
| 14.7M | 6.0066 | 5.9500 | **5.9275** |
| 19.6M | 5.7715 | 5.7218 | **5.6882** |
| 24.5M | 5.5999 | 5.5444 | **5.5196** |
| 29.4M | 5.4656 | 5.4180 | **5.3911** |
| 34.3M | 5.3771 | 5.3333 | **5.3059** |
| 39.2M | 5.3065 | 5.2676 | **5.2377** |
| 44.2M | 5.2595 | 5.2268 | **5.1964** |
| 49.1M | 5.2334 | 5.2017 | **5.1729** |
| 50.0M | 5.2300 | 5.1987 | **5.1684** |

最终步指标：

| 变体 | 总参数 | 训练 loss | val_nll | tokens/s | 显存峰值 |
|---|---|---|---|---|---|
| moe | 44.10M | 5.0734 | 5.2300 | 21869 | 3.76 GB |
| single | 27.57M | 5.0421 | 5.1987 | 23473 | 4.34 GB |
| two | 29.63M | **5.0180** | **5.1684** | 20331 | 5.35 GB |

## 4. decode（KV cache，BF16，batch=1，8 请求 × 64 新 token × 3 轮）

| 变体 | TTFT P50 | decode P50 | P95 | 吞吐 | 显存峰值 |
|---|---|---|---|---|---|
| moe | 8.48 ms | **2.98 ms** | 3.01 ms | **334 tok/s** | 113 MB |
| single | 3.95 ms | 3.42 ms | 3.44 ms | 292 tok/s | 80 MB |
| two | 5.26 ms | 4.58 ms | 4.60 ms | 218 tok/s | 84 MB |

## 5. 结论与判断

1. **代码正确性已通过冒烟**：续跑一致、因果前缀不变、增量解码 vs 完整因果前向一致、top-1 未选专家隔离，四项在 A100 BF16 下全过。
2. **质量**：双阶段（two）验证 NLL 全程最好，最终 5.168，领先单阶段 0.030、标准 MoE 0.062；且用 29.6M 参数（比标准 MoE 44.1M 少 33%）达到最好 NLL。单阶段（27.6M）次之，标准 MoE 最差。
3. **速度权衡**：双阶段 decode 最慢（4.58ms，比 single 慢 34%、比 moe 慢 54%），训练也慢（20.3K vs single 23.5K tok/s）。标准 MoE 逐专家 PyTorch 实现 decode 最快（2.98ms/334 tok/s），但注意它训练质量最差。
4. **上一版（83d4577）结论已反转**：旧版因无因果 mask、top-1 路由错误、KV cache 缺失、初始化 RMS 差异，得出"双阶段全面劣于单阶段"；修正后双阶段在质量上反转为最好。这印证了正确性前置项对结论的决定性影响。
5. **归因限制**：如任务单所述，双阶段的质量优势**尚不能归因于跨组交换**——需等预算的"无交换消融"（固定通道不交换的双阶段对照）才能判定。当前结果仅说明：在因果 + 正确 top-1 + 近单位初始化下，双阶段（含交换）质量占优，但以 decode/训练速度为代价。
6. **单 seed 仅筛选**：seed 20261123，结论为筛选性信号，不构成统计判定。

## 6. 交付物

| 文件 | 说明 |
|---|---|
| `logs/v2_train_{moe,single,two}/metrics.jsonl` | 训练/验证曲线（已提交） |
| `logs/v2_train_{moe,single,two}/decode.jsonl` | KV cache 解码测量（已提交） |
| `logs/v2_calib_{moe,single,two}/metrics.jsonl` | 校准（已提交） |
| `logs/v2_train_*/latest.pt` | checkpoint（531/333/358 MB，留远端未提交） |

命令（与任务单一致）：
```bash
python smoke_v2.py --device cuda --bf16
for kind in moe single two; do
  python train_v2.py --kind "$kind" --data data --out "logs/v2_calib_$kind" --steps 200 --warmup 50 --eval-every 100 --batch 8 --accum 8 --bf16 || break
done
for kind in moe single two; do
  python train_v2.py --kind "$kind" --data data --out "logs/v2_train_$kind" --steps 1529 --warmup 150 --eval-every 150 --batch 8 --accum 8 --bf16 || break
  python decode_v2.py --checkpoint "logs/v2_train_$kind/latest.pt" --prompts data/val.npy --prompt-len 128 --new-tokens 64 --requests 8 --rounds 3 --bf16 > "logs/v2_train_$kind/decode.jsonl" || break
done
```
