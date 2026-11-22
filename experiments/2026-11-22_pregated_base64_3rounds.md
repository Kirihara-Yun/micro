# Pre-Gated MoE switch-base-64 三方法 3 轮独立重复稳定性

> 日期：2026-11-22 17:29–18:15 (+0800)
> 口径：batch=1、seq_len=256、fp32、`--warmup_iterations 5 --iterations 30 --duration 30`
> （SE-MoE 因 `fetch_all=1` 每次全量重载 384 专家，`iterations=1`）。
> 跑法：`micro/scripts/run_pregated_3rounds.sh switch-base-64 3`（每方法独立 3 轮进程）。
> 源码：`ranggihwang/Pregated_MoE` @ `6b0a666`；容器 `pregated`（CUDA 11.8）。

## 结果（每方法 3 轮独立重复）

| 方法 | BLK (ms) r1/r2/r3 | 吞吐 (tok/s) r1/r2/r3 | 峰值显存 | max_active | cache_hit |
|---|---|---|---|---|---|
| GPU-only | 0.917 / 0.916 / 0.915 | 151 / 152 / 152 | 16.06 GB | 0 | 0 |
| Pre-gated | 1.088 / 1.089 / 1.089 | 120 / 120 / 120 | 26.06 GB | 1 | 1.0 |
| DeepSpeed | 1.755 / 1.758 / 1.755 | 81 / 81 / 81 | 26.06 GB | 1 | 1.0 |
| SE-MoE | 50.53 / 50.53 / 50.54 | 3 / 3 / 3 | 26.06 GB | 1 | 1.0 |

## 轮间波动

| 方法 | BLK mean ± std (ms) | 吞吐 mean ± std (tok/s) |
|---|---|---|
| GPU-only | 0.9161 ± 0.0008（CV 0.09%） | 151.7 ± 0.5（CV 0.3%） |
| Pre-gated | 1.0887 ± 0.0003（CV 0.03%） | 120.0 ± 0.0（CV 0%） |
| DeepSpeed | 1.7559 ± 0.0013（CV 0.07%） | 81.0 ± 0.0（CV 0%） |
| SE-MoE | 50.536 ± 0.004（CV 0.008%） | 3.0 ± 0.0（CV 0%） |

## 结论

1. **重复性极好**：四方法 3 轮独立重复的吞吐完全一致（Pre-gated/DeepSpeed/SE-MoE 为 0 波动），
   BLK 变异系数 < 0.1%，与早期 4-iter 冒烟值（GPU-only 151 / Pre-gated 120 / DeepSpeed 81 / SE-MoE 3）一致。
2. **与 base-8 的相对结论一致**：Pre-gated 相对 GPU-only 吞吐 −21%（120/151.7），显存 +10GB
   （26.06 vs 16.06，20GB arena 部分被 64 专家权重抵消）。
3. 显存：GPU-only 16.06GB（随专家数增长，base-8 为 3.37GB）；Pre-gated/DeepSpeed/SE-MoE 26.06GB
   （20GB arena + 模型），几乎不随配置缩放。

## 原始证据

- 日志：`baselines/Pregated_MoE/logs/switch-base-64_{GPU-only,Pre-gated,DeepSpeed,SE-MoE}_1_0_0_0_round{1,2,3}.log`（12 份，本机，未提交）。
