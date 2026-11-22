# Pre-Gated MoE Switch-Base-8 复现记录

## 实验信息

| 字段 | 值 |
|---|---|
| 时间 | 2026-11-22 11:00–11:15 (+0800) |
| host | `manager`（容器 `e2af132bfeea`） |
| 源码 SHA | `ranggihwang/Pregated_MoE` @ `6b0a666` |
| 模型 | `google/switch-base-8` @ revision `92fe2d22b024d9937146fe097ba3d3a7ba146e1b` |
| GPU | NVIDIA A100 80GB PCIe（compute_cap 8.0），驱动 560.35.03 |
| CPU | Intel Xeon Platinum 8360Y @ 2.40GHz，72 核，1 TiB 内存 |
| 容器 | `nvcr.io/nvidia/pytorch:22.09-py3`，CUDA 11.8，torch 1.13.0a0 |
| 权重 | fp32，`/data/ft/switch-base-8/`（转换产物 524 文件 / 4.4GB） |

## 实验配置

- 统一参数：`batch_size=1`、`seq_len=256`、`data_type=fp32`、`test_time=3`（ft-sampling）、`sampling_topk=1`、`model_type=Megatron-DeepSpeed`
- 热身：`warmup_iterations=5`
- 正式：`iterations=30`、`duration=30`（实际 30 次迭代）
- 两种方法：
  - GPU-only：`encoder_fetcher_mode=0, decoder_fetcher_mode=0`
  - Pre-gated：`encoder_fetcher_mode=1, decoder_fetcher_mode=2`
- 命令（示例）：
  ```bash
  cd /workspace/FasterTransformer/build
  python examples/pytorch/t5/perf_benchmark.py \
    --batch_size 1 --beam_width 4 --seq_len 256 --data_type fp32 \
    --test_time 3 --sampling_topk 1 --model_type Megatron-DeepSpeed \
    --ckpt_path /data/ft/switch-base-8/ --model t5-base \
    --warmup_iterations 5 --iterations 30 --duration 30
  ```

## 结果（各 35 次采样 = 5 热身 + 30 正式）

### GPU-only

| 指标 | median | min | max | mean | std |
|---|---|---|---|---|---|
| BLK (block latency) | 0.9112 ms | 0.9092 | 0.9184 | 0.9114 | 0.0019 |
| CMP (compute) | 0.7748 ms | 0.7741 | 0.7757 | 0.7749 | 0.0004 |
| NMB | 0.1390 ms | 0.1381 | 0.1455 | 0.1393 | 0.0012 |
| MEM (CPU→GPU) | 0 ms（无传输） | — | — | — | — |

- **吞吐：153 tokens/sec**
- **峰值显存：3.37 GB**
- Max active experts: 0（全部常驻 GPU）

### Pre-gated

| 指标 | median | min | max | mean | std |
|---|---|---|---|---|---|
| BLK (block latency) | 1.0799 ms | 1.0781 | 1.0867 | 1.0800 | 0.0019 |
| CMP (compute) | 0.7800 ms | 0.7793 | 0.7804 | 0.7800 | 0.0003 |
| NMB | 0.1472 ms | 0.1462 | 0.1526 | 0.1474 | 0.0014 |
| MEM (CPU→GPU) | 1.0273 ms | 1.0262 | 1.0374 | 1.0282 | 0.0023 |

- **吞吐：129 tokens/sec**
- **峰值显存：23.37 GB**（含 ~20GB arena）
- Max active experts: **1**（pre-gated 正确做到每次只 fetch 1 个激活专家）
- Average cache hit rate: 1

## 关键发现

1. **复现稳定**：两模式 35 次采样的 std 均 < 0.3%，吞吐与早期 4 次迭代冒烟值（GPU-only 154 / Pre-gated 130）一致。
2. **小模型上 Pre-gated 不占优**：switch-base-8（8 专家，fp32 仅 4.4GB）时，
   - 吞吐：Pre-gated 129 vs GPU-only 153（**-16%**，CPU→GPU 传输开销）
   - 显存：Pre-gated 23.37GB vs GPU-only 3.37GB（**+20GB**，arena 开销）
   - 这与 Pre-Gated MoE 论文的适用前提一致：其收益仅在模型大到 GPU 装不下时才体现，switch-base-8 太小，GPU 可全量驻留。
3. **Pre-gated 机制验证正确**：`Max active experts: 1`、`MEM AVG≈1.03ms`、`cache hit rate=1`，确认"提前确定下一层专家 + 只迁移被选专家"的行为符合论文描述。

## 异常与处理

- 首次 30 次迭代跑时，外部服务 **ComfyUI**（`serve_comfyui.py`，PID 846673）占用 46.6GB 显存 + 100% GPU，导致 Pre-gated 吞吐失真至 41 tok/s、峰值显存 73.4GB。
- 已征得用户同意暂停 ComfyUI 后重跑，本报告数据均为干净环境结果。

## 原始数据

- 原始日志：`/tmp/gpuonly_clean.log`、`/tmp/pregated_clean.log`（本机，未提交）
