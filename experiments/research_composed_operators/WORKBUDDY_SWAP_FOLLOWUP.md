# 跨组交换复验与解码剖析

使用原 A100、原 BF16 环境和 [V2 数据](REPORT_V2.md)。先完成两个新种子的配对训练，再剖析单阶段和双阶段的全驻留 KV cache 解码。种子 20261123 的既有数据参与汇总。

## 1. 冒烟与配对复验

在仓库根目录执行以下命令。GPU 冒烟失败时回传输出。配对脚本在训练前核对两个数据文件的完整 SHA256，默认顺序执行四轮训练，每轮 1529 步、50,004,416 个预测目标。

```bash
set -e
cd experiments/research_composed_operators
mkdir -p logs/v2_swap_followup
python smoke_v2.py --device cuda --bf16
python run_swap_pairs.py > logs/v2_swap_followup/new_pairs.json
python run_swap_pairs.py --summary-only --seeds 20261123 20261124 20261125 \
  > logs/v2_swap_followup/three_pairs.json
```

脚本会复用同一配置下正常完成的训练目录。若有中断，先保留目录和日志。把该未完成变体目录移出原位置，再执行配对脚本，已完成的变体可直接复用。

报告逐种子的最终 NLL、配对差值（恒等减交换）、全部曲线和差值范围。三个种子提供重复性证据。原曲线 11 个评估点均为正，差值在 24.5M 达到 0.02649，50M 为 0.02580。用同一组配对结果描述交换效果。

## 2. 对齐解码测速和剖析

使用同一 GPU、同一批提示、BF16、batch=1。先跑常规延迟，再跑 profiler。剖析用历史 20261123 的 checkpoints，保证与之前报告的模型对应。

```bash
for variant in single two two_no_swap; do
  case "$variant" in
    single) ckpt=logs/v2_train_single/latest.pt ;;
    *) ckpt="logs/v2_swap_ablation_${variant}_20261123/latest.pt" ;;
  esac
  test -f "$ckpt"
  python decode_v2.py --checkpoint "$ckpt" --prompts data/val.npy --bf16 \
    > "logs/v2_swap_followup/${variant}_decode.jsonl"
  python profile_decode_v2.py --checkpoint "$ckpt" --prompts data/val.npy \
    --bf16 --requests 1 --trace "logs/v2_swap_followup/${variant}_trace.json" \
    > "logs/v2_swap_followup/${variant}_profile.json"
done
```

常规测速用 8 个请求、每请求生成 64 token、3 轮。剖析用其中第一个请求的 63 个增量步，prefill 在采集区间外。汇报常规 P50/P95，并列出各 profiler 类别每 token 的主机独占时间和归因到 CPU 算子的 CUDA kernel 时间，附主要算子调用次数。Profiler 有采集和打标开销，两个时间口径分别使用。同步等待、CPU 提交和 GPU kernel 可重叠，避免将各项简单相加解释成 P50。

回传实验报告、新配对的 metrics/decode JSONL、三种子的汇总，以及三组常规测速和剖析汇总。完整 Chrome trace、数据和 checkpoints 保留在 A100 机器上。
