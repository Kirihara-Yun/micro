# 双阶段通道交换消融

基线报告见 [REPORT_V2.md](REPORT_V2.md)。首轮的 50,004,416 个预测目标上，双阶段比单阶段低 0.03035 NLL。该比较同时改变阶段数、参数和通道布局。此轮只比较 `two` 与 `two_no_swap`。两者初始权重逐元素一致，均执行同样的 `index_select`，无交换对照使用恒等索引。此次隔离跨组连接对质量的影响，延迟用于记录同样操作路径的测量。

在同一 A100、PyTorch 2.5.1+cu124 和原数据文件上运行。先把输出的两个 SHA256 与原报告 §0 核对。

```bash
set -e
cd experiments/research_composed_operators
sha256sum data/train.npy data/val.npy
python smoke_v2.py --device cuda --bf16

for kind in two two_no_swap; do
  out="logs/v2_swap_ablation_${kind}_20261123"
  python train_v2.py --kind "$kind" --data data --out "$out" \
    --steps 1529 --warmup 150 --eval-every 150 --batch 8 --accum 8 --seed 20261123 --bf16
  python decode_v2.py --checkpoint "$out/latest.pt" --prompts data/val.npy \
    --prompt-len 128 --new-tokens 64 --requests 8 --rounds 3 --bf16 > "$out/decode.jsonl"
done
```

两组均从头启动。若只够跑一组，先跑 `two_no_swap`，引用历史 `two` 结果并标明未做同轮复测。冒烟失败则回传完整输出，由本地修复。新一轮 `two` 与历史 `two` 的差值可用于检查运行间波动。

回传同一步数的完整评估序列和最终差值 `NLL(two_no_swap) - NLL(two)`，正值代表这一轮交换有利。另列实际训练吞吐、KV cache P50/P95、数据哈希、启动命令和环境，提交报告及 JSONL。checkpoint 留远端。一个配对种子用于筛选，正式结论需多种子。
