#!/usr/bin/env bash
# 串行训练三变体各 50M tokens（预算允许再扩 100M）。
set -euo pipefail
cd "$(dirname "$0")"
export HF_ENDPOINT=https://hf-mirror.com
PY=/home/qinzy/.conda/envs/rm-r1-1/bin/python
TOTAL="${1:-50000000}"

for kind in moe single two; do
  echo "===== 训练 $kind ($TOTAL tokens) ====="
  $PY train.py --ffn_kind "$kind" --total_tokens "$TOTAL" \
    --log_dir "logs/train_${kind}_${TOTAL}" \
    2>&1 | tee "logs/train_${kind}_${TOTAL}.console.log"
  echo "===== $kind 完成 ====="
done
echo "ALL TRAINING DONE"
