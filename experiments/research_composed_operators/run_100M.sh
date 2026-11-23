#!/usr/bin/env bash
# 从 45M checkpoint resume 到 100M tokens。
set -euo pipefail
cd "$(dirname "$0")"
export HF_ENDPOINT=https://hf-mirror.com
PY=/home/qinzy/.conda/envs/rm-r1-1/bin/python

for kind in moe single two; do
  echo "===== 扩展 $kind 到 100M ====="
  $PY train.py --ffn_kind "$kind" --total_tokens 100000000 \
    --resume_ckpt "logs/train_${kind}_50000000/ckpt_45M.pt" \
    --log_dir "logs/train_${kind}_100000000" \
    2>&1 | tee "logs/train_${kind}_100000000.console.log"
  echo "===== $kind 100M 完成 ====="
done
echo "ALL 100M DONE"
