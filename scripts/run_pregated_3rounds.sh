#!/usr/bin/env bash
# base-64 三方法 3 轮独立重复稳定性测试（GPU-only / Pre-gated / DeepSpeed / SE-MoE）。
# 口径与 switch-base-8 报告一致：warmup 5 + iterations 30（SE-MoE iterations=1，全量重载太慢）。
# 用法: bash run_pregated_3rounds.sh <model> [rounds]
set -euo pipefail

MODEL="${1:-switch-base-64}"
ROUNDS="${2:-3}"
FT_HOST=/home/qinzy/baselines/Pregated_MoE
LOGS_DIR="$FT_HOST/logs"
CONTAINER=pregated
WARMUP=5
ITER=30

mkdir -p "$LOGS_DIR"

declare -A ENC=( [GPU-only]=0 [Pre-gated]=1 [DeepSpeed]=1 [SE-MoE]=1 )
declare -A DEC=( [GPU-only]=0 [Pre-gated]=2 [DeepSpeed]=1 [SE-MoE]=2 )

for m in GPU-only Pre-gated DeepSpeed SE-MoE; do
  fetch_all=0; [ "$m" = "SE-MoE" ] && fetch_all=1
  this_iter="$ITER"; [ "$m" = "SE-MoE" ] && this_iter=1

  cat > "$FT_HOST/cpp_config.ini" <<EOF
[default]
arena_size = 21474836480
encoder_fetcher_mode = ${ENC[$m]}
decoder_fetcher_mode = ${DEC[$m]}
profiling = 1
detailed_timing = 0
offload_path = /data/ft/$MODEL/
disk_offload = 0
load_from_cpp = 1
use_cache = 0
quant_mode = 0
vocab_size = 32128
fetch_all = $fetch_all
forced_num_experts = 0
cache_policy = LFU
EOF

  for r in $(seq 1 "$ROUNDS"); do
    logfile="$LOGS_DIR/${MODEL}_${m}_1_0_0_0_round${r}.log"
    echo "=== [$MODEL / $m] round $r/$ROUNDS  encoder=${ENC[$m]} decoder=${DEC[$m]} fetch_all=$fetch_all iter=$this_iter ==="
    docker exec "$CONTAINER" bash -c \
      "cd /workspace/FasterTransformer/build && python /workspace/FasterTransformer/examples/pytorch/t5/perf_benchmark.py \
        --batch_size 1 --beam_width 4 --seq_len 256 --data_type fp32 \
        --test_time 3 --sampling_topk 1 --model_type Megatron-DeepSpeed \
        --ckpt_path /data/ft/$MODEL/ --model t5-base \
        --warmup_iterations $WARMUP --iterations $this_iter --duration 30" 2>&1 \
      | tee "$logfile"
    echo "=== [$MODEL / $m] round $r done ==="
  done
done

echo "ALL ROUNDS DONE"
