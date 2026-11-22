#!/usr/bin/env bash
# 在容器 pregated 内跑 Pregated_MoE 四方法（GPU-only / Pre-gated / DeepSpeed / SE-MoE），
# log 回写到宿主机 <FT_HOST>/logs/，再用 scripts/parse_pregated_logs.py 解析。
# 用法: bash run_pregated_methods.sh <model> [iterations]
set -euo pipefail

MODEL="${1:-switch-base-64}"
ITER="${2:-4}"
FT_HOST=/home/qinzy/baselines/Pregated_MoE
LOGS_DIR="$FT_HOST/logs"
CONTAINER=pregated

mkdir -p "$LOGS_DIR"

declare -A ENC=( [GPU-only]=0 [Pre-gated]=1 [DeepSpeed]=1 [SE-MoE]=1 )
declare -A DEC=( [GPU-only]=0 [Pre-gated]=2 [DeepSpeed]=1 [SE-MoE]=2 )

for m in GPU-only Pre-gated DeepSpeed SE-MoE; do
  fetch_all=0; [ "$m" = "SE-MoE" ] && fetch_all=1
  iter="$ITER";  [ "$m" = "SE-MoE" ] && iter=1

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

  echo "=== [$MODEL / $m] encoder=${ENC[$m]} decoder=${DEC[$m]} fetch_all=$fetch_all iter=$iter ==="
  docker exec "$CONTAINER" bash -c \
    "cd /workspace/FasterTransformer/build && python /workspace/FasterTransformer/examples/pytorch/t5/perf_benchmark.py \
      --batch_size 1 --beam_width 4 --seq_len 256 --data_type fp32 \
      --test_time 3 --sampling_topk 1 --model_type Megatron-DeepSpeed \
      --ckpt_path /data/ft/$MODEL/ --model t5-base \
      --duration 0 --iterations $iter" 2>&1 \
    | tee "$LOGS_DIR/${MODEL}_${m}_1_0_0_0.log"
  echo "=== [$MODEL / $m] done ==="
done

echo "ALL DONE"
