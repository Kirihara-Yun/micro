#!/usr/bin/env python3
"""正确解析 Pregated_MoE profiling log，产出四方法结果 CSV。

修复原 `eval_all.py` 解析口径（见 experiments/2026-11-22_pregated_profiling_mechanism.md §5）：

- block_lat       : 取最后一次 report 的 `BLK AVG`（正式迭代稳定值）
- throughput      : `tokens/sec`
- peak_mem        : `MEM usage` 行所有采样中的峰值字节（GPU 已用显存），
                    不减去 arena / 不做 buffer 推导（原公式在 GPU-only 下得负值）
- max_active_experts : `Max active experts`
- cache_hit_rate  : `Average cache hit rate`

用法:
    python3 parse_pregated_logs.py <logs_dir> <model> [output_csv]
"""
import csv
import re
import sys
from pathlib import Path


def parse_log(path: Path):
    lines = path.read_text().splitlines()

    def last_match(pattern):
        for line in reversed(lines):
            m = re.search(pattern, line)
            if m:
                return m
        return None

    block_lat = None
    m = last_match(r"BLK AVG: ([\d\.]+) ms")
    if m:
        block_lat = float(m.group(1))

    throughput = None
    m = last_match(r", (\d+) tokens/sec\.")
    if m:
        throughput = int(m.group(1))

    peak_mem = None
    for line in lines:
        m = re.search(r"MEM usage: (.+)$", line)
        if m:
            vals = [int(x) for x in m.group(1).split()]
            if vals:
                peak_mem = max(vals)

    max_active = None
    m = last_match(r"Max active experts: (\d+)")
    if m:
        max_active = int(m.group(1))

    cache_hit = None
    m = last_match(r"Average cache hit rate: ([\d\.]+)")
    if m:
        cache_hit = float(m.group(1))

    return {
        "block_lat_ms": block_lat,
        "throughput_tok_s": throughput,
        "peak_mem_bytes": peak_mem,
        "max_active_experts": max_active,
        "cache_hit_rate": cache_hit,
    }


def main():
    logs_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    model = sys.argv[2] if len(sys.argv) > 2 else "switch-base-8"
    out_csv = sys.argv[3] if len(sys.argv) > 3 else None

    methods = ["GPU-only", "Pre-gated", "DeepSpeed", "SE-MoE"]
    rows = []
    for method in methods:
        p = logs_dir / f"{model}_{method}_1_0_0_0.log"
        if not p.exists():
            print(f"[skip] missing {p}", file=sys.stderr)
            continue
        r = parse_log(p)
        r["method"] = method
        rows.append(r)
        print(
            f"{method:10s} BLK={r['block_lat_ms']} ms  "
            f"throughput={r['throughput_tok_s']} tok/s  "
            f"peak_mem={r['peak_mem_bytes']} B "
            f"({r['peak_mem_bytes']/1e9:.2f} GB)  "
            f"max_active={r['max_active_experts']}  "
            f"cache_hit={r['cache_hit_rate']}"
        )

    if not out_csv:
        return

    fields = ["method", "block_lat_ms", "throughput_tok_s", "peak_mem_bytes",
              "max_active_experts", "cache_hit_rate"]
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print(f"\n[written] {out_csv}")


if __name__ == "__main__":
    main()
