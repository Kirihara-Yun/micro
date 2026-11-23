"""三种变体 BF16 decode 测量：batch=1 自回归，记录显存、P50/P95 token 延迟、吞吐、理论活跃权重字节。

用法：
  python decode_bench.py --ffn_kind two --ckpt logs/two/ckpt_50M.pt
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import ModelConfig, ComposedTransformer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffn_kind", required=True, choices=["moe", "single", "two"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_gen", type=int, default=64)
    ap.add_argument("--prompt_len", type=int, default=16)
    return ap.parse_args()


def active_weight_bytes(kind, cfg):
    """每层每 token 理论活跃权重字节（BF16 = 2 bytes/参数）。"""
    d, gd = cfg.d_model, cfg.group_dim
    if kind == "moe":
        # 1 个专家：gate + up + down
        return 3 * d * cfg.moe_intermediate * 2
    elif kind == "single":
        # 共享投影 3×d×d + 1 候选 2×gd×gd
        return (3 * d * d + 2 * gd * gd) * 2
    else:  # two
        return (3 * d * d + 4 * gd * gd) * 2


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig()
    model = ComposedTransformer(cfg, args.ffn_kind).to(device).to(torch.bfloat16)
    sd = torch.load(args.ckpt, map_location=device)["model"]
    model.load_state_dict(sd)
    model.eval()

    # 预热
    x = torch.randint(0, cfg.vocab_size, (1, args.prompt_len), device=device)
    with torch.no_grad():
        for _ in range(3):
            model(x)

    torch.cuda.reset_peak_memory_stats()
    # 自回归生成，记录每 token 延迟
    ids = torch.randint(0, cfg.vocab_size, (1, args.prompt_len), device=device)
    lat = []
    with torch.no_grad():
        for i in range(args.n_gen):
            t0 = time.perf_counter()
            logits = model(ids)
            next_tok = torch.argmax(logits[0, -1])
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
            ids = torch.cat([ids, next_tok.reshape(1, 1)], dim=1)

    lat = np.array(lat) * 1000  # ms
    p50, p95 = np.percentile(lat, 50), np.percentile(lat, 95)
    mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    thr = 1000 / p50 if p50 > 0 else 0
    ab = active_weight_bytes(args.ffn_kind, cfg)

    print(f"[{args.ffn_kind}] decode 结果:")
    print(f"  显存峰值: {mem:.2f} GB")
    print(f"  token 延迟: P50={p50:.1f}ms P95={p95:.1f}ms mean={lat.mean():.1f}ms")
    print(f"  吞吐: {thr:.1f} tok/s")
    print(f"  每层每 token 理论活跃权重字节: {ab/1024:.1f} KB")

    # 写 CSV 一行
    with open("decode_results.csv", "a") as f:
        if f.tell() == 0:
            f.write("ffn_kind,mem_gb,p50_ms,p95_ms,mean_ms,tok_per_s,active_weight_bytes\n")
        f.write(f"{args.ffn_kind},{mem:.2f},{p50:.1f},{p95:.1f},{lat.mean():.1f},{thr:.1f},{ab}\n")


if __name__ == "__main__":
    main()
