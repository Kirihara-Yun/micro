"""三种变体的训练：校准模式 + 正式训练（验证 NLL + checkpoint）。

用法：
  # 校准（200 步，测速度）
  python train.py --ffn_kind two --calibrate --calibrate_steps 300
  # 正式训练 50M tokens
  python train.py --ffn_kind two --total_tokens 50000000 --log_dir logs/two
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import ModelConfig, ComposedTransformer

CONTEXT = 512


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffn_kind", required=True, choices=["moe", "single", "two"])
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calibrate_steps", type=int, default=300)
    ap.add_argument("--total_tokens", type=int, default=50_000_000)
    ap.add_argument("--micro_batch", type=int, default=8)      # 序列数
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--eval_every_tokens", type=int, default=5_000_000)
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--log_dir", type=str, default="logs")
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--dtype", type=str, default="bf16")
    ap.add_argument("--resume_ckpt", type=str, default=None)
    return ap.parse_args()


def get_batch(data: np.ndarray, rng: np.random.Generator, batch_size: int, device):
    idx = rng.integers(0, len(data), size=batch_size)
    x = torch.from_numpy(data[idx].astype(np.int64)).to(device)
    return x


@torch.no_grad()
def eval_nll(model, data, n_batches, device, dtype):
    model.eval()
    rng = np.random.default_rng(0)
    total_nll = 0.0
    total_tokens = 0
    for _ in range(n_batches):
        x = get_batch(data, rng, 8, device)          # [B, S]
        logits = model(x)                             # [B, S, V]
        shift_logits = logits[:, :-1].float()
        shift_tgt = x[:, 1:]
        nll = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                              shift_tgt.reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_tokens += shift_tgt.numel()
    model.train()
    return total_nll / total_tokens


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    cfg = ModelConfig(seed=args.seed)
    model = ComposedTransformer(cfg, args.ffn_kind).to(device)
    n_params = model.num_params()
    active = model.ffn_active_params_per_layer()
    print(f"[{args.ffn_kind}] 总参数={n_params/1e6:.2f}M, 每层活跃 MACs≈{active}", flush=True)

    resume_tokens = 0
    if args.resume_ckpt:
        ck = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        resume_tokens = ck.get("tokens", 0)
        print(f"[{args.ffn_kind}] resume from {args.resume_ckpt} at {resume_tokens/1e6:.1f}M tokens",
              flush=True)

    train_data = np.load(os.path.join(args.data_dir, "train.npy"), mmap_mode="r")
    val_data = np.load(os.path.join(args.data_dir, "val.npy"), mmap_mode="r")
    print(f"[{args.ffn_kind}] train {train_data.shape} val {val_data.shape}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))
    batch_tokens = args.micro_batch * args.grad_accum * CONTEXT
    remaining = max(0, args.total_tokens - resume_tokens)
    steps = max(1, remaining // batch_tokens)
    if args.calibrate:
        steps = args.calibrate_steps

    def lr_at(step):
        if resume_tokens > 0:
            # resume 时用低 lr 继续（训练后期）
            return args.lr * args.min_lr_ratio
        if step < args.warmup_steps:
            return args.lr * step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, steps - args.warmup_steps)
        return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * prog)))

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("step,tokens,loss,lr,step_ms,tokens_per_s,val_nll,wall_hours,gpu_mem_mb\n")

    seen_tokens = resume_tokens
    next_eval_tokens = (resume_tokens // args.eval_every_tokens + 1) * args.eval_every_tokens
    t_wall_start = time.perf_counter()
    t0 = time.perf_counter()

    model.train()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)

        accum_loss = 0.0
        for _ in range(args.grad_accum):
            x = get_batch(train_data, rng, args.micro_batch, device)
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(x)
                shift_logits = logits[:, :-1]
                shift_tgt = x[:, 1:]
                loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                                       shift_tgt.reshape(-1))
                loss = loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        tokens = args.micro_batch * args.grad_accum * CONTEXT
        seen_tokens += tokens
        step_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        tokens_per_s = tokens / (step_ms / 1000) if step_ms > 0 else 0

        # 验证
        val_nll = ""
        if (not args.calibrate) and seen_tokens >= next_eval_tokens:
            val_nll = eval_nll(model, val_data, 20, device, dtype)
            next_eval_tokens += args.eval_every_tokens
            ckpt = os.path.join(args.log_dir, f"ckpt_{seen_tokens//1_000_000}M.pt")
            torch.save({"model": model.state_dict(), "step": step,
                        "tokens": seen_tokens, "val_nll": val_nll}, ckpt)

        gpu_mem = torch.cuda.max_memory_allocated() // (1024 * 1024) if device == "cuda" else 0
        wall_h = (time.perf_counter() - t_wall_start) / 3600
        with open(log_path, "a") as f:
            f.write(f"{step},{seen_tokens},{accum_loss:.4f},{lr_at(step):.2e},"
                    f"{step_ms:.1f},{tokens_per_s:.0f},{val_nll},{wall_h:.3f},{gpu_mem}\n")
        if step % 10 == 0 or step == 1:
            print(f"[{args.ffn_kind}] step {step}/{steps} tokens {seen_tokens/1e6:.1f}M "
                  f"loss {accum_loss:.4f} {tokens_per_s:.0f} tok/s step {step_ms:.0f}ms "
                  f"mem {gpu_mem}MB val_nll {val_nll}", flush=True)

    # 保存最终 checkpoint（若最后一个验证点未触发）
    final_ckpt = os.path.join(args.log_dir, "ckpt_final.pt")
    torch.save({"model": model.state_dict(), "step": steps,
                "tokens": seen_tokens}, final_ckpt)

    print(f"[{args.ffn_kind}] done. 最终 tokens {seen_tokens/1e6:.2f}M, "
          f"墙钟 {((time.perf_counter()-t_wall_start)/3600):.2f}h", flush=True)
    if args.calibrate:
        # 汇总校准结果
        import pandas as pd
        df = pd.read_csv(log_path)
        print(f"[{args.ffn_kind}] 校准汇总: 稳定步时 {df['step_ms'].iloc[-50:].mean():.0f}ms, "
              f"tokens/s {df['tokens_per_s'].iloc[-50:].mean():.0f}, 显存峰值 {df['gpu_mem_mb'].max()}MB")


if __name__ == "__main__":
    main()
