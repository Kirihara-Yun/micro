"""Train a causal model with restartable optimizer and data state."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import ModelConfig
from model_v2 import CausalTransformer


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def batch_at(data, rng, size, device):
    rows = rng.integers(len(data), size=size)
    return torch.from_numpy(data[rows].astype(np.int64)).to(device)


def loss_for(model, batch):
    return F.cross_entropy(model(batch)[:, :-1].transpose(1, 2).float(), batch[:, 1:])


def lr_at(step, total, warmup, lr):
    if step <= warmup:
        return lr * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def run(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train_path = args.data / "train.npy"
    val_path = args.data / "val.npy"
    train = np.load(train_path, mmap_mode="r")
    val = np.load(val_path, mmap_mode="r")
    if train.ndim != 2 or val.ndim != 2 or train.shape[1] != val.shape[1] or train.shape[1] < 2:
        raise ValueError("train.npy and val.npy need the same sequence length of at least two")
    cfg = ModelConfig(context=train.shape[1], seed=args.seed)
    if args.tiny:
        cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4,
                          context=train.shape[1], n_groups=4, group_dim=8,
                          n_candidates=4, n_experts=4, moe_intermediate=32, seed=args.seed)
    model = CausalTransformer(cfg, args.kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    precision = torch.bfloat16 if args.bf16 else torch.float32
    settings = {
        "kind": args.kind, "config": asdict(cfg), "steps": args.steps,
        "warmup": args.warmup, "lr": args.lr, "batch": args.batch, "accum": args.accum,
        "bf16": args.bf16, "device": str(device),
        "data": [str(p.resolve()) for p in (train_path, val_path)],
    }
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["settings"] != settings:
            raise ValueError("Resume requires identical model, schedule, precision, and data paths")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["data_rng"]
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        start = state["step"]
    end = args.steps if args.stop_after is None else args.stop_after
    if not start < end <= args.steps:
        raise ValueError("The stop step must exceed the checkpoint step and fit the schedule")
    args.out.mkdir(parents=True, exist_ok=True)
    targets_per_step = args.batch * args.accum * (cfg.context - 1)
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    with (args.out / "metrics.jsonl").open("a" if args.resume else "w", encoding="utf-8") as log:
        for step in range(start + 1, end + 1):
            model.train()
            synchronize()
            began = time.perf_counter()
            lr = lr_at(step, args.steps, args.warmup, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(args.accum):
                batch = batch_at(train, rng, args.batch, device)
                with torch.autocast(device.type, dtype=precision, enabled=args.bf16):
                    loss = loss_for(model, batch)
                (loss / args.accum).backward()
                losses.append(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            synchronize()
            seconds = time.perf_counter() - began
            metrics = dict(step=step, target_tokens=step * targets_per_step, lr=lr,
                           loss=torch.stack(losses).mean().item(), tokens_per_s=targets_per_step / seconds)
            if step % args.eval_every == 0 or step == end:
                model.eval()
                val_rng = np.random.default_rng(0)
                with torch.no_grad(), torch.autocast(device.type, dtype=precision, enabled=args.bf16):
                    valid = [loss_for(model, batch_at(val, val_rng, args.batch, device))
                             for _ in range(args.eval_batches)]
                    metrics["val_nll"] = torch.stack(valid).mean().item()
                state = dict(settings=settings, step=step, model=model.state_dict(),
                             optimizer=optimizer.state_dict(), data_rng=rng.bit_generator.state,
                             torch_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
                save_checkpoint(args.out / "latest.pt", state)
            if device.type == "cuda":
                metrics["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            log.write(json.dumps(metrics) + "\n")
            log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("moe", "single", "two"))
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tiny", action="store_true", help="Use 64-token vocabulary for smoke checks")
    args = parser.parse_args()
    if min(args.steps, args.batch, args.accum, args.eval_every, args.eval_batches) < 1 or args.warmup < 0:
        parser.error("Steps, batch, accumulation and evaluation sizes must be positive")
    run(args)


if __name__ == "__main__":
    main()
