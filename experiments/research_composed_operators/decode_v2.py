"""Measure single-request generation from trained v2 checkpoints."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from model import ModelConfig
from model_v2 import CausalTransformer, KVCache


@torch.inference_mode()
def generate(model, prompt, count):
    if count < 1:
        raise ValueError("Generation length must be positive")
    cache = KVCache(prompt.size(1) + count - 1)
    device = prompt.device

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    synchronize()
    began = time.perf_counter()
    logits = model(prompt, cache)
    token = logits[:, -1:].argmax(dim=-1)
    synchronize()
    ttft = time.perf_counter() - began
    durations = []
    for _ in range(count - 1):
        began = time.perf_counter()
        logits = model(token, cache)
        token = logits[:, -1:].argmax(dim=-1)
        synchronize()
        durations.append(time.perf_counter() - began)
    return ttft, durations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path, help="2D .npy of fixed token sequences")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    if min(args.prompt_len, args.requests, args.rounds) < 1 or args.new_tokens < 2:
        parser.error("Positive request sizes and at least two generated tokens are required")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    settings = state["settings"]
    cfg = ModelConfig(**settings["config"])
    model = CausalTransformer(cfg, settings["kind"])
    model.load_state_dict(state["model"])
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    data = np.load(args.prompts, mmap_mode="r")
    if data.ndim != 2 or data.shape[0] < args.requests or data.shape[1] < args.prompt_len:
        parser.error("Prompt file does not contain enough rows or tokens")
    if args.prompt_len + args.new_tokens - 1 > cfg.context:
        parser.error("Prompt and generation exceed the checkpoint context")
    prompts = torch.tensor(data[:args.requests, :args.prompt_len].astype(np.int64), device=device)
    generate(model, prompts[:1], args.new_tokens)
    for round_index in range(args.rounds):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        prefill, decode = [], []
        for prompt in prompts:
            ttft, timings = generate(model, prompt.unsqueeze(0), args.new_tokens)
            prefill.append(ttft)
            decode.extend(timings)
        metrics = dict(kind=settings["kind"], round=round_index + 1, device=str(device),
                       dtype=str(dtype), requests=len(prompts), prompt_tokens=args.prompt_len,
                       decode_tokens=len(decode), ttft_p50_ms=float(np.median(prefill) * 1000),
                       p50_ms=float(np.median(decode) * 1000), p95_ms=float(np.percentile(decode, 95) * 1000),
                       decode_tokens_per_s=len(decode) / sum(decode))
        if device.type == "cuda":
            metrics["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
