"""Attribute incremental decode work without instrumenting the benchmark path."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from decode_v2 import add_decode_arguments, generate, load_decode_inputs
from model_v2 import KVCache


def tagged(name, forward):
    def run(*args, **kwargs):
        with record_function(name):
            return forward(*args, **kwargs)

    return run


@torch.inference_mode()
def collect(model, prompts, count):
    cuda = prompts.device.type == "cuda"
    activities = [ProfilerActivity.CPU]
    if cuda:
        activities.append(ProfilerActivity.CUDA)
    prepared = []
    for prompt in prompts:
        cache = KVCache(prompt.numel() + count - 1)
        token = model(prompt.unsqueeze(0), cache)[:, -1:].argmax(dim=-1)
        prepared.append((cache, token))
    if cuda:
        torch.cuda.synchronize(prompts.device)

    with ExitStack() as scopes:
        for layer in model.layers:
            ffn = layer.ffn
            scopes.enter_context(patch.object(ffn, "forward", tagged("decode.ffn", ffn.forward)))
            for name in ("w_gate", "w_up", "w_out", "router", "router1", "router2"):
                linear = getattr(ffn, name, None)
                if linear is not None:
                    label = "decode.router_linear" if name.startswith("router") else "decode.ffn_projection"
                    scopes.enter_context(patch.object(linear, "forward", tagged(label, linear.forward)))
        with profile(activities=activities) as trace:
            for cache, token in prepared:
                for _ in range(count - 1):
                    with record_function("decode.token"):
                        logits = model(token, cache)
                        token = logits[:, -1:].argmax(dim=-1)
                        if cuda:
                            torch.cuda.synchronize(prompts.device)
    return trace


FFN_SCOPES = {
    "decode.router_linear": "routing",
    "decode.ffn_projection": "shared_ffn_projections",
    "aten::softmax": "routing",
    "aten::argmax": "routing",
    "aten::div": "routing",
    "aten::gather": "routing",
    "aten::index": "block_selection",
    "aten::arange": "block_selection",
    "aten::einsum": "block_einsum",
    "aten::index_select": "channel_exchange",
}


def classify(event):
    ancestors = []
    while event is not None:
        ancestors.append(event.name)
        event = event.cpu_parent
    if "decode.token" not in ancestors:
        return None
    if "decode.ffn" in ancestors:
        return next((FFN_SCOPES[name] for name in ancestors if name in FFN_SCOPES), "other_ffn")
    return "decode_synchronization" if "cudaDeviceSynchronize" in ancestors else "outside_ffn"


def summarize(trace, token_count, cuda):
    buckets = defaultdict(Counter)
    operators = defaultdict(Counter)
    for event in trace.events():
        if event.device_type != torch.autograd.DeviceType.CPU:
            continue
        category = classify(event)
        if category is None:
            continue
        totals = Counter(host_self_us=event.self_cpu_time_total,
                         cuda_kernel_us=event.self_device_time_total if cuda else 0.0)
        buckets[category].update(totals)
        if not event.name.startswith("decode."):
            operators[category, event.name].update(totals)
            operators[category, event.name]["calls"] += 1
    if cuda and not any(row["cuda_kernel_us"] > 0 for row in buckets.values()):
        raise RuntimeError("Profiler did not attribute any CUDA kernels to decode events")
    if not {"routing", "block_selection", "block_einsum"}.issubset(buckets):
        raise RuntimeError("Missing FFN events in decode trace")

    def per_token(totals):
        return {f"{key}_per_token": value / token_count for key, value in totals.items()}

    return {
        "buckets": {name: per_token(row) for name, row in sorted(buckets.items())},
        "operators": [
            dict(category=category, name=name, **per_token(row))
            for (category, name), row in sorted(operators.items())
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_decode_arguments(parser)
    parser.add_argument("--trace", type=Path, help="Optional Chrome trace path")
    args = parser.parse_args()
    model, prompts, kind = load_decode_inputs(args, parser)
    if kind not in ("single", "two", "two_no_swap"):
        parser.error("FFN breakdown supports single, two, and two_no_swap")
    generate(model, prompts[:1], args.new_tokens)
    trace = collect(model, prompts, args.new_tokens)
    tokens = args.requests * (args.new_tokens - 1)
    result = dict(kind=kind, device=str(prompts.device), dtype=str(next(model.parameters()).dtype),
                  requests=args.requests, prompt_tokens=args.prompt_len, decode_tokens=tokens,
                  **summarize(trace, tokens, prompts.device.type == "cuda"))
    if args.trace is not None:
        trace.export_chrome_trace(str(args.trace))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
