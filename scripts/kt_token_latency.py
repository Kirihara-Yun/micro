#!/usr/bin/env python3
"""测量 Mixtral decode 每 token latency（stream 模式，精确时间戳）。

用法: python3 kt_token_latency.py [prompt_idx]
"""
import json
import time
import sys
import urllib.request

BASE = "http://127.0.0.1:30000"

# 6 个固定请求：代码、问答、叙述各 2（约 256 token 输入）
PROMPTS = {
    "code_1": "Write a Python function that computes the n-th Fibonacci number using an iterative approach with O(n) time and O(1) space. Explain each step. " + "The function should handle edge cases like n=0 and n=1 correctly. " * 3,
    "code_2": "Implement a function to reverse a linked list in place. Describe the pointer manipulation in detail and mention the time complexity. " + "Consider both iterative and recursive approaches and their trade-offs. " * 3,
    "qa_1": "What are the key differences between supervised learning and reinforcement learning? Explain with concrete examples from real-world applications. " + "Discuss when you would choose one over the other for a given problem. " * 3,
    "qa_2": "Explain how attention mechanisms work in transformer models. What is the role of queries, keys, and values in the attention computation? " + "Describe the mathematical formulation clearly. " * 3,
    "narr_1": "Describe the history of computing from the early mechanical calculators to modern quantum computers. Focus on the major milestones. " + "Highlight the people and inventions that shaped each era. " * 3,
    "narr_2": "Explain the process of photosynthesis in plants, from light absorption to glucose production. Describe the role of chlorophyll. " + "Mention the light-dependent and light-independent reactions. " * 3,
}


def gen_stream(prompt, max_tokens=128):
    payload = {"model": "mixtral", "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0, "stream": True}
    req = urllib.request.Request(
        f"{BASE}/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    ts = []  # (token_idx, timestamp_since_start)
    n = 0
    first = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            try:
                o = json.loads(d)
            except Exception:
                continue
            if first is None:
                first = time.perf_counter()
            if o.get("choices") and o["choices"][0].get("text"):
                ts.append(time.perf_counter() - t0)
                n += 1
    return ts, first - t0 if first else None, n


def analyze(ts, ttft, n):
    if n < 2:
        return
    gaps = [ts[i+1] - ts[i] for i in range(n - 1)]
    gaps_sorted = sorted(gaps)
    mean = sum(gaps) / len(gaps)
    med = gaps_sorted[len(gaps)//2]
    p95 = gaps_sorted[int(len(gaps)*0.95)]
    print(f"  tokens={n} TTFT={ttft*1000:.1f}ms  "
          f"inter-token: mean={mean*1000:.1f}ms med={med*1000:.1f}ms "
          f"p95={p95*1000:.1f}ms min={min(gaps)*1000:.1f}ms max={max(gaps)*1000:.1f}ms")


def main():
    idx = sys.argv[1] if len(sys.argv) > 1 else None
    names = list(PROMPTS.keys()) if idx is None else [idx]
    for name in names:
        print(f"[{name}]")
        gen_stream(PROMPTS[name])  # warmup
        for r in range(3):
            ts, ttft, n = gen_stream(PROMPTS[name])
            analyze(ts, ttft, n)


if __name__ == "__main__":
    main()
