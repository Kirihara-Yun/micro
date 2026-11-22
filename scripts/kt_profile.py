#!/usr/bin/env python3
"""用 sglang 内置 torch profiler 采集 decode 阶段 trace。
用法: python3 kt_profile.py <output_dir>
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:30000"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kt_trace"


def post(path, payload):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode()


def gen(prompt, max_tokens=64):
    payload = {"model": "mixtral", "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0, "stream": False}
    req = urllib.request.Request(
        f"{BASE}/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()
    return time.perf_counter() - t0


prompt = ("Explain how mixture of experts routing works in large language models, "
          "focusing on the top-k gating mechanism and load balancing. " * 6)

# 热身
gen(prompt, 32)

# 启动 profile（decode 阶段 16 个 step）
print("start_profile:", post("/start_profile", {
    "output_dir": OUT,
    "activities": ["CPU", "GPU"],
    "num_steps": 16,
    "profile_id": "mixtral_decode",
    "record_shapes": False,
}))

# 发请求（decode 会触发 16+ 个 forward step，profile 自动停止）
t = gen(prompt, 64)
print(f"generation: {t:.2f}s")

# 若未自动停止则手动停止
time.sleep(1)
print("stop_profile:", post("/stop_profile", {}))
print("done. trace 输出目录:", OUT)
