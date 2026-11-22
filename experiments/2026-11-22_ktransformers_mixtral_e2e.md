# KTransformers Mixtral-8x7B 端到端推理：成功跑通（修正前报告结论）

> 日期：2026-11-22 17:00–17:20 (+0800)
> 结论：**Mixtral-8x7B 在 KTransformers `ca198c5` 上成功跑通 CPU+GPU 端到端推理**。
> 本文修正 `2026-11-22_ktransformers_e2e_progress.md` 的核心结论——该报告称
> "Mixtral 不在官方支持列表、sglang LLAMAFILE 后端未打通"，经真机运行验证**不成立**：
> LLAMAFILE backend 是完整实现的，Mixtral 可直接跑。

## 1. 关键结论（三层）

| 层 | 状态 | 证据 |
|---|---|---|
| 代码能运行 | ✅ | sglang-kt server 成功启动，Mixtral 加载 + 推理正确 |
| 系统机制成立 | ✅ | 8 专家每层 1 GPU + 7 CPU，GPU 显存仅 16GB（全模型 bf16 需 ~94GB） |
| 端到端正确性 | ✅ | "What is 2+2?" → "The sum of 2 + 2 is 4." |

## 2. 修正前报告的错误结论

`2026-11-22_ktransformers_e2e_progress.md` 的三个结论需修正：

1. ~~"Mixtral 不在 KTransformers 官方内置支持列表"~~ → **真机可跑**。底层
   `GGUFLoader` + `LlamafileMoEWrapper` 完整支持 Mixtral（8 专家/top-2/4096/14336）。
   之前的静态代码搜索漏掉了 `kt-kernel/python/utils/llamafile.py` 与 sglang 侧的运行时
   接线（日志出现 `[LlamafileMoEWrapper] Layer 0..31 TP configuration`）。
2. ~~"sglang-kt 侧 LLAMAFILE backend 无完整实现"~~ → **完整可用**。`--kt-method LLAMAFILE`
   走 `LlamafileMoEWrapper`，GGUF 权重 mmap 直供 CPU 推理。
3. ~~"跑通 Mixtral 需自定义集成"~~ → **无需**。只需正确格式的 GGUF（见 §4）。

## 3. 两个必须满足的前置条件

### 3.1 GGUF 必须是"堆叠布局"（`ffn_gate_exps.weight`）

`LlamafileMoEWrapper.load_weights()` 按 `blk.{i}.ffn_{gate,up,down}_exps.weight` 键读取专家张量
（`kt-kernel/python/utils/llamafile.py:189-196`），即每层 3 个大张量（第一维=专家数）。

- ✅ `mixtral-exps-q8_0.gguf`（46.2GB，q8_0，**堆叠布局**，323 tensor）→ 可用
- ❌ `mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf`（24.6GB，Q4_K_M，**非堆叠** `ffn_gate.{0..7}.weight`，995 tensor）→ 不匹配

> 注：`mixtral-exps-q8_0.gguf` 由 `convert-hf-to-gguf.py`（`third_party/llama.cpp`）从
> Mixtral-8x7B HF 权重导出，`general.architecture=llama`、`llama.expert_count=8`、
> `llama.expert_used_count=2`、`llama.block_count=32`。

### 3.2 必须禁用 CodeBuddy "safe-delete" shim

默认 `PYTHONPATH` 注入 `/home/qinzy/.vscode-server/extensions/tencent-cloud.coding-copilot-*/out/vendor/shim`，
其 `sitecustomize.py` 劫持 `os.removedirs`（"把删除移入回收站"），在 triton 编译缓存清理时因
`/.Trash-1002` 权限失败而崩溃：

```
OSError: [safe-delete] 操作失败 ... Error during a `trash` operation: Permission denied
```

解决：`export CODEBUDDY_SAFE_DELETE_ENABLED=0`（`sitecustomize.py:37` 的开关）。

## 4. 复现命令

```bash
export PATH=/home/qinzy/.conda/envs/kt/bin:$PATH
export CODEBUDDY_SAFE_DELETE_ENABLED=0
HF=/data/qinzy/models/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/eba92302a2861cdc0098cc54bc9f17cb2c47eb61
GGUF=/data/qinzy/models/mixtral-8x7b-gguf/mixtral-exps-q8_0.gguf
python3.11 -m sglang.launch_server \
  --host 0.0.0.0 --port 30000 --model "$HF" \
  --kt-weight-path "$GGUF" --kt-cpuinfer 32 --kt-threadpool-count 4 \
  --kt-num-gpu-experts 1 --kt-method LLAMAFILE \
  --kt-gpu-prefill-token-threshold 4096 --kt-enable-dynamic-expert-update \
  --attention-backend flashinfer --trust-remote-code --mem-fraction-static 0.98 \
  --chunked-prefill-size 4096 --max-running-requests 32 --max-total-tokens 8192 \
  --tensor-parallel-size 1 --enable-p2p-check --disable-shared-experts-fusion
# 等 server ready 后：
curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mixtral","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":32,"temperature":0}'
```

## 5. 环境与模型

| 项 | 值 |
|---|---|
| 源码 | `kvcache-ai/ktransformers` @ `ca198c5`（kt-kernel 0.7.2） |
| sglang-kt | `third_party/sglang` @ `2c4085a`（kvcache-ai fork），`import sglang` = 0.0.0.dev0 |
| Python | conda 环境 `kt`，3.11.18，torch 2.9.1+cu128 |
| GPU | 1× NVIDIA A100 80GB PCIe，驱动 560.35.03，CUDA 12.6 |
| CPU | Xeon Platinum 8360Y，72 核，4 NUMA，AVX512_VBMI（无 AMX/BF16） |
| HF 权重 | Mixtral-8x7B-Instruct-v0.1（178GB，safetensors + consolidated.pt） |
| GGUF | `mixtral-exps-q8_0.gguf`（46.2GB，q8_0，堆叠布局） |
| dtype | torch.bfloat16 |

## 6. 加载与资源

| 指标 | 值 |
|---|---|
| 权重加载耗时 | 7.44 s（`Load weight end ... elapsed=7.44 s`） |
| GPU 显存（加载后） | 13.54 GB（日志 `mem usage=13.54 GB`） |
| GPU 显存（含 KV cache + 推理峰值） | 16.0 GB（`nvidia-smi` 16360 MiB，推理期间不变） |
| server 进程 RSS | 1.0 GB（GGUF mmap 懒加载，专家页主要靠 page cache） |
| KV cache | 8192 tokens，K/V 各 0.50 GB |
| CPU 推理线程 | 32（`--kt-cpuinfer 32`），4 NUMA threadpool |
| GPU 专家 | 每层 1 个（`--kt-num-gpu-experts 1`），其余 7 个在 CPU |

> 机制证据（日志）：`Using kt_num_gpu_experts=1 per layer, total GPU experts: 32 (= 1 × 32 MoE layers)`、
> `CPUInfer[0x...]: Hello`、`[KT] Created shared staging buffer: 32.0 MiB`。
> 对比：Mixtral bf16 全模型 ~94GB，A100 80GB 装不下；KT offload 后 GPU 仅需 16GB。

## 7. 性能（batch=1，stream，max_tokens=64，每条件热身 1 次 + 3 次）

| prompt | TTFT (s) | 总耗时 (s) | 生成 token | decode (tok/s) |
|---|---|---|---|---|
| short (~8 tok) | 0.230 / 0.227 / 0.227 | 7.33 / 7.32 / 7.32 | 64 | 8.88 |
| medium (~512 tok) | 0.229 / 0.232 / 0.232 | 7.26 / 7.25 / 7.24 | 64 | 8.96–8.99 |
| long (~2048 tok) | 0.232 / 0.230 / 0.231 | 7.24 / 7.24 / 7.24 | 64 | 8.98–8.99 |

- **decode 吞吐 ≈ 8.9 tok/s**（CPU 专家计算为瓶颈，32 线程）。
- **TTFT ≈ 0.23s**（prompt < 4096 token 走 GPU prefill，故短/中/长 prompt TTFT 几乎相同）。
- 256 token 生成：TTFT 0.877s、decode 8.97 tok/s。

## 8. 已知 bug：CPU prefill（prompt > 4096 token）崩溃

当 prompt 超过 `--kt-gpu-prefill-token-threshold 4096` 时触发 **CPU prefill**，崩溃于：

```
AttributeError: 'LlamafileMoEWrapper' object has no attribute 'submit_write_weight_scale_to_buffer'
```

调用链：`fused_moe_triton/layer.py:1045 run_moe_core` → `kt_ep_wrapper.py:5814 apply` →
`_build_full_context` → `load` → `_prepare_weight_bf16`（`:2640` `submit_write_expert`）→
`wrapper.submit_write_weight_scale_to_buffer`（`:2630`）。

根因：`_prepare_weight_bf16`（sglang CPU prefill 路径）假设 wrapper 提供
`submit_write_weight_scale_to_buffer`（把 CPU 权重写到 GPU staging buffer），但该方法仅
`AMX` wrapper（`kt-kernel/python/utils/amx.py:1065`）实现；`LlamafileMoEWrapper` 的权重是
mmap GGUF 直供 CPU，无此语义。**decode 与 GPU prefill（< 4096 token）不受影响**，任务单
要求的短/中/长 prompt（32/512/2048）均已成功测量。

## 9. 结论层级

- **代码能运行**：✅ Mixtral 端到端 server 启动 + 推理正确。
- **系统机制成立**：✅ routed experts 在 CPU（7/8 专家）、非专家模块在 GPU（显存 16GB 佐证）。
- **论文数值可比**：⚠️ 本机为 Xeon 8360Y（无 AMX/BF16）+ 单 A100 80GB + q8_0 量化，
  SOSP 论文为双路 Xeon 8452Y（AMX）+ 不同模型；**数值不可直接对比**，仅验证 AVX-512 路径的功能复现。

## 10. 原始证据

- 启动日志：`/tmp/kt_mixtral_run3.log`（含 `Load weight end`、`LlamafileMoEWrapper Layer 0..31`、CUDA graph capture）。
- 性能原始：`/tmp/kt_bench_results.json`。
- 崩溃堆栈：`/tmp/kt_mixtral_run2.log`（CPU prefill bug）。
- GGUF metadata：`general.architecture=llama`、`expert_count=8`、`expert_used_count=2`、`block_count=32`。
