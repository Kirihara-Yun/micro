# micro

> 新工作区（脱离 MECC 仓库）。本仓库记录 MoE 推理 baseline 的复现方法与来源，作为后续工作的起点。

## 目录结构

```
micro/
├── baselines/          # 软链接 → /home/qinzy/baselines/（本机资源，不入 git）
│   ├── Pregated_MoE    #   ISCA'24 Pre-Gated MoE（第三方）
│   └── ktransformers   #   KTransformers / kt-kernel（第三方）
└── data/ft/            # 软链接 → /data/ft（本机模型权重，不入 git）
```

> `baselines/` 与 `data/` 是指向本机绝对路径的软链接，clone 后需按本机环境自行重建。

## Baseline 复现记录

### 1. Pre-Gated MoE（ISCA'24）

- **来源**：<https://github.com/ranggihwang/Pregated_MoE.git> @ `6b0a666`（2024-05-04）
- **本质**：基于 NVIDIA FasterTransformer 的 T5-MoE（Switch Transformer）pre-gated 推理
- **环境**：Docker `nvcr.io/nvidia/pytorch:22.09-py3`（A100，CUDA 11.8，PyTorch 1.13.0a0）
- **编译**：
  ```bash
  cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
  make -j
  ```
- **Python 依赖**：`examples/pytorch/t5/requirement.txt`（transformers~=4.31.0、pandas、psutil 等）
- **权重**：HF `google/switch-base-8` 等，经 `scripts/convert.sh` 转换到 `/data/ft/<model>/`
- **验证结果**（冒烟级，batch=1、seq_len=256、iterations=4，非正式 benchmark）：
  - GPU-only：154 tokens/sec
  - Pre-gated：130 tokens/sec（`Max active experts: 1`）

### 2. KTransformers / kt-kernel

- **来源**：<https://github.com/kvcache-ai/ktransformers.git> @ `ca198c5`（2026-11-13）
- **本质**：CPU-GPU 异构 LLM 推理框架，MoE expert 落 CPU 内存 + 量化
- **环境**：conda 环境 `kt`（Python 3.11.18，CPU-only 构建）
- **编译**：
  ```bash
  # 拉子模块
  git submodule update --init --recursive third_party/pybind11 third_party/llama.cpp
  # 依赖（torch==2.9.1 等）+ 构建
  CPUINFER_USE_CUDA=0 CPUINFER_PARALLEL=16 pip install . --no-deps --no-build-isolation
  ```
- **系统依赖**：`libhwloc`（含 libxml2 运行时）、`libnuma`
- **验证结果**：`import kt_kernel` 成功，CPU variant 检测为 `avx512_vbmi`
  （Xeon 8360Y：AVX512F/BW/VNNI/VBMI，无 BF16/AMX）

## 环境注意事项

- 两个 baseline 的编译产物、conda 环境、Docker 镜像均依赖本机绝对路径，迁移需谨慎。
- KTransformers 当前为 CPU-only（宿主 CUDA 11.5 与 torch 2.9.1 所需 CUDA 12.x 不匹配），GPU offload 后端未启用。
