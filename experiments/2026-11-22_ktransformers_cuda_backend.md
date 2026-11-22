# KTransformers CUDA 后端可行性核对

## 结论

**kt-kernel CUDA 后端构建成功并可加载**（`torch.cuda.is_available()=True`，`kt doctor` 检测到 CUDA 12.6 + A100）。完整 CPU+GPU 端到端推理**尚未完成**，缺顶层推理框架 SGLang（见"所缺资源"）。

## 环境记录（`kt doctor` 权威诊断）

| 检查项 | 状态 | 值 |
|---|---|---|
| Python | OK | 3.11.18 |
| CUDA availability | OK | 12.6 |
| GPU detection | OK | 1× NVIDIA A100 80GB PCIe |
| CPU | OK | Xeon Platinum 8360Y @ 2.40GHz（72 核）|
| CPU Instructions | OK | AVX512F/BW/VL/CD、AVX2、AVX、FMA、SSE4.2 等（无 AMX/BF16）|
| NUMA Topology | OK | 4 nodes |
| kt-kernel | OK | v0.7.2（AVX512_VBMI）|
| System memory | OK | 995.4GB / 1007.5GB |
| SGLang Source | **Warning** | Not installed |

## 构建过程与关键障碍

- 源码：`kvcache-ai/ktransformers` @ `ca198c5`（kt-kernel 0.7.2）
- 依赖：torch 2.9.1+cu128（CUDA 12.8 runtime）、conda 环境 `kt`（Python 3.11）
- **障碍 1：宿主默认 CUDA 11.5 不匹配** → 改用系统已有的 `CUDA 12.6`（`CUDA_HOME=/usr/local/cuda-12.6`）
- **障碍 2：nvcc 用了 gcc-10 头文件**（`/usr/include/c++/10` 触发 `constexpr` 重定义）→ 显式指定 `CUDAHOSTCXX=/usr/bin/g++-12`
- 构建命令：
  ```bash
  CUDA_HOME=/usr/local/cuda-12.6 CUDAHOSTCXX=/usr/bin/g++-12 \
  CPUINFER_USE_CUDA=1 CPUINFER_CUDA_ARCHS=80 CPUINFER_PARALLEL=16 \
  pip install . --no-deps --no-build-isolation
  ```
- arch 设为 `80`（仅 A100 所需，跳过 86/89/90）

## 验证结果

```python
import torch
torch.cuda.is_available()   # True
torch.version.cuda          # 12.8（torch bundled runtime）
import kt_kernel            # OK
kt_kernel.__cpu_variant__   # avx512_vbmi
```

`kt doctor` 显示 CUDA availability 12.6（构建工具链）、GPU A100 检测成功。

## 所缺资源（完整推理受阻原因）

1. **SGLang 未安装**：kt-kernel 是内核库，完整推理需 `sglang-kt`（kvcache-ai fork）。安装方式：`pip install sglang-kt` 或从 `third_party/sglang`（子模块未拉取）源码安装。
2. 若装 SGLang，还需顶层 ktransformers 包与 transformers/tokenizers 等依赖。

## 可用模型权重

- `/data/qinzy/models/models--mistralai--Mixtral-8x7B-Instruct-v0.1`（178GB，含 consolidated.pt ×8 + safetensors ×19）
- `/data/qinzy/models/models--deepseek-ai--DeepSeek-V2-Lite`（存在）

## 备注

- Xeon 8360Y 无 AMX/BF16，仅验证 AVX-512 路径（AVX512_VBMI），不宣称复现 SOSP 论文 AMX 性能。
- 原始构建日志：`/tmp/kt_build_cuda.log`（本机，未提交）。
