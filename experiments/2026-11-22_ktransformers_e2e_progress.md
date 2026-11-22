# KTransformers 端到端推理进展与阻塞

> 日期：2026-11-22。记录 KTransformers（`ca198c5`）从「CUDA 后端可加载」向「端到端推理」推进的进展与当前阻塞。

## 已完成

1. **修复 kt-kernel 加载**：`kt_kernel_ext.so` 报 `libxml2.so.16: cannot open shared object file`（`libhwloc.so.15` 的依赖，系统只有 libxml2.so.2）。从 `/data/qinzy/ai/tools/lib/` 软链接 `libxml2.so.16` 及其依赖（`libiconv.so.2`、`libicuuc.so.78`、`libicudata.so.78`、`libicui18n.so.78`）到 kt 环境 `lib/`，恢复 `import kt_kernel` → `avx512_vbmi`。
2. **拉取 sglang 子模块**：`third_party/sglang`（`kvcache-ai/sglang` fork）checkout `2c4085a`。
3. **调整依赖版本绕过 NGC 源**：`sglang/python/pyproject.toml` 中三个 fork 包带 `.dev20260928` 后缀（仅存在于 `pypi.ngc.nvidia.com`，该源 DNS 解析失败）。改为公共 PyPI 正式版：`kt-kernel==0.7.2`、`transformers-kt==5.6.0.post6`、`accelerate-kt==1.14.0.post4`。
4. **启动 sglang-kt 核心依赖安装**（后台，`PIP_CONFIG_FILE=/dev/null` 绕过 NGC 源，公共 PyPI）。

## 当前阻塞

1. **外网带宽受限（核心阻塞）**：下载大文件稳定在 ~20–30 KB/s（无 HTTP 代理、无内网 PyPI 镜像、`pypi.ngc.nvidia.com` DNS 解析失败）。影响：
   - sglang-kt 依赖（accelerate_kt 85.5MB、flashinfer、torchao、torchvision 等，合计 ~1GB+）需 10+ 小时，安装仍在后台进行。
   - Mixtral-8x7B 的 GGUF 权重（Q4_K_M 约 26GB）下载需约 10 天，**不可行**。
2. **GGUF 权重缺失**：LLAMAFILE backend（本机无 AMX，只能用 GGUF 后端）需要预量化 GGUF 权重，本机没有，需从 HF 下载（受上述带宽限制）。
3. **DeepSeek-V2-Lite 权重不完整**：`models--deepseek-ai--DeepSeek-V2-Lite/snapshots/` 仅含 config 等元数据，无权重文件，不可用。
4. **AMX 缺失**：Xeon 8360Y 无 AMX，论文 AMX 路径无法验证（资源阻塞）。

## 可用资源

- **Mixtral-8x7B-Instruct-v0.1 权重完整**（178GB）：`/data/qinzy/models/models--mistralai--Mixtral-8x7B-Instruct-v0.1/`，含 `consolidated.00-07.pt` + `model-00001..19-of-00019.safetensors`，可作为 GPU 侧原模型。
- **CPU**：72 物理核（Thread(s) per core = 1）、4 NUMA node。
- **GPU**：A100 80GB。

## 启动参数（已备，待 sglang-kt 安装完成）

```bash
python -m sglang.launch_server \
  --host 0.0.0.0 --port 8000 \
  --model /data/qinzy/models/models--mistralai--Mixtral-8x7B-Instruct-v0.1 \
  --mem-fraction-static 0.92 \
  --kt-method LLAMAFILE \
  --kt-weight-path <Mixtral-GGUF-目录> \
  --kt-cpuinfer 72 \
  --kt-threadpool-count 4
```

## 解除阻塞所需

1. 更快外网带宽，或 HTTP 代理 / 内网 PyPI 镜像（用于完成 sglang-kt 依赖下载）。
2. 预下载的 Mixtral-8x7B-Instruct GGUF 权重（Q4_K_M ~26GB）置于本机，或提供可访问的内网 HF 镜像。

## 结论层级

- **代码能运行**：kt-kernel 恢复可加载；sglang-kt 依赖安装进行中。
- **系统机制成立**：未验证（需端到端跑通后）。
- **论文数值复现**：受 AMX 缺失 + 网络带宽阻塞，未完成。
