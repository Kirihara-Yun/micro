# KTransformers 端到端推理进展与阻塞

> 日期：2026-11-22。记录 KTransformers（`ca198c5`）端到端推理的推进、突破与当前阻塞。

## 已完成

1. **修复 kt-kernel 加载**：软链接 `libxml2.so.16` 及依赖（`libiconv.so.2`、`libicuuc/icudata/icui18n.so.78`）到 kt 环境，恢复 `import kt_kernel` → `avx512_vbmi`。
2. **拉取 sglang 子模块**：`third_party/sglang`（`kvcache-ai/sglang` fork）@ `2c4085a`。
3. **调整依赖版本绕过 NGC 源**：`pyproject.toml` 三个 fork 包去掉 `.dev20260928` 后缀（该源 DNS 不可达），改用公共 PyPI 正式版。
4. **突破网络限速，sglang-kt 完整安装成功**：
   - 原 `pypi.org` 大文件下载仅 ~30KB/s（无代理、无内网镜像、NGC 源 DNS 失败）。
   - 改用**华为云 PyPI 镜像** `repo.huaweicloud.com/repository/pypi`（实测 7–28MB/s），sglang-kt 及全部依赖（flashinfer 0.6.3、tilelang、torchao、transformers-kt、accelerate-kt 等）安装完成，`import sglang` / `sglang.srt` 通过。
5. **Mixtral-8x7B GGUF 下载中**：`hf-mirror.com` 镜像（5.7–11MB/s），`mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf`（24.6GB）后台下载。

## 关键发现（决定端到端可行性）

1. **网络阻塞已解除**：华为云 PyPI 镜像 + hf-mirror.com，两个通道均已验证可用（下载速度 7–11MB/s）。
2. **本机 CPU 能力**：Xeon Platinum 8360Y 无 `avx512_bf16`、无 AMX，仅有 `avx512f/bw/vbmi/vnni`。
   → 可选后端：RAWINT4（AVX512F+BW，官方仅支持 **Kimi-K2-Thinking**）或 LLAMAFILE（GGUF，通用）。
3. **Mixtral-8x7B 不在 KTransformers（kt-kernel 0.7.2）官方内置支持列表**：内置注册表仅含 DeepSeek-V3-0324/V3.2/R1-0528/V4-Flash、Kimi-K2-Thinking、MiniMax-M2/M2.1/M3/M2.7，**无 Mixtral**。
4. **sglang-kt 侧 LLAMAFILE backend 无完整实现**：`kt_method`/`llamafile` 在 sglang 中仅出现在参数定义（`server_args.py`）、MXFP4 专家并行（`kt_ep_wrapper.py`）与 benchmark（`benchmark_kt_ep.py`）；`mixtral.py` 无任何 kt 集成。
5. 底层 GGUF 加载器（`kt-kernel/python/utils/loader.py`）有 Mixtral MoE 格式（`block_sparse_moe.experts.{w1,w3,w2}`）识别，`analyze_moe_model.py` 也识别 `num_local_experts`（Mixtral 字段），但 sglang 集成层未打通。

## 结论与阻塞

- **任务单指定模型与 KTransformers ca198c5 官方支持模型不匹配**：
  - DeepSeek-V2-Lite：本地权重不完整（仅 config，无权重文件）。
  - Mixtral-8x7B：权重完整，但非 KTransformers 官方内置支持，sglang 集成层未打通 LLAMAFILE。
- 跑通 Mixtral + KTransformers 端到端需要自定义集成（为 Mixtral 手动注册 user model + 打通 sglang LLAMAFILE 后端），工作量和风险较高，非开箱即用。
- 若改用 KTransformers 官方支持的模型（DeepSeek-V3/V4、Kimi-K2、MiniMax），则需另行下载权重（DeepSeek-V3 ~685GB，远超本地磁盘与网络能力；Kimi-K2 ~1TB，不可行）。

## 可用资源（已就绪）

- Mixtral-8x7B-Instruct-v0.1 原模型权重（178GB，HF safetensors + consolidated.pt）。
- Mixtral GGUF Q4_K_M（24.6GB，下载中）。
- sglang-kt 完整环境（kt 环境 Python 3.11，torch 2.9.1+cu128）。
- CPU 72 物理核 / 4 NUMA；GPU A100 80GB。

## 解除阻塞所需（按优先级）

1. 确认是否接受「Mixtral 自定义集成」工作量（为 Mixtral 打通 sglang LLAMAFILE 后端），或改用 KTransformers 官方支持且权重可得的模型。
2. 若走自定义集成：需参考 kt-kernel 的 `LlamafileMoEWrapper` 与 sglang 的 `kt_ep_wrapper` 补 Mixtral 适配。
