# A100 远端验证任务单

## 目标

验证 README 中 Pre-Gated MoE 的 Switch-Base-8 冒烟结果能否稳定复现，并确认 KTransformers 是否具备 CPU+GPU 端到端推理条件。优先复用远端已有的源码、镜像、环境和权重；先盘点路径及版本，再运行，避免重复下载。

## 任务 1：Pre-Gated MoE

1. 记录 `nvidia-smi`、CPU 型号与内存、CUDA/驱动、容器镜像、源码 SHA、`google/switch-base-8` 权重版本、转换目录及确切命令。
2. 在 A100 上跑 GPU-only 和 Pre-gated 两种模式，配置统一为 batch=1、seq_len=256。先热身，再各执行足够多次迭代以报告中位数、波动范围、tokens/s、MoE block latency、峰值显存；保存原始逐次数据。
3. 检查 Pre-gated 模式实际激活专家数、CPU→GPU 传输与计算是否重叠，并抽样比较输出或任务质量。若只能用现有脚本得到吞吐，明确记录其他指标缺失。
4. 不把现有 4 次迭代的 154/130 tokens/s 当成论文 benchmark；报告新的实验条件和相对差异。

## 任务 2：KTransformers 可行性

1. 记录远端 A100 的驱动、CUDA toolkit、PyTorch、`kt-kernel` 版本、CPU 指令集、NUMA 与可用模型权重。
2. 核对 `ca198c5` 所需 CUDA 构建依赖，尝试构建并验证 CUDA 后端可加载；保留完整构建命令及失败日志。
3. 若构建成功，用该版本明确支持且已具备权重的 MoE 模型完成一次 CPU+GPU 推理，记录模型、量化、各层部署、输入输出长度、吞吐和显存/主存。没有合适权重时停止此项，列出所缺资源。
4. Xeon 8360Y 不支持 AMX；若远端 CPU 同样没有 AMX，只验证可用 AVX-512 路径，不宣称复现 SOSP 论文 AMX 性能。

## 回传格式

每次实验提供：时间、host、源码 SHA、模型及权重 revision、硬件/软件版本、完整命令、原始日志路径、热身/计时次数、结果摘要、异常。优先回传小体积 Markdown/CSV；权重和编译产物留在远端。
