# WorkBuddy 运行新版因果训练和解码

从仓库根目录进入本目录，使用现有的 PyTorch 2.5.1 CUDA 环境。以下命令假定 `python` 指向该环境。新版从头训练，历史 checkpoint 不适用。需要现有 `data/train.npy`、`data/val.npy`。回传数据来源和两个文件的 SHA256。

```bash
cd experiments/research_composed_operators
python smoke_v2.py --device cuda --bf16
```

冒烟应对三种变体分别通过续跑权重对照、改变后缀时前缀不变、分段 KV cache 与完整前向的 logits 对照。MoE 还检查未选专家隔离和 batch=1 路径的一致性。若失败，保存完整输出和环境返回本地修复。

通过后，先用三种变体各跑同样的 200 步，检查损失、显存和稳态时间。独立输出目录避免覆盖历史实验。

```bash
for kind in moe single two; do
  python train_v2.py --kind "$kind" --data data --out "logs/v2_calib_$kind" \
    --steps 200 --warmup 50 --eval-every 100 --batch 8 --accum 8 --bf16 || break
done
```

三组都完成校准后，从头跑以下同预算筛选。按实际可用时间在启动前决定步数。512 长度时，每步有 32704 个预测目标，1529 步约 50M。收集验证 NLL 随有效预测目标数的变化。数据有放回采样，目标数是处理量。训练脚本按相同的固定留出采样评估，并在结束步保存 checkpoint。

```bash
for kind in moe single two; do
  python train_v2.py --kind "$kind" --data data --out "logs/v2_train_$kind" \
    --steps 1529 --warmup 150 --eval-every 150 --batch 8 --accum 8 --bf16 || break
  python decode_v2.py --checkpoint "logs/v2_train_$kind/latest.pt" --prompts data/val.npy \
    --prompt-len 128 --new-tokens 64 --requests 8 --rounds 3 --bf16 \
    > "logs/v2_train_$kind/decode.jsonl" || break
done
```

若需要分两次运行，首次用 `--stop-after`，续跑用 `--resume logs/v2_train_<kind>/latest.pt`，原计划的 `--steps`、数据、精度与其它训练设置保持一致。日志的 `target_tokens` 计预测目标，解码吞吐为解码 token 除以总解码时间，首 token 延迟另计。

MoE 在多 token 下使用逐专家 PyTorch 分组计算，batch=1 解码在设备侧索引选中专家。三种变体都是当前 PyTorch 原型。时间和显存反映这些实现，系统性能主张需要优化后的对照和 DRAM 计数。双阶段各级块改用近单位的初始变换。当前仍需单独做等预算的无交换消融才能归因于跨组交换。首轮请回传冒烟输出、环境、命令、数据哈希、JSONL 和筛选结论。大权重留远端。
