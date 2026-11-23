"""三种 MoE/条件变换 FFN 变体的 decoder-only Transformer。

任务：比较「标准 MoE」「单阶段条件变换」「双阶段条件变换」的训练质量与解码延迟。
共享主干：token embedding（weight-tied）、RMSNorm、attention、残差。仅 FFN 计算不同。

三种 FFN（d=384）：
- 标准 MoE：SwiGLU 专家，top-1 路由，8 专家 × intermediate。
- 单阶段：共享 SwiGLU 输入投影 → 8 组 × 48 维，每组 N 个 48×48 候选矩阵 → 共享输出投影。
- 双阶段：共享 SwiGLU 输入投影 → 第一阶段条件块 → 固定通道交换 → 第二阶段条件块 → 共享输出投影。

门控：三变体统一用 softmax 概率加权的 top-1 门控（梯度可回传），可选负载均衡 aux loss。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    context: int = 512
    # 条件层结构
    n_groups: int = 8           # 组数（8 × 48 = 384）
    group_dim: int = 48
    n_candidates: int = 16      # 每组每阶段候选矩阵数
    # 标准 MoE
    n_experts: int = 8
    moe_intermediate: int = 384
    # 门控
    router_temp: float = 1.0
    load_balance_coef: float = 0.0
    top_k: int = 1
    # 随机种子
    seed: int = 20261123


def _precompute_channel_swap(n_groups: int, group_dim: int) -> torch.Tensor:
    """固定通道交换：第一阶段输出 [n_groups, group_dim] 视作 [n_groups, sub, c]，
    交换 group 轴与 sub 轴后重新分组。返回一个 [n_groups*group_dim] 的索引张量，
    使 y = x[..., swap_idx] 完成「交换两个 8 维轴后重新分组」。

    实现：group_dim = sub * c（sub = n_groups, c = group_dim // n_groups）。
    原始索引 (g, s, c) -> 线性 g*group_dim + s*c + k。
    交换后新分组 new_g 的第 j 个元素 = 原 (g=s, s=new_g, c=j//c, k=j%c) 处元素。
    """
    sub = n_groups
    c = group_dim // n_groups
    assert n_groups * c == group_dim, "group_dim 必须能被 n_groups 整除"
    # 原始坐标 -> 线性索引表：idx[g, s, k] = g*group_dim + s*c + k
    idx = torch.arange(n_groups * group_dim).reshape(n_groups, sub, c)
    # 交换 group(g) 与 sub(s)：new[new_g, j] = idx[s=new_g, g=?] 需要仔细
    # 交换后新分组 new_g 的第 j 维（j = g2*c + k）来自原 (g = g2, s = new_g, k)
    # 因为交换是把「组」和「子通道」互换：新组 = 原子通道索引，新子通道 = 原组索引
    swap = idx.permute(1, 0, 2)  # [sub, n_groups, c] -> 新组=sub, 子通道=n_groups
    return swap.reshape(-1)


def _swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def _router_top1(logits: torch.Tensor, temperature: float):
    """softmax 概率加权的 top-1 门控。返回 (prob, ids)。

    prob 形状与 logits 相同（除最后一维外），作为所选候选的缩放系数，保证路由梯度可回传。
    """
    # logits: [..., n_candidates]
    scaled = logits / temperature
    prob = torch.softmax(scaled, dim=-1)
    # 用 one-hot 近似 top-1：只保留 argmax，概率作为系数
    ids = torch.argmax(scaled, dim=-1)          # [...,]
    return prob, ids


class StandardMoEFFN(nn.Module):
    """标准 MoE：SwiGLU 专家，top-1 路由。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, m, E = cfg.d_model, cfg.moe_intermediate, cfg.n_experts
        self.cfg = cfg
        self.gate = nn.Parameter(torch.empty(E, d, m))
        self.up = nn.Parameter(torch.empty(E, d, m))
        self.down = nn.Parameter(torch.empty(E, m, d))
        self.router = nn.Linear(d, E, bias=False)
        self._reset()

    def _reset(self):
        std = 0.02
        nn.init.normal_(self.gate, std=std)
        nn.init.normal_(self.up, std=std)
        nn.init.normal_(self.down, std=std / math.sqrt(2 * self.cfg.n_experts))
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, x: torch.Tensor):
        # x: [B, S, d]
        logits = self.router(x)                     # [B, S, E]
        prob, ids = _router_top1(logits, self.cfg.router_temp)  # prob [B,S,E], ids [B,S]
        p = prob.gather(-1, ids.unsqueeze(-1))       # [B, S, 1]
        # 只计算选中的专家
        B, S, d = x.shape
        g = torch.einsum("bsd,edm->bsem", x, self.gate)   # [B,S,E,m]
        u = torch.einsum("bsd,edm->bsem", x, self.up)
        h = _swiglu(g, u)                                # [B,S,E,m]
        h = h * p.unsqueeze(-1)                          # 加权
        # 只对选中的专家做 down（用 mask 归零未选中专家，等效于只算选中）
        out = torch.einsum("bsem,emd->bsd", h, self.down)  # 稠密但等价；见性能说明
        return out

    def active_params(self):
        d, m = self.cfg.d_model, self.cfg.moe_intermediate
        # 每 token 激活 1 个专家：gate+up+down
        return 3 * d * m


class SingleStageFFN(nn.Module):
    """单阶段：共享 SwiGLU 输入投影 → 按组选小矩阵 → 共享输出投影。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, G, gd, N = cfg.d_model, cfg.n_groups, cfg.group_dim, cfg.n_candidates
        assert G * gd == d
        self.cfg = cfg
        self.w_gate = nn.Linear(d, d, bias=False)
        self.w_up = nn.Linear(d, d, bias=False)
        self.w_out = nn.Linear(d, d, bias=False)
        # 候选矩阵：[G, N, gd, gd]（out × in）
        self.blocks = nn.Parameter(torch.empty(G, N, gd, gd))
        self.router = nn.Linear(d, G * N, bias=False)
        self._reset()

    def _reset(self):
        nn.init.normal_(self.w_gate.weight, std=0.02)
        nn.init.normal_(self.w_up.weight, std=0.02)
        nn.init.normal_(self.w_out.weight, std=0.02)
        nn.init.normal_(self.blocks, std=0.02)
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, x: torch.Tensor):
        B, S, d = x.shape
        G, N, gd = self.cfg.n_groups, self.cfg.n_candidates, self.cfg.group_dim
        h = _swiglu(self.w_gate(x), self.w_up(x))          # [B,S,d]
        h = h.view(B, S, G, gd)                            # [B,S,G,gd]
        logits = self.router(x).view(B, S, G, N)           # [B,S,G,N]
        prob, ids = _router_top1(logits, self.cfg.router_temp)  # prob [B,S,G,N], ids [B,S,G]
        # 选出每组选中的候选矩阵
        sel = self.blocks[torch.arange(G, device=x.device)[None, None, :], ids]  # [B,S,G,gd,gd]
        # 组内矩阵乘（只执行选中块）
        out = torch.einsum("bsgi,bsgio->bsgo", h, sel)     # [B,S,G,gd]
        # softmax 概率加权（梯度回传）
        p = prob.gather(-1, ids.unsqueeze(-1)).unsqueeze(-1)  # [B,S,G,1,1]
        out = out * p.squeeze(-1)                          # [B,S,G,gd]
        out = out.view(B, S, d)
        return self.w_out(out)

    def active_params(self):
        d, gd = self.cfg.d_model, self.cfg.group_dim
        # 共享 SwiGLU 投影(2×d×d) + 共享输出投影(d×d) + 1 个候选块(2×gd×gd)
        return 2 * d * d + d * d + 2 * gd * gd


class TwoStageFFN(nn.Module):
    """双阶段：共享 SwiGLU 输入投影 → 第一阶段条件块 → 通道交换 → 第二阶段条件块 → 共享输出投影。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, G, gd, N = cfg.d_model, cfg.n_groups, cfg.group_dim, cfg.n_candidates
        assert G * gd == d
        self.cfg = cfg
        self.w_gate = nn.Linear(d, d, bias=False)
        self.w_up = nn.Linear(d, d, bias=False)
        self.w_out = nn.Linear(d, d, bias=False)
        self.blocks1 = nn.Parameter(torch.empty(G, N, gd, gd))
        self.blocks2 = nn.Parameter(torch.empty(G, N, gd, gd))
        self.router1 = nn.Linear(d, G * N, bias=False)
        self.router2 = nn.Linear(d, G * N, bias=False)
        # 固定通道交换索引
        self.register_buffer("swap_idx", _precompute_channel_swap(G, gd))
        self._reset()

    def _reset(self):
        nn.init.normal_(self.w_gate.weight, std=0.02)
        nn.init.normal_(self.w_up.weight, std=0.02)
        nn.init.normal_(self.w_out.weight, std=0.02)
        nn.init.normal_(self.blocks1, std=0.02)
        nn.init.normal_(self.blocks2, std=0.02)
        nn.init.normal_(self.router1.weight, std=0.02)
        nn.init.normal_(self.router2.weight, std=0.02)

    def _stage(self, h: torch.Tensor, x: torch.Tensor, blocks: torch.Tensor, router: nn.Linear):
        B, S = h.shape[:2]
        G, N, gd = self.cfg.n_groups, self.cfg.n_candidates, self.cfg.group_dim
        logits = router(x).view(B, S, G, N)
        prob, ids = _router_top1(logits, self.cfg.router_temp)
        sel = blocks[torch.arange(G, device=x.device)[None, None, :], ids]
        out = torch.einsum("bsgi,bsgio->bsgo", h, sel)
        p = prob.gather(-1, ids.unsqueeze(-1)).unsqueeze(-1).squeeze(-1)
        out = out * p
        return out.view(B, S, G * gd)

    def forward(self, x: torch.Tensor):
        B, S, d = x.shape
        G, gd = self.cfg.n_groups, self.cfg.group_dim
        h = _swiglu(self.w_gate(x), self.w_up(x))
        h = h.view(B, S, G, gd)
        # 第一阶段
        h1 = self._stage(h, x, self.blocks1, self.router1)   # [B,S,G*gd]
        # 通道交换
        h1 = h1.index_select(-1, self.swap_idx)              # [B,S,G*gd]
        h1 = h1.view(B, S, G, gd)
        # 第二阶段
        h2 = self._stage(h1, x, self.blocks2, self.router2)
        return self.w_out(h2)

    def active_params(self):
        d, gd = self.cfg.d_model, self.cfg.group_dim
        # 共享 SwiGLU 投影(2×d×d) + 共享输出投影(d×d) + 两阶段各 1 个候选块(2×2×gd×gd)
        return 2 * d * d + d * d + 2 * 2 * gd * gd


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, ffn_kind: str):
        super().__init__()
        d, H = cfg.d_model, cfg.n_heads
        self.norm1 = nn.RMSNorm(d)
        self.norm2 = nn.RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, H, batch_first=True)
        if ffn_kind == "moe":
            self.ffn = StandardMoEFFN(cfg)
        elif ffn_kind == "single":
            self.ffn = SingleStageFFN(cfg)
        elif ffn_kind == "two":
            self.ffn = TwoStageFFN(cfg)
        else:
            raise ValueError(ffn_kind)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class ComposedTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, ffn_kind: str):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # GPT-2 标准：embedding 用 N(0, 0.02)，避免 logits 方差过大导致初始 loss 异常
        nn.init.normal_(self.tok.weight, std=0.02)
        self.layers = nn.ModuleList([TransformerBlock(cfg, ffn_kind) for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model)
        # weight tying
        self.tok.weight = nn.Parameter(self.tok.weight)  # 保持单一权重
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight

    def forward(self, ids: torch.Tensor):
        x = self.tok(ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def ffn_active_params_per_layer(self):
        return self.layers[0].ffn.active_params()
