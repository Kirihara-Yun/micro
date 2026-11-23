"""Local correctness reference for the next experiment.

The historical training and decode entry points remain available under their
original names. This reference uses a causal mask, explicit positions, and
executes only the selected standard MoE expert. The per-expert Python loop is
intended for correctness checks, not a CUDA performance baseline.
"""

import torch
import torch.nn as nn

from model import (
    SingleStageFFN,
    StandardMoEFFN,
    TwoStageFFN,
    _router_top1,
    _swiglu,
)


class Top1MoEReference(StandardMoEFFN):
    def forward(self, x):
        flat = x.reshape(-1, x.size(-1))
        probabilities, ids = _router_top1(self.router(flat), self.cfg.router_temp)
        selected_probability = probabilities.gather(1, ids[:, None])
        output = torch.zeros_like(flat)
        for expert_id in range(self.cfg.n_experts):
            rows = (ids == expert_id).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            inputs = flat.index_select(0, rows)
            hidden = _swiglu(inputs @ self.gate[expert_id], inputs @ self.up[expert_id])
            values = (hidden @ self.down[expert_id]) * selected_probability[rows]
            output = output.index_copy(0, rows, values.to(output.dtype))
        return output.reshape_as(x)


class CausalBlock(nn.Module):
    def __init__(self, cfg, ffn_kind):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.ffn = {
            "moe": Top1MoEReference,
            "single": SingleStageFFN,
            "two": TwoStageFFN,
        }[ffn_kind](cfg)

    def forward(self, x, causal_mask):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


class CausalTransformerReference(nn.Module):
    def __init__(self, cfg, ffn_kind):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position = nn.Embedding(cfg.context, cfg.d_model)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)
        self.layers = nn.ModuleList(CausalBlock(cfg, ffn_kind) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight
        self.register_buffer(
            "causal_mask",
            torch.ones(cfg.context, cfg.context, dtype=torch.bool).triu(1),
            persistent=False,
        )

    def forward(self, ids):
        length = ids.size(1)
        if length > self.position.num_embeddings:
            raise ValueError("Input exceeds the configured context length")
        x = self.tok(ids) + self.position(torch.arange(length, device=ids.device))
        mask = self.causal_mask[:length, :length]
        for layer in self.layers:
            x = layer(x, mask)
        return self.lm_head(self.norm(x))
