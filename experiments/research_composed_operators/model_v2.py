"""Causal model with one attention path for training, prefill and decode."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from causal_reference import CausalTransformerReference, Top1MoEReference
from model import _router_top1, _swiglu


class Top1MoE(Top1MoEReference):
    def forward(self, x):
        if x.numel() != x.size(-1):
            return super().forward(x)
        probabilities, ids = _router_top1(self.router(x), self.cfg.router_temp)
        index = ids.reshape(1)
        gate = self.gate.index_select(0, index).squeeze(0)
        up = self.up.index_select(0, index).squeeze(0)
        down = self.down.index_select(0, index).squeeze(0)
        weight = probabilities.gather(-1, ids.unsqueeze(-1))
        return ((_swiglu(x @ gate, x @ up) @ down) * weight).to(x.dtype)


class CausalTransformer(CausalTransformerReference):
    def __init__(self, cfg, kind):
        super().__init__(cfg, kind)
        # Isolate shared backbone initialization from the size of the FFN.
        generator = torch.Generator().manual_seed(cfg.seed)
        with torch.no_grad():
            for layer in self.layers:
                nn.init.xavier_uniform_(layer.attn.in_proj_weight, generator=generator)
                nn.init.kaiming_uniform_(
                    layer.attn.out_proj.weight, a=math.sqrt(5), generator=generator
                )
        if kind == "moe":
            for layer in self.layers:
                top1 = Top1MoE(cfg)
                top1.load_state_dict(layer.ffn.state_dict())
                layer.ffn = top1
        if kind in ("single", "two"):
            # Keep each initial block near identity and preserve top-1 router
            # gradients. Multiplying by the candidate count offsets uniform p.
            for layer in self.layers:
                ffn = layer.ffn
                identity = torch.eye(cfg.group_dim)
                with torch.no_grad():
                    tables = (ffn.blocks,) if kind == "single" else (ffn.blocks1, ffn.blocks2)
                    routers = (ffn.router,) if kind == "single" else (ffn.router1, ffn.router2)
                    for table, router in zip(tables, routers):
                        table.normal_(std=0.02 / math.sqrt(cfg.group_dim))
                        table.add_(identity).mul_(cfg.n_candidates)
                        router.weight.mul_(0.1)

    def forward(self, ids, cache=None):
        """Return logits. A supplied cache is updated in-place after the forward."""
        length = ids.size(1)
        offset = 0 if cache is None else cache.length
        end = offset + length
        if length == 0 or end > self.position.num_embeddings:
            raise ValueError("Input and cached prefix must fit the configured context")
        if cache is not None and (
            torch.is_grad_enabled() or self.training or cache.capacity < end
        ):
            raise ValueError("KV cache requires eval/no-grad and sufficient capacity")
        x = self.tok(ids) + self.position(torch.arange(offset, end, device=ids.device))
        for index, layer in enumerate(self.layers):
            attn = layer.attn
            h = layer.norm1(x)
            q, k, v = (
                t.reshape(ids.size(0), length, attn.num_heads, attn.head_dim).transpose(1, 2)
                for t in F.linear(h, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
            )
            if cache is not None:
                k, v = cache.store(index, offset, end, k, v)
            mask = None
            if offset and length > 1:
                # SDPA bool masks use True for visible positions.
                mask = ~self.causal_mask[offset:end, :end]
            heads = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=(offset == 0 and length > 1)
            )
            x = x + attn.out_proj(heads.transpose(1, 2).reshape_as(h))
            x = x + layer.ffn(layer.norm2(x))
        if cache is not None:
            cache.length = end
        return self.lm_head(self.norm(x))


class KVCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.length = 0
        self.layers = {}

    def store(self, index, start, end, k, v):
        if index not in self.layers:
            shape = (*k.shape[:2], self.capacity, k.shape[-1])
            self.layers[index] = (k.new_empty(shape), v.new_empty(shape))
        keys, values = self.layers[index]
        keys[..., start:end, :].copy_(k)
        values[..., start:end, :].copy_(v)
        return keys[..., :end, :], values[..., :end, :]
