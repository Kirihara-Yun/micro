"""Reproduce the original protocol failures and verify a local reference."""

import torch
import torch.nn.functional as F

from causal_reference import CausalTransformerReference, Top1MoEReference
from model import ComposedTransformer, ModelConfig, StandardMoEFFN


def check_prefix(cfg):
    ids = torch.randint(cfg.vocab_size, (2, 12))
    changed = ids.clone()
    changed[:, 6:] = (changed[:, 6:] + 1) % cfg.vocab_size
    for variant in ("moe", "single", "two"):
        original = ComposedTransformer(cfg, variant).eval()
        corrected = CausalTransformerReference(cfg, variant).eval()
        with torch.no_grad():
            old_diff = (original(ids)[:, :6] - original(changed)[:, :6]).abs().max().item()
            result = corrected(ids)[:, :6]
            torch.testing.assert_close(result, corrected(changed)[:, :6])
            torch.testing.assert_close(result, corrected(ids[:, :6]))
        assert old_diff > 1e-5
        corrected.train()
        optimizer = torch.optim.AdamW(corrected.parameters())
        for _ in range(2):
            optimizer.zero_grad()
            logits = corrected(ids)[:, :-1].transpose(1, 2)
            loss = F.cross_entropy(logits, ids[:, 1:])
            assert torch.isfinite(loss)
            loss.backward()
            routers = [
                p for name, p in corrected.named_parameters() if ".router" in name
            ]
            assert routers and all(
                p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                for p in routers
            )
            optimizer.step()
        print(f"{variant}: original prefix max change={old_diff:.6g}; causal checks and two optimizer steps pass")


def check_top1(cfg):
    old = StandardMoEFFN(cfg)
    new = Top1MoEReference(cfg)
    x = torch.ones(1, 1, cfg.d_model)
    with torch.no_grad():
        old.router.weight.zero_()
        old.router.weight[0].fill_(1.0 / cfg.d_model)
        new.load_state_dict(old.state_dict())
        before_old, before_new = old(x), new(x)
        old.down[1].add_(1.0)
        new.down[1].add_(1.0)
        delta = (old(x) - before_old).abs().max().item()
        assert delta > 1e-5
        torch.testing.assert_close(new(x), before_new)
    print(f"changing unselected expert: original output max change={delta:.6g}; top-1 reference unchanged")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(20261123)
    config = ModelConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, context=16,
        n_groups=4, group_dim=8, n_candidates=4, n_experts=4, moe_intermediate=32,
    )
    check_prefix(config)
    check_top1(config)
