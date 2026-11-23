"""Verify v2 training, resumption, and incremental decoding on CPU or CUDA."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from decode_v2 import generate
from model import ModelConfig
from model_v2 import CausalTransformer, KVCache
from train_v2 import run


def check(kind, root, device, bf16):
    settings = dict(kind=kind, data=root, steps=4, seed=20261123, device=device,
                    lr=6e-4, warmup=2, batch=2, accum=2, eval_every=2,
                    eval_batches=2, bf16=bf16, tiny=True)
    continuous = root / f"{kind}-continuous"
    interrupted = root / f"{kind}-interrupted"
    run(SimpleNamespace(**settings, out=continuous, stop_after=None, resume=None))
    run(SimpleNamespace(**settings, out=interrupted, stop_after=2, resume=None))
    run(SimpleNamespace(**settings, out=interrupted, stop_after=None, resume=interrupted / "latest.pt"))
    first, second = [torch.load(folder / "latest.pt", weights_only=True, map_location="cpu")
                     for folder in (continuous, interrupted)]
    assert first["step"] == second["step"] == 4
    assert first["data_rng"] == second["data_rng"]
    tolerance = 0 if device == "cpu" else 1e-6
    for name, tensor in first["model"].items():
        torch.testing.assert_close(tensor, second["model"][name], rtol=0, atol=tolerance)

    cfg = ModelConfig(**first["settings"]["config"])
    model = CausalTransformer(cfg, kind)
    model.load_state_dict(first["model"])
    dtype = torch.bfloat16 if bf16 else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    ids = torch.tensor([[1, 6, 5, 4, 3, 2, 10, 11]], device=device)
    with torch.no_grad():
        full = model(ids)
        altered = ids.clone()
        altered[:, 4:] += 12
        torch.testing.assert_close(full[:, :4], model(altered)[:, :4])
        cache = KVCache(8)
        cached = torch.cat([model(ids[:, left:right], cache)
                            for left, right in ((0, 3), (3, 4), (4, 7), (7, 8))], dim=1)
        assert cache.length == 8
        torch.testing.assert_close(full, cached, atol=3e-3 if bf16 else 1e-5,
                                   rtol=1e-2 if bf16 else 1e-4)
        _, times = generate(model, ids[:, :3], 4)
        assert len(times) == 3 and min(times) > 0
        if kind == "moe":
            ffn = model.layers[0].ffn
            ffn.router.weight.zero_()
            single = torch.ones((1, 1, cfg.d_model), device=device, dtype=dtype)
            old = ffn(single)
            torch.testing.assert_close(old, ffn(single.repeat(1, 2, 1))[:, :1])
            ffn.down[1].add_(1)
            torch.testing.assert_close(old, ffn(single))
    print(f"{kind}: resume, causal prefix and incremental decode checks passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        rng = np.random.default_rng(123)
        for name in ("train", "val"):
            np.save(root / f"{name}.npy", rng.integers(64, size=(16, 12), dtype=np.uint32))
        for kind in ("moe", "single", "two"):
            check(kind, root, args.device, args.bf16)


if __name__ == "__main__":
    main()
