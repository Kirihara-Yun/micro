"""Algebra reference only; not a CPU/GPU performance prototype.

Checks a shared group-quantized weight representation, SwiGLU intermediate
partitioning, top-k weighting, and adding output bias exactly once.
"""
import numpy as np


def quantized_reference(rng, shape, group=32):
    weights = rng.normal(size=shape)
    blocks = weights.reshape(shape[0], -1, group)
    scales = np.maximum(np.max(np.abs(blocks), axis=-1, keepdims=True), 1e-12) / 127
    codes = np.clip(np.rint(blocks / scales), -127, 127).astype(np.int8)
    return (codes.astype(np.float64) * scales).reshape(shape)


def silu(x):
    return x / (1 + np.exp(-x))


def main():
    rng = np.random.default_rng(20261122)
    d, intermediate = 64, 256
    inputs = rng.normal(size=(4, d))
    experts = []
    for _ in range(2):
        experts.append((quantized_reference(rng, (intermediate, d)),
                        quantized_reference(rng, (intermediate, d)),
                        quantized_reference(rng, (d, intermediate)),
                        rng.normal(size=d)))
    route_weights = [.3, .7]
    full = np.zeros_like(inputs)
    for weight, (gate, up, down, bias) in zip(route_weights, experts):
        full += weight * ((silu(inputs @ gate.T) * (inputs @ up.T)) @ down.T + bias)
    worst_error = 0.0
    for cut in (0, 32, 64, 128, 256):
        result = np.zeros_like(full)
        for weight, (gate, up, down, bias) in zip(route_weights, experts):
            partials = []
            for part in (slice(0, cut), slice(cut, intermediate)):
                hidden = silu(inputs @ gate[part].T) * (inputs @ up[part].T)
                partials.append(hidden @ down[:, part].T)
            result += weight * (partials[0] + partials[1] + bias)
        error = np.max(np.abs(result - full))
        worst_error = max(worst_error, error)
        np.testing.assert_allclose(result, full, rtol=1e-11, atol=1e-10)
    print(f"FP64 algebra check passed: 5 splits, top-2, max abs error={worst_error:.3g}")
    print("Shared dequantized group weights; no GGUF kernel, GPU, or speed claim.")


if __name__ == "__main__":
    main()
