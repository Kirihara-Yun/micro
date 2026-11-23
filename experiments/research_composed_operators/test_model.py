"""单元测试：三种变体前向/反向 + 通道交换正确性 + 参数统计。"""
import torch
from model import ModelConfig, ComposedTransformer, _precompute_channel_swap

def test_channel_swap():
    # G=2, gd=4 -> sub=2, c=2
    G, gd = 2, 4
    swap = _precompute_channel_swap(G, gd)
    # 原始 x: [G, sub, c] = [2,2,2]，线性索引 g*4 + s*2 + k
    x = torch.arange(G * gd).reshape(G, gd)
    y = x.reshape(-1).index_select(0, swap).reshape(G, gd)
    # 期望：交换 group 与 sub 轴
    expected = x.reshape(G, 2, 2).permute(1, 0, 2).reshape(G, gd)
    assert torch.equal(y, expected), f"swap 错误:\n{y}\n!=\n{expected}"
    print("通道交换索引: OK")

def test_forward_backward():
    cfg = ModelConfig(vocab_size=1000, d_model=48, n_layers=2, n_heads=2,
                      n_groups=4, group_dim=12, n_candidates=4,
                      n_experts=4, moe_intermediate=48)
    ids = torch.randint(0, 1000, (2, 16))
    for kind in ["moe", "single", "two"]:
        m = ComposedTransformer(cfg, kind)
        out = m(ids)
        assert out.shape == (2, 16, 1000), f"{kind} 输出形状错误: {out.shape}"
        loss = out.float().mean()
        loss.backward()
        # 检查路由梯度
        ffn = m.layers[0].ffn
        router = ffn.router if kind == "moe" else (ffn.router1 if kind == "two" else ffn.router)
        g = router.weight.grad
        assert g is not None and g.abs().sum() > 0, f"{kind} 路由无梯度"
        # 检查候选矩阵梯度
        if kind == "two":
            assert ffn.blocks1.grad is not None and ffn.blocks1.grad.abs().sum() > 0
            assert ffn.blocks2.grad is not None and ffn.blocks2.grad.abs().sum() > 0
        elif kind == "single":
            assert ffn.blocks.grad is not None and ffn.blocks.grad.abs().sum() > 0
        print(f"{kind}: 前向+反向 OK, 路由梯度={g.abs().sum().item():.3f}")

def test_param_count():
    cfg = ModelConfig()
    for kind in ["moe", "single", "two"]:
        m = ComposedTransformer(cfg, kind)
        n = m.num_params()
        active = m.ffn_active_params_per_layer()
        print(f"{kind}: 总参数={n/1e6:.2f}M, 每层活跃计算参数={active}")
    # 全尺寸前向时间基准（无梯度）
    import time
    ids = torch.randint(0, cfg.vocab_size, (1, 64))
    for kind in ["moe", "single", "two"]:
        m = ComposedTransformer(cfg, kind).cuda()
        with torch.no_grad():
            for _ in range(3):
                m(ids.cuda())
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                m(ids.cuda())
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 10
        print(f"{kind}: 前向 {dt*1000:.2f} ms/step (batch=1, seq=64)")

if __name__ == "__main__":
    test_channel_swap()
    test_forward_backward()
    test_param_count()
