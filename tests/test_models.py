"""DiT の形状・初期化・学習可能性を検証する。"""

from __future__ import annotations

import torch

from learn_dit.flow import euler_sample, flow_matching_loss
from learn_dit.models.mnist_dit import MNISTDiT, patchify, unpatchify
from learn_dit.models.pusht_dit import PushTDiT


def test_patchify_roundtrip():
    x = torch.randn(2, 1, 28, 28)
    tokens = patchify(x, 4)
    assert tokens.shape == (2, 49, 16)
    assert torch.allclose(unpatchify(tokens, 4, 1, 28, 28), x)


def test_mnist_model_shape_and_zero_init():
    model = MNISTDiT()
    x = torch.randn(3, 1, 28, 28)
    t = torch.rand(3)
    label = torch.tensor([0, 5, 9])
    out = model(x, t, label)
    assert out.shape == x.shape
    # AdaLN-Zero + 出力射影のゼロ初期化により、初期の速度予測は 0
    assert torch.allclose(out, torch.zeros_like(out))


def test_pusht_model_shape():
    model = PushTDiT()
    action = torch.randn(2, 16, 2)
    t = torch.rand(2)
    images = torch.randn(2, 2, 3, 96, 96)
    agent_pos = torch.randn(2, 2, 2)
    out = model(action, t, images=images, agent_pos=agent_pos)
    assert out.shape == action.shape


def test_pusht_memory_reuse_matches_full_forward():
    torch.manual_seed(0)
    model = PushTDiT().eval()
    action = torch.randn(2, 16, 2)
    t = torch.rand(2)
    images = torch.randn(2, 2, 3, 96, 96)
    agent_pos = torch.randn(2, 2, 2)
    with torch.no_grad():
        a = model(action, t, images=images, agent_pos=agent_pos)
        b = model(action, t, memory=model.encode_obs(images, agent_pos))
    assert torch.allclose(a, b, atol=1e-6)


def test_overfit_single_batch_decreases_loss():
    """少数サンプルで損失が下がり、重みが更新されることを確認する。"""
    torch.manual_seed(0)
    model = MNISTDiT(hidden_dim=32, depth=2)
    before = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    x = torch.randn(4, 1, 28, 28)
    label = torch.zeros(4, dtype=torch.long)
    losses = []
    for _ in range(60):
        out = flow_matching_loss(model, x, cond={"label": label})
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()
        losses.append(float(out["loss"]))
    # t はサンプルごとの乱数なので損失は揺れる。前半平均と後半平均で比べる。
    first = sum(losses[:10]) / 10
    last = sum(losses[-10:]) / 10
    assert last < first
    assert any(not torch.equal(p.detach(), q) for p, q in zip(model.parameters(), before))


def test_sampling_produces_finite_values():
    model = MNISTDiT(hidden_dim=32, depth=2).eval()
    label = torch.arange(10)
    x, snapshots = euler_sample(
        model, model.sample_shape(10), steps=4, cond={"label": label}, num_snapshots=3
    )
    assert x.shape == (10, 1, 28, 28)
    assert torch.isfinite(x).all()
    assert len(snapshots) == 3
