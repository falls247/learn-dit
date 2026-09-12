"""Flow Matching の数学的な性質を検証する(SOW 受入条件: 補間端点・速度符号・Euler 更新)。"""

from __future__ import annotations

import torch

from learn_dit.flow import euler_sample, interpolate, sample_noise_and_time, target_velocity
from learn_dit.flow.flow_matching import masked_mse


def test_interpolation_endpoints():
    x1 = torch.randn(8, 3, 4)
    eps = torch.randn_like(x1)
    zeros = torch.zeros(8)
    ones = torch.ones(8)
    assert torch.allclose(interpolate(x1, eps, zeros), eps)   # t=0 はノイズ
    assert torch.allclose(interpolate(x1, eps, ones), x1)     # t=1 はデータ


def test_interpolation_midpoint_is_average():
    x1 = torch.randn(4, 2)
    eps = torch.randn_like(x1)
    half = torch.full((4,), 0.5)
    assert torch.allclose(interpolate(x1, eps, half), 0.5 * (x1 + eps), atol=1e-6)


def test_velocity_sign_points_from_noise_to_data():
    x1 = torch.tensor([[1.0, -2.0]])
    eps = torch.tensor([[0.0, 0.0]])
    v = target_velocity(x1, eps)
    assert torch.allclose(v, x1 - eps)
    # ノイズからデータへ向かう向き: データが大きい次元では正
    assert v[0, 0] > 0 and v[0, 1] < 0


def test_velocity_is_path_derivative():
    """d/dt x_t = x1 - eps。有限差分で確認する。"""
    x1 = torch.randn(4, 5)
    eps = torch.randn_like(x1)
    t = torch.full((4,), 0.3)
    dt = 1e-4
    numeric = (interpolate(x1, eps, t + dt) - interpolate(x1, eps, t)) / dt
    assert torch.allclose(numeric, target_velocity(x1, eps), atol=1e-3)


def test_sampled_time_in_unit_interval():
    x1 = torch.randn(256, 2)
    eps, t = sample_noise_and_time(x1)
    assert eps.shape == x1.shape
    assert t.shape == (256,)
    assert float(t.min()) >= 0.0 and float(t.max()) < 1.0


def test_euler_recovers_target_with_constant_velocity():
    """速度が定数なら Euler 積分は x_1 = x_0 + v を厳密に満たす。"""
    v = torch.tensor([[2.0, -1.0]])

    def model(x, t):
        return v.expand_as(x)

    x0 = torch.zeros(1, 2)
    for steps in (1, 4, 32):
        out, _ = euler_sample(model, (1, 2), steps=steps, x_init=x0)
        assert torch.allclose(out, x0 + v, atol=1e-5)


def test_euler_step_count_and_time_grid():
    seen = []

    def model(x, t):
        seen.append(float(t[0]))
        return torch.zeros_like(x)

    _, snapshots = euler_sample(model, (1, 2), steps=4, num_snapshots=5)
    assert seen == [0.0, 0.25, 0.5, 0.75]          # t=0 から 1 直前まで
    assert [t for t, _ in snapshots] == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_euler_is_deterministic_given_generator():
    def model(x, t):
        return torch.ones_like(x)

    a, _ = euler_sample(model, (3, 2), steps=8, generator=torch.Generator().manual_seed(42))
    b, _ = euler_sample(model, (3, 2), steps=8, generator=torch.Generator().manual_seed(42))
    assert torch.equal(a, b)


def test_masked_mse_ignores_masked_elements():
    pred = torch.tensor([[[1.0], [100.0]]])
    target = torch.zeros_like(pred)
    mask = torch.tensor([[[1.0], [0.0]]])
    assert torch.allclose(masked_mse(pred, target, mask), torch.tensor(1.0))
    assert masked_mse(pred, target, None) > 1.0


def test_masked_mse_all_zero_mask_is_finite():
    pred = torch.ones(2, 3, 1)
    target = torch.zeros_like(pred)
    mask = torch.zeros_like(pred)
    assert torch.isfinite(masked_mse(pred, target, mask))
