"""Flow Matching(条件付き最適輸送パス)の最小実装。

SOW 第3章の定義をそのまま実装する。

    eps ~ N(0, I)              ノイズ(t=0 側の端点)
    t   ~ Uniform(0, 1)        補間位置
    x_t = (1 - t) * eps + t * x1
    目標速度 v* = x1 - eps     (x_t を t で微分した値。t に依存しない)
    損失     = MSE(v_theta(x_t, t, cond), v*)

生成は Euler 法で t=0 -> t=1 へ積分する。

    x <- x + (1/steps) * v_theta(x, t)

Diffusion(DDPM)との比較は SOW で範囲外。
"""

from __future__ import annotations

from typing import Any, Callable

import torch


def _randn(
    shape: tuple[int, ...],
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """generator のデバイスが違っても動くようにした randn。"""
    if generator is None:
        return torch.randn(shape, device=device, dtype=dtype)
    gen_device = generator.device
    x = torch.randn(shape, device=gen_device, dtype=dtype, generator=generator)
    return x.to(device)


def sample_noise_and_time(
    x1: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """データ `x1` と同じ形のノイズと、サンプルごとの時刻 t を返す。

    t は (B,) 形で返す。x_t の計算時にブロードキャスト用の形へ変形する。
    """
    eps = _randn(tuple(x1.shape), x1.device, x1.dtype, generator)
    t = _rand_uniform(x1.shape[0], x1.device, x1.dtype, generator)
    return eps, t


def _rand_uniform(
    batch: int,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """t ~ Uniform(0, 1) を (B,) 形で返す。"""
    if generator is None:
        return torch.rand(batch, device=device, dtype=dtype)
    t = torch.rand(batch, device=generator.device, dtype=dtype, generator=generator)
    return t.to(device)


def _broadcast_t(t: torch.Tensor, ndim: int) -> torch.Tensor:
    """(B,) -> (B, 1, 1, ...) にしてテンソル演算に使えるようにする。"""
    return t.reshape(t.shape[0], *([1] * (ndim - 1)))


def interpolate(x1: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """x_t = (1 - t) * eps + t * x1。t=0 で eps、t=1 で x1 に一致する。"""
    tb = _broadcast_t(t, x1.ndim)
    return (1.0 - tb) * eps + tb * x1


def target_velocity(x1: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """目標速度 v* = x1 - eps(ノイズからデータへ向かう向き)。"""
    return x1 - eps


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """mask が与えられた要素のみで MSE を取る。

    mask は pred にブロードキャスト可能な 0/1 テンソル。
    PushT の「未来不足は損失マスクで除外する」に対応する。
    """
    se = (pred - target) ** 2
    if mask is None:
        return se.mean()
    mask = mask.to(dtype=se.dtype)
    mask = mask.expand_as(se) if mask.shape != se.shape else mask
    denom = mask.sum().clamp_min(1.0)
    return (se * mask).sum() / denom


def flow_matching_loss(
    model: Callable[..., torch.Tensor],
    x1: torch.Tensor,
    cond: dict[str, Any] | None = None,
    mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """1 バッチ分の Flow Matching 損失を計算する。

    返り値には教材用に中間値も入れる(`inspect` サブコマンドで表示する)。
    """
    cond = cond or {}
    eps, t = sample_noise_and_time(x1, generator=generator)
    x_t = interpolate(x1, eps, t)
    v_target = target_velocity(x1, eps)
    v_pred = model(x_t, t, **cond)
    loss = masked_mse(v_pred, v_target, mask)
    return {"loss": loss, "t": t, "x_t": x_t, "eps": eps, "v_target": v_target, "v_pred": v_pred}


@torch.no_grad()
def euler_sample(
    model: Callable[..., torch.Tensor],
    shape: tuple[int, ...],
    steps: int = 32,
    device: torch.device | str = "cpu",
    cond: dict[str, Any] | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    x_init: torch.Tensor | None = None,
    num_snapshots: int = 0,
) -> tuple[torch.Tensor, list[tuple[float, torch.Tensor]]]:
    """Euler 法で t=0 -> 1 へ積分して生成する。

    Returns
    -------
    x : 生成結果 (t=1 の状態)
    snapshots : [(t, x のコピー), ...] 生成途中経過。num_snapshots=0 なら空。
    """
    if steps <= 0:
        raise ValueError(f"steps は正の値にする: {steps}")
    cond = cond or {}
    if x_init is None:
        x = _randn(tuple(shape), device, dtype, generator)
    else:
        x = x_init.to(device=device, dtype=dtype).clone()

    snapshot_at = _snapshot_steps(steps, num_snapshots)
    snapshots: list[tuple[float, torch.Tensor]] = []
    if 0 in snapshot_at:
        snapshots.append((0.0, x.detach().cpu().clone()))

    dt = 1.0 / steps
    for i in range(steps):
        t_scalar = i * dt
        t = torch.full((x.shape[0],), t_scalar, device=x.device, dtype=x.dtype)
        v = model(x, t, **cond)
        x = x + dt * v
        if (i + 1) in snapshot_at:
            snapshots.append((round((i + 1) * dt, 6), x.detach().cpu().clone()))
    return x, snapshots


def _snapshot_steps(steps: int, num_snapshots: int) -> set[int]:
    if num_snapshots <= 0:
        return set()
    num = min(num_snapshots, steps + 1)
    return {round(i * steps / (num - 1)) if num > 1 else steps for i in range(num)}
