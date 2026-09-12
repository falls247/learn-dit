"""可視化。損失曲線、生成画像グリッド、生成途中経過、軌跡、動画。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from learn_dit.data.normalize import unit_range_to_image  # noqa: E402


def _prep(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_image_grid(
    images: torch.Tensor,
    path: str | Path,
    nrow: int = 10,
    title: str | None = None,
) -> Path:
    """images: (N, C, H, W) を [-1,1] 前提でグリッド保存する。"""
    path = _prep(path)
    imgs = unit_range_to_image(images.detach().float().cpu())
    n = imgs.shape[0]
    ncol = nrow
    nrows = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrows, ncol, figsize=(ncol * 0.9, nrows * 0.9))
    axes = np.atleast_1d(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            arr = imgs[i]
            ax.imshow(arr[0], cmap="gray", vmin=0, vmax=1) if arr.shape[0] == 1 else ax.imshow(
                arr.permute(1, 2, 0).numpy()
            )
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_snapshot_grid(
    snapshots: Sequence[tuple[float, torch.Tensor]],
    path: str | Path,
    sample_index: int = 0,
) -> Path:
    """生成途中経過を t 昇順で横に並べる。"""
    path = _prep(path)
    if not snapshots:
        raise ValueError("snapshots が空")
    fig, axes = plt.subplots(1, len(snapshots), figsize=(len(snapshots) * 1.1, 1.5))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, (t, x) in zip(axes, snapshots):
        img = unit_range_to_image(x[sample_index].detach().float().cpu())
        ax.imshow(img[0], cmap="gray", vmin=0, vmax=1) if img.shape[0] == 1 else ax.imshow(
            img.permute(1, 2, 0).numpy()
        )
        ax.set_title(f"t={t:.2f}", fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_loss_curve(history: list[dict[str, Any]], path: str | Path) -> Path:
    path = _prep(path)
    train = [(r["step"], r["loss"]) for r in history if r.get("kind") == "train" and "loss" in r]
    val = [(r["step"], r["loss"]) for r in history if r.get("kind") == "val" and "loss" in r]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    if train:
        ax.plot(*zip(*train), label="train", linewidth=1.0)
    if val:
        ax.plot(*zip(*val), label="val", marker="o", markersize=3, linewidth=1.0)
    ax.set_xlabel("step")
    ax.set_ylabel("flow matching loss (MSE)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_action_trajectory(
    predicted: np.ndarray,
    path: str | Path,
    reference: np.ndarray | None = None,
    current: np.ndarray | None = None,
) -> Path:
    """2 次元目標位置の系列を描く。predicted: (T, 2)"""
    path = _prep(path)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(predicted[:, 0], predicted[:, 1], "-o", markersize=3, label="predicted")
    if reference is not None:
        ax.plot(reference[:, 0], reference[:, 1], "-s", markersize=3, alpha=0.6, label="dataset")
    if current is not None:
        ax.scatter([current[0]], [current[1]], marker="*", s=120, label="current agent")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_video(frames: Sequence[np.ndarray], path: str | Path, fps: int = 10) -> Path | None:
    """フレーム列を MP4 にする。imageio が無い環境では None を返す。"""
    path = _prep(path)
    if not frames:
        return None
    try:
        import imageio.v2 as imageio
    except Exception:  # noqa: BLE001
        return None
    try:
        with imageio.get_writer(str(path), fps=fps) as writer:
            for frame in frames:
                arr = np.asarray(frame)
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                writer.append_data(arr)
    except Exception:  # noqa: BLE001 - ffmpeg 未導入など
        return None
    return path
