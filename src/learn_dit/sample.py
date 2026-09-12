"""生成。チェックポイントから Euler 法でサンプルを作り、途中経過も保存する。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from learn_dit.checkpoint import load_checkpoint
from learn_dit.config import Config
from learn_dit.data.builder import build_dataloaders, split_batch
from learn_dit.data.normalize import MinMaxNormalizer
from learn_dit.flow import euler_sample
from learn_dit.models import build_model
from learn_dit.utils.device import resolve_device
from learn_dit.utils.logging import write_json
from learn_dit.utils.seed import set_seed
from learn_dit.visualize import plot_action_trajectory, save_image_grid, save_snapshot_grid


def load_model(
    checkpoint: str | Path | None,
    cfg: Config,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any] | None, int]:
    """チェックポイントがあれば読み、無ければ未学習モデルを返す(比較用)。"""
    model = build_model(cfg).to(device)
    if checkpoint is None:
        model.eval()
        return model, None, 0
    payload = load_checkpoint(checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload.get("stats"), int(payload.get("step", 0))


def sample_mnist(
    cfg: Config,
    checkpoint: str | Path | None,
    out_dir: str | Path,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    model, _, step = load_model(checkpoint, cfg, device)

    labels = torch.arange(cfg.model.num_classes, device=device).repeat_interleave(cfg.sample.num_per_class)
    results: dict[str, Any] = {"checkpoint": str(checkpoint), "step": step, "artifacts": {}, "timing": {}}

    for steps in sorted(set(cfg.sample.steps_sweep + [cfg.sample.steps])):
        generator = torch.Generator(device="cpu").manual_seed(cfg.seed)  # 同じ初期ノイズで比較する
        started = time.time()
        x, snapshots = euler_sample(
            model,
            model.sample_shape(labels.shape[0]),
            steps=steps,
            device=device,
            cond={"label": labels},
            generator=generator,
            num_snapshots=cfg.sample.intermediate_snapshots if steps == cfg.sample.steps else 0,
        )
        elapsed = time.time() - started
        path = save_image_grid(
            x, out_dir / f"samples_steps{steps}.png", nrow=cfg.sample.num_per_class,
            title=f"0-9 conditional / euler {steps} steps / step={step}",
        )
        results["artifacts"][f"samples_steps{steps}"] = str(path)
        results["timing"][f"steps{steps}_sec"] = round(elapsed, 3)
        if snapshots:
            snap_path = save_snapshot_grid(snapshots, out_dir / "intermediate.png", sample_index=0)
            results["artifacts"]["intermediate"] = str(snap_path)
            results["snapshot_times"] = [t for t, _ in snapshots]
    write_json(out_dir / "sample_summary.json", results)
    return results


def sample_pusht(
    cfg: Config,
    checkpoint: str | Path | None,
    out_dir: str | Path,
    num_samples: int = 4,
) -> dict[str, Any]:
    """検証エピソードの観測を条件に動作系列を生成し、データセットの動作と並べて描く。"""
    out_dir = Path(out_dir)
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    model, stats, step = load_model(checkpoint, cfg, device)
    _, val_loader, stats = build_dataloaders(cfg, download=False, stats=stats)
    action_norm = MinMaxNormalizer.from_dict(stats["action"])
    state_norm = MinMaxNormalizer.from_dict(stats["state"])

    batch = next(iter(val_loader))
    x1, cond, _ = split_batch(cfg, batch, device)
    take = min(num_samples, x1.shape[0])
    cond = {k: v[:take] for k, v in cond.items()}
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)

    with torch.no_grad():
        memory = model.encode_obs(cond["images"], cond["agent_pos"])
        started = time.time()
        pred, snapshots = euler_sample(
            model,
            model.sample_shape(take),
            steps=cfg.sample.steps,
            device=device,
            cond={"memory": memory},
            generator=generator,
            num_snapshots=cfg.sample.intermediate_snapshots,
        )
        elapsed = time.time() - started

    pred_actions = action_norm.denormalize(pred.cpu())
    ref_actions = action_norm.denormalize(x1[:take].cpu())
    current_pos = state_norm.denormalize(cond["agent_pos"].cpu())[:, -1]

    artifacts = {}
    for i in range(take):
        path = plot_action_trajectory(
            pred_actions[i].numpy(),
            out_dir / f"trajectory_{i}.png",
            reference=ref_actions[i].numpy(),
            current=current_pos[i].numpy(),
        )
        artifacts[f"trajectory_{i}"] = str(path)

    # 途中経過は軌跡の収束として描く(t 昇順)
    if snapshots:
        traj_steps = np.stack([action_norm.denormalize(x)[0].numpy() for _, x in snapshots])
        fig_path = out_dir / "intermediate_trajectory.png"
        _plot_snapshot_trajectories(traj_steps, [t for t, _ in snapshots], fig_path)
        artifacts["intermediate_trajectory"] = str(fig_path)

    results = {
        "checkpoint": str(checkpoint),
        "step": step,
        "euler_steps": cfg.sample.steps,
        "sample_sec": round(elapsed, 3),
        "artifacts": artifacts,
    }
    write_json(out_dir / "sample_summary.json", results)
    return results


def _plot_snapshot_trajectories(traj: np.ndarray, times: list[float], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    cmap = plt.get_cmap("viridis")
    for i, (t, xy) in enumerate(zip(times, traj)):
        ax.plot(xy[:, 0], xy[:, 1], "-o", markersize=2.5, color=cmap(i / max(len(times) - 1, 1)),
                label=f"t={t:.2f}", alpha=0.85)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_sample(cfg: Config, checkpoint: str | Path | None, out_dir: str | Path) -> dict[str, Any]:
    if cfg.task == "mnist":
        return sample_mnist(cfg, checkpoint, out_dir)
    return sample_pusht(cfg, checkpoint, out_dir, num_samples=cfg.sample.num_per_class)
