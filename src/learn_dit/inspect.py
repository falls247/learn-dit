"""データと 1 バッチの学習を解説する。教材の第 2 ステップに対応する。

`learn-dit inspect --task mnist` は「1 バッチで何が起きているか」を実際の数値で示す。
`learn-dit inspect --task pusht` はデータ整合性(時刻対応・分割独立性・マスク)を報告する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from learn_dit.config import Config
from learn_dit.data import pusht as pusht_data
from learn_dit.data.builder import build_dataloaders, split_batch
from learn_dit.data.normalize import MinMaxNormalizer
from learn_dit.flow import flow_matching_loss, interpolate, target_velocity
from learn_dit.models import build_model
from learn_dit.utils.device import resolve_device
from learn_dit.utils.logging import write_json
from learn_dit.utils.seed import set_seed
from learn_dit.visualize import save_image_grid


def inspect_task(cfg: Config, out_dir: str | Path, download: bool = True) -> dict[str, Any]:
    out_dir = Path(out_dir)
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    train_loader, val_loader, stats = build_dataloaders(cfg, download=download)

    report: dict[str, Any] = {
        "task": cfg.task,
        "device": str(device),
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "batch_size": cfg.train.batch_size,
    }

    batch = next(iter(train_loader))
    x1, cond, mask = split_batch(cfg, batch, device)
    report["x_shape"] = list(x1.shape)
    report["x_min"] = round(float(x1.min()), 4)
    report["x_max"] = round(float(x1.max()), 4)
    report["cond_keys"] = sorted(cond)
    if mask is not None:
        report["mask_valid_ratio"] = round(float(mask.mean()), 4)

    # --- Flow Matching を数値で確認する ---
    model = build_model(cfg).to(device)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    out = flow_matching_loss(model, x1, cond=cond, mask=mask, generator=generator)
    eps, t = out["eps"], out["t"]
    x_at_0 = interpolate(x1, eps, torch.zeros_like(t))
    x_at_1 = interpolate(x1, eps, torch.ones_like(t))
    report["flow_matching"] = {
        "t_sample": [round(v, 4) for v in t[:8].tolist()],
        "x_t=0_equals_noise_max_abs_diff": float((x_at_0 - eps).abs().max()),
        "x_t=1_equals_data_max_abs_diff": float((x_at_1 - x1).abs().max()),
        "target_velocity_equals_x1_minus_eps": bool(
            torch.allclose(out["v_target"], target_velocity(x1, eps))
        ),
        "initial_pred_is_zero": float(out["v_pred"].abs().max()),  # AdaLN-Zero のため 0 から始まる
        "initial_loss": float(out["loss"]),
        "target_velocity_var": float(out["v_target"].var()),
    }

    # --- 1 バッチで数ステップ学習し、損失低下と重み更新を確認する ---
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    losses = []
    for _ in range(20):
        step_out = flow_matching_loss(model, x1, cond=cond, mask=mask)
        optimizer.zero_grad(set_to_none=True)
        step_out["loss"].backward()
        optimizer.step()
        losses.append(float(step_out["loss"].detach()))
    changed = sum(
        1 for name, p in model.named_parameters() if not torch.equal(p.detach(), before[name])
    )
    report["one_batch_training"] = {
        "losses": [round(v, 5) for v in losses],
        # t が乱数なので単点ではなく前半平均と後半平均を比べる
        "mean_loss_first_5": round(sum(losses[:5]) / 5, 5),
        "mean_loss_last_5": round(sum(losses[-5:]) / 5, 5),
        "loss_decreased": sum(losses[-5:]) < sum(losses[:5]),
        "updated_param_tensors": changed,
        "total_param_tensors": len(before),
    }

    if cfg.task == "mnist":
        report["artifacts"] = _inspect_mnist_artifacts(x1, cond, out_dir)
    else:
        report["pusht_integrity"] = _inspect_pusht(cfg, stats)
        report["artifacts"] = _inspect_pusht_artifacts(cfg, batch, stats, out_dir)

    write_json(out_dir / "inspect_report.json", report)
    return report


def _inspect_mnist_artifacts(x1: torch.Tensor, cond: dict[str, torch.Tensor], out_dir: Path) -> dict[str, str]:
    grid = save_image_grid(x1[:32].cpu(), out_dir / "batch_examples.png", nrow=8, title="data batch")
    # t を変えたときの x_t を並べ、補間の意味を目で確認する
    eps = torch.randn_like(x1[:1])
    rows = []
    for t_val in np.linspace(0.0, 1.0, 8):
        t = torch.full((1,), float(t_val), device=x1.device)
        rows.append(interpolate(x1[:1], eps, t).cpu())
    interp = save_image_grid(
        torch.cat(rows), out_dir / "interpolation_t.png", nrow=8, title="x_t: t=0(noise) -> t=1(data)"
    )
    return {"batch_examples": str(grid), "interpolation_t": str(interp)}


def _inspect_pusht(cfg: Config, stats: dict[str, Any] | None) -> dict[str, Any]:
    train_ids, val_ids = pusht_data.split_episode_indices(
        cfg.data.root, val_episodes=cfg.data.val_episodes, seed=cfg.seed
    )
    meta_path = Path(cfg.data.root) / "meta.json"
    meta = {}
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    train_eps = pusht_data.load_episodes(cfg.data.root, train_ids[:5])
    lengths = [len(ep) for ep in train_eps]
    result: dict[str, Any] = {
        "repo_id": meta.get("repo_id"),
        "resolved_revision": meta.get("resolved_revision"),
        "num_episodes": meta.get("num_episodes"),
        "train_episodes": len(train_ids),
        "val_episodes": len(val_ids),
        "split_disjoint": len(set(train_ids) & set(val_ids)) == 0,
        "sample_episode_lengths": lengths,
    }
    if stats:
        norm = MinMaxNormalizer.from_dict(stats["action"])
        raw = torch.from_numpy(train_eps[0].action[:64])
        roundtrip = norm.denormalize(norm.normalize(raw))
        result["normalize_roundtrip_max_abs_err"] = float((roundtrip - raw).abs().max())
        result["action_min"] = stats["action"]["min"]
        result["action_max"] = stats["action"]["max"]
    return result


def _inspect_pusht_artifacts(
    cfg: Config, batch: dict[str, torch.Tensor], stats: dict[str, Any] | None, out_dir: Path
) -> dict[str, str]:
    images = batch["images"][:4]  # (B, To, C, H, W)
    flat = images.reshape(-1, *images.shape[2:])
    grid = save_image_grid(
        flat.cpu(), out_dir / "obs_examples.png", nrow=cfg.data.obs_horizon,
        title=f"observation history (To={cfg.data.obs_horizon})",
    )
    artifacts = {"obs_examples": str(grid)}
    if stats:
        from learn_dit.visualize import plot_action_trajectory

        norm = MinMaxNormalizer.from_dict(stats["action"])
        actions = norm.denormalize(batch["x"][0].cpu()).numpy()
        artifacts["dataset_trajectory"] = str(
            plot_action_trajectory(actions, out_dir / "dataset_trajectory.png")
        )
    return artifacts
