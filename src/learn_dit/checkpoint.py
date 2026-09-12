"""チェックポイント。モデル・Optimizer・学習ステップ・乱数状態・設定・正規化情報を保存する。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from learn_dit.config import Config, config_from_dict
from learn_dit.utils.seed import load_rng_state, rng_state

FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    cfg: Config,
    stats: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "config": cfg.to_dict(),
        "stats": stats,          # 正規化情報(PushT)
        "rng": rng_state(),
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)            # 保存中の中断で壊れたファイルを残さない
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(f"チェックポイント形式が非対応: {version}")
    payload["config_obj"] = config_from_dict(payload["config"])
    return payload


def restore_for_resume(
    payload: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = True,
) -> int:
    """再開用の復元。戻り値は再開時点の step。"""
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        load_rng_state(payload.get("rng"))
    return int(payload.get("step", 0))
