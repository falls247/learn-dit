"""乱数の一括設定と状態の保存・復元(再開のため)。"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(n: int, n_val: int, seed: int) -> tuple[list[int], list[int]]:
    """`n` 件を seed 固定で train / val に分割する。

    - 同じ seed なら常に同じ分割になる(再現性)
    - train と val は重複しない(分割独立性)
    """
    if not 0 <= n_val <= n:
        raise ValueError(f"n_val は 0..{n} の範囲: {n_val}")
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    n_train = n - n_val
    return perm[:n_train], perm[n_train:]


def rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(_as_tuple(state["python"]))
    if "numpy" in state:
        np.random.set_state(_as_tuple(state["numpy"]))
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in state["cuda"]])
        except RuntimeError:
            # GPU 台数が変わった場合は CUDA 側の状態復元のみ諦める
            pass


def _as_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_as_tuple(v) for v in value)
    return value
