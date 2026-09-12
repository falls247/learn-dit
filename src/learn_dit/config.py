"""設定管理。YAML を dataclass に読み込み、CLI からの上書きを許す。

pydantic を使わず dataclass + 明示的な検証にしている理由は
dev/adr/ADR_20260912_small_standalone_impl.md を参照。
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

TASKS = ("mnist", "pusht")


@dataclass
class DataConfig:
    root: str = "data"
    # MNIST
    val_size: int = 5000
    # PushT
    repo_id: str = "lerobot/pusht"
    revision: str = "main"
    val_episodes: int = 21
    obs_horizon: int = 2
    action_horizon: int = 16
    exec_horizon: int = 8
    image_size: int = 96


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    # MNIST
    patch_size: int = 4
    image_size: int = 28
    in_channels: int = 1
    num_classes: int = 10
    # PushT
    action_dim: int = 2
    obs_horizon: int = 2
    action_horizon: int = 16


@dataclass
class TrainConfig:
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 30000
    time_budget_sec: float = 1800.0
    grad_clip: float = 1.0
    log_every: int = 50
    val_every: int = 500
    save_every: int = 500
    num_workers: int = 4
    amp: bool = True


@dataclass
class SampleConfig:
    steps: int = 32
    steps_sweep: list[int] = field(default_factory=lambda: [4, 8, 16, 32])
    num_per_class: int = 8
    intermediate_snapshots: int = 8


@dataclass
class EvalConfig:
    num_seeds: int = 20
    max_env_steps: int = 300
    fps: int = 10
    save_video: bool = True


@dataclass
class Config:
    task: str = "mnist"
    seed: int = 42
    device: str = "auto"
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "sample": SampleConfig,
    "eval": EvalConfig,
}


def _build_section(cls: type, raw: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"{cls.__name__} に未知のキー: {sorted(unknown)}")
    kwargs = {}
    for key, value in raw.items():
        kwargs[key] = _coerce(known[key].type, value)
    return cls(**kwargs)


def _coerce(type_repr: Any, value: Any) -> Any:
    """YAML / CLI 文字列を dataclass の型に寄せる(最小限)。"""
    name = type_repr if isinstance(type_repr, str) else getattr(type_repr, "__name__", str(type_repr))
    if value is None:
        return None
    if name.startswith("list"):
        if isinstance(value, str):
            return [int(v) for v in value.strip("[]").split(",") if v.strip()]
        return list(value)
    if name == "bool":
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)
    if name == "int":
        return int(value)
    if name == "float":
        return float(value)
    if name == "str":
        return str(value)
    return value


def config_from_dict(raw: dict[str, Any]) -> Config:
    raw = copy.deepcopy(raw)
    sections = {}
    for key, cls in _SECTIONS.items():
        sections[key] = _build_section(cls, raw.pop(key, {}) or {})
    top = {f.name: f for f in fields(Config) if f.name not in _SECTIONS}
    unknown = set(raw) - set(top)
    if unknown:
        raise ValueError(f"設定の最上位に未知のキー: {sorted(unknown)}")
    cfg = Config(**{k: _coerce(top[k].type, v) for k, v in raw.items()}, **sections)
    validate(cfg)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """YAML を読み、`train.lr=1e-3` 形式の上書きを適用する。"""
    with open(path, encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set は key=value 形式で指定する: {item!r}")
        key, value = item.split("=", 1)
        _apply_override(raw, key.strip(), value.strip())
    return config_from_dict(raw)


def _apply_override(raw: dict[str, Any], dotted: str, value: str) -> None:
    parts = dotted.split(".")
    node = raw
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"上書きできない設定パス: {dotted}")
    node[parts[-1]] = value


def validate(cfg: Config) -> None:
    if cfg.task not in TASKS:
        raise ValueError(f"task は {TASKS} のいずれか: {cfg.task!r}")
    if cfg.model.hidden_dim % cfg.model.num_heads != 0:
        raise ValueError("hidden_dim は num_heads で割り切れる必要がある")
    if cfg.task == "mnist" and cfg.model.image_size % cfg.model.patch_size != 0:
        raise ValueError("image_size は patch_size で割り切れる必要がある")
    if cfg.task == "pusht" and cfg.data.exec_horizon > cfg.data.action_horizon:
        raise ValueError("exec_horizon は action_horizon 以下にする")
    if cfg.train.batch_size <= 0 or cfg.train.max_steps <= 0:
        raise ValueError("batch_size と max_steps は正の値にする")
    if min(cfg.sample.steps_sweep + [cfg.sample.steps]) <= 0:
        raise ValueError("Euler ステップ数は正の値にする")


def default_config_path(task: str) -> Path:
    """リポジトリ同梱の configs/ を探し、無ければカレントディレクトリを見る。"""
    candidates = [
        Path(__file__).resolve().parents[2] / "configs" / f"{task}.yaml",
        Path.cwd() / "configs" / f"{task}.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"設定ファイルが見つからない: {[str(p) for p in candidates]}。--config で明示する。"
    )
