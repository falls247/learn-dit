"""設定の読み込みと検証。"""

from __future__ import annotations

import pytest

from learn_dit.config import config_from_dict, default_config_path, load_config


def test_default_configs_load():
    for task in ("mnist", "pusht"):
        cfg = load_config(default_config_path(task))
        assert cfg.task == task
        assert cfg.seed == 42
        assert cfg.model.hidden_dim == 128
        assert cfg.model.depth == 4
        assert cfg.model.num_heads == 4
        assert cfg.train.batch_size == 128
        assert cfg.train.lr == pytest.approx(1e-4)
        assert cfg.train.time_budget_sec == pytest.approx(1800.0)
        assert cfg.sample.steps == 32
        assert sorted(cfg.sample.steps_sweep) == [4, 8, 16, 32]


def test_pusht_horizons_match_sow():
    cfg = load_config(default_config_path("pusht"))
    assert cfg.data.obs_horizon == 2
    assert cfg.data.action_horizon == 16
    assert cfg.data.exec_horizon == 8
    assert cfg.data.val_episodes == 21


def test_override_applies_and_casts():
    cfg = load_config(default_config_path("mnist"), ["train.batch_size=64", "train.amp=false"])
    assert cfg.train.batch_size == 64 and cfg.train.amp is False


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        config_from_dict({"task": "mnist", "train": {"nonexistent": 1}})


def test_invalid_head_split_rejected():
    with pytest.raises(ValueError):
        config_from_dict({"task": "mnist", "model": {"hidden_dim": 130, "num_heads": 4}})
