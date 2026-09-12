"""データ整合性の検証: 分割独立性、正規化往復、時刻対応、系列マスク。"""

from __future__ import annotations

import numpy as np
import torch

from learn_dit.data.normalize import MinMaxNormalizer, image_to_unit_range, unit_range_to_image
from learn_dit.data.pusht import Episode, PushTSequenceDataset, compute_stats, split_episode_indices
from learn_dit.utils.seed import split_indices


# --- 分割 -----------------------------------------------------------------------------
def test_split_is_deterministic_and_disjoint():
    train_a, val_a = split_indices(60000, 5000, seed=42)
    train_b, val_b = split_indices(60000, 5000, seed=42)
    assert train_a == train_b and val_a == val_b          # 再現性
    assert len(train_a) == 55000 and len(val_a) == 5000   # SOW の件数
    assert not set(train_a) & set(val_a)                  # 独立性
    assert len(set(train_a) | set(val_a)) == 60000        # 全件を使う


def test_split_changes_with_seed():
    _, val_42 = split_indices(1000, 100, seed=42)
    _, val_7 = split_indices(1000, 100, seed=7)
    assert val_42 != val_7


def test_episode_split_206_to_185_and_21(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    for ep in range(206):
        np.savez(processed / f"episode_{ep:05d}.npz", image=np.zeros(1), state=np.zeros(1), action=np.zeros(1))
    train_ids, val_ids = split_episode_indices(tmp_path, val_episodes=21, seed=42)
    assert len(train_ids) == 185 and len(val_ids) == 21
    assert not set(train_ids) & set(val_ids)


# --- 正規化 ---------------------------------------------------------------------------
def test_minmax_roundtrip():
    data = np.random.default_rng(0).uniform(-30, 512, size=(500, 2))
    norm = MinMaxNormalizer.from_array(data)
    x = torch.tensor(data, dtype=torch.float32)
    y = norm.normalize(x)
    assert float(y.min()) >= -1.0 - 1e-5 and float(y.max()) <= 1.0 + 1e-5
    assert torch.allclose(norm.denormalize(y), x, atol=1e-2)


def test_minmax_constant_dimension_is_safe():
    norm = MinMaxNormalizer(min=[1.0, 0.0], max=[1.0, 1.0])
    x = torch.tensor([[1.0, 0.5]])
    y = norm.normalize(x)
    assert torch.isfinite(y).all()
    assert torch.allclose(norm.denormalize(y)[0, 1], torch.tensor(0.5), atol=1e-5)


def test_normalizer_serialization_roundtrip():
    norm = MinMaxNormalizer(min=[0.0, 1.0], max=[2.0, 3.0])
    restored = MinMaxNormalizer.from_dict(norm.to_dict())
    assert restored == norm


def test_image_range_conversion():
    images = torch.tensor([[[[0.0, 1.0]]]])
    assert torch.allclose(image_to_unit_range(images), torch.tensor([[[[-1.0, 1.0]]]]))
    assert torch.allclose(unit_range_to_image(image_to_unit_range(images)), images)


# --- 系列データセット -----------------------------------------------------------------
def _fake_episodes(lengths=(5, 7)):
    episodes = []
    for i, length in enumerate(lengths):
        # 画素値に時刻を埋め込み、時刻対応を検証できるようにする
        image = np.stack([np.full((3, 96, 96), t, dtype=np.uint8) for t in range(length)])
        state = np.stack([np.array([t, t + 100], dtype=np.float32) for t in range(length)])
        action = np.stack([np.array([t * 2, t * 2 + 1], dtype=np.float32) for t in range(length)])
        episodes.append(Episode(index=i, image=image, state=state, action=action))
    return episodes


def _identity_stats():
    return {
        "state": {"type": "minmax", "min": [-1.0, -1.0], "max": [1.0, 1.0]},
        "action": {"type": "minmax", "min": [-1.0, -1.0], "max": [1.0, 1.0]},
    }


def test_dataset_length_equals_total_frames():
    ds = PushTSequenceDataset(_fake_episodes((5, 7)), _identity_stats())
    assert len(ds) == 12


def test_history_padding_uses_first_observation():
    ds = PushTSequenceDataset(_fake_episodes((5,)), _identity_stats(), obs_horizon=2)
    first = ds[0]
    # t=0 では履歴が無いので先頭観測を 2 回使う
    assert torch.allclose(first["images"][0], first["images"][1])
    second = ds[1]
    assert not torch.allclose(second["images"][0], second["images"][1])


def test_observation_timestamps_align():
    """画素値に埋めた時刻と位置の時刻が一致する(動画と位置・動作の時刻対応)。"""
    ds = PushTSequenceDataset(_fake_episodes((6,)), _identity_stats(), obs_horizon=2)
    sample = ds[3]
    # image は [-1,1] に正規化済みなので元の値へ戻す
    values = [round(float((img[0, 0, 0] + 1) / 2 * 255)) for img in sample["images"]]
    assert values == [2, 3]
    assert sample["timestep"].item() == 3


def test_future_mask_excludes_padding_and_stays_in_episode():
    ds = PushTSequenceDataset(_fake_episodes((5,)), _identity_stats(), obs_horizon=2, action_horizon=16)
    last = ds[4]                                   # 最終時刻 -> 未来は 1 ステップだけ有効
    assert last["mask"].sum().item() == 1.0
    first = ds[0]
    assert first["mask"].sum().item() == 5.0       # エピソード長を超えない


def test_sequences_do_not_cross_episode_boundary():
    episodes = _fake_episodes((4, 4))
    ds = PushTSequenceDataset(episodes, _identity_stats(), obs_horizon=2, action_horizon=4)
    last_of_first = ds[3]
    assert last_of_first["episode_index"].item() == 0
    # 動作の有効部分はエピソード 0 の値のみ(次エピソードの値が混ざらない)
    valid = last_of_first["mask"].squeeze(-1) > 0
    actions = last_of_first["x"][valid]
    assert actions.shape[0] == 1


def test_stats_computed_from_given_episodes_only():
    stats = compute_stats(_fake_episodes((5,)))
    assert stats["action"]["min"] == [0.0, 1.0]
    assert stats["action"]["max"] == [8.0, 9.0]
