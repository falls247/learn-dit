"""保存・再開の検証(SOW 受入条件: 保存後の推論と追加学習)。"""

from __future__ import annotations

import torch

from learn_dit.checkpoint import load_checkpoint, restore_for_resume, save_checkpoint
from learn_dit.config import config_from_dict
from learn_dit.flow import euler_sample, flow_matching_loss
from learn_dit.models import build_model


def _cfg():
    return config_from_dict(
        {
            "task": "mnist",
            "seed": 42,
            "device": "cpu",
            "model": {"hidden_dim": 32, "depth": 2, "num_heads": 4},
            "train": {"batch_size": 4, "max_steps": 10},
        }
    )


def test_save_and_load_reproduces_weights_and_step(tmp_path):
    cfg = _cfg()
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(4, 1, 28, 28)
    label = torch.zeros(4, dtype=torch.long)
    out = flow_matching_loss(model, x, cond={"label": label})
    out["loss"].backward()
    opt.step()

    path = save_checkpoint(tmp_path / "ckpt.pt", model, opt, step=7, cfg=cfg, stats=None)
    payload = load_checkpoint(path)
    assert payload["step"] == 7
    assert payload["config_obj"].model.hidden_dim == 32

    restored = build_model(cfg)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    step = restore_for_resume(payload, restored, restored_opt)
    assert step == 7
    for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(a, b)


def test_inference_after_load_is_identical(tmp_path):
    cfg = _cfg()
    model = build_model(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.01)   # ゼロ初期化から動かして差が出る状態にする
    path = save_checkpoint(tmp_path / "ckpt.pt", model, None, step=1, cfg=cfg)
    restored = build_model(cfg)
    restore_for_resume(load_checkpoint(path), restored)

    label = torch.arange(4)
    a, _ = euler_sample(model.eval(), model.sample_shape(4), steps=4,
                        cond={"label": label}, generator=torch.Generator().manual_seed(0))
    b, _ = euler_sample(restored.eval(), restored.sample_shape(4), steps=4,
                        cond={"label": label}, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(a, b, atol=1e-6)


def test_resume_allows_further_training(tmp_path):
    cfg = _cfg()
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    x = torch.randn(4, 1, 28, 28)
    label = torch.zeros(4, dtype=torch.long)
    for _ in range(5):
        out = flow_matching_loss(model, x, cond={"label": label})
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()
    loss_at_save = float(flow_matching_loss(model, x, cond={"label": label})["loss"])

    path = save_checkpoint(tmp_path / "ckpt.pt", model, opt, step=5, cfg=cfg)
    resumed = build_model(cfg)
    resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=2e-3)
    step = restore_for_resume(load_checkpoint(path), resumed, resumed_opt)

    losses = []
    for _ in range(20):
        out = flow_matching_loss(resumed, x, cond={"label": label})
        resumed_opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        resumed_opt.step()
        losses.append(float(out["loss"]))
    assert step == 5
    assert min(losses) < loss_at_save    # 再開後も学習が進む


def test_rng_state_restoration_reproduces_noise(tmp_path):
    cfg = _cfg()
    model = build_model(cfg)
    torch.manual_seed(123)
    _ = torch.randn(10)
    path = save_checkpoint(tmp_path / "ckpt.pt", model, None, step=3, cfg=cfg)
    expected = torch.randn(5)

    payload = load_checkpoint(path)
    restore_for_resume(payload, build_model(cfg), None, restore_rng=True)
    assert torch.equal(torch.randn(5), expected)
