"""評価。

MNIST: 検証損失と Euler ステップ数ごとの生成物を記録する。
PushT: gym-pusht 上で動作を実行し、未学習 / 学習後を同じ初期化 seed で比較する。
       未来 16 ステップを生成し、先頭 exec_horizon ステップを実行して再観測する。
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from learn_dit.config import Config
from learn_dit.data.builder import build_dataloaders
from learn_dit.data.normalize import MinMaxNormalizer, image_to_unit_range
from learn_dit.flow import euler_sample
from learn_dit.sample import load_model, run_sample
from learn_dit.train import evaluate_loss
from learn_dit.utils.device import resolve_device
from learn_dit.utils.logging import write_json
from learn_dit.utils.seed import set_seed
from learn_dit.visualize import save_video


# --------------------------------------------------------------------------------------
# MNIST
# --------------------------------------------------------------------------------------
def evaluate_mnist(cfg: Config, checkpoint: str | Path | None, out_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    model, _, step = load_model(checkpoint, cfg, device)
    _, val_loader, _ = build_dataloaders(cfg, download=True)
    val_loss = evaluate_loss(cfg, model, val_loader, device, max_batches=40)
    sample_result = run_sample(cfg, checkpoint, out_dir)
    result = {
        "task": "mnist",
        "checkpoint": str(checkpoint),
        "step": step,
        "val_loss": val_loss,
        "sample": sample_result,
    }
    write_json(out_dir / "evaluation.json", result)
    return result


# --------------------------------------------------------------------------------------
# PushT
# --------------------------------------------------------------------------------------
def _make_pusht_env(cfg: Config, render: bool):
    try:
        import gymnasium as gym
        import gym_pusht  # noqa: F401 - 環境登録のために import する
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "gym-pusht / gymnasium が無い。`uv sync --extra pusht` を実行する。"
        ) from exc
    return gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array" if render else None,
        observation_width=cfg.data.image_size,
        observation_height=cfg.data.image_size,
    )


def _obs_to_tensors(
    history: deque,
    state_norm: MinMaxNormalizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """観測履歴を (1, To, C, H, W) と (1, To, 2) にする。"""
    pixels = np.stack([np.asarray(o["pixels"]) for o in history])        # (To, H, W, C)
    images = torch.from_numpy(pixels).permute(0, 3, 1, 2).float()       # (To, C, H, W)
    images = image_to_unit_range(images / 255.0).unsqueeze(0).to(device)
    states = torch.from_numpy(np.stack([np.asarray(o["agent_pos"], dtype=np.float32) for o in history]))
    states = state_norm.normalize(states).unsqueeze(0).to(device)
    return images, states


@torch.no_grad()
def rollout_pusht(
    cfg: Config,
    model: torch.nn.Module,
    stats: dict[str, Any],
    seed: int,
    render: bool = False,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    action_norm = MinMaxNormalizer.from_dict(stats["action"])
    state_norm = MinMaxNormalizer.from_dict(stats["state"])
    env = _make_pusht_env(cfg, render=render)
    obs, _ = env.reset(seed=seed)

    history = deque([obs] * cfg.data.obs_horizon, maxlen=cfg.data.obs_horizon)
    frames: list[np.ndarray] = []
    max_reward, total_steps, success = 0.0, 0, False
    inference_times: list[float] = []
    generator = torch.Generator(device="cpu").manual_seed(seed)

    while total_steps < cfg.eval.max_env_steps:
        images, states = _obs_to_tensors(history, state_norm, device)
        started = time.time()
        memory = model.encode_obs(images, states)
        actions, _ = euler_sample(
            model,
            model.sample_shape(1),
            steps=cfg.sample.steps,
            device=device,
            cond={"memory": memory},
            generator=generator,
        )
        inference_times.append(time.time() - started)
        plan = action_norm.denormalize(actions[0].cpu()).numpy()

        done = False
        for k in range(cfg.data.exec_horizon):
            action = np.clip(plan[k], env.action_space.low, env.action_space.high).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            history.append(obs)
            max_reward = max(max_reward, float(reward))
            success = success or bool(info.get("is_success", False))
            total_steps += 1
            if render:
                frames.append(np.asarray(env.render()))
            if terminated or truncated or total_steps >= cfg.eval.max_env_steps:
                done = True
                break
        if done:
            break
    env.close()
    return {
        "seed": seed,
        "success": success,
        "max_reward": round(max_reward, 4),
        "env_steps": total_steps,
        "mean_inference_sec": round(statistics.fmean(inference_times), 4) if inference_times else None,
        "num_predictions": len(inference_times),
        "frames": frames,
    }


def evaluate_pusht(
    cfg: Config,
    checkpoint: str | Path | None,
    out_dir: str | Path,
    compare_untrained: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    trained, stats, step = load_model(checkpoint, cfg, device)
    if stats is None:
        _, _, stats = build_dataloaders(cfg, download=False)
    seeds = [cfg.seed + i for i in range(cfg.eval.num_seeds)]  # 未学習・学習後で共有する

    variants: dict[str, torch.nn.Module] = {"trained": trained}
    if compare_untrained:
        set_seed(cfg.seed)  # 未学習モデルの初期化も seed 固定
        untrained, _, _ = load_model(None, cfg, device)
        variants["untrained"] = untrained

    results: dict[str, Any] = {
        "task": "pusht",
        "checkpoint": str(checkpoint),
        "step": step,
        "euler_steps": cfg.sample.steps,
        "exec_horizon": cfg.data.exec_horizon,
        "seeds": seeds,
        "variants": {},
    }

    for name, model in variants.items():
        episodes = []
        for i, seed in enumerate(seeds):
            render = bool(cfg.eval.save_video) and i < 2  # 最初の 2 本だけ動画にする
            episode = rollout_pusht(cfg, model, stats, seed, render=render)
            frames = episode.pop("frames")
            if render and frames:
                video = save_video(frames, out_dir / f"{name}_seed{seed}.mp4", fps=cfg.eval.fps)
                episode["video"] = str(video) if video else None
            episodes.append(episode)
            print(
                f"[{name}] seed={seed} success={episode['success']} "
                f"max_reward={episode['max_reward']:.3f}",
                flush=True,
            )
        results["variants"][name] = {
            "episodes": episodes,
            "success_rate": round(sum(e["success"] for e in episodes) / len(episodes), 4),
            "mean_max_reward": round(statistics.fmean(e["max_reward"] for e in episodes), 4),
            "max_max_reward": round(max(e["max_reward"] for e in episodes), 4),
            "mean_inference_sec": round(
                statistics.fmean([e["mean_inference_sec"] for e in episodes if e["mean_inference_sec"]]), 4
            )
            if any(e["mean_inference_sec"] for e in episodes)
            else None,
        }

    write_json(out_dir / "evaluation.json", results)
    return results


def run_evaluate(
    cfg: Config,
    checkpoint: str | Path | None,
    out_dir: str | Path,
    compare_untrained: bool = True,
) -> dict[str, Any]:
    if cfg.task == "mnist":
        return evaluate_mnist(cfg, checkpoint, out_dir)
    return evaluate_pusht(cfg, checkpoint, out_dir, compare_untrained=compare_untrained)
