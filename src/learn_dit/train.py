"""学習ループ。時間上限付き、通常中断と再開に対応する。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from learn_dit.checkpoint import load_checkpoint, restore_for_resume, save_checkpoint
from learn_dit.config import Config
from learn_dit.data.builder import build_dataloaders, split_batch
from learn_dit.flow import flow_matching_loss
from learn_dit.models import build_model
from learn_dit.utils.device import autocast_context, resolve_device
from learn_dit.utils.logging import RunLogger, write_json
from learn_dit.utils.seed import set_seed
from learn_dit.visualize import plot_loss_curve


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate_loss(cfg: Config, model: torch.nn.Module, loader, device: torch.device, max_batches: int = 20) -> float:
    model.eval()
    total, count = 0.0, 0
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)  # 検証損失を比較可能にする
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x1, cond, mask = split_batch(cfg, batch, device)
        out = flow_matching_loss(model, x1, cond=cond, mask=mask, generator=generator)
        total += float(out["loss"].detach())
        count += 1
    model.train()
    return total / max(count, 1)


def train(
    cfg: Config,
    run_dir: str | Path,
    resume: str | Path | None = None,
    download: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    logger = RunLogger(run_dir)
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    stats = None
    if resume:
        payload = load_checkpoint(resume, map_location="cpu")
        stats = payload.get("stats")

    train_loader, val_loader, stats = build_dataloaders(cfg, download=download, stats=stats)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    start_step = 0
    if resume:
        start_step = restore_for_resume(payload, model, optimizer)
        logger.info(f"[resume] step {start_step} から再開する")

    write_json(run_dir / "config.json", cfg.to_dict())
    if stats is not None:
        write_json(run_dir / "normalization.json", stats)
    logger.info(
        f"[train] task={cfg.task} device={device} params={sum(p.numel() for p in model.parameters()):,} "
        f"train_batches={len(train_loader)} time_budget={cfg.train.time_budget_sec}s"
    )

    ckpt_dir = run_dir / "checkpoints"
    best_val = float("inf")
    started = time.time()
    stop_reason = "max_steps"
    step = start_step
    batches = _infinite(train_loader)
    model.train()

    def _save(name: str, extra: dict[str, Any] | None = None) -> Path:
        return save_checkpoint(ckpt_dir / name, model, optimizer, step, cfg, stats=stats, extra=extra)

    try:
        while step < cfg.train.max_steps:
            # 時間上限は更新ステップ間で確認する(保存による超過は許容する)
            elapsed = time.time() - started
            if elapsed >= cfg.train.time_budget_sec:
                stop_reason = "time_budget"
                break

            batch = next(batches)
            x1, cond, mask = split_batch(cfg, batch, device)
            with autocast_context(device, cfg.train.amp):
                out = flow_matching_loss(model, x1, cond=cond, mask=mask)
                loss = out["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            step += 1

            if step % cfg.train.log_every == 0 or step == start_step + 1:
                logger.log(
                    kind="train",
                    step=step,
                    loss=float(loss.detach()),
                    grad_norm=float(grad_norm),
                    elapsed_sec=round(time.time() - started, 2),
                )
                print(
                    f"step {step:>6} loss {float(loss.detach()):.5f} "
                    f"({time.time() - started:.0f}s / {cfg.train.time_budget_sec:.0f}s)",
                    flush=True,
                )
            if cfg.train.val_every and step % cfg.train.val_every == 0:
                val_loss = evaluate_loss(cfg, model, val_loader, device)
                logger.log(kind="val", step=step, loss=val_loss)
                print(f"step {step:>6} val_loss {val_loss:.5f}", flush=True)
                if val_loss < best_val:
                    best_val = val_loss
                    _save("best.pt", extra={"val_loss": val_loss})
            if cfg.train.save_every and step % cfg.train.save_every == 0:
                _save("last.pt")
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        logger.info("[train] 中断を検知。チェックポイントを保存する。")

    last_path = _save("last.pt", extra={"stop_reason": stop_reason})
    history = logger.read_history()
    curve = plot_loss_curve(history, run_dir / "loss_curve.png") if history else None
    summary = {
        "task": cfg.task,
        "steps_done": step,
        "start_step": start_step,
        "stop_reason": stop_reason,
        "elapsed_sec": round(time.time() - started, 2),
        "best_val_loss": None if best_val == float("inf") else best_val,
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(ckpt_dir / "best.pt") if (ckpt_dir / "best.pt").exists() else None,
        "loss_curve": str(curve) if curve else None,
        "device": str(device),
    }
    write_json(run_dir / "train_summary.json", summary)
    logger.info(f"[train] 終了 ({stop_reason}) step={step} 経過 {summary['elapsed_sec']}s")
    return summary
