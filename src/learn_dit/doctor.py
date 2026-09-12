"""環境診断。GPU の有無と CPU スモークテストを確認する。CUDA Toolkit の更新は行わない。"""

from __future__ import annotations

import platform
import sys
import time
from typing import Any

import torch

from learn_dit.flow import euler_sample, flow_matching_loss
from learn_dit.models.mnist_dit import MNISTDiT


def collect_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    "capability": f"{props.major}.{props.minor}",
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                }
            )
    return info


def smoke_test(device: str = "cpu", steps: int = 10) -> dict[str, Any]:
    """小さな DiT で 1 バッチ学習と 4 ステップ生成が通るか確認する。"""
    dev = torch.device(device)
    torch.manual_seed(0)
    model = MNISTDiT(hidden_dim=32, depth=2, num_heads=4).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(4, 1, 28, 28, device=dev)
    label = torch.randint(0, 10, (4,), device=dev)

    losses = []
    started = time.time()
    for _ in range(steps):
        out = flow_matching_loss(model, x, cond={"label": label})
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()
        losses.append(float(out["loss"].detach()))
    sample, _ = euler_sample(
        model, (2, 1, 28, 28), steps=4, device=dev, cond={"label": label[:2]}
    )
    return {
        "device": str(dev),
        "losses": losses,
        "loss_decreased": sum(losses[-3:]) < sum(losses[:3]),
        "sample_shape": list(sample.shape),
        "sample_finite": bool(torch.isfinite(sample).all()),
        "elapsed_sec": round(time.time() - started, 3),
    }


def run_doctor(gpu: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {"environment": collect_environment()}
    report["cpu_smoke_test"] = smoke_test("cpu")
    if gpu and torch.cuda.is_available():
        report["gpu_smoke_test"] = smoke_test("cuda")
    else:
        report["gpu_smoke_test"] = {"skipped": True, "reason": "CUDA が利用できない"}
    return report
