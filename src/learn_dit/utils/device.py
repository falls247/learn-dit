"""デバイス選択。CUDA Toolkit の自動更新はしない(SOW 第3章)。"""

from __future__ import annotations

import torch


def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("cuda が指定されたが利用できない。`learn-dit doctor` で環境を確認する。")
    return torch.device(spec)


def autocast_context(device: torch.device, enabled: bool):
    """bf16 autocast。CPU や無効時は何もしない。"""
    use = bool(enabled) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use)
