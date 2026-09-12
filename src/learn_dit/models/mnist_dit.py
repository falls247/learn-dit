"""MNIST 用 DiT。28x28 を 4x4 パッチへ分割し、数字ラベルで条件付ける。VAE は使わない。"""

from __future__ import annotations

import torch
import torch.nn as nn

from learn_dit.models.dit import DiT


def patchify(x: torch.Tensor, patch: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, C*patch*patch)"""
    b, c, h, w = x.shape
    if h % patch or w % patch:
        raise ValueError(f"画像サイズ {h}x{w} が patch={patch} で割り切れない")
    gh, gw = h // patch, w // patch
    x = x.reshape(b, c, gh, patch, gw, patch)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.reshape(b, gh * gw, c * patch * patch)


def unpatchify(tokens: torch.Tensor, patch: int, channels: int, height: int, width: int) -> torch.Tensor:
    """patchify の逆変換。(B, N, C*p*p) -> (B, C, H, W)"""
    b = tokens.shape[0]
    gh, gw = height // patch, width // patch
    x = tokens.reshape(b, gh, gw, channels, patch, patch)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.reshape(b, channels, height, width)


class MNISTDiT(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        hidden_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        grid = image_size // patch_size
        token_dim = in_channels * patch_size * patch_size
        self.label_embed = nn.Embedding(num_classes, hidden_dim)
        nn.init.normal_(self.label_embed.weight, std=0.02)
        self.dit = DiT(
            token_dim=token_dim,
            num_tokens=grid * grid,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            use_cross_attn=False,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 28, 28) の x_t、返り値は同じ形の速度場。"""
        tokens = patchify(x, self.patch_size)
        cond = self.label_embed(label.long())
        out = self.dit(tokens, t, cond=cond)
        return unpatchify(out, self.patch_size, self.in_channels, self.image_size, self.image_size)

    def sample_shape(self, batch: int) -> tuple[int, ...]:
        return (batch, self.in_channels, self.image_size, self.image_size)
