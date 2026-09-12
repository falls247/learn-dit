"""PushT 用 DiT。

入力: 直近 obs_horizon(=2) 時点の 96x96 画像 と 2 次元位置
出力: 未来 action_horizon(=16) ステップの 2 次元目標位置に対する速度場

画像は小型 CNN でトークン化し、位置特徴と合わせて Cross-Attention で参照する。
動作系列側は 1 時刻 = 1 トークン(次元 2)として Self-Attention する。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from learn_dit.models.dit import DiT


class SmallCNN(nn.Module):
    """96x96 -> 6x6 特徴マップ。GroupNorm でバッチサイズ依存を避ける。"""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 128) -> None:
        super().__init__()
        chans = [in_channels, 32, 64, 128, hidden_dim]
        layers: list[nn.Module] = []
        for i in range(4):
            layers += [
                nn.Conv2d(chans[i], chans[i + 1], kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=min(8, chans[i + 1]), num_channels=chans[i + 1]),
                nn.SiLU(),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, 96, 96) -> (B, 36, hidden_dim)"""
        h = self.net(x)
        b, c, hh, ww = h.shape
        return h.reshape(b, c, hh * ww).transpose(1, 2)


class PushTDiT(nn.Module):
    def __init__(
        self,
        action_dim: int = 2,
        action_horizon: int = 16,
        obs_horizon: int = 2,
        image_size: int = 96,
        in_channels: int = 3,
        hidden_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.image_size = image_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        self.image_encoder = SmallCNN(in_channels, hidden_dim)
        self.state_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # どの時刻の観測かを区別するための埋め込み(画像側・位置側で共有)
        self.frame_embed = nn.Parameter(torch.zeros(obs_horizon, hidden_dim))
        nn.init.trunc_normal_(self.frame_embed, std=0.02)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dit = DiT(
            token_dim=action_dim,
            num_tokens=action_horizon,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            use_cross_attn=True,
        )

    def encode_obs(self, images: torch.Tensor, agent_pos: torch.Tensor) -> torch.Tensor:
        """観測を Cross-Attention 用のトークン列(memory)にする。

        images: (B, To, C, H, W) / agent_pos: (B, To, 2)
        戻り値: (B, To*36 + To, hidden_dim)
        """
        b, to = images.shape[0], images.shape[1]
        if to != self.obs_horizon:
            raise ValueError(f"obs_horizon 不一致: {to} != {self.obs_horizon}")
        flat = images.reshape(b * to, *images.shape[2:])
        img_tokens = self.image_encoder(flat)                      # (B*To, 36, H)
        img_tokens = img_tokens.reshape(b, to, -1, self.hidden_dim)
        img_tokens = img_tokens + self.frame_embed[None, :to, None, :]
        img_tokens = img_tokens.reshape(b, -1, self.hidden_dim)

        pos_tokens = self.state_proj(agent_pos) + self.frame_embed[None, :to, :]
        memory = torch.cat([img_tokens, pos_tokens], dim=1)
        return self.memory_norm(memory)

    def forward(
        self,
        action: torch.Tensor,
        t: torch.Tensor,
        images: torch.Tensor | None = None,
        agent_pos: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """action: (B, Ta, 2) の x_t。memory を渡すと観測エンコードを再利用する。"""
        if memory is None:
            if images is None or agent_pos is None:
                raise ValueError("images と agent_pos、または memory のいずれかが必要")
            memory = self.encode_obs(images, agent_pos)
        cond = self.global_proj(memory.mean(dim=1))
        return self.dit(action, t, cond=cond, memory=memory)

    def sample_shape(self, batch: int) -> tuple[int, ...]:
        return (batch, self.action_horizon, self.action_dim)
