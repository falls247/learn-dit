"""共通 DiT。MNIST(画像)と PushT(動作系列)で同じ実装を使う。

構成(SOW 第3章):
  - 4 ブロック / 隠れ次元 128 / 4 ヘッド
  - 学習可能な位置埋め込み
  - Self-Attention
  - 時刻 t の AdaLN(AdaLN-Zero: 初期状態では残差に何も足さない)
  - 観測条件を参照するための Cross-Attention(PushT のみ有効化)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_embedding(t: torch.Tensor, dim: int, scale: float = 1000.0) -> torch.Tensor:
    """t in [0,1] の (B,) を (B, dim) の正弦波埋め込みにする。"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
    )
    args = t.float().unsqueeze(-1) * scale * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeEmbedding(nn.Module):
    """時刻 t を条件ベクトルへ変換する。"""

    def __init__(self, hidden_dim: int, freq_dim: int = 128) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(t, self.freq_dim).to(self.mlp[0].weight.dtype))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN: 正規化後に条件由来の scale / shift を掛ける。"""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        if use_cross_attn:
            self.norm_cross = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        # AdaLN-Zero: shift, scale, gate を各サブ層ごとに条件から作る
        self.num_mod = 9 if use_cross_attn else 6
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, self.num_mod * hidden_dim))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        params = self.modulation(c).chunk(self.num_mod, dim=-1)
        shift_sa, scale_sa, gate_sa = params[0], params[1], params[2]
        if self.use_cross_attn:
            shift_ca, scale_ca, gate_ca = params[3], params[4], params[5]
            shift_mlp, scale_mlp, gate_mlp = params[6], params[7], params[8]
        else:
            shift_mlp, scale_mlp, gate_mlp = params[3], params[4], params[5]

        h = modulate(self.norm1(x), shift_sa, scale_sa)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_sa.unsqueeze(1) * attn_out

        if self.use_cross_attn:
            if memory is None:
                raise ValueError("Cross-Attention 有効時は memory(観測特徴)が必要")
            h = modulate(self.norm_cross(x), shift_ca, scale_ca)
            cross_out, _ = self.cross_attn(h, memory, memory, need_weights=False)
            x = x + gate_ca.unsqueeze(1) * cross_out

        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        return x + gate_mlp.unsqueeze(1) * self.mlp(h)


class DiT(nn.Module):
    """トークン列 -> 速度場。入出力次元はトークン当たりの次元。"""

    def __init__(
        self,
        token_dim: int,
        num_tokens: int,
        hidden_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        use_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(token_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.time_embed = TimeEmbedding(hidden_dim)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads, use_cross_attn=use_cross_attn) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, token_dim)
        nn.init.zeros_(self.final_modulation[1].weight)
        nn.init.zeros_(self.final_modulation[1].bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.shape[1] != self.num_tokens:
            raise ValueError(f"トークン数が不一致: {tokens.shape[1]} != {self.num_tokens}")
        c = self.time_embed(t)
        if cond is not None:
            c = c + cond
        x = self.in_proj(tokens) + self.pos_embed
        for block in self.blocks:
            x = block(x, c, memory=memory)
        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        return self.out_proj(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
