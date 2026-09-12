from learn_dit.models.dit import DiT, DiTBlock, TimeEmbedding
from learn_dit.models.mnist_dit import MNISTDiT
from learn_dit.models.pusht_dit import PushTDiT

__all__ = ["DiT", "DiTBlock", "TimeEmbedding", "MNISTDiT", "PushTDiT", "build_model"]


def build_model(cfg):
    """設定から題材ごとのモデルを作る。"""
    m = cfg.model
    if cfg.task == "mnist":
        return MNISTDiT(
            image_size=m.image_size,
            patch_size=m.patch_size,
            in_channels=m.in_channels,
            hidden_dim=m.hidden_dim,
            depth=m.depth,
            num_heads=m.num_heads,
            num_classes=m.num_classes,
        )
    if cfg.task == "pusht":
        return PushTDiT(
            action_dim=m.action_dim,
            action_horizon=m.action_horizon,
            obs_horizon=m.obs_horizon,
            image_size=m.image_size,
            in_channels=m.in_channels,
            hidden_dim=m.hidden_dim,
            depth=m.depth,
            num_heads=m.num_heads,
        )
    raise ValueError(f"未知の task: {cfg.task}")
