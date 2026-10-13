"""Mortal-inspired Q network for HEART's classic-v0 action space."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels, momentum=0.01),
            nn.Mish(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels, momentum=0.01),
        )
        self.activation = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class HeartQNetwork(nn.Module):
    """Shared residual card encoder, phase-specific Q heads, and rank auxiliary."""

    def __init__(self, input_channels: int, channels: int = 128, blocks: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels, momentum=0.01),
            nn.Mish(),
            *(ResidualBlock(channels) for _ in range(blocks)),
            nn.Conv1d(channels, 32, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Flatten(),
            nn.Linear(32 * 52, 512),
            nn.Mish(),
        )
        self.pass_head = nn.Linear(512, 286)
        self.play_head = nn.Linear(512, 52)
        self.rank_head = nn.Linear(512, 4)

    def forward(self, observation: torch.Tensor):
        latent = self.encoder(observation)
        return torch.cat((self.pass_head(latent), self.play_head(latent)), dim=-1), self.rank_head(latent)
