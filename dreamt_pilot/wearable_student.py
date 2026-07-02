"""WearableSleepFM student network."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_dreamt import EMBEDDING_DIM, WEARABLE_IN_CHANNELS


class WearableSleepFM(nn.Module):
    def __init__(
        self,
        in_channels: int = WEARABLE_IN_CHANNELS,
        emb_dim: int = EMBEDDING_DIM,
        lstm_hidden: int = 64,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(lstm_hidden * 2, emb_dim)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Helper."""
        h = self.conv(x)  # (B, 64, L)
        h = h.transpose(1, 2)  # (B, L, 64)
        h, _ = self.lstm(h)
        h = h.mean(dim=1)
        z = self.fc(h)
        if normalize:
            z = F.normalize(z, dim=1, eps=1e-8)
        return z


def cosine_distill_loss(z_student: torch.Tensor, z_teacher: torch.Tensor) -> torch.Tensor:
    """Helper."""
    z_t = F.normalize(z_teacher, dim=1, eps=1e-8)
    return 1.0 - (z_student * z_t).sum(dim=1).mean()
