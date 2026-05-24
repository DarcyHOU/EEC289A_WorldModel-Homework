"""Student world model.

Stronger residual GRU world model for InvertedPendulum-v5.
The model predicts normalized delta state, then the provided predict_next()
function adds the delta to the current state.
"""

from __future__ import annotations

import torch
from torch import nn


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 4,
        use_gru: bool = True,
        delta_limit: float = 2.0,
    ):
        super().__init__()

        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)

        in_dim = obs_dim + act_dim

        # Input encoder
        layers: list[nn.Module] = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(int(num_layers) - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())

        self.encoder = nn.Sequential(*layers)

        # GRU memory for dynamics
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None

        # Deeper prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        x = torch.cat([obs_norm, act_norm], dim=-1)

        feat = self.encoder(x)

        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(feat, hidden)
            feat = hidden

        raw_delta = self.head(feat)

        # Limit delta to prevent unstable rollout explosion
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)

        return delta, hidden
