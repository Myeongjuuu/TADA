"""Residual temporal encoders and adaptive multi-window fusion."""
from __future__ import annotations

import torch
from torch import nn


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1),
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.block(values))


class TemporalConvEncoder(nn.Module):
    """Map [batch, electrode, time] ECoG to one word-level embedding."""

    def __init__(
        self,
        n_channels: int,
        output_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv1d(n_channels, hidden_dim, 1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            *[
                ResidualTemporalBlock(hidden_dim, dilation=2**index, dropout=dropout)
                for index in range(depth)
            ]
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, ecog: torch.Tensor) -> torch.Tensor:
        hidden = self.temporal(self.spatial(ecog))
        return self.projection(hidden.mean(dim=-1))


class MultiWindowTemporalEncoder(nn.Module):
    """Fuse genuinely different word-locked ECoG windows.

    Input has shape ``[batch, window, shared_component, time]``.  A shared
    encoder is deliberately used for all windows: the comparison should test
    the information supplied by different temporal extents, rather than give
    every extent an unrelated high-capacity network.  A sample-dependent gate
    then decides how much each window contributes to the decoded embedding.
    """

    def __init__(
        self,
        n_channels: int,
        output_dim: int,
        n_windows: int = 3,
        hidden_dim: int = 96,
        depth: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_windows = n_windows
        self.encoder = TemporalConvEncoder(
            n_channels=n_channels,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )
        self.window_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_windows)]
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * n_windows, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_windows),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, ecog: torch.Tensor, return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if ecog.ndim != 4 or ecog.shape[1] != self.n_windows:
            raise ValueError(
                f"expected [batch, {self.n_windows}, channel, time], got "
                f"{tuple(ecog.shape)}"
            )
        representations = torch.stack(
            [
                self.window_norms[index](self.encoder(ecog[:, index]))
                for index in range(self.n_windows)
            ],
            dim=1,
        )
        logits = self.fusion_gate(representations.flatten(start_dim=1))
        weights = torch.softmax(logits, dim=1)
        fused = torch.sum(representations * weights.unsqueeze(-1), dim=1)
        prediction = self.output(fused)
        if return_details:
            return prediction, {
                "window_weights": weights,
                "window_representations": representations,
            }
        return prediction
