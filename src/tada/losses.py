"""Semantic hard-negative mining and ranking loss used by TADA."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .heo2026_benchmark import unit


def mine_hard_negatives(
    target: np.ndarray, indices: np.ndarray, labels: np.ndarray, count: int,
) -> np.ndarray:
    """Return local outer-train indices of semantically nearest wrong words."""
    vectors = unit(target[indices])
    similarity = vectors @ vectors.T
    local_labels = labels[indices]
    similarity[local_labels[:, None] == local_labels[None, :]] = -np.inf
    count = min(count, max(1, len(indices) - 1))
    partition = np.argpartition(similarity, -count, axis=1)[:, -count:]
    values = np.take_along_axis(similarity, partition, axis=1)
    order = np.argsort(values, axis=1)[:, ::-1]
    return np.take_along_axis(partition, order, axis=1).astype(np.int64)


def hard_negative_loss(
    prediction: torch.Tensor, positive_target: torch.Tensor,
    negative_target: torch.Tensor, temperature: float,
) -> torch.Tensor:
    prediction = F.normalize(prediction, dim=-1)
    positive = torch.sum(prediction * F.normalize(positive_target, dim=-1), dim=-1)
    negative = torch.einsum(
        "bd,bkd->bk", prediction, F.normalize(negative_target, dim=-1)
    )
    logits = torch.cat([positive[:, None], negative], dim=1) / temperature
    return F.cross_entropy(
        logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    )
