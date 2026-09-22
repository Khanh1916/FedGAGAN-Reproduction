from __future__ import annotations

import numpy as np


def weighted_average_weights(
    weighted_models: list[tuple[list[np.ndarray], int]],
) -> list[np.ndarray]:
    """Sample-count-weighted Federated Averaging over model tensors."""
    if not weighted_models:
        raise ValueError("No client weights to aggregate")
    total = sum(size for _, size in weighted_models)
    if total <= 0:
        raise ValueError("Client sample counts must be positive")
    layer_count = len(weighted_models[0][0])
    if any(len(weights) != layer_count for weights, _ in weighted_models):
        raise ValueError("All models must have the same number of tensors")
    return [
        sum(weights[layer] * (size / total) for weights, size in weighted_models)
        for layer in range(layer_count)
    ]

