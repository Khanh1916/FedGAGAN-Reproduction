from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def histogram_kl(real: np.ndarray, fake: np.ndarray, bins: int = 20) -> float:
    if real.shape[1] != fake.shape[1]:
        raise ValueError("real and fake must have the same number of columns")
    eps = 1e-8
    scores = []
    for column in range(real.shape[1]):
        lower = float(min(real[:, column].min(), fake[:, column].min()))
        upper = float(max(real[:, column].max(), fake[:, column].max()))
        if np.isclose(lower, upper):
            scores.append(0.0)
            continue
        p, edges = np.histogram(real[:, column], bins=bins, range=(lower, upper))
        q, _ = np.histogram(fake[:, column], bins=edges)
        p = p.astype(float) + eps
        q = q.astype(float) + eps
        p /= p.sum()
        q /= q.sum()
        scores.append(float(np.sum(p * np.log(p / q))))
    return float(np.mean(scores))


def correlation_distances(real: np.ndarray, fake: np.ndarray) -> tuple[float, float]:
    real_corr = np.nan_to_num(np.corrcoef(real, rowvar=False))
    fake_corr = np.nan_to_num(np.corrcoef(fake, rowvar=False))
    difference = real_corr - fake_corr
    return float(np.sqrt(np.mean(difference**2))), float(np.mean(np.abs(difference)))


def privacy_metrics(real: np.ndarray, fake: np.ndarray) -> dict[str, float | int]:
    real_rows = {row.tobytes() for row in np.ascontiguousarray(real)}
    duplicates = sum(row.tobytes() in real_rows for row in np.ascontiguousarray(fake))
    neighbors = NearestNeighbors(n_neighbors=1).fit(real)
    distances, _ = neighbors.kneighbors(fake)
    return {
        "exact_duplicate_rows": int(duplicates),
        "nearest_neighbor_mean": float(distances.mean()),
        "nearest_neighbor_std": float(distances.std()),
    }


def distribution_report(
    real: np.ndarray, fake: np.ndarray, bins: int = 20
) -> dict[str, float | int]:
    rmse, mae = correlation_distances(real, fake)
    return {
        "mean_histogram_kl": histogram_kl(real, fake, bins),
        "correlation_rmse": rmse,
        "correlation_mae": mae,
        "mean_absolute_mean_difference": float(np.mean(np.abs(real.mean(0) - fake.mean(0)))),
        "mean_absolute_std_difference": float(np.mean(np.abs(real.std(0) - fake.std(0)))),
        **privacy_metrics(real, fake),
    }

