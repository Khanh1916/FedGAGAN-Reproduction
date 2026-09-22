from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def create_demo_dataset(path: str | Path, rows: int = 2400, seed: int = 42) -> Path:
    """Create a small, imbalanced, mixed-type IDS-like CSV for a smoke test."""
    rng = np.random.default_rng(seed)
    labels = rng.choice([0, 1], size=rows, p=[0.86, 0.14])
    protocols = np.where(
        labels == 1,
        rng.choice(["tcp", "udp"], rows),
        rng.choice(["tcp", "udp", "icmp"], rows),
    )
    frame = pd.DataFrame(
        {
            "duration": rng.gamma(1.5 + labels * 1.2, 1.8),
            "source_bytes": rng.lognormal(5.0 + labels * 0.7, 1.0),
            "destination_bytes": rng.lognormal(5.4 - labels * 0.4, 0.9),
            "packet_count": rng.poisson(12 + labels * 18).astype(float),
            "protocol": protocols,
            "service": np.where(
                labels == 1,
                rng.choice(["ssh", "dns", "mqtt"], rows),
                rng.choice(["http", "dns", "mqtt"], rows),
            ),
            "label": labels,
        }
    )
    missing = rng.choice(rows, size=max(1, rows // 100), replace=False)
    frame.loc[missing, "duration"] = np.nan
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
