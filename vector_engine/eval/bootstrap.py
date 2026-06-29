"""Bootstrap confidence intervals for per-query metrics."""
from __future__ import annotations

from typing import Dict

import numpy as np


def ci(values: np.ndarray, n_resamples: int, level: float, seed: int) -> Dict:
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    mean = float(values.mean()) if n else 0.0
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "level": level}
    rng = np.random.RandomState(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample = values[rng.randint(0, n, size=n)]
        means[i] = sample.mean()
    alpha = (1.0 - level) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "level": level}


def summarize(per_query: Dict[str, np.ndarray], n_resamples: int,
              level: float, seed: int) -> Dict[str, Dict]:
    return {name: ci(vals, n_resamples, level, seed)
            for name, vals in per_query.items()}
