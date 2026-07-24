"""95% percentile-bootstrap confidence intervals.

Small self-contained utility. Callers should pass a 1-D array of per-item
scores; if they want a metric-of-arrays (e.g. AUROC) they should pass a
callable that reduces a resampled index vector to a scalar.
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) percentile CI on the sample mean."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(arr.mean()), lo, hi


def bootstrap_metric_ci(
    metric_fn: Callable[[np.ndarray], float],
    n: int,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Return (lo, hi) CI for an arbitrary metric evaluated on an index resample.

    `metric_fn` receives a 1-D int index array of length n and returns a scalar.
    """
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = metric_fn(idx)
    # nanquantile so a single degenerate resample (e.g. single-class AUROC
    # returning NaN) does not poison the whole CI.
    if np.isnan(stats).all():
        return float("nan"), float("nan")
    lo = float(np.nanquantile(stats, alpha / 2))
    hi = float(np.nanquantile(stats, 1 - alpha / 2))
    return lo, hi
