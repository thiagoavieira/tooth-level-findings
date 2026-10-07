"""Confidence intervals: Wilson score interval and the crop-level bootstrap of the test macro-F1.

The bootstrap reads the ``predictions.csv`` files written by the test-mode runs.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from ..constants import CLASSES4
from ..utils import read_csv


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return the Wilson score interval of a binomial proportion ``k / n``.

    Parameters
    ----------
    k, n : int
        Successes and trials.
    z : float
        Standard-normal quantile; the default gives a 95% interval.

    Returns
    -------
    tuple of float
        ``(low, high)``; ``(nan, nan)`` when ``n == 0``.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((center - half) / denom, (center + half) / denom)


def load_seed_predictions(
    test_dir: Path, arch: str, enhancement: str = "none", aug: str = "noaug", seeds=(0, 1, 2)
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int, list[Path]]:
    """Load the per-seed test predictions of one configuration.

    Parameters
    ----------
    test_dir : Path
        Folder with ``<enhancement>__<aug>__<arch>__seed<s>/predictions.csv``.
    arch, enhancement, aug : str
        Configuration.
    seeds : sequence of int
        Seeds to load.

    Returns
    -------
    data : list of tuple of numpy.ndarray
        One ``(y_true, y_pred)`` pair of class-name arrays per seed.
    n : int
        Number of test crops.
    files : list of Path
        The files read, in seed order.

    Raises
    ------
    ValueError
        When a file is missing or the row order differs between seeds.
    """
    data, ref_paths, files = [], None, []
    for seed in seeds:
        path = Path(test_dir) / f"{enhancement}__{aug}__{arch}__seed{seed}" / "predictions.csv"
        if not path.exists():
            raise ValueError(f"missing {path}")
        rows = read_csv(path)
        crop_paths = [r["filepath"] for r in rows]
        if ref_paths is None:
            ref_paths = crop_paths
        elif crop_paths != ref_paths:
            raise ValueError(f"seed {seed} lists the test crops in a different order")
        data.append((np.array([r["y_true"] for r in rows]), np.array([r["y_pred"] for r in rows])))
        files.append(path)
    return data, len(ref_paths), files


def bootstrap_ci(
    data: list[tuple[np.ndarray, np.ndarray]],
    n: int,
    n_resamples: int = 2000,
    seed: int = 0,
    classes: list[str] = CLASSES4,
) -> dict:
    """Bootstrap the seed-mean macro-F1 over test crops and return a percentile 95% CI.

    Each draw resamples crop indices with replacement (the same indices for every seed),
    computes each seed's macro-F1 on the draw and averages them; the 2.5th and 97.5th
    percentiles of the draws give the 95% CI. Resampling crops ignores the correlation between
    crops of the same patient, so the interval is narrower than a patient-clustered one.

    Parameters
    ----------
    data : list of tuple of numpy.ndarray
        Per-seed ``(y_true, y_pred)`` arrays, as returned by :func:`load_seed_predictions`.
    n : int
        Number of crops.
    n_resamples : int
        Number of bootstrap draws.
    seed : int
        Seed of the NumPy generator.
    classes : list of str
        Labels averaged by the macro-F1.

    Returns
    -------
    dict
        ``n_resamples``, ``n_crops``, ``rng_seed``, ``draws_mean``, ``draws_std`` (ddof=1),
        ``ci95_low`` and ``ci95_high``.
    """
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        draws[b] = float(np.mean([f1_score(y[idx], p[idx], labels=classes, average="macro") for y, p in data]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "n_resamples": n_resamples,
        "n_crops": n,
        "rng_seed": seed,
        "draws_mean": float(draws.mean()),
        "draws_std": float(draws.std(ddof=1)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }
