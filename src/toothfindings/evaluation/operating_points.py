"""Operating points at deployment prevalence.

A finding is raised only when the ensemble confidence (top mean probability) reaches a
threshold ``tau``; below it the tooth is treated as sound. Reported for each ``tau``: false
flags per exam (sound teeth flagged as a finding, over patients with at least one sound tooth),
share of exams without a false flag, recall of each finding class and impacted precision.

Reads ``experiments/prevalence/<run>/predictions.csv``; run names are
``<arch>__<tag>_s<scale>[_prior]`` (``paper``: published models; ``posneg``: models trained
with position-aware sound negatives; ``_prior``: tooth-position prior applied).
"""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np

from ..constants import CLASSES4
from ..utils import read_csv, write_json

GRID = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
DEFAULT_RUNS = [
    "inception_v3__paper_s1.0",
    "deit_small_patch16_224__paper_s1.0",
    "deit_small_patch16_224__posneg_s1.0",
    "deit_small_patch16_224__posneg_s1.0_prior",
]
H = 1  # index of Healthy in CLASSES4


def analyze_arrays(y: np.ndarray, P: np.ndarray, pid: np.ndarray, grid=GRID) -> list[dict]:
    """Compute the operating-point curve from class indices, probabilities and patient ids.

    Parameters
    ----------
    y : ndarray of int
        True class index of each crop (into ``CLASSES4``).
    P : ndarray, shape (n, 4)
        Ensemble mean probabilities.
    pid : ndarray of str
        Patient of each crop.
    grid : list of float
        Confidence thresholds ``tau``.

    Returns
    -------
    list of dict
        One entry per ``tau`` with ``false_flags_per_exam``, ``share_exams_zero``,
        ``recall`` (per finding class), ``precision_impacted`` and ``n_sound_flagged``.
    """
    pred = P.argmax(1)
    conf = P.max(1)
    out = []
    exams = sorted(set(pid[y == H]))
    for t in grid:
        flag = (pred != H) & (conf >= t)
        flags_by_patient: collections.Counter = collections.Counter()
        for p, f in zip(pid[y == H], (flag & (y == H))[y == H]):
            flags_by_patient[p] += int(f)
        flags_per_exam = np.array([flags_by_patient[p] for p in exams])
        out.append(
            {
                "tau": t,
                "false_flags_per_exam": float(flags_per_exam.mean()),
                "share_exams_zero": float((flags_per_exam == 0).mean()),
                # finding classes: Endodontics, Impacted, Implant
                "recall": {
                    CLASSES4[c]: float((flag & (pred == c) & (y == c)).sum() / (y == c).sum()) for c in [0, 2, 3]
                },
                "precision_impacted": float((flag & (pred == 2) & (y == 2)).sum() / max(1, (flag & (pred == 2)).sum())),
                "n_sound_flagged": int((flag & (y == H)).sum()),
            }
        )
    return out


def analyze(predictions_csv: Path, grid=GRID) -> list[dict]:
    """Compute the operating-point curve of one prevalence ``predictions.csv``."""
    rows = read_csv(predictions_csv)
    y = np.array([CLASSES4.index(r["class"]) for r in rows])
    P = np.array([[float(r[f"p_{c}"]) for c in CLASSES4] for r in rows])
    pid = np.array([r["patient_id"] for r in rows])
    return analyze_arrays(y, P, pid, grid)


def operating_points(prevalence_dir: Path, runs: list[str] | None = None, out: Path | None = None) -> dict:
    """Compute the operating-point curves of several prevalence runs.

    Parameters
    ----------
    prevalence_dir : Path
        Folder holding one sub-folder per run (``experiments/prevalence``).
    runs : list of str, optional
        Run names (default: ``DEFAULT_RUNS``); runs without ``predictions.csv`` are skipped.
    out : Path, optional
        JSON file to write the curves to (``operating_point.json``).

    Returns
    -------
    dict
        ``{run: curve}``, each curve as returned by :func:`analyze_arrays`.
    """
    runs = runs or DEFAULT_RUNS
    res = {
        r: analyze(Path(prevalence_dir) / r / "predictions.csv")
        for r in runs
        if (Path(prevalence_dir) / r / "predictions.csv").exists()
    }
    if out is not None:
        write_json(out, res)
    return res
