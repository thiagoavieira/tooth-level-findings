"""Aggregate the run matrix into ``aggregates/{cv,test}_summary.csv``.

One row per configuration: mean, sample standard deviation and t-based 95% CI of the macro-F1
(over 5 folds x 3 seeds for the cross-validation runs, 3 seeds for the test runs), mean and
standard deviation of the accuracy, and mean and standard deviation of each per-class F1.
Values are rounded to 4 decimals, as in the released files.
"""

from __future__ import annotations

import glob
import math
import os
from collections import defaultdict
from pathlib import Path

from scipy import stats

from ..constants import CLASSES4
from ..utils import read_json, write_dict_csv


def mean_std_ci(values: list[float]) -> dict:
    """Return the mean, sample sd (``scipy.stats.tstd``) and t-based 95% CI, rounded to 4 decimals.

    Returns
    -------
    dict
        ``mean``, ``std``, ``ci95_low``, ``ci95_high`` and ``n``. With no values every statistic
        is an empty string; with one value the sd is 0 and the CI collapses to the mean.
    """
    n = len(values)
    if n == 0:
        return {"mean": "", "std": "", "ci95_low": "", "ci95_high": "", "n": 0}
    m = sum(values) / n
    if n == 1:
        return {"mean": round(m, 4), "std": 0.0, "ci95_low": round(m, 4), "ci95_high": round(m, 4), "n": 1}
    sd = stats.tstd(values)
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return {"mean": round(m, 4), "std": round(sd, 4), "ci95_low": round(m - h, 4), "ci95_high": round(m + h, 4), "n": n}


def load_runs(exp: Path, subdir: str) -> list[dict]:
    """Return the content of every readable ``<exp>/<subdir>/*/metrics.json`` (unreadable files are skipped)."""
    out = []
    for mp in glob.glob(os.path.join(exp, subdir, "*", "metrics.json")):
        try:
            out.append(read_json(mp))
        except Exception:
            pass
    return out


def summarize(runs: list[dict], metric_key: str, classes: list[str] = CLASSES4) -> list[dict]:
    """Group runs by ``(enhancement, aug, arch)`` and summarise ``run[metric_key]``.

    Parameters
    ----------
    runs : list of dict
        Contents of ``metrics.json`` files.
    metric_key : {"val_metrics", "test_metrics"}
        Validation metrics (cv runs) or test-set metrics (test runs).
    classes : list of str
        Classes whose per-class F1 is summarised.

    Returns
    -------
    list of dict
        One row per configuration, sorted by ``(enhancement, aug, arch)``, with ``n_runs``,
        ``macro_f1_{mean,std,ci95_low,ci95_high}``, ``accuracy_{mean,std}`` (when available)
        and ``f1_<class>_{mean,std}``.
    """
    groups: dict = defaultdict(list)
    for r in runs:
        groups[(r["enhancement"], r["aug"], r["arch"])].append(r)
    rows = []
    for (enh, aug, arch), group in sorted(groups.items()):
        mets = [r[metric_key] for r in group]
        row = {"enhancement": enh, "aug": aug, "arch": arch, "n_runs": len(group)}
        ms = mean_std_ci([m["macro_f1"] for m in mets])
        row.update(
            {
                "macro_f1_mean": ms["mean"],
                "macro_f1_std": ms["std"],
                "macro_f1_ci95_low": ms["ci95_low"],
                "macro_f1_ci95_high": ms["ci95_high"],
            }
        )
        acc = [m["accuracy"] for m in mets if "accuracy" in m]
        if acc:
            a = mean_std_ci(acc)
            row.update({"accuracy_mean": a["mean"], "accuracy_std": a["std"]})
        for c in classes:
            cs = mean_std_ci([m["per_class"][c]["f1"] for m in mets])
            row[f"f1_{c}_mean"] = cs["mean"]
            row[f"f1_{c}_std"] = cs["std"]
        rows.append(row)
    return rows


def run_summaries(exp: Path, out_dir: Path | None = None) -> tuple[list[dict], list[dict]]:
    """Compute the cv and test summaries of an experiment folder and optionally write them.

    Parameters
    ----------
    exp : Path
        Experiment folder with ``runs/`` (cv) and ``test/`` sub-folders.
    out_dir : Path, optional
        When given, ``cv_summary.csv`` and ``test_summary.csv`` are written there (each only if
        non-empty).

    Returns
    -------
    cv_rows, test_rows : list of dict
        See :func:`summarize`.
    """
    cv_rows = summarize(load_runs(exp, "runs"), "val_metrics")
    test_rows = summarize(load_runs(exp, "test"), "test_metrics")
    if out_dir is not None:
        if cv_rows:
            write_dict_csv(Path(out_dir) / "cv_summary.csv", cv_rows)
        if test_rows:
            write_dict_csv(Path(out_dir) / "test_summary.csv", test_rows)
    return cv_rows, test_rows
