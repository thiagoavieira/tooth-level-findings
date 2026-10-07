"""Paired comparisons of the configurations over the 15 cross-validation blocks.

A block is one ``(fold, seed)`` pair. Every configuration is evaluated on the same 5 folds x
3 seeds, so the comparisons are paired by block:

* Friedman omnibus test over the 18 classic configurations, with the Nemenyi post-hoc test;
* every pair of configurations: paired t test, Wilcoxon signed-rank test and Cohen's d_z
  (standardised mean of the paired differences), Holm-corrected over the 153 pairs;
* each of the six backbones (three classic, three recent) against the reference InceptionV3
  (grayscale, no augmentation), Holm-corrected over these six rows.

The three seeds of a fold share its validation patients, so the 15 blocks are not independent
and the p-values and confidence intervals are optimistic; a corrected resampled t test
(Nadeau and Bengio, 2003) or averaging the seeds of each fold first would be more conservative.

Outputs: ``friedman.json``, ``nemenyi.csv``, ``paired_tests.csv``, ``new_backbones.json`` and
``new_backbones.csv``.
"""

from __future__ import annotations

import glob
import math
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ..constants import CLASSES4, CLASSIC_ARCHS, NEW_ARCHS
from ..utils import read_json, write_dict_csv, write_json

#: Reference configuration (``<enhancement>__<aug>__<arch>``) for :func:`new_backbones`.
REF = "none__noaug__inception_v3"


def holm(pvalues: list[float]) -> np.ndarray:
    """Return Holm-adjusted p-values (``statsmodels`` ``multipletests(method="holm")``)."""
    return multipletests(pvalues, method="holm")[1]


def cohens_dz(a, b) -> float:
    """Return Cohen's d_z for paired samples: mean of ``a - b`` over its sd (ddof=1); 0 if the sd is 0."""
    diff = np.array(a) - np.array(b)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def load_cv(exp: Path, archs=frozenset(CLASSIC_ARCHS)) -> dict[str, dict[tuple[int, int], float]]:
    """Load the validation macro-F1 of every cv run, keyed by configuration and block.

    Parameters
    ----------
    exp : Path
        Experiment folder; reads ``<exp>/runs/*/metrics.json`` (unreadable files are skipped).
    archs : collection of str or None
        Backbones to keep; the classic ones by default, every backbone when ``None``.

    Returns
    -------
    dict
        ``{"<enhancement>__<aug>__<arch>": {(fold, seed): macro_f1}}``.
    """
    data: dict = defaultdict(dict)
    for mp in glob.glob(os.path.join(exp, "runs", "*", "metrics.json")):
        try:
            r = read_json(mp)
        except Exception:
            continue
        if archs is not None and r["arch"] not in archs:
            continue
        cfg = f"{r['enhancement']}__{r['aug']}__{r['arch']}"
        data[cfg][(int(r["fold"]), int(r["seed"]))] = r["val_metrics"]["macro_f1"]
    return data


def block_matrix(data: dict) -> tuple[np.ndarray, list[str], list]:
    """Arrange the scores as a ``[block x configuration]`` matrix over the blocks shared by all configurations.

    Parameters
    ----------
    data : dict
        ``{config: {block: score}}``, as returned by :func:`load_cv`.

    Returns
    -------
    M : numpy.ndarray
        Scores, one row per common block and one column per configuration.
    configs : list of str
        Column labels, sorted.
    common : list
        Row labels (blocks), sorted.
    """
    common = sorted(set.intersection(*[set(v.keys()) for v in data.values()]))
    configs = sorted(c for c in data if set(common).issubset(data[c].keys()))
    return np.array([[data[c][b] for c in configs] for b in common]), configs, common


def friedman(M: np.ndarray) -> dict:
    """Run the Friedman test treating the columns of ``M`` as treatments and its rows as blocks.

    Returns
    -------
    dict
        ``statistic`` (chi-square), ``p_value``, ``n_configs`` and ``n_blocks``.
    """
    chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    return {"statistic": float(chi2), "p_value": float(p), "n_configs": M.shape[1], "n_blocks": M.shape[0]}


def nemenyi(M: np.ndarray, configs: list[str]):
    """Return the Nemenyi post-hoc p-values (``scikit_posthocs.posthoc_nemenyi_friedman``) as a labelled DataFrame."""
    import scikit_posthocs as sp

    nem = sp.posthoc_nemenyi_friedman(M)
    nem.index = configs
    nem.columns = configs
    return nem


def paired_tests(M: np.ndarray, configs: list[str]) -> list[dict]:
    """Compare every pair of columns of ``M`` with a paired t test, a Wilcoxon test and Cohen's d_z.

    A Wilcoxon test that cannot be computed (e.g. all differences zero) gets ``p = 1``; NaN
    p-values enter the Holm family as 1.

    Parameters
    ----------
    M : numpy.ndarray
        ``[block x configuration]`` matrix from :func:`block_matrix`.
    configs : list of str
        Column labels.

    Returns
    -------
    list of dict
        One row per pair with ``config_a``, ``config_b``, ``mean_a``, ``mean_b``, ``mean_diff``,
        ``t_stat``, ``t_p``, ``wilcoxon_p``, ``cohens_d`` (d_z), ``t_p_holm``,
        ``wilcoxon_p_holm`` and ``significant_holm_0.05`` (on the Holm-adjusted t-test p).
    """
    rows, pvals_t, pvals_w = [], [], []
    for i, j in combinations(range(len(configs)), 2):
        a, b = M[:, i], M[:, j]
        t_stat, t_p = stats.ttest_rel(a, b)
        try:
            _w_stat, w_p = stats.wilcoxon(a, b)
        except ValueError:
            w_p = 1.0
        rows.append(
            {
                "config_a": configs[i],
                "config_b": configs[j],
                "mean_a": round(a.mean(), 4),
                "mean_b": round(b.mean(), 4),
                "mean_diff": round(a.mean() - b.mean(), 4),
                "t_stat": round(float(t_stat), 4),
                "t_p": float(t_p),
                "wilcoxon_p": float(w_p),
                "cohens_d": round(cohens_dz(a, b), 3),
            }
        )
        pvals_t.append(t_p if not math.isnan(t_p) else 1.0)
        pvals_w.append(w_p if not math.isnan(w_p) else 1.0)
    if rows:
        t_holm, w_holm = holm(pvals_t), holm(pvals_w)
        for k, row in enumerate(rows):
            row["t_p_holm"] = float(t_holm[k])
            row["wilcoxon_p_holm"] = float(w_holm[k])
            row["significant_holm_0.05"] = bool(t_holm[k] < 0.05)
    return rows


def run_stats_tests(exp: Path, out_dir: Path | None = None) -> dict:
    """Run the Friedman, Nemenyi and pairwise tests on the classic grid; optionally write them to ``out_dir``.

    Returns
    -------
    dict
        ``friedman`` (see :func:`friedman`), ``paired`` (see :func:`paired_tests`), ``nemenyi``
        (DataFrame, or ``None`` when ``scikit_posthocs`` is unavailable), ``configs`` and
        ``blocks``.
    """
    data = load_cv(exp)
    M, configs, common = block_matrix(data)
    fr = friedman(M)
    rows = paired_tests(M, configs)
    nem = None
    try:
        nem = nemenyi(M, configs)
    except Exception as e:  # pragma: no cover - optional dependency
        print(f"Nemenyi skipped: {e}")
    if out_dir is not None:
        out_dir = Path(out_dir)
        write_json(out_dir / "friedman.json", fr)
        if nem is not None:
            nem.to_csv(out_dir / "nemenyi.csv")
        write_dict_csv(out_dir / "paired_tests.csv", rows)
    return {"friedman": fr, "paired": rows, "nemenyi": nem, "configs": configs, "blocks": common}


def new_backbones(exp: Path, out_dir: Path | None = None) -> list[dict]:
    """Compare each backbone (grayscale, no augmentation) with the reference InceptionV3 over the cv blocks.

    The cv comparison uses a paired t test, a Wilcoxon test and Cohen's d_z; the test-set
    columns summarise the test-mode seeds. The reference row itself carries ``p = 1`` and
    enters the Holm family, as in the released file. Unlike :mod:`.summaries`, the standard
    deviations here use ``ddof=0``.

    Parameters
    ----------
    exp : Path
        Experiment folder with ``runs/`` and ``test/``.
    out_dir : Path, optional
        When given, ``new_backbones.json`` and ``new_backbones.csv`` are written there.

    Returns
    -------
    list of dict
        One row per backbone found, with cv statistics, test-set means (``None`` without
        test runs), ``p_ttest_holm`` and ``significant_holm_005`` (Holm-adjusted t-test
        p < 0.05).
    """
    cv: dict = defaultdict(dict)
    test: dict = defaultdict(dict)
    for mode, sub in (("cv", "runs"), ("test", "test")):
        for mp in glob.glob(os.path.join(exp, sub, "*", "metrics.json")):
            r = read_json(mp)
            cfg = f"{r['enhancement']}__{r['aug']}__{r['arch']}"
            if mode == "cv":
                cv[cfg][(int(r["fold"]), int(r["seed"]))] = r["val_metrics"]["macro_f1"]
            else:
                test[cfg][int(r["seed"])] = r["test_metrics"]
    ref = cv[REF]
    rows, pvals = [], []
    for arch in CLASSIC_ARCHS + NEW_ARCHS:
        cfg = f"none__noaug__{arch}"
        if cfg not in cv:
            continue
        blocks = sorted(set(ref) & set(cv[cfg]))
        a = np.array([cv[cfg][b] for b in blocks])
        ref_scores = np.array([ref[b] for b in blocks])
        diff = a - ref_scores
        t_p = stats.ttest_rel(a, ref_scores).pvalue if arch != "inception_v3" else 1.0
        try:
            w_p = stats.wilcoxon(a, ref_scores).pvalue if arch != "inception_v3" else 1.0
        except ValueError:
            w_p = 1.0
        cd = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
        seeds = sorted(test.get(cfg, {}))
        test_f1 = [test[cfg][s]["macro_f1"] for s in seeds]
        per_class_f1 = (
            {c: float(np.mean([test[cfg][s]["per_class"][c]["f1"] for s in seeds])) for c in CLASSES4}
            if test_f1
            else {}
        )
        rows.append(
            {
                "arch": arch,
                "n_blocks": len(blocks),
                "cv_macro_f1_mean": float(a.mean()),
                "cv_macro_f1_std": float(a.std()),
                "diff_vs_ref_mean": float(diff.mean()),
                "cohens_d": cd,
                "p_ttest": float(t_p),
                "p_wilcoxon": float(w_p),
                "test_macro_f1_mean": float(np.mean(test_f1)) if test_f1 else None,
                "test_macro_f1_std": float(np.std(test_f1)) if test_f1 else None,
                "test_n_seeds": len(test_f1),
                **{f"test_f1_{c}": v for c, v in per_class_f1.items()},
            }
        )
        pvals.append(float(t_p))
    if rows:
        rej, p_holm, _, _ = multipletests(pvals, method="holm")
        for row, ph, rj in zip(rows, p_holm, rej):
            row["p_ttest_holm"] = float(ph)
            row["significant_holm_005"] = bool(rj)
    if out_dir is not None:
        write_json(Path(out_dir) / "new_backbones.json", rows)
        write_dict_csv(Path(out_dir) / "new_backbones.csv", rows)
    return rows
