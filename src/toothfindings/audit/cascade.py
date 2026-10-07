"""Where the experts corrected the model, and the filtering cascade (``cascade.json``).

(A) Corrections by confidence bin, raw and weighted. The weighted precision of each bin, with
    an 800-resample cluster bootstrap, is also written to ``bin_ci.json`` (the expert-audit
    table).
(B) The cascade from every prediction of a class to the operating point: confidence >= tau with
    unanimous seeds, then teeth outside the label space removed. Crops the experts could not
    use (upstream segmentation failures) are excluded from every denominator.
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from ..utils import write_json
from .core import CLIN, DATASETS, OUTSIDE, UNUSABLE, AuditData, percentile_ci, resample_groups

#: Confidence bins ``[lo, hi)``; the last one closes at 1.0.
BINS = [(0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)]


def bin_key(p: float) -> str | None:
    """Label of the confidence bin of ``p`` (``0.50-0.70`` ... ``0.99-1.00``)."""
    for lo, hi in BINS:
        if lo <= p < hi:
            return f"{lo:.2f}-{hi if hi <= 1 else 1.0:.2f}"
    return None


def weighted_bins(rows, w) -> dict[str, float]:
    """Return the weighted precision of the usable reviewed crops in each confidence bin.

    Returns
    -------
    dict
        ``{bin label: precision}`` for the bins with positive weight.
    """
    num, den = defaultdict(float), defaultdict(float)
    for r in rows:
        if r["label"] in UNUSABLE:
            continue
        k = bin_key(float(r["pred_prob"]))
        if k is None:
            continue
        wi = w[(r["model_pred"], r["stratum"])]
        den[k] += wi
        num[k] += wi * (r["label"] == r["model_pred"])
    return {k: num[k] / den[k] for k in den if den[k]}


def cascade_steps(rows, w, tau) -> dict:
    """Compute precision, estimated population and coverage of each class at each cascade step.

    Parameters
    ----------
    rows : list of dict
        Reviewed crops with ``label`` and ``stratum``.
    w : dict
        Weights ``{(class, stratum): N_h / n_h}``.
    tau : dict
        Study threshold of each class.

    Returns
    -------
    dict
        ``{class: [{"step", "precision", "est_population", "coverage_of_class"}, ...]}``;
        coverage is relative to step 0.
    """
    steps = [
        ("0. every prediction of the class", lambda r: r["label"] not in UNUSABLE),
        (
            "1. confidence >= tau, unanimous seeds",
            lambda r: (
                r["label"] not in UNUSABLE
                and float(r["pred_prob"]) >= tau[r["model_pred"]]
                and int(r["seed_agreement"]) == 3
            ),
        ),
        (
            "2. - teeth outside the label space",
            lambda r: (
                r["label"] not in UNUSABLE
                and float(r["pred_prob"]) >= tau[r["model_pred"]]
                and int(r["seed_agreement"]) == 3
                and r["label"] not in OUTSIDE
            ),
        ),
    ]
    out = {}
    for c in CLIN:
        base, class_steps = None, []
        for name, keep in steps:
            num = den = 0.0
            for r in rows:
                if r["model_pred"] != c or not keep(r):
                    continue
                wi = w[(c, r["stratum"])]
                den += wi
                num += wi * (r["label"] == c)
            if base is None:
                base = den
            class_steps.append(
                {
                    "step": name,
                    "precision": (num / den) if den else None,
                    "est_population": den,
                    "coverage_of_class": (den / base) if base else None,
                }
            )
        out[c] = class_steps
    return out


def boot_precision(rows, w, keep, cls, n: int = 800, seed: int = 0):
    """Compute the cluster-bootstrap 95% interval of the weighted precision of ``cls``.

    Only rows predicted as ``cls`` for which ``keep(row)`` is true enter the estimate. Returns
    ``[low, high]``, or ``None`` when no resample has a defined precision.
    """
    draw = resample_groups(rows, random.Random(seed))
    vals = []
    for _ in range(n):
        num = den = 0.0
        for r in draw():
            if r["model_pred"] != cls or not keep(r):
                continue
            wi = w[(cls, r["stratum"])]
            den += wi
            num += wi * (r["label"] == cls)
        if den:
            vals.append(num / den)
    return percentile_ci(vals) if vals else None


def cascade(data: AuditData, write: bool = True) -> dict:
    """Compute the corrections by confidence bin and the filtering cascade (``cascade.json``).

    Returns
    -------
    dict
        Per dataset: raw counts per bin (``reviewed``, ``confirmed``, ``corrected``), weighted
        precision per bin, mean confidence of confirmed and corrected crops, and the cascade of
        :func:`cascade_steps` with an 800-resample ``ci95`` at each step.
    """
    res = {}
    for ds in DATASETS:
        _st, rows, w, tau = data.setup(ds)
        bins: dict = defaultdict(lambda: defaultdict(int))
        wbins: dict = defaultdict(lambda: defaultdict(float))
        for r in rows:
            if r["label"] in UNUSABLE:
                continue
            k = bin_key(float(r["pred_prob"]))
            if k is None:
                continue
            bins[k]["reviewed"] += 1
            bins[k]["confirmed"] += int(r["label"] == r["model_pred"])
            bins[k]["corrected"] += int(r["label"] != r["model_pred"])
            wi = w[(r["model_pred"], r["stratum"])]
            wbins[k]["predictions"] += wi
            wbins[k]["correct"] += wi * (r["label"] == r["model_pred"])
        usable = [r for r in rows if r["label"] not in UNUSABLE]
        conf_ok = [float(r["pred_prob"]) for r in usable if r["label"] == r["model_pred"]]
        conf_bad = [float(r["pred_prob"]) for r in usable if r["label"] != r["model_pred"]]
        casc = cascade_steps(rows, w, tau)

        def step1(r):
            return (
                r["label"] not in UNUSABLE
                and float(r["pred_prob"]) >= tau[r["model_pred"]]
                and int(r["seed_agreement"]) == 3
            )

        for c in CLIN:
            casc[c][-1]["ci95"] = boot_precision(rows, w, lambda r: step1(r) and r["label"] not in OUTSIDE, c)
            casc[c][0]["ci95"] = boot_precision(rows, w, lambda r: r["label"] not in UNUSABLE, c)
            casc[c][1]["ci95"] = boot_precision(rows, w, step1, c)
        res[ds] = {
            "corrections_by_confidence": {k: dict(v) for k, v in sorted(bins.items())},
            "weighted_precision_by_confidence": {
                k: {
                    "est_predictions": v["predictions"],
                    "precision": v["correct"] / v["predictions"] if v["predictions"] else None,
                }
                for k, v in sorted(wbins.items())
            },
            "mean_confidence_confirmed": float(np.mean(conf_ok)),
            "mean_confidence_corrected": float(np.mean(conf_bad)),
            "cascade": casc,
        }
    if write:
        write_json(data.paths.review / "cascade.json", res)
    return res


def bin_ci(data: AuditData, n: int = 800, seed: int = 0, write: bool = True) -> dict:
    """Compute the weighted precision by confidence bin with cluster-bootstrap CIs (``bin_ci.json``).

    Returns
    -------
    dict
        ``{dataset: {bin label: {"precision", "ci95"}}}``.
    """
    out = {}
    for ds in DATASETS:
        _st, rows, w, _tau = data.setup(ds)
        point = weighted_bins(rows, w)
        draw = resample_groups(rows, random.Random(seed))
        samples: dict = defaultdict(list)
        for _ in range(n):
            for k, v in weighted_bins(draw(), w).items():
                samples[k].append(v)
        out[ds] = {k: {"precision": point[k], "ci95": percentile_ci(samples[k])} for k in sorted(point)}
    if write:
        write_json(data.paths.review / "bin_ci.json", out)
    return out
