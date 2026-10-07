"""Weighted precision of the audited model on the external datasets.

Main estimates by class (``audit.json``), deployed against in-scope precision (``in_scope.json``),
the expert labels behind each class's flags (``breakdown.json``), and precision under
alternative ground-truth conventions (printed, not written).
"""

from __future__ import annotations

import random
from collections import defaultdict

from ..utils import write_json
from .core import (
    CLIN,
    DATASETS,
    N_PANS,
    OUTSIDE,
    UNUSABLE,
    AuditData,
    cluster_boot,
    estimate,
    percentile_ci,
    resample_groups,
    weights,
)

#: Thresholds of the precision curve and of the wrong-flag rate.
GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995]
LABELS = CLIN + ["Other", "Exclude", "Bad crop"]


def audit(data: AuditData, write: bool = True) -> dict:
    """Compute the main audit estimates (``audit.json``).

    Returns
    -------
    dict
        Per dataset: strata, sample counts, weights (``"class|stratum"``), precision at the
        study thresholds with 1,000-resample cluster-bootstrap CIs, lenient precision (any
        finding label accepted for a finding prediction), precision and estimated kept
        predictions against a common threshold, and wrong finding flags per radiograph.
    """
    res = {}
    for ds in DATASETS:
        st = data.strata(ds)
        rows, disagree, _ = data.reviewed(ds)
        all_rows, _, n_excluded = data.reviewed(ds, with_excluded=True)
        w = weights(st, rows, sampled=all_rows)
        tau = {c: st[c]["tau"] for c in CLIN}
        base = estimate(rows, w, tau)
        ci = cluster_boot(rows, w, tau, None, False)
        lenient = estimate(rows, w, tau, lenient=True)
        curve = {}
        for t in GRID:
            e = estimate(rows, w, tau, thr=t)
            curve[str(t)] = {
                "precision": {c: e[c][0] for c in CLIN},
                "est_predictions_kept": {c: e[c][1] for c in CLIN},
            }
        flags = {}  # wrong finding flags (sound predictions excluded) per radiograph
        for t in GRID:
            wrong = 0.0
            for r in rows:
                c = r["model_pred"]
                if c == "Healthy" or float(r["pred_prob"]) < t:
                    continue
                wrong += w[(c, r["stratum"])] * (r["label"] != c)
            flags[str(t)] = wrong / N_PANS[ds]
        res[ds] = {
            "strata": st,
            "n_reviewed": len(rows),
            "n_reader_disagreements_excluded": disagree,
            "n_out_of_scope_excluded": n_excluded,
            "weights": {f"{c}|{h}": w[(c, h)] for c in CLIN for h in ("A", "B")},
            "precision_at_review_tau": {
                c: {"estimate": base[c][0], "ci95": ci.get(c), "est_population": base[c][1]} for c in CLIN
            },
            "precision_lenient": {c: lenient[c][0] for c in CLIN},
            "precision_vs_threshold": curve,
            "wrong_flags_per_radiograph": flags,
        }
    if write:
        write_json(data.paths.review / "audit.json", res)
    return res


def in_scope_parts(rows, w, tau) -> dict:
    """Split the flags of each class at its study threshold by expert label.

    Returns
    -------
    dict
        ``{class: {"deployed", "in_scope", "share_outside", "share_unusable",
        "share_wrong_class", "est_total"}}``. Deployed precision counts every flag; in-scope
        precision drops flags on teeth outside the label space and on unusable crops.
    """
    acc: dict = defaultdict(lambda: defaultdict(float))
    for r in rows:
        c = r["model_pred"]
        if float(r["pred_prob"]) < tau[c]:
            continue
        wi = w[(c, r["stratum"])]
        acc[c]["total"] += wi
        if r["label"] == c:
            acc[c]["correct"] += wi
        elif r["label"] in OUTSIDE:
            acc[c]["outside"] += wi
        elif r["label"] in UNUSABLE:
            acc[c]["unusable"] += wi
        else:
            acc[c]["wrong_class"] += wi
    out = {}
    for c in CLIN:
        a = acc[c]
        scope = a["total"] - a["outside"] - a["unusable"]
        out[c] = {
            "deployed": a["correct"] / a["total"] if a["total"] else None,
            "in_scope": a["correct"] / scope if scope else None,
            "share_outside": a["outside"] / a["total"] if a["total"] else None,
            "share_unusable": a["unusable"] / a["total"] if a["total"] else None,
            "share_wrong_class": a["wrong_class"] / a["total"] if a["total"] else None,
            "est_total": a["total"],
        }
    return out


def in_scope(data: AuditData, n: int = 1000, seed: int = 0, write: bool = True) -> dict:
    """Compute deployed and in-scope precision with cluster-bootstrap CIs (``in_scope.json``).

    Returns
    -------
    dict
        ``{dataset: {class: {**in_scope_parts, "ci95": {"deployed": [lo, hi], "in_scope": [lo, hi]}}}}``.
    """
    res = {}
    for ds in DATASETS:
        _st, rows, w, tau = data.setup(ds)
        parts = in_scope_parts(rows, w, tau)
        draw = resample_groups(rows, random.Random(seed))
        samples: dict = defaultdict(lambda: defaultdict(list))
        for _ in range(n):
            for c, v in in_scope_parts(draw(), w, tau).items():
                for k in ("deployed", "in_scope"):
                    if v[k] is not None:
                        samples[c][k].append(v[k])
        ci = {c: {k: percentile_ci(v) for k, v in d.items()} for c, d in samples.items()}
        res[ds] = {c: {**parts[c], "ci95": ci.get(c, {})} for c in CLIN}
    if write:
        write_json(data.paths.review / "in_scope.json", res)
    return res


def breakdown(data: AuditData, write: bool = True) -> dict:
    """Estimate the expert labels behind each class's flags at the study thresholds (``breakdown.json``).

    Returns
    -------
    dict
        ``{dataset: {predicted class: {expert label: estimated count, rounded to 0.1}}}``.
    """
    out = {}
    for ds in DATASETS:
        _st, rows, w, tau = data.setup(ds)
        confusion: dict = defaultdict(lambda: defaultdict(float))
        for r in rows:
            c = r["model_pred"]
            if float(r["pred_prob"]) < tau[c]:
                continue
            confusion[c][r["label"]] += w[(c, r["stratum"])]
        out[ds] = {c: {lab: round(confusion[c][lab], 1) for lab in LABELS} for c in CLIN}
    if write:
        write_json(data.paths.review / "breakdown.json", out)
    return out


def rows_for_convention(data: AuditData, ds: str, mode: str) -> list[dict]:
    """Return the reviewed crops of dataset ``ds`` labelled under a ground-truth convention.

    Parameters
    ----------
    data : AuditData
        Audit inputs.
    ds : str
        Dataset.
    mode : str
        ``default`` (agreed label where read twice, the single reader's elsewhere; crops with
        disagreement dropped), ``consensus_only`` (only crops read twice with agreement), or
        ``reader_<id>`` (that reader's label wherever they read the crop).

    Returns
    -------
    list of dict
        Copies of the ``review_wide.csv`` rows with ``label`` and ``stratum`` added.
        Out-of-scope crops are not dropped here.
    """
    out = []
    for r0 in data.wide():
        if r0["dataset"] != ds:
            continue
        labels = [r0[k] for k in ("label1", "label2") if r0.get(k)]
        readers = [r0[k] for k in ("reader1", "reader2") if r0.get(k)]
        if mode == "default":
            if len(labels) == 2 and labels[0] != labels[1]:
                continue
            lab = labels[0]
        elif mode == "consensus_only":
            if len(labels) != 2 or labels[0] != labels[1]:
                continue
            lab = labels[0]
        else:
            rid = mode.split("_")[1]
            idx = [i for i, x in enumerate(readers) if x == rid]
            if not idx:
                continue
            lab = labels[idx[0]]
        r = dict(r0)
        r["label"] = lab
        r["stratum"] = "A" if r["bucket"] == "confirm" else "B"
        out.append(r)
    return out


def reader_conventions(data: AuditData) -> dict:
    """Compute precision (%) at the study thresholds under each ground-truth convention.

    The weights use the sample sizes of each convention's own rows (see
    :func:`rows_for_convention`).

    Returns
    -------
    dict
        ``{dataset: {mode: {"n", "precision_pct": {class: value}} or None}}``.
    """
    res = {}
    for ds in DATASETS:
        st = data.strata(ds)
        tau = {c: st[c]["tau"] for c in CLIN}
        readers = sorted(
            {r[k] for r in data.wide() if r["dataset"] == ds for k in ("reader1", "reader2") if r.get(k)},
            key=lambda x: (len(x), x),
        )
        res[ds] = {}
        for mode in ["default", "consensus_only"] + [f"reader_{x}" for x in readers]:
            rows = rows_for_convention(data, ds, mode)
            if not rows:
                res[ds][mode] = None
                continue
            w = weights(st, rows)
            num: dict = defaultdict(float)
            den: dict = defaultdict(float)
            for r in rows:
                c = r["model_pred"]
                if float(r["pred_prob"]) < tau[c]:
                    continue
                wi = w[(c, r["stratum"])]
                num[c] += wi * (r["label"] == c)
                den[c] += wi
            res[ds][mode] = {
                "n": len(rows),
                "precision_pct": {c: (100 * num[c] / den[c] if den[c] else None) for c in CLIN},
            }
    return res
