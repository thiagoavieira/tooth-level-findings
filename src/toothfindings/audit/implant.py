"""The implant class on DENTEX against Tufts: threshold, base rate and discrimination.

Separates three things that one precision mixes: the operating threshold (0.50 for implants,
almost the raw argmax), the base rate (implants are 0.25% of DENTEX teeth against 0.82% of
Tufts teeth) and discrimination proper (false alarms per tooth or radiograph). Reports the
population by confidence bin, in-scope precision against the threshold with cluster-bootstrap
CIs, rates per 1,000 teeth and per radiograph, and a prevalence-standardised precision
(each dataset's own false-alarm rate with the other's abundance of true implants).
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from ..utils import write_json
from .core import CLIN, N_PANS, OUTSIDE, UNUSABLE, AuditData, percentile_ci, resample_groups

CLS = "Implant"
#: Thresholds of the implant curve.
GRID = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]


def counts(rows, w, t: float) -> tuple[float, float, float, float]:
    """Return weighted true and false positives of the implant class at threshold ``t``.

    Returns
    -------
    tuple of float
        ``(tp, fp wrong class, fp outside the label space, fp unusable crop)``.
    """
    tp = fpc = fpo = fpu = 0.0
    for r in rows:
        if r["model_pred"] != CLS or float(r["pred_prob"]) < t:
            continue
        wi = w[(CLS, r["stratum"])]
        lab = r["label"]
        if lab == CLS:
            tp += wi
        elif lab in UNUSABLE:
            fpu += wi
        elif lab in OUTSIDE:
            fpo += wi
        else:
            fpc += wi
    return tp, fpc, fpo, fpu


def point(rows, w, t, n_teeth, n_pans) -> dict:
    """Compute precision and rates of the implant class at threshold ``t``.

    "Deployed" precision counts every flag; "in scope" precision drops flags on teeth outside
    the label space and on unusable crops. Rates use ``n_teeth`` (all external teeth of the
    dataset) and ``n_pans`` (radiographs).
    """
    tp, fpc, fpo, fpu = counts(rows, w, t)
    tot = tp + fpc + fpo + fpu
    return {
        "tau": t,
        "precision_deployed": tp / tot if tot else None,
        "precision_in_scope": tp / (tp + fpc) if (tp + fpc) else None,
        "tp": tp,
        "fp_wrong_class": fpc,
        "fp_outside": fpo,
        "fp_unusable": fpu,
        "tp_per_1k_teeth": 1000 * tp / n_teeth,
        "fp_per_1k_teeth": 1000 * fpc / n_teeth,
        "fp_per_radiograph": fpc / n_pans,
    }


def boot(rows, w, t, n_teeth, n_pans, n: int = 1000, seed: int = 0):
    """Compute cluster-bootstrap 95% intervals of in-scope precision and false positives per 1,000 teeth.

    Returns
    -------
    tuple of list
        ``([low, high] in-scope precision, [low, high] false positives per 1,000 teeth)``.
    """
    draw = resample_groups(rows, random.Random(seed))
    in_scope, fp_per_1k = [], []
    for _ in range(n):
        p = point(draw(), w, t, n_teeth, n_pans)
        if p["precision_in_scope"] is not None:
            in_scope.append(p["precision_in_scope"])
        fp_per_1k.append(p["fp_per_1k_teeth"])
    return percentile_ci(in_scope), percentile_ci(fp_per_1k)


def implant_view(data: AuditData, write: bool = True) -> dict:
    """Compare the implant class on Tufts and DENTEX (``implant_view.json``).

    Returns
    -------
    dict
        ``datasets``: per dataset, the implant population by confidence bin, review coverage,
        mean scores of confirmed and corrected crops, radiographs with true and false implant
        flags, crop aspect ratios and the threshold ``curve`` (:func:`point` with CIs).
        ``prevalence_standardised``: per threshold, each dataset's in-scope precision with its
        own false-positive rate and the other dataset's rate of true implants.
        ``all_classes_at_own_tau``: the same standardisation for every class at its study
        threshold. ``implant_prevalence_pct_of_teeth`` and ``implants_per_radiograph``: true
        implants estimated at threshold 0.5.
    """
    out: dict = {}
    per_ds = {}
    for ds in ("tufts", "dentex"):
        preds = data.predictions(ds)
        n_teeth = len(preds)
        _st, rows, w, _tau = data.setup(ds)
        pop: dict = defaultdict(int)  # implant predictions per confidence bin
        for r in preds:
            if r["pred"] != CLS:
                continue
            p = float(r["pred_prob"])
            pop["<0.70" if p < 0.7 else "0.70-0.90" if p < 0.9 else "0.90-0.99" if p < 0.99 else ">=0.99"] += 1
        n_pred = sum(pop.values())
        curve = []
        for t in GRID:
            pt = point(rows, w, t, n_teeth, N_PANS[ds])
            pt["ci95_in_scope"], pt["ci95_fp_per_1k"] = boot(rows, w, t, n_teeth, N_PANS[ds])
            curve.append(pt)
        reviewed = [r for r in rows if r["model_pred"] == CLS]
        conf_confirmed = [float(r["pred_prob"]) for r in reviewed if r["label"] == CLS]
        conf_corrected = [float(r["pred_prob"]) for r in reviewed if r["label"] not in [CLS] + UNUSABLE]
        geo = {}
        pred_by_tooth = {(r["pan_id"], r["tooth_idx"]): r for r in preds}
        for name, sel in (
            ("true", [r for r in reviewed if r["label"] == CLS]),
            ("false_sound", [r for r in reviewed if r["label"] == "Healthy"]),
        ):
            aspect = []  # box height / width
            for r in sel:
                p = pred_by_tooth.get((r["pan_id"], r["tooth_idx"]))
                if not p:
                    continue
                W = float(p["x2"]) - float(p["x1"])
                H = float(p["y2"]) - float(p["y1"])
                if W > 0:
                    aspect.append(H / W)
            geo[name] = {"n": len(aspect), "median_aspect_ratio": float(np.median(aspect)) if aspect else None}
        per_ds[ds] = {
            "n_teeth": n_teeth,
            "n_radiographs": N_PANS[ds],
            "n_implant_predictions": n_pred,
            "population_by_bin": dict(pop),
            "share_at_or_above_0.99": pop[">=0.99"] / n_pred if n_pred else None,
            "n_reviewed": len(reviewed),
            "review_coverage": len(reviewed) / n_pred if n_pred else None,
            "mean_score_confirmed": float(np.mean(conf_confirmed)) if conf_confirmed else None,
            "mean_score_corrected": float(np.mean(conf_corrected)) if conf_corrected else None,
            "radiographs_with_a_true_implant": len({r["pan_id"] for r in reviewed if r["label"] == CLS}),
            "radiographs_with_a_false_flag": len(
                {r["pan_id"] for r in reviewed if r["label"] not in [CLS] + UNUSABLE + OUTSIDE}
            ),
            "crop_geometry": geo,
            "curve": curve,
        }
    # Prevalence standardisation: precision = true rate / (true rate + false-positive rate),
    # with the true-implant rate of the other dataset.
    std = []
    for t, a, b in zip(GRID, per_ds["tufts"]["curve"], per_ds["dentex"]["curve"]):
        std.append(
            {
                "tau": t,
                "dentex_observed_in_scope": b["precision_in_scope"],
                "dentex_at_tufts_prevalence": a["tp_per_1k_teeth"] / (a["tp_per_1k_teeth"] + b["fp_per_1k_teeth"]),
                "tufts_observed_in_scope": a["precision_in_scope"],
                "tufts_at_dentex_prevalence": b["tp_per_1k_teeth"] / (b["tp_per_1k_teeth"] + a["fp_per_1k_teeth"]),
            }
        )
    allc: dict = {}  # every class at its own study threshold
    for ds in ("tufts", "dentex"):
        n_teeth = len(data.predictions(ds))
        st, rows, w, _tau = data.setup(ds)
        allc[ds] = {}
        for c in CLIN:
            tau_c = st[c]["tau"]
            tp = fpc = 0.0
            for r in rows:
                if r["model_pred"] != c or float(r["pred_prob"]) < tau_c:
                    continue
                wi = w[(c, r["stratum"])]
                if r["label"] == c:
                    tp += wi
                elif r["label"] not in UNUSABLE + OUTSIDE:
                    fpc += wi
            tp05 = sum(
                w[(c, r["stratum"])]
                for r in rows
                if r["model_pred"] == c and float(r["pred_prob"]) >= 0.5 and r["label"] == c
            )
            allc[ds][c] = {
                "tau": tau_c,
                "in_scope": tp / (tp + fpc) if (tp + fpc) else None,
                "confirmed_per_1k_teeth": 1000 * tp / n_teeth,
                "fp_per_1k_teeth": 1000 * fpc / n_teeth,
                "confirmed_pct_of_teeth_tau05": 100 * tp05 / n_teeth,
            }
    shifts = {}
    for c in CLIN:
        t, d = allc["tufts"][c], allc["dentex"][c]
        den = t["confirmed_per_1k_teeth"] + d["fp_per_1k_teeth"]
        d_at_t = t["confirmed_per_1k_teeth"] / den if den else None
        shifts[c] = {
            "dentex_observed": d["in_scope"],
            "dentex_at_tufts_prevalence": d_at_t,
            "shift_points": 100 * (d_at_t - d["in_scope"])
            if d_at_t is not None and d["in_scope"] is not None
            else None,
            "confirmed_pct_of_teeth_at_own_tau": {
                "tufts": t["confirmed_per_1k_teeth"] / 10,
                "dentex": d["confirmed_per_1k_teeth"] / 10,
            },
            "confirmed_pct_of_teeth_tau05": {
                "tufts": t["confirmed_pct_of_teeth_tau05"],
                "dentex": d["confirmed_pct_of_teeth_tau05"],
            },
        }
    out["all_classes_at_own_tau"] = {"per_dataset": allc, "standardisation_shift": shifts}
    out["datasets"] = per_ds
    out["prevalence_standardised"] = std
    out["implant_prevalence_pct_of_teeth"] = {
        ds: 100 * per_ds[ds]["curve"][0]["tp"] / per_ds[ds]["n_teeth"] for ds in per_ds
    }
    out["implants_per_radiograph"] = {ds: per_ds[ds]["curve"][0]["tp"] / per_ds[ds]["n_radiographs"] for ds in per_ds}
    if write:
        write_json(data.paths.review / "implant_view.json", out, indent=1)
    return out
