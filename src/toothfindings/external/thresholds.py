"""Study thresholds and the selection of external crops for expert review.

1. Calibration on the in-domain deployment-prevalence predictions of the audited model: for
   each class ``c``, the study threshold ``tau_c`` is the smallest grid value at which the
   predictions ``pred = c`` with confidence ``>= tau_c`` reach the target precision (0.95) on
   at least 20 crops. The published thresholds are 0.70 (endodontically treated), 0.50 (sound
   and implant) and 0.99 (impacted).
2. Selection from each external prediction file: per class a ``confirm`` stratum (confidence
   ``>= tau_c`` and all three seeds agreeing; a random sample of the class quota, sound teeth
   taking up to half the quota from the first and last three detections of a radiograph,
   i.e. posterior teeth) and a ``below_tau`` stratum (random sample of 50).

Class names follow the code (``Endodontics``, ``Healthy`` = sound, ``Impacted``, ``Implant``).
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path

import numpy as np

from ..constants import CLASSES4
from ..utils import read_csv, write_dict_csv

#: Candidate study thresholds.
GRID = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99]


def calibrate(prev_csv: Path, target: float = 0.95, grid=GRID) -> dict:
    """Calibrate the per-class study thresholds on in-domain predictions.

    Parameters
    ----------
    prev_csv : Path
        Deployment-prevalence predictions with a ``class`` (ground truth) column and one
        ``p_<class>`` probability column per class.
    target : float
        Target precision.
    grid : sequence of float
        Candidate thresholds, tried in order.

    Returns
    -------
    dict
        ``{class: {"tau", "base_precision", "curve"}}``; ``tau`` is ``None`` when no grid value
        reaches the target on at least 20 crops, ``base_precision`` is the precision of all
        predictions of the class, and ``curve`` lists per threshold ``precision``,
        ``kept_frac`` (share of the class predictions kept), ``recall`` and ``n`` (kept).
    """
    rows = read_csv(prev_csv)
    y = np.array([CLASSES4.index(r["class"]) for r in rows])
    P = np.array([[float(r[f"p_{c}"]) for c in CLASSES4] for r in rows])
    pred, conf = P.argmax(1), P.max(1)
    out = {}
    for ci, c in enumerate(CLASSES4):
        predicted = pred == ci
        curve, tau = [], None
        for t in grid:
            kept = predicted & (conf >= t)
            prec = float((y[kept] == ci).mean()) if kept.sum() else float("nan")
            curve.append(
                {
                    "tau": t,
                    "precision": prec,
                    "kept_frac": float(kept.sum() / max(1, predicted.sum())),
                    "recall": float((kept & (y == ci)).sum() / max(1, (y == ci).sum())),
                    "n": int(kept.sum()),
                }
            )
            if tau is None and kept.sum() >= 20 and prec >= target:
                tau = t
        out[c] = {"tau": tau, "base_precision": float((y[predicted] == ci).mean()), "curve": curve}
    return out


def select_for_review(
    prev_csv: Path,
    external: list[Path],
    tag: str,
    target: float = 0.95,
    quota: str = "Impacted:400,Implant:300,Endodontics:250,Healthy:250",
    below: int = 50,
    fallback_tau: float = 0.90,
) -> dict:
    """Calibrate the study thresholds and draw the review sample of each external dataset.

    Writes ``keep_<ds>_<tag>.csv`` (selected rows with a ``bucket`` column), ``keep_<ds>_<tag>.txt``
    (crop stems) and ``thresholds_<tag>.json`` next to each external predictions file. One
    ``Random(0)`` generator is shared by all datasets and classes, so the samples depend on the
    order of ``external``.

    Parameters
    ----------
    prev_csv : Path
        In-domain deployment-prevalence predictions used by :func:`calibrate`.
    external : list of Path
        External prediction files (``external/<ds>/predictions_<tag>.csv``); the dataset name
        is the parent folder.
    tag : str
        Suffix of the output files.
    target : float
        Target precision of the calibration.
    quota : str
        Size of the ``confirm`` sample per class, as ``Class:n,Class:n,...``.
    below : int
        Size of the ``below_tau`` sample per class.
    fallback_tau : float
        Threshold used for a class whose calibration found none.

    Returns
    -------
    dict
        ``{dataset: {class: {"tau", "predicted", "above_tau_3of3", "below_tau", "sent_confirm",
        "sent_below"}}}``.
    """
    cal = calibrate(prev_csv, target)
    quotas = {k: int(v) for k, v in (x.split(":") for x in quota.split(","))}
    rng = random.Random(0)
    summaries = {}
    for ext in external:
        ext = Path(ext)
        ds = ext.parent.name
        rows = read_csv(ext)
        teeth_per_pan = collections.Counter(r["pan_id"] for r in rows)
        keep, summary = [], {}
        for c in CLASSES4:
            tau = cal[c]["tau"] or fallback_tau
            preds = [r for r in rows if r["pred"] == c]
            above = [r for r in preds if float(r["pred_prob"]) >= tau and int(r["seed_agreement"]) == 3]
            below_tau = [r for r in preds if float(r["pred_prob"]) < tau]
            if c == "Healthy":
                # Posterior teeth: the first and last three detections (numbered left to right).
                posterior = [
                    r for r in above if int(r["tooth_idx"]) < 3 or int(r["tooth_idx"]) >= teeth_per_pan[r["pan_id"]] - 3
                ]
                rest = [r for r in above if r not in posterior]
                rng.shuffle(posterior)
                rng.shuffle(rest)
                sel_confirm = posterior[: quotas[c] // 2] + rest[: quotas[c] - min(len(posterior), quotas[c] // 2)]
            else:
                rng.shuffle(above)
                sel_confirm = above[: quotas[c]]
            rng.shuffle(below_tau)
            sel_below = below_tau[:below]
            for r in sel_confirm:
                r["bucket"] = "confirm"
            for r in sel_below:
                r["bucket"] = "below_tau"
            keep += sel_confirm + sel_below
            summary[c] = {
                "tau": tau,
                "predicted": len(preds),
                "above_tau_3of3": len(above),
                "below_tau": len(below_tau),
                "sent_confirm": len(sel_confirm),
                "sent_below": len(sel_below),
            }
        base = ext.parent / f"keep_{ds}_{tag}"
        write_dict_csv(f"{base}.csv", keep)
        Path(f"{base}.txt").write_text("\n".join(Path(r["crop_file"]).name[:-4] for r in keep) + "\n")
        with open(ext.parent / f"thresholds_{tag}.json", "w") as fh:
            json.dump({"calibration": cal, "target": target, "summary": summary}, fh, indent=2)
        summaries[ds] = summary
    return summaries
