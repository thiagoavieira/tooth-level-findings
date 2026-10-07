"""Consistency checks against the DENTEX annotations.

* :func:`impacted_check`: the impacted class against the DENTEX disease-subset annotations
  (detection recall, classifier recall on detected teeth, end-to-end recall, a lower bound on
  precision, and what the classifier predicts for teeth with the other DENTEX diagnoses).
  Ground-truth and predicted boxes are matched greedily at IoU >= 0.5. Writes
  ``external/dentex/dentex_impacted_eval.json``.
* :func:`trace_implants`: finds the source radiograph of each of the 142 manually cropped DENTEX
  implants by multi-scale normalised cross-correlation (NCC). Writes
  ``external/dentex/dentex_implant_provenance.csv``.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from ..config import Paths
from ..constants import CLASSES4
from ..evaluation.robustness import iou
from ..utils import read_csv, read_json, write_dict_csv

#: Minimum IoU for a predicted box to match a DENTEX ground-truth box.
IOU_THR = 0.5
#: DENTEX disease-subset annotations, relative to the DENTEX root.
DISEASE_JSON = "dentex_classification/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json"


def impacted_check(annotations: dict, predictions: list[dict]) -> dict:
    """Compare the impacted predictions with the DENTEX disease annotations.

    Only DENTEX diagnoses are annotated, so a predicted impacted tooth with no matching
    "Impacted" box is not necessarily wrong: ``precision_lower_bound`` counts it as wrong.

    Parameters
    ----------
    annotations : dict
        COCO-style DENTEX quadrant-enumeration-disease annotations (``categories_3`` are the
        diagnoses, ``category_id_1``/``category_id_2`` the zero-based quadrant and tooth number).
    predictions : list of dict
        Rows of ``external/dentex/predictions.csv``; only radiographs whose ``pan_id`` starts
        with ``disease__`` are used.

    Returns
    -------
    dict
        Content of ``dentex_impacted_eval.json``: counts per diagnosis, detection recall per
        diagnosis, the ``impacted`` summary and, for each diagnosis, the classes predicted for
        its detected teeth.
    """
    diagnosis_name = {c["id"]: c["name"] for c in annotations["categories_3"]}
    img_by_id = {im["id"]: im for im in annotations["images"]}
    # gt[stem]: (box xyxy, diagnosis, quadrant, tooth number); preds[stem]: (box, class, prob, seeds agreeing)
    gt = collections.defaultdict(list)
    for a in annotations["annotations"]:
        im = img_by_id[a["image_id"]]
        x, y, w, h = a["bbox"]
        gt[im["file_name"][:-4]].append(
            ((x, y, x + w, y + h), diagnosis_name[a["category_id_3"]], a["category_id_1"] + 1, a["category_id_2"] + 1)
        )
    preds = collections.defaultdict(list)
    for r in predictions:
        if not r["pan_id"].startswith("disease__"):
            continue
        preds[r["pan_id"].split("__", 1)[1]].append(
            (
                (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])),
                r["pred"],
                float(r["pred_prob"]),
                int(r["seed_agreement"]),
            )
        )
    n_gt, n_detected, pred_of_gt = collections.Counter(), collections.Counter(), collections.Counter()
    imp_total = imp_match_imp = imp_match_other = imp_unmatched = 0
    imp_gt_type, imp_hit_type = collections.Counter(), collections.Counter()
    impacted_not_in_dentex_per_pan = []
    for stem, g in gt.items():
        p = preds.get(stem, [])
        # Greedy one-to-one matching, highest IoU first.
        candidates = sorted(
            ((iou(gb, pb), gi, pi) for gi, (gb, *_) in enumerate(g) for pi, (pb, *_) in enumerate(p)), reverse=True
        )
        gt_to_pred, pred_to_gt = {}, {}
        for v, gi, pi in candidates:
            if v < IOU_THR:
                break
            if gi in gt_to_pred or pi in pred_to_gt:
                continue
            gt_to_pred[gi], pred_to_gt[pi] = pi, gi
        for gi, (_gb, dz, _q, t) in enumerate(g):
            n_gt[dz] += 1
            if dz == "Impacted":
                imp_gt_type[t] += 1
            if gi in gt_to_pred:
                n_detected[dz] += 1
                pl = p[gt_to_pred[gi]][1]
                pred_of_gt[(dz, pl)] += 1
                if dz == "Impacted" and pl == "Impacted":
                    imp_hit_type[t] += 1
        not_in_dentex = 0
        for pi, (_pb, pl, _pp, _ag) in enumerate(p):
            if pl != "Impacted":
                continue
            imp_total += 1
            if pi in pred_to_gt:
                if g[pred_to_gt[pi]][1] == "Impacted":
                    imp_match_imp += 1
                else:
                    imp_match_other += 1
                    not_in_dentex += 1
            else:
                imp_unmatched += 1
                not_in_dentex += 1
        impacted_not_in_dentex_per_pan.append(not_in_dentex)
    return {
        "n_pans_with_gt": len(gt),
        "n_pans_with_preds": sum(1 for s in gt if s in preds),
        "gt_counts": dict(n_gt),
        "detection_recall_by_diagnosis": {k: n_detected[k] / n_gt[k] for k in n_gt},
        "impacted": {
            "gt": n_gt["Impacted"],
            "detected": n_detected["Impacted"],
            "recall_end_to_end": pred_of_gt[("Impacted", "Impacted")] / n_gt["Impacted"],
            "recall_classifier_only_on_detected": pred_of_gt[("Impacted", "Impacted")] / max(1, n_detected["Impacted"]),
            "pred_total": imp_total,
            "pred_matching_dentex_impacted": imp_match_imp,
            "pred_matching_other_dentex_diagnosis": imp_match_other,
            "pred_unmatched_no_dentex_annotation": imp_unmatched,
            "precision_lower_bound": imp_match_imp / max(1, imp_total),
            "impacted_flags_not_in_dentex_per_pan_mean": sum(impacted_not_in_dentex_per_pan)
            / len(impacted_not_in_dentex_per_pan),
            "gt_by_tooth_number": dict(sorted(imp_gt_type.items())),
            "hit_by_tooth_number": dict(sorted(imp_hit_type.items())),
        },
        "prediction_of_detected_dentex_teeth": {dz: {c: pred_of_gt[(dz, c)] for c in CLASSES4} for dz in n_gt},
    }


def run_impacted_check(paths: Paths) -> dict:
    """Run :func:`impacted_check` and write ``external/dentex/dentex_impacted_eval.json``.

    Needs the DENTEX annotations under ``paths.dentex``.
    """
    res = impacted_check(
        read_json(paths.dentex / DISEASE_JSON), read_csv(paths.external / "dentex" / "predictions.csv")
    )
    with open(paths.external / "dentex" / "dentex_impacted_eval.json", "w") as fh:
        json.dump(res, fh, indent=2)
    return res


# --------------------------------------------------------------------------- implant provenance
#: Downsampling factor of the radiographs in the coarse search.
COARSE = 4
#: Crop scales tried in the coarse search.
SCALES = (0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.30)
#: Bounds of the crop scale in the fine search.
SCALE_MIN, SCALE_MAX = 0.45, 2.00
#: Number of distinct candidate radiographs refined at full resolution.
TOPK = 6
#: Minimum NCC score of an accepted match.
ACCEPT = 0.97


def _pan_id(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"{'disease' if 'quadrant-enumeration-disease' in path else 'enum'}__{stem}"


def trace_implants(paths: Paths, limit: int = 0) -> Path:
    """Match each pre-cropped DENTEX implant to its source radiograph by multi-scale NCC.

    Coarse search: every crop scale in ``SCALES`` against every radiograph downsampled by
    ``COARSE``; the ``TOPK`` best distinct radiographs are kept. Fine search: at full
    resolution, scales in steps of 0.01 around the coarse scale, widening the range until the
    best scale is interior or hits ``SCALE_MIN``/``SCALE_MAX``. The margin against the best
    score on a different radiograph separates a real match from a generic posterior-tooth
    pattern; matches scoring at least ``ACCEPT`` (0.97) are accepted.

    Parameters
    ----------
    paths : Paths
        Configured paths; needs ``paths.dentex`` (radiographs) and ``paths.dentex_implant_crops``.
    limit : int
        Process only the first ``limit`` crops (0: all).

    Returns
    -------
    Path
        The written ``external/dentex/dentex_implant_provenance.csv``.
    """
    import cv2

    def gray(p):
        return cv2.imread(p, cv2.IMREAD_GRAYSCALE)

    def rescale(img, s):
        h, w = img.shape
        return cv2.resize(
            img,
            (max(4, int(round(w * s))), max(4, int(round(h * s)))),
            interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC,
        )

    def best_match(tpl, img):
        if tpl.shape[0] > img.shape[0] or tpl.shape[1] > img.shape[1]:
            return -1.0, (0, 0)
        _, mx, _, loc = cv2.minMaxLoc(cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED))
        return float(mx), loc

    cv2.setNumThreads(4)
    pans = sorted(glob.glob(os.path.join(paths.dentex, "**", "xrays", "*.png"), recursive=True))
    coarse = []
    for p in pans:
        g = gray(p)
        if g is not None:
            coarse.append(
                (p, cv2.resize(g, (g.shape[1] // COARSE, g.shape[0] // COARSE), interpolation=cv2.INTER_AREA))
            )
    crops = sorted(
        glob.glob(os.path.join(paths.dentex_implant_crops, "*.png")),
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0]),
    )
    if limit:
        crops = crops[:limit]
    rows = []
    for cf in crops:
        t1 = time.time()
        c = gray(cf)
        ranked = sorted(
            ((best_match(rescale(c, s / COARSE), im)[0], p, s) for s in SCALES for p, im in coarse), reverse=True
        )
        seen, candidates = set(), []
        for _sc, p, s in ranked:
            if p in seen:
                continue
            seen.add(p)
            candidates.append((p, s))
            if len(candidates) >= TOPK:
                break
        fine = []
        for p, s0 in candidates:
            g = gray(p)
            if g is None:
                continue
            # cache: scale -> (score, location); widen [lo, hi] until the best scale is interior.
            cache, lo, hi = {}, s0 - 0.05, s0 + 0.05
            for _ in range(8):
                for s in np.arange(lo, hi + 0.001, 0.01):
                    k = round(float(s), 2)
                    if k in cache or not (SCALE_MIN <= k <= SCALE_MAX):
                        continue
                    cache[k] = best_match(rescale(c, k), g)
                k_best = max(cache, key=lambda k: cache[k][0])
                if (min(cache) + 0.005 < k_best < max(cache) - 0.005) or not (SCALE_MIN < k_best < SCALE_MAX):
                    break
                lo, hi = min(cache) - 0.05, max(cache) + 0.05
            k_best = max(cache, key=lambda k: cache[k][0])
            sc, loc = cache[k_best]
            fine.append((sc, p, k_best, loc))
        fine.sort(reverse=True)
        s1, p1, scale1, loc = fine[0]
        other = next((f[0] for f in fine[1:] if f[1] != p1), -1.0)
        rows.append(
            {
                "crop": os.path.basename(cf),
                "patient_id": "dentex_" + os.path.splitext(os.path.basename(cf))[0],
                "pan_id": _pan_id(p1),
                "pan_path": Path(p1).relative_to(paths.dentex).as_posix(),
                "score": round(s1, 4),
                "scale": round(scale1, 3),
                "best_other_pan": round(other, 4),
                "margin": round(s1 - other, 4),
                "x1": loc[0],
                "y1": loc[1],
                "x2": loc[0] + int(round(c.shape[1] * scale1)),
                "y2": loc[1] + int(round(c.shape[0] * scale1)),
                "accepted": int(s1 >= ACCEPT),
            }
        )
        print(f"{os.path.basename(cf):>8s} -> {_pan_id(p1)} score {s1:.4f} ({time.time() - t1:.1f}s)", flush=True)
    out = paths.external / "dentex" / "dentex_implant_provenance.csv"
    write_dict_csv(out, rows)
    return out
