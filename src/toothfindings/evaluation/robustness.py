"""Robustness of the classifiers to imperfect crops.

* :func:`crop_jitter`: synthetic box noise on the primary test crops, without retraining.
  Jitter level ``d``: centre shift ``U(-d, d) * (w, h)`` and per-axis scale ``U(1-d, 1+d)``, one
  draw per crop from ``default_rng(1234)``, re-seeded for each level. Scale condition ``s``:
  the box is scaled deterministically about its centre. The 37 pre-cropped DENTEX implants
  have no radiograph and are excluded.
* :func:`end_to_end`: the real upstream modules (mouth detector and tooth segmenter) run on the
  test radiographs; detected boxes are matched to the annotated teeth greedily at IoU >= 0.5
  and the classifiers are scored on both the annotated and the detected boxes.

Outputs go to ``experiments/robustness/``.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import Paths
from ..constants import CLASSES4
from ..data.crops import gray3
from ..data.lyria_blocks import iter_blocks, parse_crop_name
from ..utils import read_csv

JITTER = [0.05, 0.10, 0.20, 0.30]  # jitter levels d
SCALES = [0.7, 0.85, 1.15, 1.3, 1.5]  # deterministic box scales s


def perturb(box, W: int, H: int, rng=None, d: float = 0.0, s: float | None = None):
    """Jitter and/or scale a box, clipped to the image.

    Parameters
    ----------
    box : tuple of int
        ``(x1, y1, x2, y2)``.
    W, H : int
        Image width and height.
    rng : numpy.random.Generator, optional
        Random generator; jitter is applied only when given.
    d : float
        Jitter level (see the module docstring).
    s : float, optional
        Deterministic scale about the box centre.

    Returns
    -------
    tuple of int
        The new box, or the original box if the result is narrower or shorter than 4 px.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    if rng is not None:
        cx += rng.uniform(-d, d) * w
        cy += rng.uniform(-d, d) * h
        w *= rng.uniform(1 - d, 1 + d)
        h *= rng.uniform(1 - d, 1 + d)
    if s is not None:
        w *= s
        h *= s
    nx1, ny1 = max(0, int(round(cx - w / 2))), max(0, int(round(cy - h / 2)))
    nx2, ny2 = min(W, int(round(cx + w / 2))), min(H, int(round(cy + h / 2)))
    if nx2 - nx1 < 4 or ny2 - ny1 < 4:
        return box
    return (nx1, ny1, nx2, ny2)


def iou(a, b) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def greedy_match(gt_boxes: list, pred_boxes: list, thr: float) -> dict[int, tuple[int, float]]:
    """Match annotated and predicted boxes one to one, greedily by decreasing IoU.

    Pairs below ``thr`` are never matched. Returns ``{gt index: (pred index, iou)}``.
    """
    candidates = sorted(
        ((iou(g, pb), gi, pi) for gi, g in enumerate(gt_boxes) for pi, pb in enumerate(pred_boxes)), reverse=True
    )
    matched, used = {}, set()
    for v, gi, pi in candidates:
        if v < thr:
            break
        if gi in matched or pi in used:
            continue
        matched[gi] = (pi, v)
        used.add(pi)
    return matched


def crop_jitter(paths: Paths, archs: list[str]) -> None:
    """Score the three-seed ensembles on jittered and rescaled test crops.

    Needs the restricted images and the trained weights. Writes
    ``experiments/robustness/jitter_<arch>.json`` (per condition: macro-F1 mean and population
    sd over seeds, per-class F1, recall and precision means) and ``jitter_<arch>.csv``.
    """
    from ..models.ensemble import Ensemble, metrics

    test = [r for r in read_csv(paths.split_dir("splits") / "test.csv") if r["source"] == "lyria"]
    by_pid: dict[str, list] = {}
    for r in test:
        pid, idx = parse_crop_name(r["filepath"])
        by_pid.setdefault(pid, []).append((r["class"], idx))
    items = []  # (pid, class, box, W, H) of every test crop found in the annotations
    for pid, blocks in by_pid.items():
        W, H = Image.open(paths.lyria_image(pid)).size
        need = set(blocks)
        for cls, idx, box in iter_blocks(paths.lyria_json, paths.lyria_images, pid, (W, H)):
            if (cls, idx) in need:
                items.append((pid, cls, box, W, H))
    print(f"primary test crops located: {len(items)} / {len(test)}")
    y = [CLASSES4.index(c) for _, c, _, _, _ in items]
    conditions = (
        [("orig", None, None)] + [(f"jitter_{d}", d, None) for d in JITTER] + [(f"scale_{s}", None, s) for s in SCALES]
    )
    out_dir = paths.experiments / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    def build(d, s_):
        """Crop every item under jitter level ``d`` and/or scale ``s_``."""
        rng = np.random.default_rng(1234) if d is not None else None
        crops, cur_pid, img = [], None, None
        for pid, _cls, box, W, H in items:
            if pid != cur_pid:
                img = Image.open(paths.lyria_image(pid)).convert("L")
                cur_pid = pid
            crops.append(gray3(img.crop(perturb(box, W, H, rng, d or 0.0, s_))))
        return crops

    for arch in archs:
        ens = Ensemble(paths.experiments, arch)
        results, rows = {}, []
        for name, d, s in conditions:
            probs = ens.predict(build(d, s))
            per_seed = [metrics(y, probs[:, si].argmax(1)) for si in range(probs.shape[1])]
            macro_f1 = [m["macro_f1"] for m in per_seed]
            res = {
                "macro_f1_mean": float(np.mean(macro_f1)),
                "macro_f1_std": float(np.std(macro_f1)),
                "per_class_f1_mean": {c: float(np.mean([m["per_class"][c]["f1"] for m in per_seed])) for c in CLASSES4},
                "per_class_recall_mean": {
                    c: float(np.mean([m["per_class"][c]["recall"] for m in per_seed])) for c in CLASSES4
                },
                "per_class_precision_mean": {
                    c: float(np.mean([m["per_class"][c]["precision"] for m in per_seed])) for c in CLASSES4
                },
            }
            results[name] = res
            rows.append(
                [name, f"{res['macro_f1_mean']:.4f}", f"{res['macro_f1_std']:.4f}"]
                + [f"{res['per_class_f1_mean'][c]:.4f}" for c in CLASSES4]
            )
            print(arch, name, f"macroF1={res['macro_f1_mean']:.4f}", flush=True)
        with open(out_dir / f"jitter_{arch}.json", "w") as fh:
            json.dump({"n": len(items), "results": results}, fh, indent=2)
        with open(out_dir / f"jitter_{arch}.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["condition", "macro_f1_mean", "macro_f1_std"] + [f"f1_{c}" for c in CLASSES4])
            writer.writerows(rows)


def detection_recall_from_teeth(teeth_csv: Path) -> dict[str, tuple[int, int]]:
    """Count detected and annotated teeth per class from ``end2end_teeth.csv``.

    Returns ``{class: (detected, annotated)}``; the CSV has one row per annotated tooth.
    """
    out = {c: [0, 0] for c in CLASSES4}
    for r in read_csv(teeth_csv):
        out[r["class"]][1] += 1
        out[r["class"]][0] += int(r["detected"])
    return {c: (v[0], v[1]) for c, v in out.items()}


def end_to_end(paths: Paths, archs: list[str], iou_thr: float = 0.5, limit: int = 0) -> dict:
    """Score the classifiers on boxes detected by the upstream modules on the test radiographs.

    Needs the restricted images, the trained weights and the ONNX mouth detector and tooth
    detector. Writes ``experiments/robustness/end2end_teeth.csv`` (one row per annotated tooth:
    matched or not, IoU, annotated and detected boxes) and ``end2end_<arch>.json``.

    Parameters
    ----------
    paths : Paths
        Repository and data locations.
    archs : list of str
        Backbones (three-seed ensembles).
    iou_thr : float
        Minimum IoU to match a detected box to an annotated tooth.
    limit : int
        Process only the first ``limit`` patients (0: all).

    Returns
    -------
    dict
        Detection summary (``n_pans`` radiographs, ``n_annotated_teeth``, ``n_matched``,
        ``detection_recall`` per class, ``iou_matched``) plus one entry per backbone with
        macro-F1 on annotated (``gt_boxes``) and detected boxes, the prediction agreement
        between both, and ``end_to_end_recall`` (correct on a detected box, over all
        annotated teeth of the class).
    """
    import cv2

    from ..external.pipeline import MouthDetector, ToothDetector
    from ..models.ensemble import Ensemble, metrics

    test = [r for r in read_csv(paths.split_dir("splits") / "test.csv") if r["source"] == "lyria"]
    want: dict[str, set] = {}
    for r in test:
        pid, idx = parse_crop_name(r["filepath"])
        want.setdefault(pid, set()).add((r["class"], idx))
    pids = sorted(want)[:limit] if limit else sorted(want)
    mouth = MouthDetector(paths.mouth_onnx, 0.5)
    tooth = ToothDetector(paths.tooth_onnx, 0.5, device="cpu")
    ens = {a: Ensemble(paths.experiments, a) for a in archs}
    out_dir = paths.experiments / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, y_gt, gt_crops, det_crops, ious = [], [], [], [], []
    t0 = time.time()
    for k, pid in enumerate(pids):
        bgr = cv2.imread(str(paths.lyria_image(pid)))
        H, W = bgr.shape[:2]
        det = mouth(bgr)
        mx1, my1, mx2, my2 = det[0] if det else [0, 0, W, H]
        boxes, _scores = tooth(bgr[my1:my2, mx1:mx2])
        # detected boxes, shifted from the mouth crop back to radiograph coordinates
        pboxes = [
            (int(round(b[0])) + mx1, int(round(b[1])) + my1, int(round(b[2])) + mx1, int(round(b[3])) + my1)
            for b in boxes
        ]
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert("L")
        gts = [
            (c, i, b)
            for c, i, b in iter_blocks(paths.lyria_json, paths.lyria_images, pid, (W, H))
            if (c, i) in want[pid]
        ]
        matched = greedy_match([g[2] for g in gts], pboxes, iou_thr)
        for gi, (cls, idx, gbox) in enumerate(gts):
            if gi in matched:
                pi, v = matched[gi]
                pb = pboxes[pi]
                y_gt.append(CLASSES4.index(cls))
                gt_crops.append(gray3(pil.crop(gbox)))
                det_crops.append(gray3(pil.crop(pb)))
                ious.append(v)
                rows.append([pid, cls, idx, 1, round(v, 3), *gbox, *pb])
            else:
                rows.append([pid, cls, idx, 0, 0.0, *gbox, "", "", "", ""])
        if (k + 1) % 50 == 0:
            print(f"[{k + 1}/{len(pids)}] {time.time() - t0:.0f}s matched={len(y_gt)}", flush=True)

    det_by_cls = {c: [0, 0] for c in CLASSES4}
    for r in rows:
        det_by_cls[r[1]][1] += 1
        det_by_cls[r[1]][0] += r[3]
    out = {
        "n_pans": len(pids),
        "n_annotated_teeth": len(rows),
        "n_matched": len(y_gt),
        "iou_thr": iou_thr,
        "detection_recall": {c: (v[0] / v[1] if v[1] else None) for c, v in det_by_cls.items()},
        "iou_matched": {
            "mean": float(np.mean(ious)),
            "median": float(np.median(ious)),
            "p10": float(np.percentile(ious, 10)),
        }
        if ious
        else {},
    }
    for a, e in ens.items():
        # per-seed probabilities on the annotated (pg) and detected (pd) boxes
        pg, pd = e.predict(gt_crops), e.predict(det_crops)
        res_g = [metrics(y_gt, pg[:, s].argmax(1)) for s in range(pg.shape[1])]
        res_d = [metrics(y_gt, pd[:, s].argmax(1)) for s in range(pd.shape[1])]
        agree = float(np.mean([(pg[:, s].argmax(1) == pd[:, s].argmax(1)).mean() for s in range(pg.shape[1])]))
        e2e = {}
        for ci, c in enumerate(CLASSES4):
            n_c = det_by_cls[c][1]
            correct = np.mean([((pd[:, s].argmax(1) == ci) & (np.array(y_gt) == ci)).sum() for s in range(pd.shape[1])])
            e2e[c] = float(correct / n_c) if n_c else None
        out[a] = {
            "gt_boxes": {
                "macro_f1_mean": float(np.mean([m["macro_f1"] for m in res_g])),
                "macro_f1_std": float(np.std([m["macro_f1"] for m in res_g])),
                "per_class_f1": {c: float(np.mean([m["per_class"][c]["f1"] for m in res_g])) for c in CLASSES4},
            },
            "detected_boxes": {
                "macro_f1_mean": float(np.mean([m["macro_f1"] for m in res_d])),
                "macro_f1_std": float(np.std([m["macro_f1"] for m in res_d])),
                "per_class_f1": {c: float(np.mean([m["per_class"][c]["f1"] for m in res_d])) for c in CLASSES4},
                "per_class_recall": {c: float(np.mean([m["per_class"][c]["recall"] for m in res_d])) for c in CLASSES4},
                "per_class_precision": {
                    c: float(np.mean([m["per_class"][c]["precision"] for m in res_d])) for c in CLASSES4
                },
            },
            "prediction_agreement_gt_vs_detected": agree,
            "end_to_end_recall": e2e,
        }
        # `out` accumulates every backbone, so each file also holds the backbones scored before it
        with open(out_dir / f"end2end_{a}.json", "w") as fh:
            json.dump(out, fh, indent=2)
    with open(out_dir / "end2end_teeth.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["patient_id", "class", "idx", "detected", "iou", "gx1", "gy1", "gx2", "gy2", "px1", "py1", "px2", "py2"]
        )
        writer.writerows(rows)
    return out


__all__ = ["perturb", "iou", "greedy_match", "crop_jitter", "end_to_end", "detection_recall_from_teeth"]
