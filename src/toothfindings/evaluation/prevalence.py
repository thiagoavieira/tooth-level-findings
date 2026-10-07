"""Evaluation at deployment prevalence.

The *capped* test set is the canonical test split, in which the sound class was capped (1,300
sound crops over the whole dataset, about two per patient) so that the classes are roughly
balanced. A real panoramic
exam contains far more sound teeth, so the deployment-prevalence set adds every remaining
strictly sound tooth of the 586 primary-source (InReDD-PAN924, ``lyria`` in file names) test
patients, 8,672 crops in total. Options used by the mitigation study:

``scale``
    Re-extract every primary crop with its box scaled by ``s`` about the centre (1.0: the
    published crops; 1.3: context-preserving crops). The pre-cropped DENTEX implants are
    unchanged.
``prior``
    Post-hoc tooth-position prior ``p'(c|x,t) ∝ p(c|x) P(c|t) / P_train(c)``, where ``t`` is the
    tooth type, ``P(c|t)`` is estimated with Laplace smoothing (+1/+4) from every four-class
    annotation block of the training-partition patients, and ``P_train(c) = 1/4``.

Outputs per run, in ``experiments/prevalence/<arch>__<tag>/``: ``metrics.json``,
``per_seed.json`` and ``predictions.csv`` (ensemble mean probabilities).
"""

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import Paths
from ..constants import CLASSES4, tooth_type
from ..data.crops import gray3
from ..data.lyria_blocks import block_index, iter_blocks, scaled_box
from ..utils import read_csv, write_json

METRIC_KEYS = ["precision", "recall", "f1"]


# --------------------------------------------------------------------------- pure functions (results/ only)
def mean_sd(values) -> list[float]:
    """Return ``[mean, sample sd (ddof=1)]``, the format of every prevalence ``metrics.json``."""
    return [float(np.mean(values)), float(np.std(values, ddof=1))]


def aggregate_per_seed(per_seed: dict, classes: list[str] = CLASSES4) -> dict:
    """Average the per-seed metrics of ``per_seed.json`` into seed means and sample sds.

    Parameters
    ----------
    per_seed : dict
        ``{"capped": [metrics of seed 0, ...], "full": [...]}``; each metrics dict holds
        ``macro_f1`` and ``per_class[<class>][precision|recall|f1]``.
    classes : list of str
        Class names to report.

    Returns
    -------
    dict
        ``macro_f1_capped`` and ``macro_f1_full`` as ``[mean, sd]``, and ``per_class_full`` and
        ``per_class_capped`` as ``{class: {metric: [mean, sd]}}``.
    """
    out = {
        "macro_f1_capped": mean_sd([m["macro_f1"] for m in per_seed["capped"]]),
        "macro_f1_full": mean_sd([m["macro_f1"] for m in per_seed["full"]]),
    }
    for split in ("full", "capped"):
        out[f"per_class_{split}"] = {
            c: {key: mean_sd([m["per_class"][c][key] for m in per_seed[split]]) for key in METRIC_KEYS} for c in classes
        }
    return out


def false_flags_per_exam(true_cls, pred_idx, patient_ids, classes: list[str] = CLASSES4) -> dict:
    """Summarise the false flags per exam over patients with at least one sound tooth.

    A false flag is a sound tooth predicted as any other class; each patient is one exam.

    Parameters
    ----------
    true_cls : sequence of str
        True class of each crop.
    pred_idx : sequence of int
        Ensemble prediction of each crop, as an index into ``classes``.
    patient_ids : sequence of str
        Patient of each crop.
    classes : list of str
        Class names; must contain ``"Healthy"``.

    Returns
    -------
    dict
        ``mean``, ``median``, ``p90`` (90th percentile) of the flags per exam, ``share_zero``
        (share of exams without a false flag) and ``n_exams``.
    """
    healthy = classes.index("Healthy")
    # patient -> [false flags, sound teeth]
    by_patient: dict = defaultdict(lambda: [0, 0])
    for true, pred, pid in zip(true_cls, pred_idx, patient_ids):
        if true == "Healthy":
            by_patient[pid][1] += 1
            if pred != healthy:
                by_patient[pid][0] += 1
    flags_per_exam = [v[0] for v in by_patient.values()]
    return {
        "mean": float(np.mean(flags_per_exam)),
        "median": float(np.median(flags_per_exam)),
        "p90": float(np.percentile(flags_per_exam, 90)),
        "share_zero": float(np.mean(np.array(flags_per_exam) == 0)),
        "n_exams": len(flags_per_exam),
    }


def apply_prior(probs: np.ndarray, types: list[str], prior: dict[str, np.ndarray], k: int = 4) -> np.ndarray:
    """Reweight per-seed probabilities by the tooth-position prior and renormalise.

    Parameters
    ----------
    probs : ndarray, shape (n, seeds, k)
        Softmax probabilities of each crop and seed.
    types : list of str
        Tooth type of each crop. Types missing from ``prior`` (e.g. ``"unknown"`` for the
        pre-cropped DENTEX implants) get a uniform prior, i.e. no change.
    prior : dict of str to ndarray
        ``P(c | t)`` for each tooth type ``t``.
    k : int
        Number of classes; dividing by ``P_train(c) = 1/k`` is a multiplication by ``k``.

    Returns
    -------
    ndarray, shape (n, seeds, k)
        Adjusted probabilities, summing to one over the last axis.
    """
    weights = np.stack([prior.get(t, np.ones(k) / k) for t in types]) * float(k)
    probs = probs * weights[:, None, :]
    return probs / probs.sum(-1, keepdims=True)


def laplace_prior(counts: dict[str, Counter], classes: list[str] = CLASSES4) -> dict[str, np.ndarray]:
    """Estimate ``P(c | t) = (n_tc + 1) / (n_t + 4)`` for every tooth type ``t``.

    ``counts`` maps each tooth type to a counter of annotated classes. The denominator uses
    +4 whatever the length of ``classes``.
    """
    prior = {}
    for t, class_counts in counts.items():
        n = sum(class_counts.values())
        prior[t] = np.array([(class_counts[c] + 1) / (n + 4) for c in classes])
    return prior


# --------------------------------------------------------------------------- with images and weights
def build_rows(paths: Paths) -> list[list]:
    """List the crops of the deployment-prevalence set.

    The capped test set comes first, in ``test.csv`` order, followed by the remaining sound
    teeth of every primary test patient. Missing sound crops are cut from the radiograph into
    ``crops_prevalence/Healthy`` on first use.

    Returns
    -------
    list of list
        One row ``[filepath, class, source, patient_id, block_idx, in_capped]`` per crop;
        ``block_idx`` is the annotation block index (-1 for non-primary crops) and
        ``in_capped`` is 1 for capped-test crops, else 0.
    """
    test = read_csv(paths.split_dir("splits") / "test.csv")
    rows, capped_sound = [], set()
    for r in test:
        base = os.path.basename(r["filepath"])[:-4]
        # primary crop names are "lyria__<pid>__b<idx>.png"
        idx = int(base.split("__")[2][1:]) if r["source"] == "lyria" else -1
        rows.append([r["filepath"], r["class"], r["source"], r["patient_id"], idx, 1])
        if r["class"] == "Healthy":
            capped_sound.add(base)
    pids = sorted({r["patient_id"] for r in test if r["source"] == "lyria"})
    sound_dir = paths.crops_prevalence / "Healthy"
    sound_dir.mkdir(parents=True, exist_ok=True)
    for pid in pids:
        img = None
        for cls, idx, box in iter_blocks(paths.lyria_json, paths.lyria_images, pid):
            if cls != "Healthy":
                continue
            base = f"lyria__{pid}__b{idx}"
            if base in capped_sound:
                continue
            fpath = sound_dir / f"{base}.png"
            if not fpath.exists():
                if img is None:
                    img = Image.open(paths.lyria_image(pid))
                img.crop(box).save(fpath)
            rows.append([paths.relative_to_crops(fpath), "Healthy", "lyria", pid, idx, 0])
    return rows


def position_prior(paths: Paths, classes: list[str] = CLASSES4) -> dict[str, np.ndarray]:
    """Estimate the tooth-position prior from the annotations of the training-partition patients."""
    pool = read_csv(paths.split_dir("splits") / "train_pool.csv")
    pids = sorted({r["patient_id"] for r in pool if r["source"] == "lyria"})
    counts: dict[str, Counter] = defaultdict(Counter)
    for pid in pids:
        for (c, _i), (_box, code, _W, _H) in block_index(paths.lyria_json, paths.lyria_images, pid).items():
            counts[tooth_type(code)][c] += 1
    return laplace_prior(counts, classes)


def evaluate_prevalence(
    paths: Paths, archs: list[str], exp: Path | None = None, scale: float = 1.0, prior: bool = False, tag: str = ""
) -> dict:
    """Score the deployment-prevalence set with the three-seed models of each backbone.

    Needs the restricted images and the trained weights. Writes one output folder per
    backbone (see the module docstring) and prints a one-line summary.

    Parameters
    ----------
    paths : Paths
        Repository and data locations.
    archs : list of str
        Backbones; models are read from ``<exp>/test/none__noaug__<arch>__seed{0,1,2}/model.pt``.
    exp : Path, optional
        Experiment folder holding the models (default: the published models, ``experiments``).
    scale : float
        Crop scale (1.0 or 1.3).
    prior : bool
        Apply the tooth-position prior.
    tag : str
        Output tag; defaults to ``paper`` for the published models, else the experiment folder
        name. ``_s<scale>`` and, with the prior, ``_prior`` are appended.

    Returns
    -------
    dict
        ``{"<arch>__<tag>": metrics}`` with the content of each ``metrics.json``.
    """
    from ..models.ensemble import Ensemble, metrics

    exp = Path(exp or paths.experiments)
    classes = CLASSES4
    rows = build_rows(paths)
    y = np.array([classes.index(r[1]) for r in rows])
    capped = np.array([bool(r[5]) for r in rows])
    # load crops patient by patient so that only one radiograph is open at a time
    order = sorted(range(len(rows)), key=lambda i: (rows[i][3], rows[i][4]))
    imgs: list = [None] * len(rows)
    types = ["unknown"] * len(rows)
    cache: dict = {}
    for i in order:
        f, c, src, pid, idx, _ = rows[i]
        if src == "lyria" and (scale != 1.0 or prior):
            if pid not in cache:
                cache = {
                    pid: (
                        block_index(paths.lyria_json, paths.lyria_images, pid),
                        Image.open(paths.lyria_image(pid)) if scale != 1.0 else None,
                    )
                }
            blocks, img = cache[pid]
            box, code, W, H = blocks[(c, idx)]
            types[i] = tooth_type(code)
            imgs[i] = (
                gray3(img.crop(scaled_box(box, W, H, scale)))
                if scale != 1.0
                else gray3(Image.open(paths.crop_file(f, src)))
            )
        else:
            imgs[i] = gray3(Image.open(paths.crop_file(f, src)))
    prior_table = position_prior(paths) if prior else None

    tag = tag or ("paper" if exp == Path(paths.experiments) else exp.name)
    tag += f"_s{scale}" + ("_prior" if prior else "")
    summary = {}
    for arch in archs:
        probs = Ensemble(exp, arch).predict(imgs)
        if prior_table is not None:
            probs = apply_prior(probs, types, prior_table)
        out = paths.experiments / "prevalence" / f"{arch}__{tag}"
        out.mkdir(parents=True, exist_ok=True)
        per_seed: dict = {"capped": [], "full": []}
        for seed in range(probs.shape[1]):
            pred = probs[:, seed].argmax(1)
            per_seed["full"].append(metrics(y, pred))
            per_seed["capped"].append(metrics(y[capped], pred[capped]))
        mean = probs.mean(1)
        pred = mean.argmax(1)  # ensemble prediction
        agg = aggregate_per_seed(per_seed)
        res = {
            "arch": arch,
            "tag": tag,
            "n_full": len(rows),
            "n_capped": int(capped.sum()),
            "macro_f1_capped": agg["macro_f1_capped"],
            "macro_f1_full": agg["macro_f1_full"],
            "per_class_full": agg["per_class_full"],
            "per_class_capped": agg["per_class_capped"],
            "ensemble_full_confusion": metrics(y, pred)["confusion"],
            "false_flags_per_exam": false_flags_per_exam([r[1] for r in rows], pred, [r[3] for r in rows]),
        }
        write_json(out / "metrics.json", res)
        write_json(out / "per_seed.json", per_seed)
        with open(out / "predictions.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["filepath", "class", "patient_id", "tooth_type", "in_capped_test", "pred"]
                + [f"p_{c}" for c in classes]
            )
            for r, t, in_capped, p_idx, p_mean in zip(rows, types, capped, pred, mean):
                writer.writerow([r[0], r[1], r[3], t, int(in_capped), classes[p_idx]] + [f"{v:.4f}" for v in p_mean])
        impacted = res["per_class_full"]["Impacted"]
        print(
            f"[{arch} {tag}] full macro-F1 {100 * res['macro_f1_full'][0]:.1f} | impacted P "
            f"{100 * impacted['precision'][0]:.1f} R {100 * impacted['recall'][0]:.1f} | flags/exam "
            f"{res['false_flags_per_exam']['mean']:.2f}"
        )
        summary[f"{arch}__{tag}"] = res
    return summary
