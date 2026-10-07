"""Four-class against five-class models: does a class for restored teeth change what matters.

Three comparisons, each between three-seed ensembles (mean softmax) of a four-class control
and a five-class model trained under the same protocol and patient partition:

A. In-domain cost on the four-class test crops. A Restored prediction counts as an error: it
   lowers the recall of the true class and is a false positive of none of the four classes.
B. Behaviour of the new class on the restored test crops.
C. The expert-reviewed Tufts/DENTEX crops: restored teeth called endodontically treated or
   implant by the four-class model, and recovered by the five-class model.

The published report (``experiments/clean_e1_report.json``) comes from the clean partition,
which is the default: ``ctrl = clean_ctrl4``, ``treat = clean_5class``,
``splits = splits_clean5``. Class indices follow ``CLASSES4``/``CLASSES5`` (Endodontics 0,
Healthy 1, Impacted 2, Implant 3, Restored 4).
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from ..config import Paths
from ..constants import ARCH_INPUT, CLASSES4, CLASSES5, IMAGENET_MEAN, IMAGENET_STD, TIMM_NAME
from ..utils import read_csv

C4, C5 = CLASSES4, CLASSES5


def per_class(y, p, classes: list[str]) -> dict:
    """Compute per-class precision, recall, F1 (rounded to 4 decimals) and support.

    ``y`` and ``p`` are class indices into ``classes``; returns ``{class: {metric: value}}``.
    """
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=list(range(len(classes))), zero_division=0)
    return {
        c: {
            "precision": round(float(pr[i]), 4),
            "recall": round(float(rc[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(sup[i]),
        }
        for i, c in enumerate(classes)
    }


def four_class_view(y5: np.ndarray, p4: np.ndarray, p5: np.ndarray) -> dict:
    """Compute comparison A on the four-class test crops.

    Parameters
    ----------
    y5 : ndarray of int
        True class indices into ``CLASSES5`` of every test crop.
    p4 : ndarray of int
        Four-class ensemble predictions (indices into ``CLASSES4``).
    p5 : ndarray of int
        Five-class ensemble predictions (indices into ``CLASSES5``).

    Returns
    -------
    dict
        ``n`` and, for ``ctrl4`` and ``five``, the macro-F1 and per-class metrics; ``five``
        also counts the Restored predictions, overall and by true class.
    """
    m = y5 < 4
    y4 = y5[m]
    p5_on4 = p5[m]
    p5_as4 = np.where(p5_on4 == 4, -1, p5_on4)  # Restored -> -1, wrong for every class
    return {
        "n": int(m.sum()),
        "ctrl4": {
            "macro_f1": round(float(f1_score(y4, p4[m], average="macro", zero_division=0)), 4),
            "per_class": per_class(y4, p4[m], C4),
        },
        "five": {
            "macro_f1": round(float(f1_score(y4, p5_as4, average="macro", labels=list(range(4)), zero_division=0)), 4),
            "per_class": per_class(y4, p5_as4, C4),
            "predicted_restored": int((p5_on4 == 4).sum()),
            "predicted_restored_by_true_class": {C5[c]: int(((p5_on4 == 4) & (y4 == c)).sum()) for c in range(4)},
        },
    }


def external_view(ry: np.ndarray, rp4: np.ndarray, rp5: np.ndarray, sources: list[str]) -> dict:
    """Compute comparison C per source (``tufts``, ``dentex``) and overall (``all``).

    Parameters
    ----------
    ry : ndarray of int
        Expert labels as indices into ``CLASSES5``.
    rp4 : ndarray of int
        Four-class ensemble predictions.
    rp5 : ndarray of int
        Five-class ensemble predictions.
    sources : list of str
        Source dataset of each crop.

    Returns
    -------
    dict
        ``n``, ``by_source`` and, per source, the restored-tooth counts, the four-class
        macro-F1 of both models on the non-restored crops, ``view1_restored_excluded`` and
        ``view2_all_crops``.
    """
    ext: dict = {"n": len(sources), "by_source": dict(Counter(sources))}
    for src in ("tufts", "dentex", "all"):
        sel = np.array([src == "all" or s == src for s in sources])
        ys, a, b = ry[sel], rp4[sel], rp5[sel]  # a: four-class, b: five-class predictions
        r_true = ys == 4
        b_as4 = np.where(b[~r_true] == 4, -1, b[~r_true])
        ext[src] = {
            "n": int(sel.sum()),
            "restored_crops": int(r_true.sum()),
            "restored_recovered_by_five": int((b[r_true] == 4).sum()),
            "restored_called_endo_or_implant_ctrl4": int(((a[r_true] == 0) | (a[r_true] == 3)).sum()),
            "restored_called_endo_or_implant_five": int(((b[r_true] == 0) | (b[r_true] == 3)).sum()),
            "four_class_macro_f1_ctrl4": round(
                float(f1_score(ys[~r_true], a[~r_true], average="macro", labels=list(range(4)), zero_division=0)), 4
            ),
            "four_class_macro_f1_five": round(
                float(f1_score(ys[~r_true], b_as4, average="macro", labels=list(range(4)), zero_division=0)), 4
            ),
            # view 1: restored crops removed, both models on the same crops
            "view1_restored_excluded": {
                "n": int((~r_true).sum()),
                "ctrl4": per_class(ys[~r_true], a[~r_true], C4),
                "five": per_class(ys[~r_true], b_as4, C4),
            },
            # view 2: every reviewed crop, as a deployment sees them
            "view2_all_crops": {
                "n": int(sel.sum()),
                "ctrl4": per_class(ys, a, C5),
                "five": per_class(ys, b, C5),
                "ctrl4_macro_f1_4cls": round(
                    float(f1_score(ys, a, average="macro", labels=list(range(4)), zero_division=0)), 4
                ),
                "five_macro_f1_5cls": round(
                    float(f1_score(ys, b, average="macro", labels=list(range(5)), zero_division=0)), 4
                ),
            },
        }
    return ext


def _load_models(exp: Path, arch: str, classes: list[str], device):
    """Load the available seeds 0-2 of ``<exp>/test/none__noaug__<arch>``; exit if none exists."""
    import timm
    import torch

    models = []
    for seed in (0, 1, 2):
        p = Path(exp) / "test" / f"none__noaug__{arch}__seed{seed}" / "model.pt"
        if not p.exists():
            continue
        m = timm.create_model(TIMM_NAME.get(arch, arch), pretrained=False, num_classes=len(classes))
        m.load_state_dict(torch.load(p, map_location="cpu"))
        models.append(m.to(device).eval())
    if not models:
        raise SystemExit(f"no models under {exp}")
    return models


def predict_mean(models, paths_list: list, size: int, device, bs: int = 64) -> np.ndarray:
    """Average the softmax of several models over crops read as grayscale and replicated to RGB.

    Returns an ``(n, C)`` array in the order of ``paths_list``.
    """
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    class Crops(Dataset):
        def __len__(self):
            return len(paths_list)

        def __getitem__(self, i):
            g = Image.open(paths_list[i]).convert("L")
            return tf(Image.merge("RGB", (g, g, g))), i

    dl = DataLoader(Crops(), batch_size=bs, num_workers=4)
    total = None
    with torch.no_grad():
        for m in models:
            p = np.concatenate([torch.softmax(m(x.to(device)), 1).cpu().numpy() for x, _ in dl])
            total = p if total is None else total + p
    return total / len(models)


def evaluate_restored(
    paths: Paths,
    arch: str = "deit_small_patch16_224",
    ctrl: str = "clean_ctrl4",
    treat: str = "clean_5class",
    splits: str = "splits_clean5",
    out: Path | None = None,
    frozen_only: bool = False,
) -> dict:
    """Run comparisons A, B and C and write the report (needs images and weights).

    Also writes the external predictions of both models to
    ``experiments/e1_external_predictions.csv``.

    Parameters
    ----------
    paths : Paths
        Repository and data locations.
    arch : str
        Backbone of both models.
    ctrl, treat : str
        Experiment sub-folders of the four-class and five-class models.
    splits : str
        Split folder holding the five-class ``test.csv``.
    out : Path, optional
        Report path (``experiments/clean_e1_report.json`` by default).
    frozen_only : bool
        Score comparison C only on the frozen holdout of the reviewed crops.

    Returns
    -------
    dict
        The report: ``arch``, ``n_seeds``, ``A_in_domain_four_class``,
        ``B_restored_in_domain`` and ``C_external_expert_labels``.
    """
    import torch

    from ..data.manifests import manifest5_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = ARCH_INPUT[arch]
    out = Path(out or paths.experiments / "clean_e1_report.json")
    report: dict = {"arch": arch}
    m4 = _load_models(paths.experiments / ctrl, arch, C4, device)
    m5 = _load_models(paths.experiments / treat, arch, C5, device)
    report["n_seeds"] = {"ctrl4": len(m4), "five": len(m5)}

    test = read_csv(paths.split_dir(splits) / "test.csv")
    files = [paths.crop_file(r["filepath"], r["source"]) for r in test]
    y5 = np.array([C5.index(r["class"]) for r in test])
    p4 = predict_mean(m4, files, size, device).argmax(1)
    p5 = predict_mean(m5, files, size, device).argmax(1)
    report["A_in_domain_four_class"] = four_class_view(y5, p4, p5)
    restored = y5 == 4
    report["B_restored_in_domain"] = {
        "n": int(restored.sum()),
        "five_recall": round(float((p5[restored] == 4).mean()), 4),
        "five_predicted": {C5[i]: int((p5[restored] == i).sum()) for i in range(5)},
        "ctrl4_predicted": {C4[i]: int((p4[restored] == i).sum()) for i in range(4)},
        "five_confusion_full": confusion_matrix(y5, p5, labels=list(range(5))).tolist(),
    }

    reviewed = [r for r in read_csv(manifest5_path(paths)) if r["origin"] == "reviewed"]
    if frozen_only:
        reviewed = [r for r in reviewed if r["frozen"] == "1"]
    rfiles = [paths.crop_file(r["filepath"], r["source"]) for r in reviewed]
    ry = np.array([C5.index(r["class"]) for r in reviewed])
    rp4 = predict_mean(m4, rfiles, size, device).argmax(1)
    rp5 = predict_mean(m5, rfiles, size, device).argmax(1)
    with open(paths.experiments / "e1_external_predictions.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filepath", "source", "group", "expert", "pred_ctrl4", "pred_five"])
        for r, a, b in zip(reviewed, rp4, rp5):
            writer.writerow([r["filepath"], r["source"], r["group"], r["class"], C4[a], C5[b]])
    report["C_external_expert_labels"] = external_view(ry, rp4, rp5, [r["source"] for r in reviewed])
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    return report
