"""Published figures, drawn from ``results/``.

Each figure is written as PDF and PNG.

* ``fig_cd_diagram``: critical-difference diagram (Friedman + Nemenyi) of the 18 classic
  configurations over the 15 cross-validation blocks, coloured by backbone (Okabe-Ito).
* ``fig_mitigation``: impacted precision/recall and false flags per exam of every
  configuration of ``tab_mitigation``.
* ``fig_confusion_test``: held-out confusion matrix of the reference model, summed over the
  three seeds' ``predictions.csv`` (6,195 predictions).
* ``fig_gradcam``: crops and Grad-CAM overlays; needs patient images, so it is drawn only
  when an image folder is given.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..config import Paths
from ..constants import CLASSES4
from ..utils import read_csv, read_json

#: Okabe-Ito colour-blind-safe palette.
OKABE = {
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#666666",
}
#: Short codes of the configuration labels in the critical-difference diagram.
ENH_CODE = {"none": "N", "msthgr": "M", "msthgr-clahe": "MC"}
ARCH_CODE = {"resnetv2_50": "Res", "vgg16": "VGG", "inception_v3": "Inc"}
ARCH_COLOR = {"Inc": OKABE["blue"], "VGG": OKABE["orange"], "Res": OKABE["green"]}
#: Configurations of ``fig_mitigation``: (prevalence run, stage, label), where stage 0 is the
#: frozen model, 1 a mitigation without retraining and 2 a retrained model. Run names are
#: ``<arch>__<tag>_s<crop scale>[_prior]``; ``posneg`` = position-aware negatives, ``ctx13`` =
#: training on crops enlarged 1.3x.
MITIGATION = [
    ("inception_v3__paper_s1.0", 0, "InceptionV3, frozen"),
    ("deit_small_patch16_224__paper_s1.0", 0, "DeiT-S, frozen"),
    ("inception_v3__paper_s1.0_prior", 1, "InceptionV3, prior"),
    ("inception_v3__paper_s1.3", 1, "InceptionV3, context 1.3"),
    ("inception_v3__paper_s1.3_prior", 1, "InceptionV3, context + prior"),
    ("deit_small_patch16_224__paper_s1.0_prior", 1, "DeiT-S, prior"),
    ("deit_small_patch16_224__paper_s1.3", 1, "DeiT-S, context 1.3"),
    ("deit_small_patch16_224__paper_s1.3_prior", 1, "DeiT-S, context + prior"),
    ("inception_v3__posneg_s1.0", 2, "InceptionV3, negatives"),
    ("inception_v3__posneg_s1.0_prior", 2, "InceptionV3, negatives + prior"),
    ("deit_small_patch16_224__posneg_s1.0", 2, "DeiT-S, negatives"),
    ("deit_small_patch16_224__posneg_s1.0_prior", 2, "DeiT-S, negatives + prior"),
    ("inception_v3__ctx13_s1.3", 2, "InceptionV3, context in training"),
    ("deit_small_patch16_224__ctx13_s1.3", 2, "DeiT-S, context in training"),
    ("deit_small_patch16_224__posneg_ctx13_s1.3", 2, "DeiT-S, negatives + context"),
    ("deit_small_patch16_224__posneg_ctx13_s1.3_prior", 2, "DeiT-S, negatives + context + prior"),
]
STAGE_COLOR = {0: OKABE["grey"], 1: OKABE["skyblue"], 2: OKABE["blue"]}
STAGE_LABEL = {0: "Frozen model", 1: "Without retraining", 2: "Retrained"}
CONFUSION_LABELS = ["Endo. treated", "Sound", "Impacted", "Implant"]
#: Panels of ``fig_gradcam``: (crop file, overlay file, caption).
GRADCAM = [
    ("orig_Implant.png", "gradcam_succ_Implant.png", "(a) Implant\ncorrect"),
    ("orig_Endodontics.png", "gradcam_succ_Endodontics.png", "(b) Endo. treated\ncorrect"),
    ("orig_Sound.png", "gradcam_succ_Sound.png", "(c) Sound\ncorrect"),
    ("orig_Impacted.png", "gradcam_fail_Impacted.png", "(d) Impacted\npredicted sound"),
]


def _style():
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig, out: Path, stem: str) -> Path:
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out / f"{stem}.pdf"


def cd_data(paths: Paths):
    """Compute the inputs of the critical-difference diagram from the cross-validation runs.

    Only the classic backbones are kept; a block is one ``(fold, seed)`` pair present in every
    configuration.

    Returns
    -------
    matrix : pandas.DataFrame
        Validation macro-F1, blocks x configurations (labels like ``N|NA|Inc``).
    ranks : pandas.Series
        Mean rank of each configuration (rank 1 = best macro-F1 in the block).
    nem : pandas.DataFrame
        Nemenyi post-hoc p-values, configurations x configurations.
    cd : float
        Nemenyi critical difference at alpha = 0.05.
    """
    import glob

    import pandas as pd
    import scikit_posthocs as sp
    from scipy.stats import studentized_range

    data: dict = defaultdict(dict)
    for metrics_path in glob.glob(str(paths.experiments / "runs" / "*" / "metrics.json")):
        r = read_json(metrics_path)
        if r["arch"] not in ARCH_CODE:
            continue
        cfg = f"{ENH_CODE[r['enhancement']]}|{'A' if r['aug'] == 'aug' else 'NA'}|{ARCH_CODE[r['arch']]}"
        data[cfg][(int(r["fold"]), int(r["seed"]))] = r["val_metrics"]["macro_f1"]
    blocks = sorted(set.intersection(*[set(v) for v in data.values()]))
    configs = sorted(data)
    matrix = pd.DataFrame({c: [data[c][b] for b in blocks] for c in configs})
    ranks = matrix.rank(axis=1, ascending=False).mean(axis=0)
    nem = sp.posthoc_nemenyi_friedman(matrix.values)
    nem.index = configs
    nem.columns = configs
    k, n = len(configs), len(blocks)
    cd = float(studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2.0) * np.sqrt(k * (k + 1) / (6.0 * n)))
    return matrix, ranks, nem, cd


def cd_diagram(paths: Paths, out: Path) -> Path:
    """Draw ``fig_cd_diagram`` into ``out`` and return the PDF path."""
    import matplotlib.pyplot as plt
    import scikit_posthocs as sp

    _style()
    _matrix, ranks, nem, _cd = cd_data(paths)
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    sp.critical_difference_diagram(
        ranks,
        nem,
        ax=ax,
        color_palette={c: ARCH_COLOR[c.split("|")[-1]] for c in ranks.index},
        label_props={"fontsize": 9},
        marker_props={"s": 26},
        elbow_props={"linewidth": 1.1},
        crossbar_props={"color": "black", "linewidth": 2.0},
        text_h_margin=0.02,
    )
    return _save(fig, out, "fig_cd_diagram")


def mitigation_data(paths: Paths) -> list[dict]:
    """Read the impacted precision and recall and the false flags of each mitigation configuration.

    Returns
    -------
    list of dict
        One dict per entry of :data:`MITIGATION`, with ``label``, ``stage``, ``precision`` and
        ``recall`` (mean over seeds, percent) and ``flags`` (mean false flags per exam).
    """
    rows = []
    for name, stage, label in MITIGATION:
        m = read_json(paths.experiments / "prevalence" / name / "metrics.json")
        rows.append(
            {
                "label": label,
                "stage": stage,
                "precision": 100.0 * m["per_class_full"]["Impacted"]["precision"][0],
                "recall": 100.0 * m["per_class_full"]["Impacted"]["recall"][0],
                "flags": m["false_flags_per_exam"]["mean"],
            }
        )
    return rows


def mitigation(paths: Paths, out: Path) -> Path:
    """Draw ``fig_mitigation`` into ``out`` and return the PDF path."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    _style()
    rows = sorted(mitigation_data(paths), key=lambda r: r["precision"])
    y = np.arange(len(rows))
    colors = [STAGE_COLOR[r["stage"]] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 5.1), sharey=True, gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    ax.barh(y, [r["precision"] for r in rows], color=colors, height=0.68)
    (recall_handle,) = ax.plot(
        [r["recall"] for r in rows],
        y,
        marker="D",
        linestyle="none",
        markersize=4.5,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=0.9,
        label="Impacted recall",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("Impacted precision and recall (%)")
    ax.set_xlim(0, 105)
    axes[1].barh(y, [r["flags"] for r in rows], color=colors, height=0.68)
    axes[1].set_xlabel("False flags per exam")
    axes[1].set_xlim(0, 0.8)
    handles = [mpl.patches.Patch(facecolor=STAGE_COLOR[s], label=STAGE_LABEL[s]) for s in (0, 1, 2)] + [recall_handle]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    return _save(fig, out, "fig_mitigation")


def confusion_matrix_test(paths: Paths, config: str = "none__noaug__inception_v3") -> np.ndarray:
    """Sum the confusion matrices of the three seeds' test predictions.

    Parameters
    ----------
    paths : Paths
        Path configuration.
    config : str
        Run configuration ``<enhancement>__<aug>__<arch>``.

    Returns
    -------
    numpy.ndarray
        4 x 4 integer counts, rows true and columns predicted, in :data:`CLASSES4` order.
    """
    M = np.zeros((4, 4), dtype=int)
    for s in range(3):
        for r in read_csv(paths.experiments / "test" / f"{config}__seed{s}" / "predictions.csv"):
            M[CLASSES4.index(r["y_true"]), CLASSES4.index(r["y_pred"])] += 1
    return M


def confusion(paths: Paths, out: Path) -> Path:
    """Draw ``fig_confusion_test`` into ``out`` and return the PDF path."""
    import matplotlib.pyplot as plt

    _style()
    M = confusion_matrix_test(paths)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(M, cmap="Blues")
    for i in range(4):
        for j in range(4):
            v = M[i, j]
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=10, color="white" if v > 1500 else "#1C2321")
    ax.set_xticks(range(4), CONFUSION_LABELS, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(4), CONFUSION_LABELS, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    return _save(fig, out, "fig_confusion_test")


def gradcam_grid(images: Path, out: Path, side: int = 299) -> Path:
    """Draw ``fig_gradcam`` from the crops and overlays in ``images`` and return the PDF path.

    The images show patient data and are not distributed; each is resized to ``side`` pixels.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    _style()
    fig, axes = plt.subplots(2, 4, figsize=(6.6, 3.7))
    for col, (orig, cam, label) in enumerate(GRADCAM):
        for row, name in enumerate((orig, cam)):
            img = Image.open(Path(images) / name).convert("RGB").resize((side, side), Image.LANCZOS)
            ax = axes[row, col]
            ax.imshow(np.asarray(img))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#444444")
                spine.set_linewidth(0.6)
        axes[1, col].set_xlabel(label, fontsize=8.0, labelpad=4)
    axes[0, 0].set_ylabel("Tooth crop", fontsize=9)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=9)
    fig.subplots_adjust(wspace=0.06, hspace=0.06)
    return _save(fig, out, "fig_gradcam")


def build_figures(paths: Paths, out: Path, gradcam_images: Path | None = None) -> list[Path]:
    """Draw every figure that needs only ``results/``, plus Grad-CAM when an image folder is given.

    Returns
    -------
    list of Path
        PDF path of each figure (a PNG is written next to it).
    """
    figs = [cd_diagram(paths, out), mitigation(paths, out), confusion(paths, out)]
    if gradcam_images is not None:
        figs.append(gradcam_grid(gradcam_images, out))
    return figs
