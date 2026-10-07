"""Five-class study with all mitigations combined.

The mitigations are: a fifth class for restored teeth, position-aware sound negatives
(``posneg``: sound training crops drawn from third molars, second molars and other teeth),
context-preserving crops (``ctx13``: boxes scaled x1.3 about the centre) and the post-hoc
tooth-position prior. ``LODO`` is leave-one-dataset-out: trained on the primary source
(InReDD-PAN924, ``lyria``) only and evaluated on the external datasets.

Models (three DeiT-S seeds each, trained in test mode on the whole training pool):

========  ===========================================  =========  ===========
name      experiment folder                            classes    crop scale
========  ===========================================  =========  ===========
NEW       variants/five_mitigated                      5          1.3
NEW_LODO  variants/five_mitigated_lodo_lyria           5          1.3
REF4      variants/posneg_ctx13                        4          1.3
FIVE0     clean_5class                                 5          1.0
CTRL4     clean_ctrl4                                  4          1.0
========  ===========================================  =========  ===========

:func:`run_inference` stores the per-seed softmax WITHOUT the prior
(``experiments/variants/five_mitigated/eval/<MODEL>__<set>.npz``); :func:`analyze` applies the
priors and writes every number of the study to
``experiments/variants/five_mitigated/results/results.json``. Conventions:

* per-seed metrics use each seed's argmax and are reported as mean and sample sd (ddof=1) over
  the 3 seeds; the ensemble is the argmax of the mean of the (prior-adjusted) softmax;
* prior: ``p'(c|x,t) = p(c|x) P(c|t) / P_train(c)``, renormalised, ``P_train(c) = 1/|C|``;
  unknown tooth type (pre-cropped DENTEX implants): no adjustment;
* on the four original classes a Restored prediction is an error;
* false flag (``ff`` in result keys): a sound tooth (or, in the ``plus_restored`` variant, a
  sound or restored tooth) predicted Endodontics, Impacted or Implant; a Restored prediction
  raises no flag (the ``strict`` variant counts it).
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support
from statsmodels.stats.multitest import multipletests

from ..config import Paths
from ..constants import CLASSES4, CLASSES5
from ..utils import read_csv, read_json

C4, C5 = CLASSES4, CLASSES5
FIND = {"Endodontics", "Impacted", "Implant"}  # finding classes
GRID = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]  # confidence thresholds tau
# model -> tooth-position prior file in the variant split folder (None: no prior)
PRIORS = {"REF4": "prior4.json", "NEW": "prior5.json", "FIVE0": "prior5.json", "CTRL4": "prior4.json", "NEW_LODO": None}
MODELS = {  # name -> (experiment sub-folder, classes, crop scale)
    "NEW": ("variants/five_mitigated", C5, "1.3"),
    "NEW_LODO": ("variants/five_mitigated_lodo_lyria", C5, "1.3"),
    "REF4": ("variants/posneg_ctx13", C4, "1.3"),
    "FIVE0": ("clean_5class", C5, "1.0"),
    "CTRL4": ("clean_ctrl4", C4, "1.0"),
}
SETS = ["indomain_test", "prevalence", "external"]  # evaluation manifests in eval_sets/
ARCH, TIMM = "deit_small_patch16_224", "deit_small_patch16_224.fb_in1k"


class Study:
    """Locations of the study files.

    Attributes
    ----------
    exp : Path
        Experiments folder.
    v : Path
        ``experiments/variants/five_mitigated``.
    evd : Path
        Stored probabilities (``<v>/eval``).
    res : Path
        Results folder (``<v>/results``).
    dv : Path
        Variant split folder, holding the priors.
    evs : Path
        Evaluation manifests (``<dv>/eval_sets``).
    """

    def __init__(self, paths: Paths):
        self.paths = paths
        self.exp = paths.experiments
        self.v = paths.experiments / "variants" / "five_mitigated"
        self.evd = self.v / "eval"
        self.res = self.v / "results"
        self.dv = paths.split_dir("variants/five_mitigated")
        self.evs = self.dv / "eval_sets"

    def has(self, model: str, evset: str) -> bool:
        """Whether the stored probabilities of ``model`` on ``evset`` exist."""
        return (self.evd / f"{model}__{evset}.npz").exists()

    def load(self, model: str, evset: str, prior: bool = False, rows: list[dict] | None = None):
        """Load the stored per-seed probabilities, optionally prior-adjusted.

        Parameters
        ----------
        model : str
            Model name (a key of ``MODELS``).
        evset : str
            Evaluation set (one of ``SETS``).
        prior : bool
            Apply the model's tooth-position prior (``PRIORS``).
        rows : list of dict, optional
            Manifest rows of ``evset`` with a ``tooth_type`` column; required when ``prior``.

        Returns
        -------
        tuple
            ``(P, classes)``: probabilities ``(n, 3, C)`` as float64 and the class names.
        """
        z = np.load(self.evd / f"{model}__{evset}.npz")
        P, classes = z["probs"].astype(np.float64), [str(c) for c in z["classes"]]
        if prior:
            prior_json = read_json(self.dv / PRIORS[model])
            assert prior_json["classes"] == classes
            k = len(classes)
            weights = np.stack(
                [
                    np.array(prior_json["prior"][t]) * k if t in prior_json["prior"] else np.ones(k)
                    for t in (r["tooth_type"] for r in rows)
                ]
            )
            P = P * weights[:, None, :]
            P = P / P.sum(-1, keepdims=True)
        return P, classes


# --------------------------------------------------------------------------- metrics
def sd(v) -> float:
    """Return the sample standard deviation (ddof=1), or 0 for a single value."""
    v = np.asarray(v, dtype=float)
    return float(v.std(ddof=1)) if len(v) > 1 else 0.0


def prf(y, p, labels):
    """Compute per-class precision, recall, F1 and support over string ``labels``, and their macro-F1.

    Returns ``({label: {metric: value}}, macro_f1)``.
    """
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return (
        {
            c: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])}
            for i, c in enumerate(labels)
        },
        float(np.mean(f1)),
    )


def block(y, P, classes, labels, mask=None) -> dict:
    """Compute per-seed and ensemble metrics over ``labels``.

    Parameters
    ----------
    y : ndarray of str
        True class names.
    P : ndarray, shape (n, seeds, C)
        Per-seed probabilities.
    classes : list of str
        Class names of the columns of ``P``.
    labels : list of str
        Labels over which the metrics and the macro-F1 are computed.
    mask : ndarray of bool, optional
        Restrict to these crops.

    Returns
    -------
    dict
        ``n``, ``labels``, ``macro_f1`` (``mean``, ``sd``, ``per_seed``, ``ensemble``) and
        ``per_class[label][precision|recall|f1]`` (``mean``, ``sd``, ``ensemble``) with
        ``support``.
    """
    if mask is not None:
        y, P = y[mask], P[mask]
    cls = np.array(classes)
    per_seed = []
    for s in range(P.shape[1]):
        pc, mf = prf(y, cls[P[:, s].argmax(1)], labels)
        per_seed.append({"per_class": pc, "macro_f1": mf})
    ens_pc, ens_mf = prf(y, cls[P.mean(1).argmax(1)], labels)
    out = {
        "n": int(len(y)),
        "labels": labels,
        "macro_f1": {
            "mean": float(np.mean([m["macro_f1"] for m in per_seed])),
            "sd": sd([m["macro_f1"] for m in per_seed]),
            "per_seed": [m["macro_f1"] for m in per_seed],
            "ensemble": ens_mf,
        },
        "per_class": {},
    }
    for c in labels:
        out["per_class"][c] = {
            k: {
                "mean": float(np.mean([m["per_class"][c][k] for m in per_seed])),
                "sd": sd([m["per_class"][c][k] for m in per_seed]),
                "ensemble": ens_pc[c][k],
            }
            for k in ("precision", "recall", "f1")
        }
        out["per_class"][c]["support"] = ens_pc[c]["support"]
    return out


def flags(y, pred, pid, neg=("Healthy",), strict=False) -> dict:
    """Summarise the false flags per exam over patients with at least one tooth of a class in ``neg``.

    Parameters
    ----------
    y, pred : ndarray of str
        True and predicted class names.
    pid : ndarray of str
        Patient of each crop; each patient is one exam.
    neg : tuple of str
        Negative classes: a tooth of these classes predicted as a finding is a false flag.
    strict : bool
        Also count a Restored prediction as a flag.

    Returns
    -------
    dict
        ``mean``, ``median``, ``p90`` (90th percentile), ``share_zero`` (share of exams
        without a flag), ``n_exams`` and ``n_flagged``.
    """
    neg_mask = np.isin(y, list(neg))
    flagged = neg_mask & (pred != "Healthy") & ((pred != "Restored") | strict)
    flags_by_patient: Counter = Counter()
    for p_, f_ in zip(pid[neg_mask], flagged[neg_mask]):
        flags_by_patient[p_] += int(f_)
    exams = sorted(set(pid[neg_mask]))
    v = np.array([flags_by_patient[e] for e in exams])
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "share_zero": float((v == 0).mean()),
        "n_exams": len(exams),
        "n_flagged": int(flagged.sum()),
    }


def deploy(y, P, classes, pid, neg=("Healthy",), labels=C4, mask=None) -> dict:
    """Compute per-class metrics and false flags on a deployment set.

    Returns the :func:`block` metrics plus ``ff`` and ``ff_strict`` (ensemble false flags, see
    :func:`flags`), and ``ff_per_seed`` and ``share_zero_per_seed`` (values, mean and sd over
    seeds).
    """
    if mask is not None:
        y, P, pid = y[mask], P[mask], pid[mask]
    cls = np.array(classes)
    out = block(y, P, classes, labels)
    ens = cls[P.mean(1).argmax(1)]
    out["ff"] = flags(y, ens, pid, neg)
    out["ff_strict"] = flags(y, ens, pid, neg, strict=True)
    per_seed = [flags(y, cls[P[:, s].argmax(1)], pid, neg)["mean"] for s in range(P.shape[1])]
    out["ff_per_seed"] = {"values": per_seed, "mean": float(np.mean(per_seed)), "sd": sd(per_seed)}
    per_seed_z = [flags(y, cls[P[:, s].argmax(1)], pid, neg)["share_zero"] for s in range(P.shape[1])]
    out["share_zero_per_seed"] = {"values": per_seed_z, "mean": float(np.mean(per_seed_z)), "sd": sd(per_seed_z)}
    return out


def operating(y, P, classes, pid, neg=("Healthy",)) -> list[dict]:
    """Compute the operating-point curve of the ensemble.

    A finding is raised only when the ensemble confidence (top mean probability) reaches
    ``tau``. Returns one dict per ``tau`` of ``GRID`` with ``ff_per_exam``, ``share_exams_zero``,
    ``recall`` per finding class, ``precision_impacted`` and ``n_flagged_neg``.
    """
    cls = np.array(classes)
    M = P.mean(1)
    pred, conf = cls[M.argmax(1)], M.max(1)
    neg_mask = np.isin(y, list(neg))
    exams = sorted(set(pid[neg_mask]))
    out = []
    for t in GRID:
        flag = np.isin(pred, list(FIND)) & (conf >= t)
        flags_by_patient: Counter = Counter()
        for p_, f_ in zip(pid[neg_mask], flag[neg_mask]):
            flags_by_patient[p_] += int(f_)
        v = np.array([flags_by_patient[e] for e in exams])
        # iterates the set FIND, so the key order of "recall" may vary between runs
        rec = {c: float((flag & (pred == c) & (y == c)).sum() / max(1, (y == c).sum())) for c in FIND}
        out.append(
            {
                "tau": t,
                "ff_per_exam": float(v.mean()),
                "share_exams_zero": float((v == 0).mean()),
                "recall": rec,
                "precision_impacted": float(
                    (flag & (pred == "Impacted") & (y == "Impacted")).sum()
                    / max(1, (flag & (pred == "Impacted")).sum())
                ),
                "n_flagged_neg": int((flag & neg_mask).sum()),
            }
        )
    return out


def counts(y, P, classes, cls_true, mask=None) -> dict:
    """Count the ensemble and per-seed predicted classes of the crops of true class ``cls_true``."""
    cls = np.array(classes)
    m = (y == cls_true) if mask is None else (mask & (y == cls_true))
    ens = Counter(cls[P[m].mean(1).argmax(1)])
    per_seed = [Counter(cls[P[m][:, s].argmax(1)]) for s in range(P.shape[1])]
    return {
        "n": int(m.sum()),
        "ensemble": {c: int(ens[c]) for c in classes},
        "per_seed": [{c: int(ps[c]) for c in classes} for ps in per_seed],
    }


def boot_diff(fn, pid, n=2000, seed=0) -> dict:
    """Bootstrap a statistic difference by resampling patients with replacement.

    Parameters
    ----------
    fn : callable
        ``fn(index_array) -> float``, the difference evaluated on the selected crops.
    pid : sequence of str
        Patient of each crop (the resampling cluster).
    n : int
        Bootstrap replicates.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    dict
        ``point`` (on all crops), ``ci95`` (percentile interval), ``share_gt0`` (share of
        replicates above zero), ``n_boot`` and ``n_clusters``.
    """
    rng = np.random.default_rng(seed)
    groups: dict = defaultdict(list)
    for i, p in enumerate(pid):
        groups[p].append(i)
    keys = list(groups)
    idx = [np.array(groups[k]) for k in keys]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        vals.append(fn(np.concatenate([idx[j] for j in pick])))
    vals = np.array(vals)
    return {
        "point": float(fn(np.arange(len(pid)))),
        "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
        "share_gt0": float((vals > 0).mean()),
        "n_boot": n,
        "n_clusters": len(keys),
    }


def ens_pred(P, classes):
    """Return the ensemble prediction (argmax of the mean softmax) as class names."""
    return np.array(classes)[P.mean(1).argmax(1)]


# --------------------------------------------------------------------------- analysis
def analyze(paths: Paths, write: bool = True, bootstrap: bool = True) -> dict:
    """Compute every number of the study from the stored probabilities (CPU only).

    Parameters
    ----------
    paths : Paths
        Repository and data locations.
    write : bool
        Write ``results/results.json``.
    bootstrap : bool
        Compute the patient-clustered bootstraps (the slowest part; skipped when only point
        estimates are needed).

    Returns
    -------
    dict
        Sections ``A_indomain`` (in-domain test), ``B_prevalence`` and ``B_operating``
        (deployment prevalence; ``base_8672`` is the four-class prevalence set,
        ``plus_restored`` adds the restored teeth), ``C_external`` (expert-reviewed external
        crops), ``D_lodo`` (leave-one-dataset-out), ``E_cost`` (training time), ``S_cv``
        (paired test on the cross-validation blocks) and ``bootstrap`` (NEW minus REF4).
        Models whose stored probabilities are missing are skipped.
    """
    st = Study(paths)
    R: dict = {}

    # A. in-domain test
    ind = read_csv(st.evs / "indomain_test.csv")
    y = np.array([r["class"] for r in ind])
    m4 = np.isin(y, C4)
    A = {}
    for name, model, use_prior in [
        ("REF4_prior", "REF4", True),
        ("REF4_noprior", "REF4", False),
        ("CTRL4", "CTRL4", False),
        ("FIVE0", "FIVE0", False),
        ("FIVE0_prior5", "FIVE0", True),
        ("NEW_prior", "NEW", True),
        ("NEW_noprior", "NEW", False),
    ]:
        if not st.has(model, "indomain_test"):
            continue
        P, cl = st.load(model, "indomain_test", use_prior, ind)
        A[name] = {"four_class_subset": block(y, P, cl, C4, m4)}
        if "Restored" in cl:
            A[name]["five_class_all"] = block(y, P, cl, C5)
            A[name]["four_subset_predicted_restored"] = {
                c: int(((y == c) & (ens_pred(P, cl) == "Restored")).sum()) for c in C4
            }
        A[name]["restored_test_crops"] = counts(y, P, cl, "Restored")
        A[name]["endo_test_crops"] = counts(y, P, cl, "Endodontics")
        A[name]["confusion_ensemble"] = {t: dict(Counter(ens_pred(P, cl)[y == t])) for t in C5}
    R["A_indomain"] = A

    # B. deployment prevalence
    prev = read_csv(st.evs / "prevalence.csv")
    py = np.array([r["class"] for r in prev])
    ppid = np.array([r["patient_id"] for r in prev])
    base = np.array([r["set"] == "base" for r in prev])
    lyr = np.array([r["source"] == "lyria" for r in prev])
    B, OP = {}, {}
    for name, model, use_prior in [
        ("REF4_prior", "REF4", True),
        ("REF4_noprior", "REF4", False),
        ("FIVE0", "FIVE0", False),
        ("FIVE0_prior5", "FIVE0", True),
        ("NEW_prior", "NEW", True),
        ("NEW_noprior", "NEW", False),
    ]:
        if not st.has(model, "prevalence"):
            continue
        P, cl = st.load(model, "prevalence", use_prior, prev)
        B[name] = {
            "base_8672": deploy(py, P, cl, ppid, mask=base),
            "base_lyria_only": deploy(py, P, cl, ppid, mask=base & lyr),
            "plus_restored_4labels": deploy(py, P, cl, ppid, neg=("Healthy", "Restored"), labels=C4),
        }
        if "Restored" in cl:
            B[name]["plus_restored_5labels"] = deploy(py, P, cl, ppid, neg=("Healthy", "Restored"), labels=C5)
        B[name]["plus_restored_restored_teeth"] = counts(py, P, cl, "Restored")
        OP[name] = {
            "base_8672": operating(py[base], P[base], cl, ppid[base]),
            "plus_restored": operating(py, P, cl, ppid, neg=("Healthy", "Restored")),
        }
        tt = np.array([r["tooth_type"] for r in prev])
        e = ens_pred(P, cl)
        B[name]["sound_flagged_impacted_by_type"] = {
            t: {
                "n": int(((py == "Healthy") & base & (tt == t)).sum()),
                "flagged_impacted": int(((py == "Healthy") & base & (tt == t) & (e == "Impacted")).sum()),
            }
            for t in sorted(set(tt[base]))
        }
    R["B_prevalence"] = B
    R["B_operating"] = OP

    # C. restored confusion on the external expert-reviewed crops
    ext = read_csv(st.evs / "external.csv")
    ey = np.array([r["class"] for r in ext])
    es = np.array([r["source"] for r in ext])
    efz = np.array([r["frozen"] == "1" for r in ext])
    C = {}
    for name in ("CTRL4", "FIVE0", "REF4", "NEW", "NEW_LODO"):
        if not st.has(name, "external"):
            continue
        P, cl = st.load(name, "external", False)
        cls = np.array(cl)
        d = {}
        for src in ("tufts", "dentex", "all"):
            sm = (es == src) if src != "all" else np.ones(len(ey), bool)
            ens = cls[P[sm].mean(1).argmax(1)]
            r_ = ey[sm] == "Restored"
            d[src] = {
                "n": int(sm.sum()),
                "restored_crops": int(r_.sum()),
                "restored_called_endo_or_implant": int(np.isin(ens[r_], ["Endodontics", "Implant"]).sum()),
                "restored_called_endo_or_implant_per_seed": [
                    int(np.isin(cls[P[sm][:, s].argmax(1)][r_], ["Endodontics", "Implant"]).sum()) for s in range(3)
                ],
                "restored_called_endo": int((ens[r_] == "Endodontics").sum()),
                "restored_called_endo_per_seed": [
                    int((cls[P[sm][:, s].argmax(1)][r_] == "Endodontics").sum()) for s in range(3)
                ],
                "restored_called_implant": int((ens[r_] == "Implant").sum()),
                "restored_pred_counts": dict(Counter(ens[r_])),
                "restored_recovered": int((ens[r_] == "Restored").sum()),
                "view2_all_crops": block(ey[sm], P[sm], cl, C5),
            }
        C[name] = d
    R["C_external"] = C

    # D. leave-one-dataset-out (trained on the primary source only)
    D: dict = {}
    if st.has("NEW_LODO", "external"):
        P, cl = st.load("NEW_LODO", "external", False)
        for src in ("dentex", "tufts"):
            b = block(ey, P, cl, C5, (es == src) & ~efz)
            b["macro_f1"]["sd_ddof0"] = float(np.std(b["macro_f1"]["per_seed"]))
            D[f"NEW_LODO__{src}"] = b
    for src in ("dentex", "tufts"):
        per_seed = []
        rows: list = []
        for s in range(3):
            rows = [
                r
                for r in read_csv(
                    paths.experiments / "lodo_lyria" / "test" / f"none__noaug__{ARCH}__seed{s}" / "predictions.csv"
                )
                if r["source"] == src
            ]
            per_seed.append(prf(np.array([r["y_true"] for r in rows]), np.array([r["y_pred"] for r in rows]), C5))
        D[f"LODO_LYRIA_published__{src}"] = {
            "n": len(rows),
            "macro_f1": {
                "mean": float(np.mean([m for _, m in per_seed])),
                "sd": sd([m for _, m in per_seed]),
                "sd_ddof0": float(np.std([m for _, m in per_seed])),
                "per_seed": [m for _, m in per_seed],
            },
            "per_class": {c: {"f1": {"mean": float(np.mean([pc[c]["f1"] for pc, _ in per_seed]))}} for c in C5},
        }
    # check that the stored probabilities reproduce the predictions written at training time
    chk = {}
    if st.has("NEW_LODO", "external"):
        P, cl = st.load("NEW_LODO", "external", False)
        bypath = {os.path.basename(r["path_s1.3"]): i for i, r in enumerate(ext)}
        for s in range(3):
            pp = (
                paths.experiments
                / "variants"
                / "five_mitigated_lodo_lyria"
                / "test"
                / f"none__noaug__{ARCH}__seed{s}"
                / "predictions.csv"
            )
            if not pp.exists():
                continue
            rows = read_csv(pp)
            agree = sum(cl[P[bypath[os.path.basename(r["filepath"])], s].argmax()] == r["y_pred"] for r in rows)
            chk[f"seed{s}"] = {"n": len(rows), "agree": int(agree)}
    D["consistency_trainpy_vs_stored"] = chk
    R["D_lodo"] = D

    # E. training cost
    E = {}
    for name, sub in [("NEW", "five_mitigated"), ("NEW_LODO", "five_mitigated_lodo_lyria")]:
        reg = paths.experiments / "variants" / sub / "registry.csv"
        if reg.exists():
            f = "%Y-%m-%d %H:%M:%S"
            rr = [r for r in read_csv(reg) if r["status"] == "done"]
            secs = [
                (dt.datetime.strptime(r["end_ts"], f) - dt.datetime.strptime(r["start_ts"], f)).total_seconds()
                for r in rr
            ]
            E[name] = {
                "runs": len(rr),
                "hours": sum(secs) / 3600,
                "median_min": float(np.median(secs) / 60),
                "epochs_run": [int(r["epochs_run"]) for r in rr],
            }
    R["E_cost"] = E

    # S. paired statistics on the 15 cross-validation blocks (5 folds x 3 seeds)
    def cv_blocks(root: Path) -> dict:
        """Validation macro-F1 of every ``(fold, seed)`` run under ``root/runs``."""
        out = {}
        for mp in glob.glob(str(root / "runs" / "*" / "metrics.json")):
            r = read_json(mp)
            out[(int(r["fold"]), int(r["seed"]))] = r["val_metrics"]["macro_f1"]
        return out

    S = {}
    new_cv = cv_blocks(paths.experiments / "variants" / "five_mitigated")
    old_cv = cv_blocks(paths.experiments / "clean_5class")
    if len(new_cv) == 15 and len(old_cv) == 15:
        blocks = sorted(new_cv)
        a = np.array([new_cv[b] for b in blocks])
        r_ = np.array([old_cv[b] for b in blocks])
        dlt = a - r_
        S["cv_new_vs_five0"] = {
            "n_blocks": 15,
            "new_mean": float(a.mean()),
            "new_sd": sd(a),
            "five0_mean": float(r_.mean()),
            "five0_sd": sd(r_),
            "diff_mean": float(dlt.mean()),
            "dz": float(dlt.mean() / dlt.std(ddof=1)),
            "p_ttest_rel": float(stats.ttest_rel(a, r_).pvalue),
            "p_wilcoxon": float(stats.wilcoxon(a, r_).pvalue),
            "p_welch_unpaired": float(stats.ttest_ind(a, r_, equal_var=False).pvalue),
            "note": "validation folds differ (sound crops replaced, folds recomputed); "
            "pairing is by (fold, seed) index only",
        }
        # Holm over a single comparison, so p_holm equals p_ttest_rel.
        S["cv_new_vs_five0"]["p_holm"] = float(
            multipletests([S["cv_new_vs_five0"]["p_ttest_rel"]], method="holm")[1][0]
        )
    R["S_cv"] = S

    # bootstrap of NEW minus REF4 (ensembles, both with the prior)
    BT = {}
    if bootstrap and st.has("NEW", "prevalence") and st.has("REF4", "prevalence"):
        Pn, cn = st.load("NEW", "prevalence", True, prev)
        Pr, cr = st.load("REF4", "prevalence", True, prev)
        en, er = ens_pred(Pn, cn), ens_pred(Pr, cr)
        yb, pb = py[base], ppid[base]
        enb, erb = en[base], er[base]

        def imp_prec(pred, idx):
            """Impacted precision on crops ``idx``."""
            sel = pred[idx] == "Impacted"
            return (yb[idx][sel] == "Impacted").mean() if sel.any() else 0.0

        def macro4(pred, idx):
            """Four-class macro-F1 on crops ``idx``."""
            return prf(yb[idx], pred[idx], C4)[1]

        BT["impacted_precision_new_minus_ref4"] = boot_diff(lambda i: imp_prec(enb, i) - imp_prec(erb, i), pb)
        BT["macro_f1_4_new_minus_ref4"] = boot_diff(lambda i: macro4(enb, i) - macro4(erb, i), pb)
        # false flags per exam: paired per-exam differences, resampled over exams
        sound = yb == "Healthy"
        flags_new, flags_ref = Counter(), Counter()
        for p_, is_sound, a_, b_ in zip(pb, sound, np.isin(enb, list(FIND)), np.isin(erb, list(FIND))):
            if is_sound:
                flags_new[p_] += int(a_)
                flags_ref[p_] += int(b_)
        exams = sorted(set(pb[sound]))
        diffs = np.array([flags_new[e] - flags_ref[e] for e in exams], float)
        rng = np.random.default_rng(0)
        boot = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(2000)]
        BT["ff_per_exam_new_minus_ref4"] = {
            "point": float(diffs.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n_boot": 2000,
            "n_exams": len(exams),
        }
    if bootstrap and st.has("NEW", "indomain_test") and st.has("REF4", "indomain_test"):
        Pn, cn = st.load("NEW", "indomain_test", True, ind)
        Pr, cr = st.load("REF4", "indomain_test", True, ind)
        en, er = ens_pred(Pn, cn)[m4], ens_pred(Pr, cr)[m4]
        y4 = y[m4]
        pid4 = np.array([r["patient_id"] for r in ind])[m4]
        BT["indomain_macro_f1_4_new_minus_ref4"] = boot_diff(
            lambda i: prf(y4[i], en[i], C4)[1] - prf(y4[i], er[i], C4)[1], pid4
        )
    R["bootstrap"] = BT

    if write:
        st.res.mkdir(parents=True, exist_ok=True)
        with open(st.res / "results.json", "w") as fh:
            json.dump(R, fh, indent=1)
    return R


# --------------------------------------------------------------------------- inference (images, weights)
def resolve_eval_path(paths: Paths, stored: str, column: str, evset: str, source: str) -> Path:
    """Resolve a crop path stored in an evaluation manifest.

    Paths written by this package carry their sub-folder (``variants/...``,
    ``crops_prevalence/...``); the released manifests had their roots stripped, so the
    candidate locations of each column are tried in order and the first existing file wins
    (the first candidate if none exists).

    Parameters
    ----------
    paths : Paths
        Repository and data locations.
    stored : str
        Path as stored in the manifest.
    column : str
        Manifest column (``path_s1.0`` or ``path_s1.3``).
    evset : str
        Evaluation set.
    source : str
        Source dataset of the crop.
    """
    p = Path(stored)
    if p.is_absolute() or p.parts[0] in ("crops_source", "crops_enhanced", "crops_prevalence", "variants"):
        return paths.crop_file(stored, source)
    root = paths.variant_crops("five_mitigated")
    if column == "path_s1.3":
        cands = ([root / "prevalence" / "s1.3" / p] if evset == "prevalence" else []) + [
            root / "enhanced" / "none" / p,
            paths.crop_file(stored, source),
        ]
    else:
        cands = [paths.crop_file(stored, source), paths.crops_prevalence / p, root / "prevalence" / "s1.0" / p]
    return next((c for c in cands if c.exists()), cands[0])


def run_inference(
    paths: Paths, models: list[str] | None = None, sets: list[str] | None = None, force: bool = False, bs: int = 128
) -> None:
    """Store the per-seed softmax of each model on each evaluation set (needs images and weights).

    Parameters
    ----------
    paths : Paths
        Repository and data locations.
    models : list of str, optional
        Model names (default: every key of ``MODELS``).
    sets : list of str, optional
        Evaluation sets (default: ``SETS``).
    force : bool
        Recompute existing ``.npz`` files.
    bs : int
        Batch size.
    """
    import timm
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from ..constants import IMAGENET_MEAN, IMAGENET_STD

    st = Study(paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    for model in models or list(MODELS):
        sub, classes, scale = MODELS[model]
        for evset in sets or SETS:
            out = st.evd / f"{model}__{evset}.npz"
            if out.exists() and not force:
                print(f"skip {model} {evset} (exists)")
                continue
            t0 = time.time()
            rows = read_csv(st.evs / f"{evset}.csv")
            col = f"path_s{scale}"
            files = [resolve_eval_path(paths, r[col], col, evset, r["source"]) for r in rows]

            class Crops(Dataset):
                def __len__(self):
                    return len(files)

                def __getitem__(self, i):
                    g = Image.open(files[i]).convert("L")
                    return tf(Image.merge("RGB", (g, g, g))), i

            dl = DataLoader(Crops(), batch_size=bs, num_workers=2)
            probs = np.zeros((len(files), 3, len(classes)), dtype=np.float32)
            for si in range(3):
                m = timm.create_model(TIMM, pretrained=False, num_classes=len(classes))
                m.load_state_dict(
                    torch.load(
                        paths.experiments / sub / "test" / f"none__noaug__{ARCH}__seed{si}" / "model.pt",
                        map_location="cpu",
                    )
                )
                m = m.eval().to(device)
                with torch.no_grad():
                    for x, idx in dl:
                        probs[idx.numpy(), si] = torch.softmax(m(x.to(device)).float(), 1).cpu().numpy()
                del m
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            st.evd.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out, probs=probs, classes=np.array(classes), paths=np.array([r[col] for r in rows]), scale=scale
            )
            print(f"{model:9s} {evset:14s} n={len(files):6d} {time.time() - t0:6.1f}s", flush=True)
