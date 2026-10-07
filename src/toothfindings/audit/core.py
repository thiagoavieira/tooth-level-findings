"""Strata, reviewed crops, weights and weighted precision estimates shared by the audit analyses.

Vocabulary used throughout :mod:`toothfindings.audit`:

* ``tau`` (``tau_c``): the study threshold of class ``c`` (see
  :mod:`toothfindings.external.thresholds`).
* Strata of the predictions of each class: ``A`` = score >= tau with three agreeing seeds
  (sampled, review bucket ``confirm``), ``B`` = score < tau (sampled, bucket ``below_tau``),
  ``C`` = score >= tau without unanimity (not sampled; reported as uncovered).
* Weight of a reviewed crop: ``N_h / n_h``, the stratum size over the number of crops reviewed
  in it, so weighted sums estimate counts over all predictions.
* Expert labels: the four clinical classes, ``Other`` (a tooth outside the four-class label
  space, e.g. restored), and ``Exclude``/``Bad crop`` (crop not usable, e.g. a segmentation
  failure).
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from ..config import Paths
from ..utils import read_csv, read_json

#: The four clinical classes (``Healthy`` = sound tooth).
CLIN = ["Endodontics", "Healthy", "Impacted", "Implant"]
#: Finding classes (every clinical class except sound).
FIND = ["Endodontics", "Impacted", "Implant"]
#: The seven categories of the review form.
ALL7 = CLIN + ["Other", "Exclude", "Bad crop"]
#: Labels of crops the experts could not use.
UNUSABLE = ["Exclude", "Bad crop"]
#: Labels of teeth outside the four-class label space.
OUTSIDE = ["Other"]
#: Tag of the audited model: DeiT-S trained with position-aware negatives.
TAG = "deit_posneg"
#: Number of radiographs per external dataset.
N_PANS = {"tufts": 1000, "dentex": 1067}
DATASETS = ("tufts", "dentex")


class AuditData:
    """Inputs of the audit, read once and cached.

    Parameters
    ----------
    paths : Paths
        Configured paths (``review_wide.csv`` under ``paths.review``, predictions and
        thresholds under ``paths.external``).
    tag : str
        Tag of the audited model's prediction and threshold files.
    """

    def __init__(self, paths: Paths, tag: str = TAG):
        self.paths = paths
        self.tag = tag
        self._wide: list[dict] | None = None
        self._preds: dict[str, list[dict]] = {}

    def wide(self) -> list[dict]:
        """Return the rows of ``review_wide.csv`` (one per reviewed crop)."""
        if self._wide is None:
            self._wide = read_csv(self.paths.review / "review_wide.csv")
        return self._wide

    def predictions(self, ds: str) -> list[dict]:
        """Return the external predictions of the audited model on dataset ``ds``."""
        if ds not in self._preds:
            self._preds[ds] = read_csv(self.paths.external / ds / f"predictions_{self.tag}.csv")
        return self._preds[ds]

    def strata(self, ds: str) -> dict:
        """Return the review strata of dataset ``ds``.

        Returns
        -------
        dict
            ``{class: {"tau", "n_pop", "A", "B", "C_uncovered"}}``: the study threshold, the
            number of predictions of the class and the sizes of strata A, B and C.
        """
        th = read_json(self.paths.external / ds / f"thresholds_{self.tag}.json")["summary"]
        preds = self.predictions(ds)
        out = {}
        for c, s in th.items():
            tau = s["tau"]
            pop = [r for r in preds if r["pred"] == c]
            out[c] = {
                "tau": tau,
                "n_pop": len(pop),
                "A": sum(1 for r in pop if float(r["pred_prob"]) >= tau and int(r["seed_agreement"]) == 3),
                "B": sum(1 for r in pop if float(r["pred_prob"]) < tau),
                "C_uncovered": sum(1 for r in pop if float(r["pred_prob"]) >= tau and int(r["seed_agreement"]) < 3),
            }
        return out

    def reviewed(self, ds: str, with_excluded: bool = False) -> tuple[list[dict], int, int]:
        """Return the reviewed crops of dataset ``ds`` that have a consensus label.

        A crop read twice is kept only if both readers agree. Crops declared out of scope
        (``excluded = 1``) are dropped unless ``with_excluded``; they still count in the
        stratum sample size (see :meth:`setup`).

        Returns
        -------
        rows : list of dict
            Copies of the ``review_wide.csv`` rows with ``label`` and ``stratum`` added.
        n_disagree : int
            Crops dropped because the two readers disagree.
        n_excluded : int
            Crops declared out of scope (dropped or not).
        """
        keep, disagree, excluded = [], 0, 0
        for r0 in self.wide():
            if r0["dataset"] != ds:
                continue
            r = dict(r0)
            labels = [r[k] for k in ("label1", "label2") if r.get(k)]
            if len(labels) == 2 and labels[0] != labels[1]:
                disagree += 1
                continue
            r["label"] = labels[0]
            r["stratum"] = "A" if r["bucket"] == "confirm" else "B"
            if r.get("excluded") == "1":
                excluded += 1
                if not with_excluded:
                    continue
            keep.append(r)
        return keep, disagree, excluded

    def setup(self, ds: str) -> tuple[dict, list[dict], dict, dict]:
        """Return ``(strata, rows, weights, tau by class)`` of dataset ``ds``.

        ``rows`` excludes out-of-scope crops, while the weights count them in ``n_h``. Crops on
        which the two readers disagree are dropped from both, so ``n_h`` is smaller than the
        sample as drawn and those crops are treated as missing at random.
        """
        st = self.strata(ds)
        rows = self.reviewed(ds)[0]
        w = weights(st, rows, sampled=self.reviewed(ds, with_excluded=True)[0])
        return st, rows, w, {c: st[c]["tau"] for c in CLIN}


def weights(st: dict, rows: list[dict], sampled: list[dict] | None = None) -> dict:
    """Compute the weight ``N_h / n_h`` of each (class, stratum).

    Parameters
    ----------
    st : dict
        Strata from :meth:`AuditData.strata` (``N_h`` = ``st[class][stratum]``).
    rows : list of dict
        Reviewed crops (with ``model_pred`` and ``stratum``).
    sampled : list of dict, optional
        Crops on which ``n_h`` is counted; ``rows`` by default.

    Returns
    -------
    dict
        ``{(class, "A" | "B"): weight}``; 0.0 for an empty stratum sample.
    """
    n: dict = defaultdict(int)
    for r in sampled if sampled is not None else rows:
        n[(r["model_pred"], r["stratum"])] += 1
    w = {}
    for c in CLIN:
        for h in ("A", "B"):
            n_h = n[(c, h)]
            w[(c, h)] = (st[c][h] / n_h) if n_h else 0.0
    return w


def estimate(rows: list[dict], w: dict, tau_by_class: dict, thr: float | None = None, lenient: bool = False) -> dict:
    """Estimate the precision of each class for the rule "flag ``c`` if score >= threshold".

    Parameters
    ----------
    rows : list of dict
        Reviewed crops with ``label`` and ``stratum``.
    w : dict
        Weights from :func:`weights`.
    tau_by_class : dict
        Study threshold of each class, used when ``thr`` is ``None``.
    thr : float, optional
        One threshold for every class.
    lenient : bool
        Also count as correct a finding prediction labelled with another finding class.

    Returns
    -------
    dict
        ``{class: (precision or None, estimated number of predictions kept)}``.
    """
    num: dict = defaultdict(float)
    den: dict = defaultdict(float)
    for r in rows:
        c = r["model_pred"]
        t = thr if thr is not None else tau_by_class[c]
        if float(r["pred_prob"]) < t:
            continue
        wi = w[(c, r["stratum"])]
        ok = (r["label"] == c) if not lenient else (r["label"] == c or (c != "Healthy" and r["label"] in FIND))
        num[c] += wi * ok
        den[c] += wi
    return {c: (num[c] / den[c] if den[c] else None, den[c]) for c in CLIN}


def resample_groups(rows: list[dict], rng: random.Random, key: str = "pan_id"):
    """Return a function that draws one cluster-bootstrap sample of ``rows``.

    Each call draws as many clusters (radiographs, grouped by ``key``) as there are, with
    replacement, and returns the concatenation of their rows.
    """
    groups: dict = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    keys = list(groups)

    def draw() -> list[dict]:
        sample: list = []
        for _ in range(len(keys)):
            sample += groups[keys[rng.randrange(len(keys))]]
        return sample

    return draw


def percentile_ci(values) -> list[float]:
    """Return the 2.5th and 97.5th percentiles of ``values`` (a 95% percentile interval)."""
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def cluster_boot(rows, w, tau_by_class, thr, lenient, n_boot: int = 1000, seed: int = 0) -> dict:
    """Compute cluster-bootstrap 95% intervals of :func:`estimate`.

    ``rows``, ``w``, ``tau_by_class``, ``thr`` and ``lenient`` are as in :func:`estimate`;
    the weights are kept fixed across resamples.

    Returns
    -------
    dict
        ``{class: [low, high]}`` for the classes with at least one defined resample.
    """
    draw = resample_groups(rows, random.Random(seed))
    acc: dict = defaultdict(list)
    for _ in range(n_boot):
        for c, (p, _n) in estimate(draw(), w, tau_by_class, thr, lenient).items():
            if p is not None:
                acc[c].append(p)
    return {c: percentile_ci(v) for c, v in acc.items() if v}
