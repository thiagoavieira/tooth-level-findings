"""Recomputed inputs of the tables, cached per configuration.

Everything here is derived from ``results/`` with the package's own analysis code: run
summaries and statistics from the per-run ``metrics.json``, prevalence aggregates from
``per_seed.json`` and the ensemble predictions, operating points and false flags from the
prediction files, the audit from ``review_wide.csv``. Only measurements and evaluations that
need images or weights (latency, crop jitter, end-to-end detection, DENTEX matching, the
restored-class report) are read as released.

Terms used below: the *capped* test set is the canonical test split, in which the sound class is
capped so that the four classes are roughly balanced; the *deployment-prevalence* set adds every
other sound tooth of the same test patients. A *false flag* is a sound tooth predicted as a
finding. *LODO* (leave one dataset out) trains without one source and tests on it.
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property

import numpy as np

from ..config import Paths
from ..constants import CLASSES4, CLASSES5
from ..evaluation import operating_points as op
from ..evaluation.fp_by_tooth import fp_by_tooth_from_predictions
from ..evaluation.prevalence import aggregate_per_seed, false_flags_per_exam
from ..utils import read_csv, read_json


class Sources:
    """Lazily computed inputs of every table and text number.

    Each input is computed on first access and cached, so tables that share an input (for
    example the run summaries) compute it once.

    Parameters
    ----------
    paths : Paths
        Path configuration; :attr:`E` is a shortcut for ``paths.experiments``.
    """

    def __init__(self, paths: Paths):
        self.paths = paths
        self.E = paths.experiments

    # ------------------------------------------------------------------ grid and backbones
    @cached_property
    def summaries(self) -> tuple[dict, dict]:
        """Return the ``(cross-validation, test)`` summaries, each keyed by ``(enhancement, aug, arch)``."""
        from ..stats.summaries import run_summaries

        cv, test = run_summaries(self.E)
        key = lambda r: (r["enhancement"], r["aug"], r["arch"])  # noqa: E731
        return {key(r): r for r in cv}, {key(r): r for r in test}

    @cached_property
    def grid_stats(self) -> dict:
        """Friedman, Nemenyi and paired tests of the 18 classic configurations."""
        from ..stats.tests import run_stats_tests

        return run_stats_tests(self.E)

    @cached_property
    def new_backbones(self) -> dict:
        """Backbone comparison against InceptionV3, keyed by arch."""
        from ..stats.tests import new_backbones

        return {r["arch"]: r for r in new_backbones(self.E)}

    @cached_property
    def cost(self) -> dict:
        """``inference_cost.csv`` rows keyed by arch (measured; read as released)."""
        return {r["arch"]: r for r in read_csv(self.E / "cost" / "inference_cost.csv")}

    @cached_property
    def latency(self) -> dict:
        """Tooth-level rows of the unified latency session keyed by model label (measured)."""
        return {r["model"]: r for r in read_csv(self.paths.reporting / "latency_all.csv") if r["task"] == "task1"}

    @cached_property
    def bootstrap(self) -> dict:
        """Crop-level bootstrap of the reference test macro-F1 (2,000 draws, seed 0)."""
        from ..stats.bootstrap import bootstrap_ci, load_seed_predictions

        data, n, _ = load_seed_predictions(self.E / "test", "inception_v3")
        return bootstrap_ci(data, n, 2000, 0)

    # ------------------------------------------------------------------ prevalence
    def prevalence(self, run: str) -> dict:
        """Recompute the aggregates of one prevalence run from ``per_seed.json`` and ``predictions.csv``.

        Parameters
        ----------
        run : str
            Run folder under ``experiments/prevalence/``, ``<arch>__<tag>_s<crop scale>[_prior]``.

        Returns
        -------
        dict
            The output of ``aggregate_per_seed`` (``macro_f1_capped``, ``macro_f1_full``,
            ``per_class_capped``, ``per_class_full`` as ``[mean, sd]`` over seeds) plus
            ``false_flags_per_exam`` of the ensemble, ``n_full`` (crops at deployment prevalence)
            and ``n_capped`` (crops of the capped test set).
        """
        cache = self.__dict__.setdefault("_prev", {})
        if run not in cache:
            base = self.E / "prevalence" / run
            agg = aggregate_per_seed(read_json(base / "per_seed.json"))
            rows = read_csv(base / "predictions.csv")
            agg["false_flags_per_exam"] = false_flags_per_exam(
                [r["class"] for r in rows], [CLASSES4.index(r["pred"]) for r in rows], [r["patient_id"] for r in rows]
            )
            agg["n_full"], agg["n_capped"] = len(rows), sum(r["in_capped_test"] == "1" for r in rows)
            cache[run] = agg
        return cache[run]

    def operating(self, run: str) -> list[dict]:
        """Return the operating-point curve (one row per confidence threshold) of a prevalence run."""
        return op.analyze(self.E / "prevalence" / run / "predictions.csv")

    @cached_property
    def fp_by_tooth(self) -> dict:
        """Sound teeth flagged impacted by coarse tooth type (reference model, deployment prevalence)."""
        prev = self.E / "prevalence"
        return fp_by_tooth_from_predictions(
            prev / "inception_v3" / "predictions.csv", prev / "inception_v3__paper_s1.0_prior" / "predictions.csv"
        )

    # ------------------------------------------------------------------ splits and robustness
    @cached_property
    def split_counts(self) -> Counter:
        """Crops by ``(class, source, partition)`` of the canonical partition."""
        return Counter(
            (r["class"], r["source"], r["partition"])
            for r in read_csv(self.paths.split_dir("splits") / "split_manifest.csv")
        )

    @cached_property
    def jitter(self) -> dict:
        """Crop-noise results (needs images to recompute; read as released)."""
        return {r["condition"]: r for r in read_csv(self.E / "robustness" / "jitter_inception_v3.csv")}

    @cached_property
    def end2end(self) -> dict:
        """End-to-end results (needs images and the ONNX models; read as released)."""
        return read_json(self.E / "robustness" / "end2end_inception_v3.json")

    @cached_property
    def detection_counts(self) -> dict:
        """``{class: (detected, annotated)}`` recomputed from the per-tooth file."""
        from ..evaluation.robustness import detection_recall_from_teeth

        return detection_recall_from_teeth(self.E / "robustness" / "end2end_teeth.csv")

    @cached_property
    def dentex(self) -> dict:
        """DENTEX impacted check (needs the DENTEX annotations; read as released)."""
        return read_json(self.paths.external / "dentex" / "dentex_impacted_eval.json")

    # ------------------------------------------------------------------ audit, restored class, LODO
    @cached_property
    def audit(self):
        """Return the expert-review data (:class:`~toothfindings.audit.core.AuditData`)."""
        from ..audit.core import AuditData

        return AuditData(self.paths)

    @cached_property
    def audit_bins(self) -> dict:
        """Return the audit by dataset and confidence bin.

        Returns
        -------
        dict
            ``{dataset: {bin label: {"est_predictions", "precision", "ci95"}}}``, where
            ``est_predictions`` is the population estimate of the number of predictions in the bin
            (sum of the stratum weights ``N_h / n_h`` of the usable reviewed crops).
        """
        from ..audit.cascade import bin_ci, bin_key

        ci = bin_ci(self.audit, write=False)
        out = {}
        for ds in ("tufts", "dentex"):
            _strata, rows, weights, _tau = self.audit.setup(ds)
            est_predictions: dict = Counter()
            for r in rows:
                if r["label"] in ("Exclude", "Bad crop"):
                    continue
                k = bin_key(float(r["pred_prob"]))
                if k is not None:
                    est_predictions[k] += weights[(r["model_pred"], r["stratum"])]
            out[ds] = {k: {"est_predictions": est_predictions[k], **ci[ds][k]} for k in ci[ds]}
        return out

    @cached_property
    def restored(self) -> dict:
        """Four- vs five-class report (needs images and weights; read as released)."""
        return read_json(self.E / "clean_e1_report.json")

    def lodo(self, exp: str, source: str) -> dict:
        """Score a five-class leave-one-dataset-out experiment on one test source.

        Parameters
        ----------
        exp : str
            Experiment folder under ``experiments/`` (``lodo_lyria``, ``lodo_dentex`` or ``lodo_tufts``).
        source : str
            Source whose test crops are scored (``dentex`` or ``tufts``).

        Returns
        -------
        dict
            ``n`` (crops), ``f1`` (per-class F1, mean over the 3 seeds) and ``macro``
            (macro-F1 as ``(mean, population sd)`` over the seeds).
        """
        from ..evaluation.five_mitigated import prf

        per_seed = []
        for s in range(3):
            rows = [
                r
                for r in read_csv(
                    self.E / exp / "test" / f"none__noaug__deit_small_patch16_224__seed{s}" / "predictions.csv"
                )
                if r["source"] == source
            ]
            per_seed.append(prf(np.array([r["y_true"] for r in rows]), np.array([r["y_pred"] for r in rows]), CLASSES5))
        return {
            "n": len(rows),
            "f1": {c: float(np.mean([per_class[c]["f1"] for per_class, _ in per_seed])) for c in CLASSES5},
            "macro": (float(np.mean([m for _, m in per_seed])), float(np.std([m for _, m in per_seed]))),
        }

    @cached_property
    def five_mitigated(self) -> dict:
        """Recompute every number of the five-class mitigated study from the stored probabilities."""
        from ..evaluation.five_mitigated import analyze

        return analyze(self.paths, write=False, bootstrap=False)
