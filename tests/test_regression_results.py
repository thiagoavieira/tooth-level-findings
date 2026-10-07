"""The released intermediate files under results/ are reproduced by the package from their inputs."""

import csv
import json
import math

import pytest

from _compare import cmp
from toothfindings.audit import agreement, cascade, implant, operating, precision
from toothfindings.audit.core import AuditData
from toothfindings.evaluation.fp_by_tooth import STAT_KEYS
from toothfindings.evaluation.operating_points import operating_points
from toothfindings.utils import read_csv, read_json


def _same_csv(rows, path):
    ref = read_csv(path)
    assert len(rows) == len(ref)
    for a, b in zip(rows, ref):
        for k, v in b.items():
            x = a[k]
            if isinstance(x, bool) or v in ("True", "False"):
                assert str(x) == v, (k, x, v)
            elif isinstance(x, (int, float)):
                assert math.isclose(float(x), float(v), rel_tol=1e-9, abs_tol=1e-12), (k, x, v)
            else:
                assert str(x) == v, (k, x, v)


def test_run_summaries(paths, sources):
    cv, te = sources.summaries
    _same_csv(list(cv.values()), paths.experiments / "aggregates" / "cv_summary.csv")
    _same_csv(list(te.values()), paths.experiments / "aggregates" / "test_summary.csv")


def test_grid_statistics(paths, sources):
    st = sources.grid_stats
    assert not cmp(st["friedman"], read_json(paths.experiments / "stats" / "friedman.json"))
    _same_csv(st["paired"], paths.experiments / "stats" / "paired_tests.csv")
    with open(paths.experiments / "stats" / "nemenyi.csv") as fh:
        nem = list(csv.reader(fh))
    assert [r[0] for r in nem[1:]] == list(st["nemenyi"].index)
    for row, ours in zip(nem[1:], st["nemenyi"].values):
        assert all(math.isclose(float(a), b, rel_tol=1e-9) for a, b in zip(row[1:], ours))


def test_new_backbones(paths, sources):
    ref = read_json(paths.experiments / "stats" / "new_backbones.json")
    assert not cmp(list(sources.new_backbones.values()), ref)


def test_bootstrap_test_ci(paths, sources):
    ref = read_json(paths.reporting / "bootstrap_test_ci.json")
    assert not cmp(sources.bootstrap, {k: v for k, v in ref.items() if k not in ("arch", "enhancement", "aug")})


# inception_v3/ is the older output format of the reference run (checked with summary.json below)
PREV_RUNS = sorted(
    p.name
    for p in (__import__("toothfindings.config").config.load_paths().experiments / "prevalence").iterdir()
    if (p / "per_seed.json").exists() and p.name != "inception_v3"
)


@pytest.mark.parametrize("run", PREV_RUNS)
def test_prevalence_metrics(paths, sources, run):
    ref = read_json(paths.experiments / "prevalence" / run / "metrics.json")
    ours = sources.prevalence(run)
    keys = ["macro_f1_capped", "macro_f1_full", "per_class_full", "per_class_capped", "false_flags_per_exam"]
    assert not cmp({k: ours[k] for k in keys}, {k: ref[k] for k in keys})
    assert (ours["n_full"], ours["n_capped"]) == (ref["n_full"], ref["n_capped"])


def test_prevalence_summary_of_the_reference_model(paths, sources):
    # summary.json was written by an earlier version of the evaluation (no writer in the code base);
    # it uses the population sd (ddof=0) where metrics.json uses the sample sd
    import numpy as np

    ref = read_json(paths.experiments / "prevalence" / "summary.json")["inception_v3"]
    per_seed = read_json(paths.experiments / "prevalence" / "inception_v3" / "per_seed.json")
    ours = sources.prevalence("inception_v3")
    for split in ("full", "capped"):
        f1 = [m["macro_f1"] for m in per_seed[split]]
        acc = [m["accuracy"] for m in per_seed[split]]
        assert math.isclose(ref[f"seed_mean_{split}"]["macro_f1"], np.mean(f1), rel_tol=1e-9)
        assert math.isclose(ref[f"seed_std_{split}"]["macro_f1"], np.std(f1), rel_tol=1e-9)
        assert math.isclose(ref[f"seed_mean_{split}"]["accuracy"], np.mean(acc), rel_tol=1e-9)
    ff, rff = ours["false_flags_per_exam"], ref["sound_false_flags_per_exam"]
    assert (ff["mean"], ff["median"], ff["p90"], ff["n_exams"], ff["share_zero"]) == (
        rff["mean"],
        rff["median"],
        rff["p90"],
        rff["n_exams"],
        rff["share_exams_zero_flags"],
    )


def test_operating_points(paths):
    prev = paths.experiments / "prevalence"
    assert not cmp(operating_points(prev), read_json(prev / "operating_point.json"))


def test_false_flags_by_tooth(paths, sources):
    ref = read_json(paths.experiments / "prevalence" / "fp_by_tooth_type.json")
    coarse = {
        "central incisor": "incisor",
        "lateral incisor": "incisor",
        "1st premolar": "premolar",
        "2nd premolar": "premolar",
    }
    for key in STAT_KEYS:
        merged = {}
        for t, n in ref[key].items():
            merged[coarse.get(t, t)] = merged.get(coarse.get(t, t), 0) + n
        assert sources.fp_by_tooth[key] == merged, key


@pytest.fixture(scope="module")
def audit_data(paths):
    return AuditData(paths)


AUDIT = {
    "audit": precision.audit,
    "in_scope": precision.in_scope,
    "breakdown": precision.breakdown,
    "cascade": cascade.cascade,
    "bin_ci": cascade.bin_ci,
    "agreement": agreement.agreement,
    "agreement_extra": agreement.agreement_extra,
    "implant_view": implant.implant_view,
    "operating_points": operating.operating_points,
    "fairness_view": operating.fairness_view,
    "impacted_recall": operating.impacted_recall,
}


@pytest.mark.parametrize("name", list(AUDIT))
def test_audit_outputs(paths, audit_data, name):
    ours = AUDIT[name](audit_data, write=False)
    assert not cmp(ours, read_json(paths.review / f"{name}.json"))


def test_bridge_metrics(paths, audit_data):
    assert not cmp(operating.bridge_metrics(audit_data, write=False), read_json(paths.review / "bridge_metrics.json"))


def test_five_class_study(paths, sources):
    ref = read_json(paths.experiments / "variants" / "five_mitigated" / "results" / "results.json")
    ours = json.loads(json.dumps(sources.five_mitigated))
    ref = {k: v for k, v in ref.items() if k != "bootstrap"}  # test_five_class_study_bootstrap
    assert not cmp(ours, ref)


@pytest.mark.slow
def test_five_class_study_bootstrap(paths):
    from toothfindings.evaluation.five_mitigated import analyze

    ref = read_json(paths.experiments / "variants" / "five_mitigated" / "results" / "results.json")["bootstrap"]
    ours = analyze(paths, write=False)["bootstrap"]
    # docs/REPRODUCE.md: this interval is one exam step (1/568) wider than the released one
    stray = "ff_per_exam_new_minus_ref4"
    assert cmp(ours[stray]["ci95"], ref[stray]["ci95"])
    assert ours[stray]["point"] == ref[stray]["point"]
    assert not cmp({k: v for k, v in ours.items() if k != stray}, {k: v for k, v in ref.items() if k != stray})


def test_training_times(paths):
    from toothfindings.reporting.costs import training_times

    ours = training_times(paths.experiments / "registry.csv")
    ref = [r for r in read_csv(paths.reporting / "training_times.csv") if r["task"] == "task1_iei"]
    assert [(r["run_id"], r["group"], r["dur_s"]) for r in ours] == [(r["run_id"], r["group"], r["dur_s"]) for r in ref]
