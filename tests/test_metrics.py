from collections import Counter

import numpy as np
import pytest

from toothfindings.constants import CLASSES4
from toothfindings.evaluation import operating_points, prevalence
from toothfindings.evaluation.gradcam import cam_scores
from toothfindings.evaluation.robustness import greedy_match, iou, perturb
from toothfindings.models.ensemble import metrics
from toothfindings.stats.bootstrap import bootstrap_ci, wilson_ci
from toothfindings.stats.summaries import mean_std_ci, summarize
from toothfindings.stats.tests import block_matrix, cohens_dz, friedman, holm, paired_tests

ENDO, SOUND, IMPACTED, IMPLANT = range(4)


def test_laplace_prior():
    prior = prevalence.laplace_prior({"3rd molar": Counter({"Impacted": 6, "Healthy": 2})})
    np.testing.assert_allclose(prior["3rd molar"], [1 / 12, 3 / 12, 7 / 12, 1 / 12])


def test_prior_reweights_and_renormalises_every_seed():
    probs = np.full((2, 3, 4), 0.25)
    prior = {"3rd molar": np.array([0.1, 0.2, 0.6, 0.1])}
    out = prevalence.apply_prior(probs, ["3rd molar", "unknown"], prior)
    np.testing.assert_allclose(out.sum(-1), 1.0)
    np.testing.assert_allclose(out[0, 1], [0.1, 0.2, 0.6, 0.1])
    np.testing.assert_allclose(out[1], probs[1])  # unknown tooth type: uniform prior, no change


def test_false_flags_count_only_sound_teeth_of_each_exam():
    true = ["Healthy", "Healthy", "Implant", "Healthy", "Healthy", "Impacted"]
    pred = [IMPACTED, SOUND, ENDO, SOUND, SOUND, IMPACTED]
    pids = ["a", "a", "a", "b", "b", "c"]
    ff = prevalence.false_flags_per_exam(true, pred, pids)
    assert ff == {"mean": 0.5, "median": 0.5, "p90": 0.9, "share_zero": 0.5, "n_exams": 2}


def test_aggregate_per_seed_uses_the_sample_sd():
    def seed(f):
        return {"macro_f1": f, "per_class": {c: {"precision": f, "recall": f, "f1": f} for c in CLASSES4}}

    agg = prevalence.aggregate_per_seed({"capped": [seed(0.8), seed(0.9)], "full": [seed(0.5), seed(0.7)]})
    assert agg["macro_f1_capped"] == pytest.approx([0.85, np.std([0.8, 0.9], ddof=1)])
    assert agg["per_class_full"]["Implant"]["recall"] == pytest.approx([0.6, np.std([0.5, 0.7], ddof=1)])


def test_operating_points_on_a_toy_exam():
    y = np.array([SOUND, SOUND, SOUND, IMPACTED, IMPACTED, ENDO, IMPLANT])
    P = np.array(
        [
            [0.0, 0.1, 0.9, 0.0],  # sound tooth flagged as impacted at 0.9
            [0.0, 0.4, 0.6, 0.0],  # sound tooth flagged as impacted at 0.6
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.05, 0.95, 0.0],
            [0.0, 0.3, 0.7, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pid = np.array(["a", "a", "b", "a", "b", "c", "c"])
    curve = {r["tau"]: r for r in operating_points.analyze_arrays(y, P, pid, grid=[0.0, 0.8])}
    assert curve[0.0]["false_flags_per_exam"] == 1.0
    assert curve[0.0]["share_exams_zero"] == 0.5
    assert curve[0.0]["recall"]["Impacted"] == 1.0
    assert curve[0.0]["precision_impacted"] == 0.5
    assert curve[0.8]["n_sound_flagged"] == 1
    assert curve[0.8]["recall"] == {"Endodontics": 1.0, "Impacted": 0.5, "Implant": 1.0}
    assert curve[0.8]["precision_impacted"] == 0.5


def test_metrics_with_zero_division():
    m = metrics([0, 1, 1, 2], [0, 1, 2, 2])
    assert m["per_class"]["Implant"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert m["accuracy"] == 0.75
    assert m["confusion"][1] == [0, 1, 1, 0]
    assert m["macro_f1"] == pytest.approx((1 + 2 / 3 + 2 / 3 + 0) / 4)


def test_perturb_scales_about_the_centre_and_rejects_degenerate_boxes():
    assert perturb((10, 10, 30, 50), 100, 100, s=1.5) == (5, 0, 35, 60)
    assert perturb((10, 10, 30, 50), 100, 100, s=0.1) == (10, 10, 30, 50)
    rng = np.random.default_rng(1234)
    x1, y1, x2, y2 = perturb((40, 40, 60, 80), 100, 100, rng, d=0.1)
    assert abs((x1 + x2) / 2 - 50) <= 2 + 1 and abs((y1 + y2) / 2 - 60) <= 4 + 1
    assert 17 <= x2 - x1 <= 23 and 35 <= y2 - y1 <= 45


def test_iou_and_greedy_matching():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    gt = [(0, 0, 10, 10), (20, 0, 30, 10)]
    pred = [(21, 0, 31, 10), (0, 0, 10, 9), (100, 100, 110, 110)]
    m = greedy_match(gt, pred, 0.5)
    assert {g: p for g, (p, _) in m.items()} == {0: 1, 1: 0}


def test_wilson_interval():
    lo, hi = wilson_ci(81, 263)
    assert (lo, hi) == pytest.approx((0.2553, 0.3662), abs=1e-4)
    assert all(np.isnan(wilson_ci(0, 0)))


def test_bootstrap_of_perfect_predictions():
    y = np.array(CLASSES4 * 5)
    ci = bootstrap_ci([(y, y), (y, y)], len(y), n_resamples=50)
    assert ci["ci95_low"] == ci["ci95_high"] == ci["draws_mean"] == 1.0


def test_mean_std_ci():
    assert mean_std_ci([]) == {"mean": "", "std": "", "ci95_low": "", "ci95_high": "", "n": 0}
    assert mean_std_ci([0.5])["std"] == 0.0
    s = mean_std_ci([0.8, 0.9, 1.0])
    assert (s["mean"], s["std"], s["n"]) == (0.9, 0.1, 3)
    assert s["ci95_low"] == pytest.approx(0.9 - 4.302653 * 0.1 / 3**0.5, abs=1e-4)


def test_summarize_groups_runs_by_configuration():
    def run(arch, f):
        pc = {c: {"f1": f} for c in CLASSES4}
        return {"enhancement": "none", "aug": "noaug", "arch": arch, "val_metrics": {"macro_f1": f, "per_class": pc}}

    rows = summarize([run("vgg16", 0.8), run("vgg16", 0.9), run("inception_v3", 0.7)], "val_metrics")
    assert [(r["arch"], r["n_runs"], r["macro_f1_mean"]) for r in rows] == [
        ("inception_v3", 1, 0.7),
        ("vgg16", 2, 0.85),
    ]
    assert "accuracy_mean" not in rows[0]


def test_holm_and_cohens_dz():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert cohens_dz([1, 2, 3], [0, 0, 0]) == pytest.approx(2.0)
    assert cohens_dz([1, 2], [1, 2]) == 0.0


def test_friedman_and_paired_tests_on_blocks():
    rng = np.random.default_rng(0)
    data = {
        cfg: {(f, s): base + rng.normal(0, 0.01) for f in range(5) for s in range(3)}
        for cfg, base in (("a", 0.80), ("b", 0.85), ("c", 0.90))
    }
    data["d"] = {(0, 0): 0.5}  # incomplete configuration: excluded from the shared blocks
    M, configs, blocks = block_matrix({k: v for k, v in data.items() if k != "d"})
    assert M.shape == (15, 3) and configs == ["a", "b", "c"] and len(blocks) == 15
    assert friedman(M)["p_value"] < 1e-4
    rows = paired_tests(M, configs)
    assert [(r["config_a"], r["config_b"]) for r in rows] == [("a", "b"), ("a", "c"), ("b", "c")]
    assert all(r["significant_holm_0.05"] and r["cohens_d"] < 0 for r in rows)
    assert all(r["t_p_holm"] >= r["t_p"] for r in rows)


def test_cam_scores():
    g = np.zeros((10, 10))
    g[2:4, 2:4] = 1.0
    g[8, 8] = -5.0  # negative attributions are clipped
    mask = np.zeros((10, 10), bool)
    mask[:5, :5] = True
    inside, iou_top, hit = cam_scores(g, mask, top=0.04)
    assert inside == pytest.approx(1.0)
    assert iou_top == pytest.approx(4 / 25)
    assert hit == 1
