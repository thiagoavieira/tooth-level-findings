"""Numbers quoted in the published text (outside the tables), recomputed from ``results/``.

Each entry maps a name to a value in the unit the text uses (percentages as percent). The
expected values, transcribed from the published text, live in
``tests/expected/published_text_numbers.json``.
In the names, ``ff`` = false flags (sound teeth predicted as a finding), ``dz`` = Cohen's d_z,
``tau`` = confidence threshold, ``ref4`` = the four-class reference model and ``new`` = the
five-class mitigated model.
"""

from __future__ import annotations

from collections import Counter

from ..constants import CLASSES4
from ..external.thresholds import calibrate
from ..stats.bootstrap import wilson_ci
from ..stats.tests import block_matrix, cohens_dz, load_cv
from ..utils import read_csv, read_json
from .sources import Sources


def _aug_effects(S: Sources) -> dict[str, float]:
    """Cohen's d_z of augmented against non-augmented, per enhancement and classic backbone."""
    M, configs, _ = block_matrix(load_cv(S.E))
    out = {}
    for enh in ("none", "msthgr", "msthgr-clahe"):
        for arch in ("resnetv2_50", "vgg16", "inception_v3"):
            a = M[:, configs.index(f"{enh}__aug__{arch}")]
            b = M[:, configs.index(f"{enh}__noaug__{arch}")]
            out[f"{enh}__{arch}"] = cohens_dz(a, b)
    return out


def training_time(S: Sources) -> dict[str, float]:
    """Summarise the training wall-clock of the tooth-level grid from ``experiments/registry.csv``.

    Returns
    -------
    dict
        ``runs``, ``hours`` (total), ``median_min`` (element ``n // 2`` of the sorted durations,
        i.e. the upper median for an even count) and the run count and hours of the reference
        configuration (``inception_runs``, ``inception_hours``).
    """
    from .costs import training_times

    rows = training_times(S.E / "registry.csv")
    durations = sorted(float(r["dur_s"]) for r in rows)
    reference_seconds = sum(float(r["dur_s"]) for r in rows if r["group"] == "none|noaug|inception_v3")
    n_reference = sum(1 for r in rows if r["group"] == "none|noaug|inception_v3")
    return {
        "runs": len(durations),
        "hours": sum(durations) / 3600,
        "median_min": durations[len(durations) // 2] / 60,
        "inception_runs": n_reference,
        "inception_hours": reference_seconds / 3600,
    }


def class_mix(rows: list[dict]) -> dict[str, float]:
    """Return the share (%) of each predicted class (``pred`` column) among ``rows``."""
    counts = Counter(r["pred"] for r in rows)
    return {k: 100 * counts[k] / len(rows) for k in CLASSES4}


def text_numbers(S: Sources) -> dict[str, float]:
    """Recompute every recomputable number of the published text, keyed by name."""
    out: dict[str, float] = {}
    grid = S.grid_stats
    out["friedman_chi2"] = grid["friedman"]["statistic"]
    out["friedman_p"] = grid["friedman"]["p_value"]
    out["paired_pairs"] = len(grid["paired"])
    out["paired_significant_holm"] = sum(r["significant_holm_0.05"] for r in grid["paired"])
    aug_effects = _aug_effects(S)
    out["dz_aug_grayscale_resnetv2_50"] = aug_effects["none__resnetv2_50"]
    out["dz_aug_grayscale_vgg16"] = aug_effects["none__vgg16"]
    out["dz_aug_most_negative"] = min(aug_effects.values())

    out.update({f"training_{k}": v for k, v in training_time(S).items()})

    flags = S.prevalence("inception_v3__paper_s1.0")["false_flags_per_exam"]
    lo, hi = wilson_ci(round(flags["share_zero"] * flags["n_exams"]), flags["n_exams"])
    out.update(
        {
            "ff_per_exam": flags["mean"],
            "ff_median": flags["median"],
            "ff_p90": flags["p90"],
            "exams_without_ff_pct": 100 * flags["share_zero"],
            "exams_without_ff_ci_low": 100 * lo,
            "exams_without_ff_ci_high": 100 * hi,
            "ff_exams": flags["n_exams"],
        }
    )

    gradcam = read_json(S.E / "gradcam_masks" / "summary_inception_v3.json")
    out.update(
        {
            "gradcam_maps": gradcam["all"]["n"],
            "gradcam_no_mask": gradcam["n_teeth_no_mask"],
            "gradcam_mass_inside_pct": 100 * gradcam["all"]["mass_inside"],
            "gradcam_uniform_pct": 100 * gradcam["all"]["mask_area_frac"],
            "gradcam_iou_top20": gradcam["all"]["iou_top20"],
            "gradcam_pointing_pct": 100 * gradcam["all"]["pointing"],
        }
    )

    external = S.paths.external
    dentex_pred = read_csv(external / "dentex" / "predictions.csv")
    tufts_pred = read_csv(external / "tufts" / "predictions.csv")
    dentex_mix, tufts_mix = class_mix(dentex_pred), class_mix(tufts_pred)
    # predictions of the audited model (DeiT-S with position-aware negatives)
    tufts_audited_csv = external / "tufts" / "predictions_deit_posneg.csv"
    out.update(
        {
            "dentex_teeth": len(dentex_pred),
            "dentex_radiographs": len({r["pan_id"] for r in dentex_pred}),
            "dentex_sound_pct": dentex_mix["Healthy"],
            "dentex_impacted_pct": dentex_mix["Impacted"],
            "dentex_implant_pct": dentex_mix["Implant"],
            "dentex_endo_pct": dentex_mix["Endodontics"],
            "tufts_teeth": len(tufts_pred),
            "tufts_radiographs": len(read_csv(external / "tufts" / "pans.csv")),
            "tufts_radiographs_with_detection": len({r["pan_id"] for r in tufts_pred}),
            "tufts_implant_pct": tufts_mix["Implant"],
            "tufts_implant_pct_audited": class_mix(read_csv(tufts_audited_csv))["Implant"],
        }
    )

    calibration = calibrate(S.E / "prevalence" / "deit_small_patch16_224__posneg_s1.0" / "predictions.csv")
    out.update({f"study_tau_{c}": calibration[c]["tau"] for c in CLASSES4})

    prevalence = S.five_mitigated["B_prevalence"]
    ref, new = prevalence["REF4_prior"], prevalence["NEW_prior"]
    out.update(
        {
            "five_restored_teeth": ref["plus_restored_restored_teeth"]["n"],
            "five_ref4_restored_called_endo": ref["plus_restored_restored_teeth"]["ensemble"]["Endodontics"],
            "five_ref4_ff_with_restored": ref["plus_restored_4labels"]["ff"]["mean"],
            "five_ref4_ff_base": ref["base_8672"]["ff"]["mean"],
            "five_new_restored_called_endo": new["plus_restored_restored_teeth"]["ensemble"]["Endodontics"],
            "five_new_ff_with_restored": new["plus_restored_5labels"]["ff"]["mean"],
            "five_new_exams_without_ff_pct": 100 * new["plus_restored_5labels"]["ff"]["share_zero"],
            "five_ref4_exams_without_ff_pct": 100 * ref["plus_restored_4labels"]["ff"]["share_zero"],
            "five_new_impacted_precision_pct": 100 * new["base_8672"]["per_class"]["Impacted"]["precision"]["mean"],
            "five_new_impacted_precision_sd": 100 * new["base_8672"]["per_class"]["Impacted"]["precision"]["sd"],
            "five_ref4_impacted_precision_pct": 100 * ref["base_8672"]["per_class"]["Impacted"]["precision"]["mean"],
            "five_ref4_impacted_precision_sd": 100 * ref["base_8672"]["per_class"]["Impacted"]["precision"]["sd"],
            "five_new_ff_base": new["base_8672"]["ff"]["mean"],
        }
    )
    return out
