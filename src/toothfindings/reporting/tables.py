"""Generators of the published tables (``tab_*``), recomputed from ``results/``.

Each generator takes a :class:`~toothfindings.reporting.sources.Sources` and
returns a :class:`~toothfindings.reporting.latex.Table` whose layout follows the published
table; the notes are short and factual (the published notes were edited by hand). Abbreviations
in the tables: FF = false flag (a sound tooth predicted as a finding), P/R = precision/recall,
CV = cross-validation, sd = standard deviation.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Paths
from ..constants import CLASSES4, CLASSES5
from ..stats.bootstrap import wilson_ci
from .latex import (
    MIDRULE,
    Row,
    Table,
    Tabular,
    bold,
    ci_range,
    integer,
    join,
    meansd,
    multicol,
    num,
    paren,
    pct,
    pvalue,
    signed,
    text,
)
from .sources import Sources

#: Display labels of the classes, backbones, enhancements and augmentation settings.
CLASS_LABEL = {
    "Endodontics": "Endo. treated",
    "Healthy": "Sound",
    "Impacted": "Impacted",
    "Implant": "Implant",
    "Restored": "Restored",
}
ARCH_LABEL = {
    "resnetv2_50": "ResNet50V2",
    "vgg16": "VGG16",
    "inception_v3": "InceptionV3",
    "convnext_tiny": "ConvNeXt-T",
    "tf_efficientnetv2_s": "EfficientNetV2-S",
    "deit_small_patch16_224": "DeiT-S",
}
ENH_LABEL = {
    "none": "Grayscale",
    "msthgr": "MSTHGR",
    "msthgr-clahe": "MSTHGR+CLAHE",
}
AUG_LABEL = {"noaug": "No augmentation", "aug": "Augmented"}
#: The three classic backbones of the 18-configuration grid.
CLASSIC = ["resnetv2_50", "vgg16", "inception_v3"]


def class_distribution(S: Sources) -> Table:
    """``tab_class_distribution``: crops per class and source in the training pool and test set."""
    cnt = S.split_counts
    rows, tr_tot, te_tot = [], 0, 0
    for c in CLASSES4:
        srcs = sorted({s for (cc, s, _p) in cnt if cc == c}, key=lambda s: (s != "lyria", s))
        for i, s in enumerate(srcs):
            tr, te = cnt[(c, s, "train_pool")], cnt[(c, s, "test")]
            tr_tot += tr
            te_tot += te
            rows.append(
                Row(
                    [CLASS_LABEL[c] if i == 0 else "", "InReDD" if s == "lyria" else "DENTEX"],
                    [integer(tr, False), integer(te, False)],
                )
            )
    rows += [MIDRULE, Row(["Total", ""], [integer(tr_tot, False), integer(te_tot, False)])]
    return Table(
        "tab_class_distribution",
        [Tabular("llcc", [r"Class & Source & Train pool $n$ & Test $n$ \\"], rows, resize=None)],
        "the training pool draws 180 crops per class from the training patients; the test set keeps "
        "every crop of the other patients.",
    )


def cv_ablation(S: Sources) -> Table:
    """``tab_cv_ablation``: the 18 classic configurations ordered by cross-validated macro-F1."""
    cv, te = S.summaries
    configs = sorted((k for k in cv if k[2] in CLASSIC), key=lambda k: -float(cv[k]["macro_f1_mean"]))
    rows = [
        Row(
            [ENH_LABEL[k[0]], AUG_LABEL[k[1]], ARCH_LABEL[k[2]]],
            [
                meansd(cv[k]["macro_f1_mean"], cv[k]["macro_f1_std"]),
                meansd(te[k]["macro_f1_mean"], te[k]["macro_f1_std"]),
            ],
        )
        for k in configs
    ]
    fr = S.grid_stats["friedman"]
    header = [r"Enhancement & Augmentation & Backbone & CV macro-F1 (15 runs) & Test macro-F1 (3 seeds) \\"]
    return Table(
        "tab_cv_ablation",
        [Tabular("lllcc", header, rows)],
        rf"Friedman test over the 18 configurations with 15 CV blocks each, "
        rf"$\chi^2={fr['statistic']:.1f}$, $p={fr['p_value']:.1e}$.",
    )


def modern_backbones(S: Sources) -> Table:
    """``tab_modern_backbones``: classic and recent backbones against InceptionV3."""
    cv, te = S.summaries
    rows = []
    for i, arch in enumerate(
        ["resnetv2_50", "vgg16", "inception_v3", "convnext_tiny", "tf_efficientnetv2_s", "deit_small_patch16_224"]
    ):
        k = ("none", "noaug", arch)
        comparison, ref = S.new_backbones[arch], arch == "inception_v3"
        rows.append(
            Row(
                [ARCH_LABEL[arch]],
                [
                    num(float(S.cost[arch]["params_M"]), 1),
                    meansd(cv[k]["macro_f1_mean"], cv[k]["macro_f1_std"]),
                    text("ref.") if ref else signed(comparison["cohens_d"]),
                    text("ref.") if ref else pvalue(comparison["p_ttest_holm"]),
                    meansd(te[k]["macro_f1_mean"], te[k]["macro_f1_std"]),
                    *[pct(te[k][f"f1_{c}_mean"]) for c in CLASSES4],
                ],
            )
        )
        if i == 2:
            rows.append(MIDRULE)
    header = [
        r"Backbone & Params (M) & CV macro-F1 & $d$ vs.\ Inc. & $p_{Holm}$ & Test macro-F1 & "
        r"Endo. treated & Sound & Impacted & Implant \\"
    ]
    return Table(
        "tab_modern_backbones",
        [Tabular("l c c c c c c c c c", header, rows, r"\textwidth")],
        r"CV macro-F1 is the mean $\pm$ sd over 15 blocks; $d$ and $p_{Holm}$ come from the "
        r"paired $t$ test against InceptionV3, Holm-corrected; test values are means over 3 seeds.",
    )


#: Year of publication of each backbone.
YEAR = {
    "vgg16": 2015,
    "inception_v3": 2016,
    "resnetv2_50": 2016,
    "tf_efficientnetv2_s": 2021,
    "deit_small_patch16_224": 2021,
    "convnext_tiny": 2022,
}
#: Model label of a backbone in ``latency_all.csv`` when it differs from the backbone name.
LATENCY_LABEL = {"deit_small_patch16_224": "deit_small"}


def inference_cost(S: Sources) -> Table:
    """``tab_inference_cost``: size, latency (unified session) and test macro-F1."""
    _cv, te = S.summaries
    rows = []
    for arch in [
        "vgg16",
        "inception_v3",
        "resnetv2_50",
        "tf_efficientnetv2_s",
        "deit_small_patch16_224",
        "convnext_tiny",
    ]:
        lat = S.latency[LATENCY_LABEL.get(arch, arch)]
        rows.append(
            Row(
                [ARCH_LABEL[arch]],
                [
                    integer(YEAR[arch], False),
                    num(float(S.cost[arch]["params_M"]), 1),
                    num(float(S.cost[arch]["model_MB"]), 0),
                    num(float(lat["cpu_median_ms"]), 1),
                    num(float(lat["gpu_median_ms"]), 1),
                    pct(te[("none", "noaug", arch)]["macro_f1_mean"]),
                ],
            )
        )
    header = [
        r"Model & Year & Params (M) & Size (MB) & CPU (ms) & GPU (ms) & "
        r"Test macro-F1 \\"
    ]
    return Table(
        "tab_inference_cost",
        [Tabular("l c c c c c c", header, rows)],
        "median latency per crop at batch size 1 in one session; year of publication of each backbone.",
    )


def _per_class_cells(per_class: dict, c: str) -> list:
    return [meansd(*per_class[c][k]) for k in ("precision", "recall", "f1")]


def test_per_class(S: Sources) -> Table:
    """``tab_test_per_class``: reference model on the held-out test set, with the bootstrap CI."""
    prev = S.prevalence("inception_v3__paper_s1.0")
    _cv, te = S.summaries
    t = te[("none", "noaug", "inception_v3")]
    b = S.bootstrap
    rows = [Row([CLASS_LABEL[c]], _per_class_cells(prev["per_class_capped"], c)) for c in CLASSES4]
    rows += [
        MIDRULE,
        Row(
            ["Macro-F1"],
            [
                multicol(
                    3,
                    join(
                        meansd(t["macro_f1_mean"], t["macro_f1_std"]),
                        text(r", bootstrap 95\% CI"),
                        ci_range(b["ci95_low"], b["ci95_high"]),
                        sep="",
                    ),
                )
            ],
        ),
    ]
    return Table(
        "tab_test_per_class",
        [Tabular("lccc", [r"Class & Precision & Recall & F1 \\"], rows, resize=None)],
        "the macro-F1 interval resamples the test crops 2,000 times, averaging the 3 seeds in each draw.",
    )


def prevalence(S: Sources) -> Table:
    """``tab_prevalence``: capped test set against deployment prevalence."""
    prev = S.prevalence("inception_v3__paper_s1.0")
    _cv, te = S.summaries
    t = te[("none", "noaug", "inception_v3")]
    rows = [
        Row(
            [CLASS_LABEL[c]],
            _per_class_cells(prev["per_class_capped"], c) + _per_class_cells(prev["per_class_full"], c),
        )
        for c in CLASSES4
    ]
    rows += [
        MIDRULE,
        Row(
            ["Macro-F1"],
            [multicol(3, meansd(t["macro_f1_mean"], t["macro_f1_std"])), multicol(3, meansd(*prev["macro_f1_full"]))],
        ),
    ]
    ff = prev["false_flags_per_exam"]
    lo, hi = wilson_ci(round(ff["share_zero"] * ff["n_exams"]), ff["n_exams"])
    header = [
        rf"& \multicolumn{{3}}{{c}}{{Capped test ($n={prev['n_capped']:,}$)}} & "
        rf"\multicolumn{{3}}{{c}}{{Deployment prevalence ($n={prev['n_full']:,}$)}} \\".replace(",", "{,}"),
        r"Class & P & R & F1 & P & R & F1 \\",
    ]
    return Table(
        "tab_prevalence",
        [Tabular("l ccc ccc", header, rows)],
        rf"per exam, the three-seed ensemble flags on average {ff['mean']:.2f} sound teeth as a finding "
        rf"(median {ff['median']:.0f}, 90th percentile {ff['p90']:.0f}), and {100 * ff['share_zero']:.1f}\% "
        rf"of the {ff['n_exams']} exams (Wilson 95\% CI {100 * lo:.1f} to {100 * hi:.1f}) "
        "receive no false flag.",
    )


#: Rows of ``tab_fp_by_tooth``: (label, tooth types summed into the row).
TOOTH_ROWS = [
    ("Third molar", ["3rd molar"]),
    ("Second molar", ["2nd molar"]),
    ("First molar", ["1st molar"]),
    ("Premolars", ["premolar"]),
    ("Canines", ["canine"]),
    ("Incisors", ["incisor"]),
]


def fp_by_tooth(S: Sources) -> Table:
    """``tab_fp_by_tooth``: sound teeth flagged impacted, by tooth type, with Wilson CIs.

    The interval is shown only from 10 flagged teeth on; below that the rate has two decimals.
    """
    counts = S.fp_by_tooth
    rows = []
    for label, tooth_types in TOOTH_ROWS:
        n = sum(counts["sound_total"].get(t, 0) for t in tooth_types)
        k = sum(counts["sound_pred_impacted"].get(t, 0) for t in tooth_types)
        rate = k / n if n else 0.0
        if k >= 10:
            lo, hi = wilson_ci(k, n)
            cell = join(pct(rate), paren(ci_range(lo, hi)))
            cell.tex = cell.tex.replace("( ", "(").replace(" )", ")")
        else:
            cell = pct(rate, 2) if k else text("0.0")
            if not k:
                cell.values = [0.0]
        rows.append(Row([label], [integer(n), integer(k, False), cell]))
    header = [r"Tooth type (sound) & $n$ & Flagged impacted & Rate, \% (95\% CI) \\"]
    return Table("tab_fp_by_tooth", [Tabular("l r r r", header, rows)])


#: Rows of ``tab_mitigation``: (retrained, label, prevalence run) or :data:`MIDRULE`. Run
#: names are ``<arch>__<tag>_s<crop scale>[_prior]``: ``paper`` = published models, ``posneg`` =
#: position-aware negatives (sound training crops drawn by tooth position), ``ctx13`` =
#: trained on crops enlarged 1.3x, ``_s1.3`` = crops enlarged 1.3x at test time, ``_prior`` =
#: tooth-position prior applied.
MITIGATION_ROWS = [
    ("no", "InceptionV3, frozen", "inception_v3__paper_s1.0"),
    ("no", "DeiT-S, frozen", "deit_small_patch16_224__paper_s1.0"),
    MIDRULE,
    ("no", "InceptionV3, + tooth-position prior", "inception_v3__paper_s1.0_prior"),
    ("no", "InceptionV3, + test-time context 1.3", "inception_v3__paper_s1.3"),
    ("no", "InceptionV3, + context + prior", "inception_v3__paper_s1.3_prior"),
    ("no", "DeiT-S, + tooth-position prior", "deit_small_patch16_224__paper_s1.0_prior"),
    ("no", "DeiT-S, + test-time context 1.3", "deit_small_patch16_224__paper_s1.3"),
    ("no", "DeiT-S, + context + prior", "deit_small_patch16_224__paper_s1.3_prior"),
    MIDRULE,
    ("yes", "InceptionV3, position-aware negatives", "inception_v3__posneg_s1.0"),
    ("yes", r"\quad + prior", "inception_v3__posneg_s1.0_prior"),
    ("yes", "DeiT-S, position-aware negatives", "deit_small_patch16_224__posneg_s1.0"),
    ("yes", r"\quad + prior", "deit_small_patch16_224__posneg_s1.0_prior"),
    ("yes", "InceptionV3, context in training (control)", "inception_v3__ctx13_s1.3"),
    ("yes", "DeiT-S, context in training (control)", "deit_small_patch16_224__ctx13_s1.3"),
    ("yes", "DeiT-S, negatives + context in training", "deit_small_patch16_224__posneg_ctx13_s1.3"),
    ("yes", r"\quad + prior", "deit_small_patch16_224__posneg_ctx13_s1.3_prior"),
]
#: Configuration highlighted in bold in the mitigation and operating-point tables.
BEST_RUN = "deit_small_patch16_224__posneg_ctx13_s1.3_prior"


def mitigation(S: Sources) -> Table:
    """``tab_mitigation``: impacted precision and false flags at deployment prevalence."""
    rows = []
    for item in MITIGATION_ROWS:
        if item == MIDRULE:
            rows.append(MIDRULE)
            continue
        retrained, label, run = item
        d = S.prevalence(run)
        impacted, ff = d["per_class_full"]["Impacted"], d["false_flags_per_exam"]
        best = run == BEST_RUN
        b = bold if best else (lambda c: c)
        share = pct(ff["share_zero"])
        rows.append(
            Row(
                [retrained, label],
                [
                    b(meansd(*impacted["precision"])),
                    pct(impacted["recall"][0]),
                    b(pct(impacted["f1"][0])),
                    b(meansd(*d["macro_f1_full"])),
                    b(num(ff["mean"], 2)),
                    join(b(share), text(r"\%"), sep=""),
                ],
            )
        )
    header = [r"Retrained & Configuration & Imp.\ P & Imp.\ R & Imp.\ F1 & Macro-F1 & FF/exam & Exams w/o FF \\"]
    return Table(
        "tab_mitigation",
        [Tabular("l l c c c c c c", header, rows, r"\textwidth")],
        "impacted precision, recall and F1 and macro-F1 are means over 3 seeds (with the sd for precision and "
        "macro-F1); FF counts false flags of the three-seed ensemble over the test patients with a sound tooth.",
    )


#: Rows of ``tab_operating``: (label, prevalence run, thresholds; ``"none"`` = no threshold).
OPERATING_ROWS = [
    ("InceptionV3, reference", "inception_v3__paper_s1.0", ["none", 0.70, 0.80, 0.90, 0.95]),
    ("DeiT-S, pos.-aware negatives", "deit_small_patch16_224__posneg_s1.0", ["none", 0.80, 0.90]),
    (r"\quad + prior", "deit_small_patch16_224__posneg_s1.0_prior", [0.80, 0.90]),
    (r"\quad + enlarged crops and prior", BEST_RUN, ["none", 0.80, 0.90]),
]


def operating(S: Sources) -> Table:
    """``tab_operating``: false flags and recall against the confidence threshold."""
    rows = []
    for label, run, taus in OPERATING_ROWS:
        by_tau = {r["tau"]: r for r in S.operating(run)}
        for i, tau in enumerate(taus):
            r = by_tau[0.0 if tau == "none" else tau]
            focus = run == BEST_RUN and tau == 0.80
            b = bold if focus else (lambda c: c)
            first = label if i == 0 else (r"\tabfocus" if focus else "")
            rows.append(
                Row(
                    [first],
                    [
                        text("none") if tau == "none" else b(num(tau, 2)),
                        b(num(r["false_flags_per_exam"], 2)),
                        b(pct(r["share_exams_zero"])),
                        b(pct(r["recall"]["Impacted"])),
                        b(pct(r["recall"]["Endodontics"])),
                    ],
                )
            )
    header = [r"Model & $\tau$ & FF/exam & Exams w/o FF & R\textsubscript{imp} & R\textsubscript{endo} \\"]
    return Table(
        "tab_operating",
        [Tabular("l c c c c c", header, rows)],
        r"a finding is raised only when the three-seed ensemble confidence reaches $\tau$.",
    )


#: Rows of the synthetic-noise part of ``tab_robustness``: (condition in the results, label).
JITTER = [
    ("orig", "Original boxes"),
    ("jitter_0.05", r"Jitter $\delta=0.05$"),
    ("jitter_0.1", r"Jitter $\delta=0.10$"),
    ("jitter_0.2", r"Jitter $\delta=0.20$"),
    ("jitter_0.3", r"Jitter $\delta=0.30$"),
    ("scale_0.7", r"Scale $s=0.70$ (truncate)"),
    ("scale_0.85", r"Scale $s=0.85$"),
    ("scale_1.15", r"Scale $s=1.15$ (context)"),
    ("scale_1.3", r"Scale $s=1.30$"),
    ("scale_1.5", r"Scale $s=1.50$"),
]


def robustness(S: Sources) -> Table:
    """``tab_robustness``: synthetic crop noise and the real upstream modules."""
    rows1 = [
        Row(
            [label],
            [
                meansd(float(S.jitter[k]["macro_f1_mean"]), float(S.jitter[k]["macro_f1_std"])),
                *[pct(float(S.jitter[k][f"f1_{c}"])) for c in CLASSES4],
            ],
        )
        for k, label in JITTER
    ]
    n_test = {c: S.split_counts[(c, "lyria", "test")] for c in CLASSES4}
    det = S.detection_counts
    e2e = S.end2end
    inc = e2e["inception_v3"]
    cis = {c: wilson_ci(*det[c]) for c in CLASSES4}
    rows2 = [
        Row([r"Annotated teeth ($n$)"], [integer(n_test[c], False) for c in CLASSES4]),
        Row(["Detection recall"], [pct(det[c][0] / det[c][1]) for c in CLASSES4]),
        Row([r"\quad 95\% CI"], [ci_range(*cis[c]) for c in CLASSES4]),
        Row(["F1, annotated boxes"], [pct(inc["gt_boxes"]["per_class_f1"][c]) for c in CLASSES4]),
        Row(["F1, detected boxes"], [pct(inc["detected_boxes"]["per_class_f1"][c]) for c in CLASSES4]),
        Row(["End-to-end recall"], [pct(inc["end_to_end_recall"][c]) for c in CLASSES4]),
    ]
    h1 = [r"Synthetic noise (2{,}028 InReDD crops) & Macro-F1 & Endo. treated & Sound & Impacted & Implant \\"]
    h2 = [
        r"Real upstream modules (586 test radiographs, IoU $\geq 0.5$) & Endo. treated & "
        r"Sound & Impacted & Implant \\"
    ]
    return Table(
        "tab_robustness",
        [Tabular("l c c c c c", h1, rows1), Tabular("l c c c c", h2, rows2)],
        "macro-F1 is the mean and sd over 3 seeds; detection recall carries a Wilson 95% interval.",
    )


def dentex_impacted(S: Sources) -> Table:
    """``tab_dentex_impacted``: the impacted class against the DENTEX annotations."""
    imp = S.dentex["impacted"]
    gt, det = imp["gt"], imp["detected"]
    clf_hit = round(imp["recall_classifier_only_on_detected"] * det)
    e2e_hit = round(imp["recall_end_to_end"] * gt)

    def frac(k, n, rate):
        # "k/n" count cell and the rate with its Wilson interval
        lo, hi = wilson_ci(k, n)
        return [
            join(integer(k, False), text("/"), integer(n, False)),
            join(pct(rate), paren(ci_range(lo, hi))),
        ]

    rows = [
        Row(["Annotated impacted teeth detected"], frac(det, gt, det / gt)),
        Row(["Detected ones predicted impacted"], frac(clf_hit, det, imp["recall_classifier_only_on_detected"])),
        Row(["End-to-end recall"], frac(e2e_hit, gt, imp["recall_end_to_end"])),
        Row(["Impacted predictions, total"], [integer(imp["pred_total"]), text("")]),
    ]
    lo, hi = wilson_ci(imp["pred_matching_dentex_impacted"], imp["pred_total"])
    rows += [
        Row(
            [r"\quad on an annotated impacted tooth"],
            [
                integer(imp["pred_matching_dentex_impacted"]),
                join(pct(imp["precision_lower_bound"]), paren(ci_range(lo, hi))),
            ],
        ),
        Row(
            [r"\quad on a tooth with another DENTEX label"],
            [integer(imp["pred_matching_other_dentex_diagnosis"]), text("")],
        ),
        Row([r"\quad on a tooth without annotation"], [integer(imp["pred_unmatched_no_dentex_annotation"]), text("")]),
    ]
    return Table(
        "tab_dentex_impacted",
        [Tabular("l r c", [r" & Count & \% (95\% CI) \\"], rows)],
        f"{S.dentex['n_pans_with_gt']} radiographs of the disease subset, {gt} annotated impacted teeth; "
        "Wilson 95% intervals; the precision row is a lower bound.",
    )


#: Confidence bins of ``tab_audit``: (bin label in the audit results, display label).
AUDIT_BINS = [
    ("0.50-0.70", "0.50 to 0.70"),
    ("0.70-0.90", "0.70 to 0.90"),
    ("0.90-0.99", "0.90 to 0.99"),
    ("0.99-1.00", r"$\geq 0.99$"),
]


def audit(S: Sources) -> Table:
    """``tab_audit``: expert-confirmed share of the audited model's predictions by confidence."""
    rows = []
    for key, label in AUDIT_BINS:
        cells = []
        for ds in ("tufts", "dentex"):
            b = S.audit_bins[ds][key]
            share = pct(b["precision"], suffix=r"\%")
            if key == "0.99-1.00":
                share = bold(share)
            cells += [
                num(b["est_predictions"], 0, thousands=True),
                join(share, paren(ci_range(*b["ci95"]))),
            ]
        rows.append(Row([label], cells))
    header = [
        r"& \multicolumn{2}{c}{Tufts} & \multicolumn{2}{c}{DENTEX} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Confidence & Predictions & Confirmed & Predictions & Confirmed \\",
    ]
    return Table(
        "tab_audit",
        [Tabular("l rr rr", header, rows)],
        "population estimates from the stratified review, with 95% intervals from a bootstrap over radiographs.",
    )


def restored(S: Sources) -> Table:
    """``tab_restored``: four-class against five-class DeiT-S on the four-class test crops."""
    a = S.restored["A_in_domain_four_class"]
    rows = [
        Row(
            [CLASS_LABEL[c]],
            [pct(a[m]["per_class"][c][k]) for k in ("precision", "recall", "f1") for m in ("ctrl4", "five")],
        )
        for c in CLASSES4
    ]
    rows += [
        MIDRULE,
        Row(
            ["Macro-F1 over the four"],
            [
                multicol(
                    6,
                    join(
                        pct(a["ctrl4"]["macro_f1"], 2),
                        text(r"(four-class model) \quad against \quad"),
                        pct(a["five"]["macro_f1"], 2),
                        text("(five-class model)"),
                    ),
                )
            ],
        ),
    ]
    header = [
        r"& \multicolumn{2}{c}{Precision} & \multicolumn{2}{c}{Recall} & \multicolumn{2}{c}{F1} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Class & Four-class & Five-class & Four-class & Five-class & Four-class & Five-class \\",
    ]
    return Table(
        "tab_restored",
        [Tabular("l cc cc cc", header, rows)],
        f"both models are DeiT-S under the same protocol and partition, scored on the same {a['n']:,} "
        "test crops; a restored prediction counts as an error.",
    )


def lodo(S: Sources) -> Table:
    """``tab_lodo``: five-class DeiT-S with one dataset held out (leave one dataset out, LODO)."""
    cols = [
        S.lodo("lodo_lyria", "dentex"),
        S.lodo("lodo_dentex", "dentex"),
        S.lodo("lodo_lyria", "tufts"),
        S.lodo("lodo_tufts", "tufts"),
    ]
    rows = [Row([CLASS_LABEL[c]], [pct(col["f1"][c]) for col in cols]) for c in CLASSES5]
    rows += [MIDRULE, Row(["Macro-F1"], [meansd(*col["macro"]) for col in cols])]
    header = [
        rf"& \multicolumn{{2}}{{c}}{{Test: DENTEX ({cols[0]['n']:,} crops)}} & "
        rf"\multicolumn{{2}}{{c}}{{Test: Tufts ({cols[2]['n']:,} crops)}} \\".replace(",", "{,}"),
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Class & InReDD only & \(+\) Tufts & InReDD only & \(+\) DENTEX \\",
    ]
    return Table(
        "tab_lodo",
        [Tabular("l cc cc", header, rows)],
        "per-class F1 as means over three seeds, macro-F1 as mean and population sd.",
    )


#: Table name -> generator.
GENERATORS = {
    "tab_class_distribution": class_distribution,
    "tab_cv_ablation": cv_ablation,
    "tab_modern_backbones": modern_backbones,
    "tab_inference_cost": inference_cost,
    "tab_test_per_class": test_per_class,
    "tab_prevalence": prevalence,
    "tab_fp_by_tooth": fp_by_tooth,
    "tab_mitigation": mitigation,
    "tab_operating": operating,
    "tab_robustness": robustness,
    "tab_dentex_impacted": dentex_impacted,
    "tab_audit": audit,
    "tab_restored": restored,
    "tab_lodo": lodo,
}


def build_tables(
    paths: Paths, out_dir: Path | None = None, names: list[str] | None = None, sources: Sources | None = None
) -> dict[str, Table]:
    """Generate the published tables and write them to ``out_dir`` when given.

    Parameters
    ----------
    paths : Paths
        Path configuration.
    out_dir : Path, optional
        Folder for the ``.tex`` files; nothing is written when None.
    names : list of str, optional
        Tables to generate (keys of :data:`GENERATORS`); all by default.
    sources : Sources, optional
        Shared inputs, to reuse computations across calls; built from ``paths`` by default.

    Returns
    -------
    dict
        Generated :class:`~toothfindings.reporting.latex.Table` objects keyed by name.
    """
    S = sources or Sources(paths)
    tables = {n: GENERATORS[n](S) for n in (names or GENERATORS)}
    if out_dir is not None:
        for t in tables.values():
            t.write(out_dir)
    return tables
