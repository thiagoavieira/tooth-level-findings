"""Command-line interface: ``toothfindings <group> <command> [options]``.

Every path comes from the configuration (``configs/paths.toml``, ``TOOTHFINDINGS_*`` variables
or ``--set key=value``). Commands that need the restricted images or trained weights say so in
their help. ``toothfindings <group> -h`` lists the commands of a group; ``docs/REPRODUCE.md``
maps every table and figure to its command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Paths, load_paths
from .constants import AUGS, CLASSES4, CLASSES5, ENHANCEMENTS, config_id

# Experiment presets: one entry per published experiment. Each gives the experiment folder under
# ``experiments/`` (``exp``), the partition folder under ``results/splits/`` (``splits``), the
# enhanced-crop root under ``data/crops/`` (``enhanced``), the label space (``classes``), the
# filters of the run matrix (``archs``, ``enhancements``, ``augs``; ``None`` = all), the matrix
# mode (``only``: cross-validation, test or both) and, when published that way, the number of
# data-loader workers.
#
# Variant names: ``posneg`` = position-aware negatives (sound training crops drawn by tooth
# position: third molars, second molars and other teeth in equal parts), ``ctx13`` = crops
# enlarged 1.3x around the tooth to add context, ``lodo_<src>`` = leave-one-dataset-out
# (``lodo_dentex``/``lodo_tufts`` hold that dataset out for testing; ``lodo_lyria`` trains on the
# primary source only and tests on both external datasets), ``clean`` = the contamination-free
# crop pool.
PRESETS = {
    # 18 classic configurations (the 3 recent backbones are trained with the ``paper_new`` preset)
    "paper": dict(
        exp=".",
        splits="splits",
        enhanced="crops_enhanced",
        classes=CLASSES4,
        archs=None,
        enhancements=None,
        augs=None,
        only="all",
    ),
    "paper_new": dict(
        exp=".",
        splits="splits",
        enhanced="crops_enhanced",
        classes=CLASSES4,
        archs=["convnext_tiny", "tf_efficientnetv2_s", "deit_small_patch16_224"],
        enhancements=["none"],
        augs=["noaug"],
        only="all",
    ),
    # mitigation variants (InceptionV3 and DeiT-S, grayscale, no augmentation)
    **{
        v: dict(
            exp=f"variants/{v}",
            splits=f"variants/{v}/splits",
            enhanced=f"variants/{v}/enhanced",
            classes=CLASSES4,
            archs=["inception_v3", "deit_small_patch16_224"],
            enhancements=["none"],
            augs=["noaug"],
            only="all",
        )
        for v in ("posneg", "ctx13", "posneg_ctx13")
    },
    # restored class: four- and five-class DeiT-S on the contamination-free pool
    "clean_ctrl4": dict(
        exp="clean_ctrl4",
        splits="splits_clean4",
        enhanced="crops_enhanced",
        classes=CLASSES4,
        archs=["deit_small_patch16_224"],
        enhancements=["none"],
        augs=["noaug"],
        only="all",
        workers=0,
    ),
    "clean_5class": dict(
        exp="clean_5class",
        splits="splits_clean5",
        enhanced="crops_enhanced",
        classes=CLASSES5,
        archs=["deit_small_patch16_224"],
        enhancements=["none"],
        augs=["noaug"],
        only="all",
        workers=0,
    ),
    **{
        f"lodo_{s}": dict(
            exp=f"lodo_{s}",
            splits=f"splits_lodo_{s}",
            enhanced="crops_enhanced",
            classes=CLASSES5,
            archs=["deit_small_patch16_224"],
            enhancements=["none"],
            augs=["noaug"],
            only="test",
            workers=0,
        )
        for s in ("lyria", "dentex", "tufts")
    },
    # five classes with the mitigations
    "five_mitigated": dict(
        exp="variants/five_mitigated",
        splits="splits5_mitigated",
        enhanced="variants/five_mitigated/enhanced",
        classes=CLASSES5,
        archs=["deit_small_patch16_224"],
        enhancements=["none"],
        augs=["noaug"],
        only="all",
        workers=0,
    ),
    "five_mitigated_lodo_lyria": dict(
        exp="variants/five_mitigated_lodo_lyria",
        splits="splits5_mitigated_lodo_lyria",
        enhanced="variants/five_mitigated/enhanced",
        classes=CLASSES5,
        archs=["deit_small_patch16_224"],
        enhancements=["none"],
        augs=["noaug"],
        only="test",
        workers=0,
    ),
}

#: Prevalence evaluations of the mitigation study, as tuples ``(experiment sub-folder or None for
#: the published models, tag, archs, crop scale, apply the tooth-position prior)``.
PREVALENCE_RUNS = [
    (None, "paper", ["inception_v3", "deit_small_patch16_224", "convnext_tiny"], 1.0, False),
    (None, "paper", ["inception_v3", "deit_small_patch16_224"], 1.3, False),
    (None, "paper", ["inception_v3", "deit_small_patch16_224"], 1.0, True),
    (None, "paper", ["inception_v3", "deit_small_patch16_224"], 1.3, True),
    *[
        (f"variants/{v}", v, ["inception_v3", "deit_small_patch16_224"], 1.3 if "ctx13" in v else 1.0, pr)
        for v in ("posneg", "ctx13", "posneg_ctx13")
        for pr in (False, True)
    ],
]


def _data_config(paths: Paths, preset: dict, pretrained: bool = True):
    from .models.train import DataConfig

    return DataConfig(
        splits=paths.split_dir(preset["splits"]),
        enhanced_root=paths.crops / preset["enhanced"],
        classes=list(preset["classes"]),
        pretrained=pretrained,
    )


# --------------------------------------------------------------------------- handlers
# Every handler takes the resolved :class:`~toothfindings.config.Paths` and the parsed
# ``argparse.Namespace`` (``a.command`` names the command of the group).


def cmd_data(paths: Paths, a) -> None:
    """Run a ``data`` command: crops, manifests, partitions and variants (needs the restricted images)."""
    from .data import crops, manifests, splits, variants
    from .enhance.clahe import apply_enhancement

    if a.command == "crops":
        crops.generate_crops(paths)
    elif a.command == "extra-crops":
        crops.generate_extra_crops(paths)
    elif a.command == "enhance":
        apply_enhancement(paths, variants=a.variants, workers=a.workers)
    elif a.command == "manifest5":
        manifests.build_5class_manifest(paths)
    elif a.command == "manifest-clean":
        manifests.build_manifest_clean(paths)
    elif a.command == "split":
        splits.make_split(
            paths.splits / "crops_source" / "manifest.csv", Path(a.out) if a.out else paths.split_dir("splits")
        )
    elif a.command == "splits-clean":
        clean_manifest = paths.splits / "crops_source" / "manifest_clean.csv"
        published_split = paths.split_dir("splits") / "split_manifest.csv"
        for name, kw in CLEAN_SPLITS.items():
            if a.only and name not in a.only:
                continue
            splits.make_splits_clean(clean_manifest, paths.split_dir(name), published_split, **kw)
    elif a.command == "variants":
        for v in a.names:
            variants.build_variant(paths, v)
    elif a.command == "five-mitigated":
        variants.build_five_mitigated(paths)


#: Arguments of the clean partitions. In domain (``splits_clean4``/``splits_clean5``) the implant
#: class is filled with the reviewed external crops; in every leave-one-dataset-out partition
#: (``splits_lodo_*``) the implant class is capped at 93 crops.
CLEAN_SPLITS = {
    "splits_clean4": dict(
        classes="Endodontics,Healthy,Impacted,Implant", fill_class="Implant", fill_sources="tufts,dentex"
    ),
    "splits_clean5": dict(fill_class="Implant", fill_sources="tufts,dentex"),
    "splits_lodo_lyria": dict(train_sources="lyria", test_sources="tufts,dentex", class_caps="Implant=93"),
    "splits_lodo_dentex": dict(train_sources="lyria,tufts", test_sources="dentex", class_caps="Implant=93"),
    "splits_lodo_tufts": dict(train_sources="lyria,dentex", test_sources="tufts", class_caps="Implant=93"),
}


def cmd_train(paths: Paths, a) -> None:
    """Run a ``train`` command: one run or the resumable run matrix (needs the enhanced crops)."""
    from .models.run_matrix import run_matrix
    from .models.train import RunConfig, run

    preset = dict(PRESETS[a.preset])
    data = _data_config(paths, preset, pretrained=not a.no_pretrained)
    workers = preset.get("workers", RunConfig.workers) if a.workers is None else a.workers
    if a.command == "run":
        exp = paths.experiments / preset["exp"]
        fold = f"__fold{a.fold}" if a.mode == "cv" else ""
        run_id = f"{config_id(a.enhancement, a.aug, a.arch)}{fold}__seed{a.seed}"
        run(
            RunConfig(
                mode=a.mode,
                enhancement=a.enhancement,
                aug=a.aug,
                arch=a.arch,
                fold=a.fold,
                seed=a.seed,
                run_id=run_id,
                run_dir=str(exp / ("runs" if a.mode == "cv" else "test") / run_id),
                batch=a.batch,
                lr=a.lr,
                epochs=a.epochs,
                patience=a.patience,
                workers=workers,
            ),
            data,
        )
    elif a.command == "matrix":
        run_matrix(
            paths.experiments / preset["exp"],
            data,
            only=a.only or preset["only"],
            epochs=a.epochs,
            patience=a.patience,
            lr=a.lr,
            workers=workers,
            max_runs=a.max_runs,
            enhancements=a.enhancements or preset["enhancements"],
            augs=a.augs or preset["augs"],
            archs=a.archs or preset["archs"],
        )


def cmd_eval(paths: Paths, a) -> None:
    """Run an ``eval`` command; all but ``operating-points`` and ``five-mitigated-analyze`` need images and weights."""
    experiments = paths.experiments
    if a.command == "prevalence":
        from .evaluation.prevalence import evaluate_prevalence

        runs = PREVALENCE_RUNS if a.all else [(a.exp, a.tag, a.archs, a.scale, a.prior)]
        for exp, tag, archs, scale, prior in runs:
            evaluate_prevalence(paths, archs, experiments / exp if exp else None, scale, prior, tag)
    elif a.command == "operating-points":
        from .evaluation.operating_points import operating_points

        operating_points(experiments / "prevalence", a.runs, experiments / "prevalence" / "operating_point.json")
    elif a.command == "fp-by-tooth":
        from .evaluation.fp_by_tooth import fp_by_tooth_from_annotations

        fp_by_tooth_from_annotations(
            experiments / "prevalence" / a.run / "predictions.csv",
            paths.lyria_json,
            experiments / "prevalence" / "fp_by_tooth_type.json",
        )
    elif a.command == "jitter":
        from .evaluation.robustness import crop_jitter

        crop_jitter(paths, a.archs)
    elif a.command == "end2end":
        from .evaluation.robustness import end_to_end

        end_to_end(paths, a.archs)
    elif a.command == "gradcam":
        from .evaluation.gradcam import gradcam_overlays

        gradcam_overlays(paths, a.run_config, a.seed, a.n)
    elif a.command == "gradcam-masks":
        from .evaluation.gradcam import gradcam_masks

        for arch in a.archs:
            gradcam_masks(paths, arch)
    elif a.command == "restored":
        from .evaluation.restored import evaluate_restored

        evaluate_restored(paths)
    elif a.command == "five-mitigated-infer":
        from .evaluation.five_mitigated import run_inference

        run_inference(paths, a.models, a.sets, a.force)
    elif a.command == "five-mitigated-analyze":
        from .evaluation.five_mitigated import analyze

        analyze(paths)
    elif a.command == "cost":
        from .models.inference_cost import measure

        measure(experiments / "cost", a.iters)
    elif a.command == "latency":
        from .models.inference_cost import measure_latency_all

        measure_latency_all(paths.reporting / "latency_all.csv")


def cmd_stats(paths: Paths, a) -> None:
    """Run a ``stats`` command from the released per-run metrics (no data needed)."""
    experiments = paths.experiments
    if a.command in ("summaries", "all"):
        from .stats.summaries import run_summaries

        run_summaries(experiments, experiments / "aggregates")
    if a.command in ("tests", "all"):
        from .stats.tests import run_stats_tests

        print(run_stats_tests(experiments, experiments / "stats")["friedman"])
    if a.command in ("new-backbones", "all"):
        from .stats.tests import new_backbones

        new_backbones(experiments, experiments / "stats")
    if a.command in ("bootstrap", "all"):
        import json

        from .stats.bootstrap import bootstrap_ci, load_seed_predictions

        data, n, _files = load_seed_predictions(experiments / "test", "inception_v3")
        result = dict(bootstrap_ci(data, n, 2000, 0), arch="inception_v3", enhancement="none", aug="noaug")
        (paths.reporting / "bootstrap_test_ci.json").write_text(json.dumps(result, indent=1))


def cmd_external(paths: Paths, a) -> None:
    """Run an ``external`` command (pipeline and rescoring need radiographs, weights and the ONNX models)."""
    if a.command == "pipeline":
        from .external.pipeline import run_pipeline

        run_pipeline(paths, a.dataset, Path(a.images), a.name_prefix)
    elif a.command == "rescore":
        from .external.rescore import rescore

        rescore(paths, paths.experiments / a.exp, a.tag, arch=a.arch, scale=a.scale, batch=a.batch)
    elif a.command == "thresholds":
        from .external.thresholds import select_for_review

        select_for_review(
            paths.experiments / "prevalence" / a.prev_run / "predictions.csv",
            [paths.external / ds / f"predictions_{a.tag}.csv" for ds in ("tufts", "dentex")],
            a.tag,
        )
    elif a.command == "blind-tasks":
        from .external.blind_tasks import build_blind_tasks

        for ds in ("tufts", "dentex"):
            build_blind_tasks(paths, ds, a.tag)
    elif a.command == "dentex-impacted":
        from .external.dentex import run_impacted_check

        run_impacted_check(paths)
    elif a.command == "trace-implants":
        from .external.dentex import trace_implants

        trace_implants(paths)


#: Steps of ``toothfindings audit all``, in execution order.
AUDIT_STEPS = [
    "audit",
    "in-scope",
    "breakdown",
    "cascade",
    "bin-ci",
    "agreement",
    "agreement-extra",
    "implant-view",
    "operating-points",
    "bridge-metrics",
    "fairness-view",
    "impacted-recall",
]


def cmd_audit(paths: Paths, a) -> None:
    """Run an ``audit`` command from ``review_wide.csv`` and the external predictions (no data needed)."""
    from .audit import agreement, cascade, implant, operating, precision
    from .audit.core import AuditData

    if a.command == "review-tables":
        from .audit.review_export import build_review_tables

        build_review_tables(paths)
        return
    if a.command == "label-space":
        from .audit.label_space import label_space_exclusion

        label_space_exclusion(paths)
        return
    step_functions = {
        "audit": precision.audit,
        "in-scope": precision.in_scope,
        "breakdown": precision.breakdown,
        "cascade": cascade.cascade,
        "bin-ci": cascade.bin_ci,
        "agreement": agreement.agreement,
        "agreement-extra": agreement.agreement_extra,
        "implant-view": implant.implant_view,
        "operating-points": operating.operating_points,
        "bridge-metrics": operating.bridge_metrics,
        "fairness-view": operating.fairness_view,
        "impacted-recall": operating.impacted_recall,
    }
    data = AuditData(paths)
    for step in AUDIT_STEPS if a.command == "all" else [a.command]:
        step_functions[step](data)
        print(f"wrote {step}")
    if a.command in ("all", "reader-conventions"):
        import json

        print(json.dumps(precision.reader_conventions(data), indent=1))


def cmd_report(paths: Paths, a) -> None:
    """Run a ``report`` command: tables, figures and the numeric check against the published tables (no data needed)."""
    from .reporting.sources import Sources

    out = Path(a.out) if a.out else paths.build
    sources = Sources(paths)
    if a.command in ("tables", "all", "check"):
        from .reporting.tables import build_tables

        tables = build_tables(paths, out / "tables" if a.command != "check" else None, sources=sources)
        if a.command != "check":
            print(f"wrote {len(tables)} tables to {out / 'tables'}")
    if a.command in ("figures", "all"):
        from .reporting.figures import build_figures

        for f in build_figures(paths, out / "figures", Path(a.gradcam_images) if a.gradcam_images else None):
            print(f"wrote {f}")
    if a.command in ("check", "all"):
        from .reporting.check import check_tables

        mismatches = check_tables(
            tables, Path(a.reference) if a.reference else paths.root / "reference" / "published_tables.json"
        )
        for m in mismatches:
            print(m)
        print(f"{len(mismatches)} mismatches")
        if mismatches:
            raise SystemExit(1)


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of the ``toothfindings`` command."""
    p = argparse.ArgumentParser(
        prog="toothfindings", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", help="TOML file with a [paths] table (default: configs/paths.toml)")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one configured path, e.g. --set data=../pan924 (repeatable)",
    )
    groups = p.add_subparsers(dest="group", required=True)

    g = groups.add_parser("data", help="crops, manifests, partitions and variants (needs the restricted images)")
    s = g.add_subparsers(dest="command", required=True)
    s.add_parser("crops", help="cut the canonical four-class crop pool")
    s.add_parser("extra-crops", help="cut the Restored and DevelopingM3 (developing third molar) crops")
    c = s.add_parser("enhance", help="write the enhanced crops: none, msthgr and msthgr-clahe")
    c.add_argument("--variants", nargs="+", choices=ENHANCEMENTS, help="enhancements to write (default: all)")
    c.add_argument("--workers", type=int, help="parallel worker processes")
    s.add_parser("manifest5", help="five-class manifest: reviewed external crops and the frozen holdout")
    s.add_parser("manifest-clean", help="manifest of the contamination-free crop pool")
    c = s.add_parser(
        "split",
        help="canonical partition (not reproducible, see docs/REPRODUCE.md; the saved one is the source of truth)",
    )
    c.add_argument("--out", required=True, help="output folder (never overwrite results/splits/splits)")
    c = s.add_parser("splits-clean", help="clean partitions splits_clean4/5 and splits_lodo_* (leave one dataset out)")
    c.add_argument("--only", nargs="+", choices=list(CLEAN_SPLITS), help="partitions to build (default: all)")
    c = s.add_parser("variants", help="four-class mitigation variants (enlarged crops, position-aware negatives)")
    c.add_argument(
        "names", nargs="*", default=["ctx13", "posneg", "posneg_ctx13"], help="variants to build (default: all three)"
    )
    s.add_parser("five-mitigated", help="five-class mitigated splits, crops, evaluation sets and priors")
    g.set_defaults(func=cmd_data)

    g = groups.add_parser("train", help="training (needs the enhanced crops; GPU recommended)")
    s = g.add_subparsers(dest="command", required=True)
    for name in ("run", "matrix"):
        c = s.add_parser(name, help="train one run" if name == "run" else "train the resumable run matrix of a preset")
        c.add_argument(
            "--preset",
            default="paper",
            choices=list(PRESETS),
            help="published experiment: partition, crops, classes and matrix filters (default: paper)",
        )
        c.add_argument("--epochs", type=int, default=150, help="maximum number of epochs (default: 150)")
        c.add_argument("--patience", type=int, default=20, help="early-stopping patience in epochs (default: 20)")
        c.add_argument("--lr", type=float, default=1e-4, help="learning rate (default: 1e-4)")
        c.add_argument("--workers", type=int, help="data-loader workers (default: as published for the preset)")
        c.add_argument("--no-pretrained", action="store_true", help="random initialisation (offline smoke tests)")
        if name == "run":
            c.add_argument(
                "--mode",
                choices=["cv", "test"],
                required=True,
                help="cv: one cross-validation fold; test: train on the whole pool and score the test set",
            )
            c.add_argument("--enhancement", choices=ENHANCEMENTS, default="none", help="crop enhancement")
            c.add_argument("--aug", choices=AUGS, default="noaug", help="data augmentation")
            c.add_argument("--arch", default="inception_v3", help="backbone name (default: inception_v3)")
            c.add_argument("--fold", type=int, default=-1, help="validation fold in cv mode (0-4)")
            c.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
            c.add_argument("--batch", type=int, help="batch size (default: the published batch size of the backbone)")
        else:
            c.add_argument(
                "--only", choices=["cv", "test", "all"], help="run only cv or test runs (default: as in the preset)"
            )
            c.add_argument("--max-runs", type=int, default=0, help="stop after this many new runs (0: no limit)")
            c.add_argument(
                "--enhancements", nargs="+", choices=ENHANCEMENTS, help="restrict the matrix to these enhancements"
            )
            c.add_argument("--augs", nargs="+", choices=AUGS, help="restrict the matrix to these augmentation settings")
            c.add_argument("--archs", nargs="+", help="restrict the matrix to these backbones")
    g.set_defaults(func=cmd_train)

    g = groups.add_parser("eval", help="evaluations of trained models")
    s = g.add_subparsers(dest="command", required=True)
    c = s.add_parser("prevalence", help="score the deployment-prevalence set (images, weights)")
    c.add_argument("--all", action="store_true", help="every run of the mitigation study (ignores the other options)")
    c.add_argument("--archs", nargs="+", default=["inception_v3"], help="backbones to evaluate")
    c.add_argument("--exp", help="experiment sub-folder holding the models (default: published models)")
    c.add_argument("--scale", type=float, default=1.0, help="crop scale around the tooth box (1.0 or 1.3)")
    c.add_argument("--prior", action="store_true", help="apply the tooth-position prior")
    c.add_argument("--tag", default="", help="output tag: paper for the published models, else the variant name")
    c = s.add_parser("operating-points", help="operating-point curves from prevalence predictions (no data)")
    c.add_argument("--runs", nargs="+", help="prevalence runs to analyse (default: the published ones)")
    c = s.add_parser("fp-by-tooth", help="false flags by tooth position (needs the annotations)")
    c.add_argument("--run", default="inception_v3", help="prevalence run to analyse (default: inception_v3)")
    c = s.add_parser("jitter", help="robustness to synthetic crop noise (images, weights)")
    c.add_argument("--archs", nargs="+", default=["inception_v3"], help="backbones to evaluate")
    c = s.add_parser("end2end", help="end to end with the real detection modules (images, weights, ONNX)")
    c.add_argument("--archs", nargs="+", default=["inception_v3"], help="backbones to evaluate")
    c = s.add_parser("gradcam", help="Grad-CAM overlays (images, weights)")
    c.add_argument(
        "--config",
        dest="run_config",
        default="none__noaug__inception_v3",
        help="run configuration <enhancement>__<aug>__<arch>",
    )
    c.add_argument("--seed", type=int, default=0, help="seed of the model (default: 0)")
    c.add_argument("--n", type=int, default=4, help="success and failure overlays per class (default: 4)")
    c = s.add_parser("gradcam-masks", help="Grad-CAM against tooth-segmenter masks (images, weights, ONNX)")
    c.add_argument(
        "--archs", nargs="+", default=["inception_v3", "deit_small_patch16_224"], help="backbones to evaluate"
    )
    s.add_parser("restored", help="four- against five-class comparison (images, weights)")
    c = s.add_parser("five-mitigated-infer", help="store per-seed softmax of the five-class study (images, weights)")
    c.add_argument("--models", nargs="+", help="models to score (default: all of the study)")
    c.add_argument("--sets", nargs="+", help="evaluation sets to score (default: all)")
    c.add_argument("--force", action="store_true", help="recompute probabilities that are already stored")
    s.add_parser("five-mitigated-analyze", help="every number of the five-class study (no data)")
    c = s.add_parser("cost", help="parameters, size and latency per backbone")
    c.add_argument("--iters", type=int, default=50, help="timed GPU iterations per backbone (default: 50)")
    s.add_parser("latency", help="unified latency session (CPU/GPU columns of the cost table)")
    g.set_defaults(func=cmd_eval)

    g = groups.add_parser("stats", help="statistics from the per-run metrics (no data)")
    g.add_argument(
        "command",
        choices=["summaries", "tests", "new-backbones", "bootstrap", "all"],
        help="summaries: per-configuration means; tests: Friedman, Nemenyi and paired tests; "
        "new-backbones: recent backbones against InceptionV3; bootstrap: test macro-F1 interval",
    )
    g.set_defaults(func=cmd_stats)

    g = groups.add_parser("external", help="external evaluation on Tufts and DENTEX")
    s = g.add_subparsers(dest="command", required=True)
    c = s.add_parser("pipeline", help="zero-shot pipeline (radiographs, weights, ONNX)")
    c.add_argument("--dataset", choices=["tufts", "dentex"], required=True, help="external dataset")
    c.add_argument("--images", required=True, help="folder of radiographs inside the configured dataset folder")
    c.add_argument("--name-prefix", default="", help="prefix of the radiograph ids (e.g. the DENTEX subset)")
    c = s.add_parser("rescore", help="re-score the external crops with another classifier (radiographs, weights)")
    c.add_argument("--exp", default="variants/posneg", help="experiment sub-folder holding the models")
    c.add_argument("--arch", default="deit_small_patch16_224", help="backbone of the models")
    c.add_argument("--scale", type=float, default=1.0, help="crop scale around the tooth box")
    c.add_argument("--tag", default="deit_posneg", help="tag of the output predictions_<tag>.csv")
    c.add_argument("--batch", type=int, default=32, help="inference batch size")
    c = s.add_parser("thresholds", help="study thresholds and review selection (no data)")
    c.add_argument(
        "--prev-run",
        default="deit_small_patch16_224__posneg_s1.0",
        help="prevalence run the thresholds are calibrated on",
    )
    c.add_argument("--tag", default="deit_posneg", help="tag of the external predictions to select from")
    c = s.add_parser("blind-tasks", help="blind Label Studio tasks for the expert review")
    c.add_argument("--tag", default="deit_posneg", help="tag of the external predictions")
    s.add_parser("dentex-impacted", help="DENTEX impacted-class check (DENTEX annotations)")
    s.add_parser("trace-implants", help="provenance of the 142 DENTEX implant crops (DENTEX images)")
    g.set_defaults(func=cmd_external)

    g = groups.add_parser("audit", help="expert-review analysis (no data, except review-tables and label-space)")
    g.add_argument(
        "command",
        choices=["all", *AUDIT_STEPS, "reader-conventions", "review-tables", "label-space"],
        help="analysis step, or all for every step in order",
    )
    g.set_defaults(func=cmd_audit)

    g = groups.add_parser("report", help="tables and figures from results/ (no data)")
    g.add_argument(
        "command",
        choices=["tables", "figures", "check", "all"],
        help="check: compare the recomputed tables with the published ones",
    )
    g.add_argument("--out", help="output folder (default: build/)")
    g.add_argument(
        "--reference", help="JSON file of the published table numbers (default: reference/published_tables.json)"
    )
    g.add_argument("--gradcam-images", help="folder of the Grad-CAM crops and overlays (patient images)")
    g.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the ``toothfindings`` console script and return its exit code."""
    parser = build_parser()
    a = parser.parse_args(argv)
    if bad := [kv for kv in a.set if "=" not in kv]:
        parser.error(f"--set expects KEY=VALUE, got {bad[0]!r}")
    overrides = dict(kv.split("=", 1) for kv in a.set)
    paths = load_paths(a.config, **overrides)
    a.func(paths, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
