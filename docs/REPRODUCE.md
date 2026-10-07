# Reproducing the results

There are three levels. The first two need only this repository and run on a CPU in a few
minutes. The third needs the restricted data (`docs/DATA.md`) and a GPU.

| Level | Needs | Gives |
|---|---|---|
| 1. Tables and figures | `results/` | every table and figure, checked against the published numbers |
| 2. Intermediate results | `results/` | the statistics and audit files under `results/` |
| 3. Full pipeline | images, annotations, GPU | crops, partitions, trained models, predictions |

```bash
pip install -e ".[dev]"              # levels 1 and 2
pip install -e ".[train,external]"   # level 3
```

Paths come from `configs/paths.toml`, `TOOTHFINDINGS_<KEY>` environment variables or
`toothfindings --set key=value`. By default `results/` is read from the repository, restricted
data from `data/`, and generated tables and figures go to `build/`.

## 1. Tables and figures

```bash
toothfindings report check     # compare the 14 tables with the published numbers; exits 1 on a mismatch
toothfindings report tables    # write build/tables/tab_*.tex
toothfindings report figures   # write build/figures/*.pdf
pytest                         # the same check, plus the numbers quoted in the published text
```

`reference/published_tables.json` holds the published numbers of each table, row by row. A
published number must equal the recomputed value rounded to the published precision.

| Table or figure | Output | Generator (`toothfindings.reporting`) | Reads from `results/` |
|---|---|---|---|
| Crops per class and source | `tab_class_distribution` | `tables.class_distribution` | `splits/splits/split_manifest.csv` |
| Cross-validation grid: 3 enhancements x 2 augmentation settings x 3 backbones | `tab_cv_ablation` | `tables.cv_ablation` | `experiments/aggregates/*_summary.csv`, `experiments/stats/friedman.json` |
| Recent backbones against InceptionV3 | `tab_modern_backbones` | `tables.modern_backbones` | the two summaries, `experiments/stats/new_backbones.json`, `experiments/cost/inference_cost.csv` |
| Inference cost per backbone | `tab_inference_cost` | `tables.inference_cost` | `reporting/latency_all.csv`, `experiments/cost/inference_cost.csv`, `test_summary.csv` |
| Held-out test results per class | `tab_test_per_class` | `tables.test_per_class` | `experiments/prevalence/inception_v3__paper_s1.0/`, test predictions of InceptionV3 |
| Capped test set against deployment prevalence | `tab_prevalence` | `tables.prevalence` | `experiments/prevalence/inception_v3__paper_s1.0/` |
| False flags by tooth position | `tab_fp_by_tooth` | `tables.fp_by_tooth` | `experiments/prevalence/inception_v3*/predictions.csv` |
| Mitigations of the false flags | `tab_mitigation` | `tables.mitigation` | the 16 runs under `experiments/prevalence/` |
| Operating points (confidence thresholds) | `tab_operating` | `tables.operating` | `experiments/prevalence/*/predictions.csv` |
| Robustness to crop noise and to the real detector | `tab_robustness` | `tables.robustness` | `experiments/robustness/` |
| Impacted teeth on DENTEX | `tab_dentex_impacted` | `tables.dentex_impacted` | `external/dentex/dentex_impacted_eval.json` |
| Expert audit of the external predictions | `tab_audit` | `tables.audit` | `review_analysis/review_wide.csv`, `external/*/predictions_deit_posneg.csv` |
| Four- against five-class model (restored teeth) | `tab_restored` | `tables.restored` | `experiments/clean_e1_report.json` |
| Leave-one-dataset-out | `tab_lodo` | `tables.lodo` | `experiments/lodo_*/test/*/predictions.csv` |
| Critical-difference diagram | `fig_cd_diagram` | `figures.cd_diagram` | `experiments/runs/*/metrics.json` |
| Precision, recall and false flags of the mitigations | `fig_mitigation` | `figures.mitigation` | the 16 mitigation runs |
| Test confusion matrix | `fig_confusion_test` | `figures.confusion` | test predictions of InceptionV3 |
| Grad-CAM examples | `fig_gradcam` | `figures.gradcam_grid` | patient crops, so only with `report figures --gradcam-images DIR` |
| Numbers quoted in the text | | `text_numbers`, `costs` | listed in `tests/expected/published_text_numbers.json` |

Example radiograph crops and the enhancement examples are patient images and are not generated.

## 2. Intermediate results

These commands rebuild files under `results/` from other files under `results/`. They overwrite
the released files with the same content, so run them on a copy if you want to compare. The tests
in `tests/test_regression_results.py` recompute the same files in memory and compare them with the
released ones.

| Command | Writes |
|---|---|
| `toothfindings stats summaries` | `experiments/aggregates/{cv,test}_summary.csv` |
| `toothfindings stats tests` | `experiments/stats/{friedman.json,nemenyi.csv,paired_tests.csv}` |
| `toothfindings stats new-backbones` | `experiments/stats/new_backbones.{json,csv}` |
| `toothfindings stats bootstrap` | `reporting/bootstrap_test_ci.json` |
| `toothfindings eval operating-points` | `experiments/prevalence/operating_point.json` |
| `toothfindings eval five-mitigated-analyze` | `experiments/variants/five_mitigated/results/results.json` |
| `toothfindings external thresholds` | `external/*/thresholds_deit_posneg.json` |
| `toothfindings audit all` | every JSON under `review_analysis/` |

`toothfindings stats all` runs the first four.

## 3. Full pipeline

Commands can be resumed: a run whose `metrics.json` exists is skipped. Since `results/experiments/`
already holds every released run, write new runs elsewhere and run the commands from the
repository root:

```bash
export TOOTHFINDINGS_EXPERIMENTS=data/experiments
```

1. **Crops**
   ```bash
   toothfindings data crops          # four-class crop pool (3,109 crops)
   toothfindings data extra-crops    # Restored and DevelopingM3
   toothfindings data enhance        # none, msthgr, msthgr-clahe
   ```
   Use the released partitions under `results/splits/`. `toothfindings data split --out DIR` shows
   how the main partition was made but cannot rebuild it exactly (see *Notes* below).
2. **Backbone grid and recent backbones**
   ```bash
   toothfindings train matrix --preset paper       # 18 configurations x (15 cv + 3 test) runs
   toothfindings train matrix --preset paper_new   # ConvNeXt-T, EfficientNetV2-S, DeiT-S
   toothfindings eval cost && toothfindings eval latency
   toothfindings stats all
   ```
3. **Deployment prevalence and mitigations**
   ```bash
   toothfindings data variants
   for v in posneg ctx13 posneg_ctx13; do toothfindings train matrix --preset $v; done
   toothfindings eval prevalence --all
   toothfindings eval operating-points
   toothfindings eval fp-by-tooth
   ```
4. **Robustness and Grad-CAM** (`end2end` and `gradcam-masks` need the ONNX models)
   ```bash
   toothfindings eval jitter
   toothfindings eval end2end
   toothfindings eval gradcam
   toothfindings eval gradcam-masks
   ```
5. **External evaluation and expert audit**
   ```bash
   toothfindings external pipeline --dataset tufts --images data/tufts/Radiographs
   toothfindings external pipeline --dataset dentex --images data/dentex/pan_unique
   toothfindings external rescore
   toothfindings external thresholds
   toothfindings external blind-tasks
   toothfindings audit all
   toothfindings external dentex-impacted
   ```
6. **Restored teeth and leave-one-dataset-out**
   ```bash
   toothfindings external trace-implants
   toothfindings data manifest5 && toothfindings data manifest-clean && toothfindings data splits-clean
   for p in clean_ctrl4 clean_5class lodo_lyria lodo_dentex lodo_tufts; do toothfindings train matrix --preset $p; done
   toothfindings eval restored
   ```
7. **Five classes with the mitigations**
   ```bash
   toothfindings data five-mitigated
   toothfindings eval five-mitigated-infer --models REF4 FIVE0 CTRL4
   toothfindings train matrix --preset five_mitigated
   toothfindings train matrix --preset five_mitigated_lodo_lyria
   toothfindings eval five-mitigated-infer --models NEW NEW_LODO
   toothfindings eval five-mitigated-analyze
   ```
8. **Tables and figures**: level 1.

Every published run starts from ImageNet weights. `--no-pretrained` starts from random weights,
which is only useful for a quick test without internet access.

## Notes

- **Partitions.** The partitions under `results/splits/` are the reference. The main partition was
  drawn by iterating a Python `set` of patient ids, so `toothfindings data split` gives the same
  test set but a different training pool and folds on each interpreter run. Rebuilding
  `splits_clean4` and `splits_clean5` from the released manifest gives test sets that differ by
  16 and 34 crops. Training from the released files reproduces the published runs.
- **Hardware.** The published runs were trained on a CUDA GPU with mixed precision. On a CPU,
  training runs in full precision and is not bit-identical.
- **Bootstrap.** One interval of the five-class study (false flags per exam, new minus reference)
  comes out one exam step (1/568) wider on each side than the released one; every other number
  is reproduced.
- **Released files.** Level 2 commands overwrite files under `results/` with the same content,
  and `toothfindings eval latency` keeps only the tooth-level rows of `latency_all.csv`. A few
  released files (`restored_class_record.json`, `prevalence/summary.json`,
  `position_prior_posneg_ctx13.json`, `tables_source.json`) have no writer here; every
  published number taken from them is recomputed by `report check` or by the tests.
