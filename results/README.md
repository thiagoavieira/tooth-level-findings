# results/

Every result of the study that can be released: per-run metrics, per-crop predictions with
pseudonymised patient ids, partitions, statistics and audit outputs. No image, annotation or model
weight. The command that writes each file is in `docs/REPRODUCE.md`.

Conventions used throughout:

- **Classes** are `Endodontics` (endodontically treated), `Healthy` (sound), `Impacted`, `Implant`
  and, in the five-class studies, `Restored`. Model outputs and `p_<class>` columns follow this
  alphabetical order.
- **Configurations** are `<enhancement>__<aug>__<arch>`: enhancement `none`, `msthgr` or
  `msthgr-clahe`; `noaug` or `aug`; backbone `resnetv2_50`, `vgg16`, `inception_v3`,
  `convnext_tiny`, `tf_efficientnetv2_s` or `deit_small_patch16_224`.
- **Runs** are `<config>__fold<k>__seed<s>` in cross-validation (`runs/`) and `<config>__seed<s>`
  on the fixed test set (`test/`), with folds 0 to 4 and seeds 0 to 2.
- **Crop paths** are relative: `<Class>/lyria__<patient>__b<k>.png` for the primary source (see
  `docs/DATA.md`), `Implant/dentex__<n>.png` for the hand-cut DENTEX implants, and
  `crops/<radiograph>__t<NN>.png` for crops of the external pipeline. Patient ids are `P0001`,
  `P0002`, ... ; `group` columns are `lyria:<patient>` or `<dataset>:<radiograph>`.
- Probabilities are rounded to 4 decimals.

## experiments/

One folder per experiment, each with the same run layout:

```
runs/<run>/      cross-validation run
test/<run>/      test-mode run
registry.csv     one row per run: run_id, mode, enhancement, aug, arch, fold, seed, status,
                 batch_size, start_ts, end_ts, epochs_run, best_val_macro_f1, test_macro_f1,
                 peak_vram_MB, error
progress.json    progress counters of the run matrix (total, done, failed, current, ...)
```

Files of a run:

| File | Content |
|---|---|
| `run_config.yaml` | the run's arguments, as JSON: mode, enhancement, aug, arch, fold, seed, run_id, run_dir, batch, lr, epochs, patience, workers |
| `metrics.json` | run_id, mode, configuration, fold, seed, batch_size, `best_val_macro_f1`, `val_metrics` and, in test mode, `test_metrics` (`accuracy`, `macro_f1`, `per_class.<class>.{precision,recall,f1}`) |
| `training_log.csv` | epoch, train_loss, train_acc, val_loss, val_acc, val_macro_f1, lr, epoch_time_s |
| `split_used.csv` | filepath, class, role (`train` or `val`) |
| `status.json` | last heartbeat (state, epoch, best and last validation macro-F1) |
| `predictions.csv` | test mode: filepath, source, y_true, y_pred |
| `confusion_matrix.csv` | test mode: rows true class, columns predicted class |

The experiments:

| Folder | Content |
|---|---|
| `runs/`, `test/`, `registry.csv` | the grid: 18 classic configurations and 3 recent backbones (grayscale, no augmentation); 315 cv and 63 test runs on `splits/splits` |
| `aggregates/{cv,test}_summary.csv` | per configuration: n_runs, macro_f1 mean, sd, t-based 95% CI, accuracy mean and sd, per-class F1 mean and sd |
| `stats/friedman.json` | Friedman test over the 18 classic configurations (statistic, p_value, n_configs, n_blocks) |
| `stats/nemenyi.csv` | Nemenyi p-values, configuration x configuration |
| `stats/paired_tests.csv` | the 153 pairs: means, t statistic, t and Wilcoxon p-values, Cohen's d_z, Holm-adjusted p-values, significance |
| `stats/new_backbones.{json,csv}` | each backbone against InceptionV3 over the 15 cv blocks, test macro-F1 and per-class F1 |
| `cost/inference_cost.csv` | arch, params_M, model_MB, latency_cpu_ms, latency_gpu_ms, peak_vram_MB |
| `prevalence/<arch>__<tag>_s<scale>[_prior]/` | deployment-prevalence evaluation (8,672 crops); tag `paper` for the grid models, otherwise the mitigation variant; scale 1.0 or 1.3; `_prior` with the tooth-position prior. `metrics.json`: n_full, n_capped, macro-F1 and per-class metrics on the full and capped sets as [mean, sd] over seeds, ensemble confusion, false flags per exam. `per_seed.json`: the same metrics per seed. `predictions.csv`: filepath, class, patient_id, tooth_type, in_capped_test, pred, p_<class> (ensemble mean) |
| `prevalence/inception_v3/` | older output format of the reference run (no tooth_type column) |
| `prevalence/operating_point.json` | per run, per threshold tau: false_flags_per_exam, share_exams_zero, recall per finding class, precision_impacted, n_sound_flagged |
| `prevalence/fp_by_tooth_type.json` | impacted recall and sound teeth flagged per tooth type |
| `prevalence/summary.json` | false flags per exam of the reference model (mean, median, p90, share of exams without one) |
| `robustness/jitter_inception_v3.{json,csv}` | macro-F1 (mean, sd over seeds) and per-class F1 under box jitter and scaling |
| `robustness/end2end_inception_v3.json`, `end2end_teeth.csv` | real mouth detector and tooth segmenter: detection recall at IoU 0.5 and classification of the matched teeth; per annotated tooth: detected, iou, annotated and predicted boxes |
| `gradcam_masks/summary_<arch>.json`, `per_crop_<arch>.csv` | Grad-CAM against segmenter masks: mask area, CAM mass inside the mask, IoU of the top 20% pixels, pointing game; per crop and seed |
| `clean_ctrl4/`, `clean_5class/` | restored class: four- and five-class DeiT-S on `splits/splits_clean{4,5}` |
| `lodo_{lyria,dentex,tufts}/` | leave-one-dataset-out, test runs only |
| `clean_e1_report.json` | four- vs five-class comparison: in-domain four-class view (A), restored teeth in domain (B), external expert labels (C) |
| `restored_class_record.json` | record of the restored-class experiment (protocol, splits, in-domain and LODO results) |
| `variants/{posneg,ctx13,posneg_ctx13}/` | four-class mitigation variants (InceptionV3 and DeiT-S) |
| `variants/five_mitigated/` | five classes with the mitigations; `README.txt` describes it. `eval/<model>__<set>.npz`: `probs` (crops x seeds x classes, no prior), `classes`, `paths`, `scale`. `results/results.json`: every number of the study; `results/params.json`: parameter counts; `results/tables.txt`: plain-text tables; `results/pool_composition_by_tooth_type.json`; `results/results_reference_only_pretraining.json`: the reference-model numbers computed before the new model was trained |
| `variants/five_mitigated_lodo_lyria/` | the leave-InReDD-out model of the same study |

## splits/

Partition folders hold `train_pool.csv` and `test.csv` (filepath, class, source, patient_id and,
in the clean and five-class ones, group and origin), `folds.json`
(`{"0": {"train": [row indices into train_pool.csv], "val": [...]}, ...}`) and
`split_summary.json` (counts per class and side, groups, fold sizes, arguments). These saved files
are the source of truth (`docs/REPRODUCE.md`).

| Folder | Content |
|---|---|
| `crops_source/manifest.csv` | the four-class crop pool (3,109 crops): filepath, class, source, patient_id, width, height |
| `crops_source/manifest_extra.csv` | Restored and DevelopingM3 crops, same columns |
| `crops_source/manifest_clean.csv` | the contamination-free pool: filepath, class, source, group, origin, frozen (1 for the frozen external holdout) |
| `splits/` | canonical four-class partition: 720 training crops (180 per class), 2,065 test crops, `folds/fold<k>_{train,val}.csv`, `split_manifest.csv` (every crop with its partition) |
| `variants/{posneg,ctx13,posneg_ctx13}/splits/` | mitigation variants of the canonical partition |
| `splits_clean4/`, `splits_clean5/` | restored class, in domain |
| `splits_lodo_{lyria,dentex,tufts}/` | leave-one-dataset-out |
| `splits5_mitigated/`, `splits5_mitigated_lodo_lyria/` | five-class mitigated study (`README.txt`) |
| `variants/five_mitigated/` | `eval_sets/{indomain_test,prevalence,external}.csv` (crop paths at scale 1.0 and 1.3, class, source, patient or group), `prior4.json` and `prior5.json` (tooth-position priors with their counts), `build_log.json` |

## external/

| File | Content |
|---|---|
| `<ds>/pans.csv` | one row per radiograph: dataset, pan_id, pan_file, width, height, mouth box and score, n_teeth, seconds |
| `<ds>/predictions.csv` | zero-shot pipeline with the reference InceptionV3: dataset, pan_id, pan_file, tooth_idx, crop_file, box (x1, y1, x2, y2), det_score, pred, pred_prob, seed_agreement, margin, p_<class> (mean) and p_<class>_s<seed> |
| `<ds>/predictions_deit_posneg.csv` | the same teeth re-scored by the audited DeiT-S (position-aware negatives) |
| `<ds>/thresholds_deit_posneg.json` | study thresholds: per-class calibration curve on the in-domain prevalence set, target precision, and the review selection counts |
| `dentex/dentex_impacted_eval.json` | DENTEX impacted-class check against the challenge annotations |

## review_analysis/

| File | Content |
|---|---|
| `review_wide.csv` | one row per reviewed tooth: blind_id, bucket (`confirm` or `below_tau`), dataset, pan_id, tooth_idx, model prediction and probabilities, seed_agreement, the two readers' ids and labels, exclusion flag |
| `audit.json` | weighted precision per class and threshold, with cluster-bootstrap CIs |
| `in_scope.json`, `breakdown.json` | precision restricted to the label space; what the corrections were |
| `cascade.json`, `bin_ci.json` | corrections by confidence bin and the filtering cascade (the expert-audit table) |
| `agreement.json`, `agreement_extra.json` | inter-reader agreement (kappa, AC1, Krippendorff's alpha, marginals) |
| `implant_view.json` | the implant class seen as a detection task |
| `operating_points.json`, `curves_<ds>.csv`, `bridge_metrics.json`, `fairness_view.json`, `impacted_recall.json`, `label_space_exclusion.json` | complementary analyses (operating curves, detection-style metrics, per-subgroup views, impacted recall, teeth outside the label space) |

## reporting/

| File | Content |
|---|---|
| `bootstrap_test_ci.json` | crop-level bootstrap of the reference test macro-F1 (2,000 draws, seed 0) |
| `latency_all.csv` | unified latency session: input, model, params_M, task, CPU and GPU latency statistics of each backbone |
| `training_times.csv` | duration of every run of the grid, recovered from `experiments/registry.csv` |
| `tables_source.json` | which files each table was generated from |
| `position_prior_posneg_ctx13.json` | the tooth-position prior used with the best mitigation |
