# Tooth-level findings on panoramic radiographs

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![DOI](https://img.shields.io/badge/DOI-to%20be%20assigned-lightgrey)](#citation)
[![Code license: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![Results license: CC BY 4.0](https://img.shields.io/badge/results-CC%20BY%204.0-green)](results/LICENSE)

Code and released results for the tooth-level classification of **implants**, **endodontically
treated teeth**, **impacted teeth** and **sound teeth** on panoramic radiographs, from the MSc
dissertation *Deep learning for multimodal analysis of dental radiographs: a decision-support platform* (PPGCA, FFCLRP, University of São Paulo).

## Summary

Each tooth of a panoramic radiograph is cropped and classified by a convolutional or transformer
backbone. The study:

- trains under a **leakage-safe protocol**: patient-level partitions, a grid of 18 configurations
  (3 image enhancements x 2 augmentation settings x 3 backbones) and 3 recent backbones, compared
  with Friedman, Nemenyi and Holm-corrected paired tests;
- evaluates the model at **deployment prevalence**, where every sound tooth of an exam is scored,
  and measures how false findings grow, then reduces them with position-aware negatives, context
  crops and a tooth-position prior;
- tests the model **without retraining on two external datasets** (Tufts and DENTEX), with an
  expert audit of its predictions;
- adds a fifth class for **restored teeth**.

Every table and figure is recomputed from `results/` on a CPU, without images or weights, and
checked against the published numbers.

| Result | Value |
|---|---|
| Held-out test macro-F1, InceptionV3 (3 seeds) | 93.0 ± 1.1 |
| Macro-F1 at deployment prevalence (8,672 teeth) | 82.1 ± 3.5 |
| Impacted precision, capped test set → deployment prevalence | 75.4 → 31.3 |
| False flags per exam, reference → best mitigation | 0.51 → 0.14 |
| Impacted precision with the best mitigation (DeiT-S) | 70.6 ± 3.3 |
| Expert-confirmed precision at confidence ≥ 0.99, Tufts / DENTEX | 96.3% / 96.5% |
| Impacted F1, four- vs five-class model | 81.9 vs 87.0 |

## Data

No radiograph, annotation or trained weight is distributed here. Each dataset is obtained from its
provider under its own terms; [`docs/DATA.md`](docs/DATA.md) shows where the code expects it.

| Dataset | Role | Access |
|---|---|---|
| **InReDD-PAN924** | training and held-out test | [PhysioNet](https://physionet.org/content/inredd-dataset-pan924/1.0.0/), credentialed access, [doi:10.13026/r5nt-we67](https://doi.org/10.13026/r5nt-we67) |
| **Tufts Dental Database** | external evaluation and expert audit | [Tufts University](https://tdd.ece.tufts.edu/), on request |
| **DENTEX** (MICCAI 2023) | external evaluation, expert audit, implant crops | [DENTEX challenge](https://dentex.grand-challenge.org/) |

> **Related dataset.** **InReDD-Dataset-PAN924-Ext-BoneLoss-769** extends PAN924 with 769
> panoramic radiographs and 21,064 expert-validated annotations (tooth crowns, alveolar ridge and
> reference lines) for periodontal bone-loss assessment. Available on PhysioNet:
> [doi:10.13026/s6wh-nq14](https://doi.org/10.13026/s6wh-nq14).

## Installation

Python 3.10 or later.

```bash
pip install -e ".[dev]"             # tables, figures, statistics and tests (CPU, no data)
pip install -e ".[train,external]"  # training, inference and the external pipeline
```

The published numbers were produced with PyTorch 2.9.1, timm 1.0.22 and scikit-learn 1.7.2 on a
single RTX 5070. `constraints.txt` pins every version; add `-c constraints.txt` to either command
to install exactly that environment. Paths come from `configs/paths.toml`,
`TOOTHFINDINGS_<KEY>` environment variables or `toothfindings --set key=value`.

## Reproducing the results

From `results/` only (CPU, a few minutes):

```bash
toothfindings report check     # recompute the 14 tables and compare them with the published numbers
toothfindings report tables    # write them to build/tables/
toothfindings report figures   # critical-difference diagram, mitigation chart, confusion matrix
toothfindings stats all        # summaries, Friedman, Nemenyi, paired tests with Holm, bootstrap CI
toothfindings audit all        # expert-audit analysis
```

With the datasets (GPU recommended):

```bash
export TOOTHFINDINGS_EXPERIMENTS=data/experiments        # keep new runs apart from the released ones
toothfindings data crops && toothfindings data enhance   # crop pool and enhancement variants
toothfindings train matrix --preset paper                # 18 configurations x (15 cv + 3 test) runs
toothfindings train matrix --preset paper_new            # ConvNeXt-T, EfficientNetV2-S, DeiT-S
toothfindings eval prevalence --all                      # deployment prevalence and mitigations
```

[`docs/REPRODUCE.md`](docs/REPRODUCE.md) maps every table and figure to its command and input
files and gives the full pipeline. Trained models are not released, since they were trained on data
under a data use agreement.

## Tests

```bash
pytest                                  # unit tests and regression against the published numbers
pytest -m "not slow"                    # skips the 90-second five-class bootstrap
ruff check . && ruff format --check .
```

Continuous integration runs the whole suite on CPU, without data.

## Repository layout

```
src/toothfindings/  data, enhance, models, evaluation, stats, external, audit, reporting, cli.py
results/            released results, described in results/README.md
reference/          published table numbers, for the regression test
tests/              unit and regression tests
docs/               REPRODUCE.md, DATA.md
configs/paths.toml  path configuration
constraints.txt     exact package versions of the published results
```

## Citation

If you use this code or these results, please cite the dissertation and the datasets
(`CITATION.cff` holds the software entry).

```bibtex
@software{toothfindings,
  title   = {tooth-level-findings: code and results for leakage-safe tooth-level classification of radiographic findings in panoramic radiographs},
  author  = {{Alves Vieira de Matos}, Thiago and {Alaniz Macedo}, Alessandra},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/thiagoavieira/tooth-level-findings}
}

@mastersthesis{matos2026dissertation,
  title  = {Deep learning for multimodal analysis of dental radiographs: a decision-support platform},
  author = {{Alves Vieira de Matos}, Thiago},
  school = {University of S{\~a}o Paulo, FFCLRP, PPGCA},
  year   = {2026}
}

@misc{inredd_pan924,
  title     = {{InReDD-Dataset-PAN924} (version 1.0.0)},
  publisher = {PhysioNet},
  year      = {2025},
  doi       = {10.13026/r5nt-we67}
}

@article{PhysioNet-inredd-pan924-ext-boneloss-769-1.0.0,
  author  = {{Alves Vieira de Matos}, Thiago and Bauman, Jo{\~a}o Donato and Ramos, Maria Julia and Vitareli, Thais and {Uehara Martins}, Caio and Castro, Antonio and Tirapelli, Camila and {Alaniz Macedo}, Alessandra},
  title   = {{InReDD-Dataset-PAN924-Ext-BoneLoss-769}},
  journal = {{PhysioNet}},
  year    = {2026},
  month   = sep,
  note    = {Version 1.0.0},
  doi     = {10.13026/s6wh-nq14},
  url     = {https://doi.org/10.13026/s6wh-nq14}
}

@article{panetta2022tufts,
  title   = {Tufts Dental Database: A Multimodal Panoramic X-Ray Dataset for Benchmarking Diagnostic Systems},
  author  = {Panetta, Karen and Rajendran, Rahul and Ramesh, Aruna and Rao, Shishir Paramathma and Agaian, Sos},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  volume  = {26},
  number  = {4},
  pages   = {1650--1659},
  year    = {2022}
}

@article{hamamci2023dentex,
  title   = {DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays},
  author  = {Hamamci, Ibrahim Ethem and others},
  journal = {arXiv preprint arXiv:2305.19112},
  year    = {2023}
}
```

PhysioNet also asks for its standard citation: Goldberger A, et al. PhysioBank, PhysioToolkit,
and PhysioNet. *Circulation* 101(23):e215–e220, 2000.

## Funding

This work was supported by CNPq (grant 133030/2025-3) and FAPESP (grant 2024/15912-0).

## License

- **Code**: MIT ([`LICENSE`](LICENSE)). The optional `external` dependencies include Ultralytics,
  which is distributed under AGPL-3.0; its terms apply when that part of the pipeline is used.
- **Results** (`results/`): CC BY 4.0 ([`results/LICENSE`](results/LICENSE)). Content derived from
  DENTEX also remains under the DENTEX license, CC BY-NC-SA 4.0.
- **Datasets** keep their own licenses and terms of use.
