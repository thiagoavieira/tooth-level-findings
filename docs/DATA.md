# Data

This repository contains no radiograph, annotation or trained model. The tables, figures and
statistics are recomputed from `results/` alone. Only the full pipeline (training, inference,
external evaluation) needs the datasets below, which each user obtains under the provider's terms.

By default the code looks for them under `data/` (ignored by git). Another location can be set in
`configs/paths.toml`, with a `TOOTHFINDINGS_<KEY>` environment variable or with
`toothfindings --set key=value`.

## InReDD-PAN924

Panoramic radiographs with tooth-level polygon annotations, distributed by PhysioNet under a
credentialed data use agreement (URL and version: *to be filled in by the authors*).

The code reads one annotation file and one radiograph per patient:

```
data/lyria/json/<patient_id>.json
data/lyria/images/<patient_id>.jpg
```

Each annotation file lists the teeth of the exam. A tooth has an FDI number (`code`), a polygon
(`points`) and its findings (`states`): `Im` implant, `Te`/`TeM` endodontically treated,
`M3i`/`I` impacted, `H` sound, `R` restored, `M3f` developing third molar. "Lyria" is the name of
this annotation format in the code.

The patient ids in `results/` (`P0001`, `P0002`, ...) are pseudonyms and cannot be linked to the
PhysioNet records.

## Tufts Dental Database

Panoramic radiographs released by Tufts University on request for research use. Expected under
`data/tufts/Radiographs/<n>.JPG`.

## DENTEX

Panoramic radiographs of the MICCAI 2023 DENTEX challenge, released for non-commercial research.
Expected layout:

```
data/dentex/dentex_classification/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json
data/dentex/**/xrays/*.png                       radiographs as distributed
data/dentex/pan_unique/disease__train_<n>.png    one copy of each training radiograph, renamed
data/dentex_implant_crops/*.png                  142 implant crops cut by hand from DENTEX
```

The renamed copies in `pan_unique/` and the hand-cut implant crops were prepared manually and are
not redistributable. `results/splits/crops_source/manifest.csv` lists the crops, and
`toothfindings external trace-implants` finds the radiograph each one came from.

## Models

- Trained classifiers are not released. Training writes a `model.pt` next to each run.
- The external pipeline and the end-to-end robustness check also use a mouth detector and a tooth
  segmenter (`data/models/mouth.onnx`, `data/models/tooth_seg.onnx`), which are not public.
- ImageNet weights are downloaded by `timm` on the first training run.

## Expert review

The raw exports of the expert review are not released. `results/review_analysis/review_wide.csv`
holds their de-identified content and is what `toothfindings audit all` reads.
