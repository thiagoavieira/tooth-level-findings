"""Multi-source crop manifests.

* :func:`build_5class_manifest`: the expert-reviewed Tufts/DENTEX crops plus the primary
  manifest, with the frozen holdout (``review_analysis/manifest5/manifest5.csv``). The frozen
  holdout is a fixed set of whole radiographs of reviewed crops that never enters training.
* :func:`build_manifest_clean`: the contamination-free pool (``crops_source/manifest_clean.csv``):
  the primary source without the 142 untraceable DENTEX implant crops, plus the reviewed crops.

Every row carries a ``group`` (``<source>:<patient or radiograph id>``) used to keep the same
radiograph or patient on one side of every partition.
"""

from __future__ import annotations

import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import Paths
from ..utils import read_csv, write_dict_csv, write_json

#: The four clinical classes; the reviewers' label ``Other`` becomes ``Restored``.
CLIN = ["Endodontics", "Healthy", "Impacted", "Implant"]
LABEL_MAP = {c: c for c in CLIN} | {"Other": "Restored"}
#: Reviewer labels that remove a crop.
DROP = {"Exclude", "Bad crop"}
FROZEN_N = 160
FROZEN_SEED = 20260906
MANIFEST5_COLS = ["filepath", "class", "source", "group", "origin", "pred", "pred_prob", "stratum", "frozen"]
CLEAN_COLS = ["filepath", "class", "source", "group", "origin", "frozen"]


def manifest5_path(paths: Paths) -> Path:
    """Return the location of ``manifest5.csv``."""
    return paths.review / "manifest5" / "manifest5.csv"


def provenance_path(paths: Paths) -> Path:
    """Return the location of the DENTEX implant provenance table (written by the template matcher)."""
    return paths.external / "dentex" / "dentex_implant_provenance.csv"


def reviewed_rows(paths: Paths) -> list[dict]:
    """Return one manifest row per reviewed crop with a usable label.

    A label is usable when both readers agree (or only one read the crop), it is not in
    :data:`DROP`, the crop is not listed in ``exclusions.csv`` and its image exists.
    ``stratum`` is ``A`` for crops sampled to confirm a model prediction (bucket ``confirm``)
    and ``B`` otherwise.

    Returns
    -------
    list of dict
        Rows with the :data:`MANIFEST5_COLS` keys except ``frozen``.
    """
    out = []
    excluded = set()
    exclusions_file = paths.review / "exclusions.csv"
    if exclusions_file.exists():
        excluded = {(r["dataset"], r["pan_id"], r["tooth_idx"]) for r in read_csv(exclusions_file)}
    for r in read_csv(paths.review / "review_wide.csv"):
        labels = [r[k] for k in ("label1", "label2") if r.get(k)]
        if len(labels) == 2 and labels[0] != labels[1]:
            continue
        lab = labels[0]
        if lab in DROP or lab not in LABEL_MAP:
            continue
        if (r["dataset"], r["pan_id"], r["tooth_idx"]) in excluded:
            continue
        rel = f"crops/{r['pan_id']}__t{int(r['tooth_idx']):02d}.png"
        if not paths.crop_file(rel, r["dataset"]).exists():
            continue
        out.append(
            {
                "filepath": rel,
                "class": LABEL_MAP[lab],
                "source": r["dataset"],
                "group": f"{r['dataset']}:{r['pan_id']}",
                "origin": "reviewed",
                "pred": r["model_pred"],
                "pred_prob": r["pred_prob"],
                "stratum": "A" if r["bucket"] == "confirm" else "B",
            }
        )
    return out


def primary_rows(paths: Paths) -> tuple[list[dict], int, int]:
    """Return the primary manifest rows; DENTEX implants get their real radiograph id when traced.

    Returns
    -------
    rows : list of dict
        Rows with the :data:`MANIFEST5_COLS` keys except ``frozen``.
    n_remapped : int
        DENTEX implant crops whose group was replaced by the traced radiograph id.
    n_untraced : int
        DENTEX implant crops still grouped by their own file stem (``dentex:dentex_<stem>``).
    """
    provenance, n_remap = {}, 0
    provenance_file = provenance_path(paths)
    if provenance_file.exists():
        for r in read_csv(provenance_file):
            if int(r["accepted"]):
                provenance[r["patient_id"]] = r["pan_id"]
    out = []
    for r in read_csv(paths.splits / "crops_source" / "manifest.csv"):
        pid = r["patient_id"]
        if r["source"] == "dentex" and pid in provenance:
            pid, n_remap = provenance[pid], n_remap + 1
        out.append(
            {
                "filepath": r["filepath"],
                "class": r["class"],
                "source": r["source"],
                "group": f"{r['source']}:{pid}",
                "origin": "primary",
                "pred": "",
                "pred_prob": "",
                "stratum": "",
            }
        )
    n_untraced = sum(1 for r in out if r["source"] == "dentex" and r["group"].startswith("dentex:dentex_"))
    return out, n_remap, n_untraced


def freeze_groups(rows: list[dict], target_n: int = FROZEN_N, seed: int = FROZEN_SEED) -> set:
    """Draw the frozen holdout: whole radiographs, round robin over (source, class) cells.

    Each round visits the cells in sorted order and freezes the group of the next not yet
    frozen crop of that cell, until the frozen groups hold ``target_n`` reviewed crops or no
    cell has crops left.

    Parameters
    ----------
    rows : list of dict
        Manifest rows; only ``origin == "reviewed"`` rows are eligible.
    target_n : int
        Target number of reviewed crops in the holdout, after group expansion.
    seed : int
        Seed of the ``random.Random`` that shuffles each cell.

    Returns
    -------
    set
        Frozen groups.
    """
    rng = random.Random(seed)
    by_cell: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r["origin"] == "reviewed":
            by_cell[(r["source"], r["class"])].append(i)
    cells = sorted(by_cell)
    order = {c: sorted(by_cell[c]) for c in cells}
    for c in cells:
        rng.shuffle(order[c])
    reviewed_of_group: dict[str, int] = defaultdict(int)
    for r in rows:
        if r["origin"] == "reviewed":
            reviewed_of_group[r["group"]] += 1
    frozen_groups, n_frozen, exhausted = set(), 0, False
    while n_frozen < target_n and not exhausted:
        exhausted = True
        for c in cells:
            while order[c]:
                g = rows[order[c].pop()]["group"]
                if g in frozen_groups:
                    continue
                frozen_groups.add(g)
                n_frozen += reviewed_of_group[g]
                exhausted = False
                break
            if n_frozen >= target_n:
                break
    return frozen_groups


def build_5class_manifest(paths: Paths) -> Path:
    """Write ``manifest5.csv`` and ``summary.json`` (primary pool + reviewed crops + frozen holdout).

    Returns
    -------
    Path
        The written ``manifest5.csv``.
    """
    out = manifest5_path(paths).parent
    out.mkdir(parents=True, exist_ok=True)
    primary, n_remap, n_untraced = primary_rows(paths)
    rows = primary + reviewed_rows(paths)
    frozen_groups = freeze_groups(rows)
    for r in rows:
        r["frozen"] = int(r["group"] in frozen_groups)
    write_dict_csv(out / "manifest5.csv", rows, MANIFEST5_COLS)
    by_source_class = Counter((r["source"], r["class"]) for r in rows)
    summary = {
        "n_rows": len(rows),
        "n_groups": len({r["group"] for r in rows}),
        "by_source_class": {f"{s}|{c}": n for (s, c), n in sorted(by_source_class.items())},
        "by_class": dict(Counter(r["class"] for r in rows)),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "groups_by_source": {
            s: len({r["group"] for r in rows if r["source"] == s}) for s in {r["source"] for r in rows}
        },
        "frozen": {
            "seed": FROZEN_SEED,
            "target_n": FROZEN_N,
            "n_rows": sum(r["frozen"] for r in rows),
            "n_groups": len(frozen_groups),
            "by_source_class": {
                f"{s}|{c}": n
                for (s, c), n in sorted(Counter((r["source"], r["class"]) for r in rows if r["frozen"]).items())
            },
        },
        "sources_per_class": {
            c: sorted({r["source"] for r in rows if r["class"] == c}) for c in sorted({r["class"] for r in rows})
        },
        "dentex_implant_provenance": {"remapped": n_remap, "still_untraced": n_untraced},
    }
    write_json(out / "summary.json", summary, indent=1)
    return out / "manifest5.csv"


def build_manifest_clean(
    paths: Paths, classes: str = "Endodontics,Healthy,Impacted,Implant,Restored", materialise: bool = True
) -> Path:
    """Write ``manifest_clean.csv``: primary-source crops without DENTEX, plus the reviewed external crops.

    Parameters
    ----------
    paths : Paths
        Reads the primary manifests and ``manifest5.csv``.
    classes : str
        Comma-separated classes to keep.
    materialise : bool
        Also write the grayscale ("none") variant of the reviewed crops under ``crops_enhanced``.

    Returns
    -------
    Path
        The written manifest.
    """
    keep = set(classes.split(","))
    rows, dropped = [], 0
    for name in ("manifest.csv", "manifest_extra.csv"):
        path = paths.splits / "crops_source" / name
        if not path.exists():
            continue
        for r in read_csv(path):
            if r["source"] != "lyria":
                dropped += 1
                continue
            if r["class"] not in keep:
                continue
            rows.append(
                {
                    "filepath": r["filepath"],
                    "class": r["class"],
                    "source": "lyria",
                    "group": f"lyria:{r['patient_id']}",
                    "origin": "primary",
                    "frozen": 0,
                }
            )
    n_lyria = len(rows)
    for r in read_csv(manifest5_path(paths)):
        if r["origin"] != "reviewed" or r["class"] not in keep:
            continue
        rows.append(
            {
                "filepath": r["filepath"],
                "class": r["class"],
                "source": r["source"],
                "group": r["group"],
                "origin": "reviewed",
                "frozen": int(r["frozen"]),
            }
        )
    out = paths.splits / "crops_source" / "manifest_clean.csv"
    write_dict_csv(out, rows, CLEAN_COLS)
    print(
        f"dropped {dropped} untraceable DENTEX crops; {len(rows)} crops: {n_lyria} primary, "
        f"{len(rows) - n_lyria} reviewed"
    )
    if materialise:
        root = paths.crops_enhanced / "none"
        made = 0
        for r in (r for r in rows if r["origin"] == "reviewed"):
            dst = root / r["class"] / os.path.basename(r["filepath"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                continue
            gray = np.array(Image.open(paths.crop_file(r["filepath"], r["source"])).convert("L"))
            Image.fromarray(np.stack([gray] * 3, -1)).save(dst)
            made += 1
        print(f"materialised {made} grayscale crops under {root}")
    return out
