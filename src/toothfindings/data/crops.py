"""Crop pools of the primary source (InReDD-PAN924, called ``lyria`` in file and source names).

* :func:`generate_crops`: the canonical four-class pool (``crops_source/manifest.csv``).
  Implant (Im), Impacted (M3i, I), Endodontics (Te, TeM) and the abundant sound class, strictly
  ``states == {"H"}``, capped at 1,300 crops with a patient-diverse round robin. The 142
  pre-cropped DENTEX implants are copied into the implant class.
* :func:`generate_extra_crops`: the two discarded states that the restored-class study needs,
  Restored (R, capped at 1,300) and DevelopingM3 (M3f) (``manifest_extra.csv``).

Manifest paths are written relative to ``crops_source`` (``<Class>/<file>.png``).
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import Paths
from ..utils import write_csv
from .lyria_blocks import bbox_from_points, class_and_multi, extra_class_for_states

CLASSES = ["Implant", "Impacted", "Endodontics", "Healthy"]
HEALTHY_CAP = 1300
HEALTHY_SEED = 0
NEW_CLASSES = ["Restored", "DevelopingM3"]
RESTORED_CAP = 1300
SEED = 0
MANIFEST_HEADER = ["filepath", "class", "source", "patient_id", "width", "height"]


def gray3(pil: Image.Image) -> Image.Image:
    """Convert an image to grayscale replicated over three channels (the ``none`` enhancement variant)."""
    g = np.array(pil.convert("L"))
    return Image.fromarray(np.stack([g] * 3, axis=-1))


def gray3_save(pil: Image.Image, dst: str | Path) -> None:
    """Save :func:`gray3` of ``pil`` to ``dst``."""
    gray3(pil).save(dst)


def round_robin(
    cands: dict[str, list], cap: int, rng: random.Random, shuffle_in_patient_order: bool = False, with_pid: bool = True
) -> list[tuple]:
    """Select items patient-diversely: shuffle patients and their items, then take one per patient per round.

    Parameters
    ----------
    cands : dict
        Patient id -> list of candidate tuples (the lists are shuffled in place).
    cap : int
        Maximum number of items.
    rng : random.Random
        Source of randomness: the patient list is shuffled first, then each patient's list.
    shuffle_in_patient_order : bool
        Shuffle the per-patient lists in the shuffled patient order (extra-crop generator)
        instead of insertion order (sound class, position-aware negatives). The two orders
        consume the generator differently, so each caller keeps the one it was published with.
    with_pid : bool
        Prepend the patient id to each selected item.

    Returns
    -------
    list of tuple
        Selected items, ``(pid, *item)`` when ``with_pid``.
    """
    pids = list(cands.keys())
    rng.shuffle(pids)
    for c in [cands[p] for p in pids] if shuffle_in_patient_order else cands.values():
        rng.shuffle(c)
    selected: list[tuple] = []
    round_i = 0
    while len(selected) < cap:
        added = False
        for pid in pids:
            if round_i < len(cands[pid]):
                item = cands[pid][round_i]
                selected.append((pid, *item) if with_pid else item)
                added = True
                if len(selected) >= cap:
                    break
        if not added:
            break
        round_i += 1
    return selected


def generate_crops(paths: Paths, healthy_cap: int = HEALTHY_CAP) -> Path:
    """Build the canonical four-class crop pool and its manifest.

    Parameters
    ----------
    paths : Paths
        Reads ``lyria_json``, ``lyria_images`` and ``dentex_implant_crops``; writes under
        ``crops_source``.
    healthy_cap : int
        Maximum number of sound crops.

    Returns
    -------
    Path
        The written ``manifest.csv``.
    """
    out = paths.crops_source
    for cls in CLASSES:
        (out / cls).mkdir(parents=True, exist_ok=True)
    manifest, counts, patients = [], Counter(), defaultdict(set)
    multi_class_blocks = missing_images = 0
    healthy_candidates: dict[str, list] = defaultdict(list)

    json_files = sorted(f.name for f in Path(paths.lyria_json).iterdir() if f.name.endswith(".json"))
    print(f"Processing {len(json_files)} annotation files...")
    for jf in json_files:
        patient_id = jf[:-5]
        img_path = paths.lyria_image(patient_id)
        if not img_path.exists():
            missing_images += 1
            continue
        with open(Path(paths.lyria_json) / jf) as fh:
            data = json.load(fh)
        image = None
        idx_per_class: Counter = Counter()
        for ann in data.get("annotations", []):
            for block in ann.get("blocks", []):
                cls, is_multi = class_and_multi(block.get("states", []))
                if cls is None:
                    continue
                if image is None:
                    image = Image.open(img_path)
                box = bbox_from_points(block.get("points", []), image.width, image.height)
                if box is None:
                    continue
                idx = idx_per_class[cls]
                idx_per_class[cls] += 1
                if cls == "Healthy":
                    healthy_candidates[patient_id].append((img_path, box, idx))
                    continue
                if is_multi:
                    multi_class_blocks += 1
                crop = image.crop(box)
                rel = f"{cls}/lyria__{patient_id}__b{idx}.png"
                crop.save(out / rel)
                manifest.append([rel, cls, "lyria", patient_id, crop.width, crop.height])
                counts[cls] += 1
                patients[cls].add(patient_id)

    # Sound class: round robin over shuffled patients, up to the cap (about 2 per patient).
    selected = round_robin(healthy_candidates, healthy_cap, random.Random(HEALTHY_SEED))
    for pid, img_path, box, idx in selected:
        with Image.open(img_path) as image:
            crop = image.crop(box)
            rel = f"Healthy/lyria__{pid}__b{idx}.png"
            crop.save(out / rel)
            manifest.append([rel, "Healthy", "lyria", pid, crop.width, crop.height])
            counts["Healthy"] += 1
            patients["Healthy"].add(pid)

    # Pre-cropped DENTEX implants: each file is an independent sample (no radiograph id).
    src_dir = Path(paths.dentex_implant_crops)
    for fname in sorted(f.name for f in src_dir.iterdir() if f.name.lower().endswith((".png", ".jpg", ".jpeg"))):
        stem = Path(fname).stem
        rel = f"Implant/dentex__{stem}.png"
        with Image.open(src_dir / fname) as im:
            w, h = im.size
            if fname.lower().endswith(".png"):
                shutil.copy(src_dir / fname, out / rel)
            else:
                im.save(out / rel)
        manifest.append([rel, "Implant", "dentex", f"dentex_{stem}", w, h])
        counts["Implant"] += 1
        patients["Implant"].add(f"dentex_{stem}")

    manifest_path = out / "manifest.csv"
    write_csv(manifest_path, manifest, MANIFEST_HEADER)
    for cls in CLASSES:
        print(f"  {cls:12s}: {counts[cls]:5d} crops | {len(patients[cls]):4d} patients/samples")
    print(
        f"  healthy cap {healthy_cap} (available {sum(len(v) for v in healthy_candidates.values())}); "
        f"missing images {missing_images}; multi-class blocks {multi_class_blocks}"
    )
    return manifest_path


def generate_extra_crops(paths: Paths, restored_cap: int = RESTORED_CAP) -> Path:
    """Build the Restored and DevelopingM3 crops (``manifest_extra.csv``).

    Restored is capped with the same patient-diverse round robin as the sound class;
    DevelopingM3 keeps every block. The canonical manifest is not touched.

    Parameters
    ----------
    paths : Paths
        Reads the primary annotations and radiographs; writes under ``crops_source``.
    restored_cap : int
        Maximum number of restored crops.

    Returns
    -------
    Path
        The written ``manifest_extra.csv``.
    """
    out = paths.crops_source
    for cls in NEW_CLASSES:
        (out / cls).mkdir(parents=True, exist_ok=True)
    candidates: dict[str, dict[str, list]] = {c: defaultdict(list) for c in NEW_CLASSES}
    missing = 0
    json_files = sorted(f.name for f in Path(paths.lyria_json).iterdir() if f.name.endswith(".json"))
    for jf in json_files:
        pid = jf[:-5]
        img_path = paths.lyria_image(pid)
        if not img_path.exists():
            missing += 1
            continue
        with open(Path(paths.lyria_json) / jf) as fh:
            data = json.load(fh)
        image, idx_per_class = None, Counter()
        for ann in data.get("annotations", []):
            for block in ann.get("blocks", []):
                cls = extra_class_for_states(block.get("states", []))
                if cls is None:
                    continue
                if image is None:
                    image = Image.open(img_path)
                box = bbox_from_points(block.get("points", []), image.width, image.height)
                if box is None:
                    continue
                idx = idx_per_class[cls]
                idx_per_class[cls] += 1
                candidates[cls][pid].append((img_path, box, idx))
        if image is not None:
            image.close()
    print(f"{len(json_files)} annotation files, {missing} without an image")

    rng = random.Random(SEED)
    manifest, counts = [], Counter()
    for cls in NEW_CLASSES:
        cap = restored_cap if cls == "Restored" else 10**9  # DevelopingM3 is not capped
        for pid, img_path, box, idx in round_robin(candidates[cls], cap, rng, shuffle_in_patient_order=True):
            with Image.open(img_path) as image:
                crop = image.crop(box)
                rel = f"{cls}/lyria__{pid}__b{idx}.png"
                crop.save(out / rel)
                manifest.append([rel, cls, "lyria", pid, crop.width, crop.height])
                counts[cls] += 1
    manifest_path = out / "manifest_extra.csv"
    write_csv(manifest_path, manifest, MANIFEST_HEADER)
    print(f"wrote {manifest_path}: {dict(counts)}")
    return manifest_path
