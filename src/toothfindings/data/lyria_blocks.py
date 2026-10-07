"""Annotation blocks of the primary source (InReDD-PAN924 export, "Lyria" JSON format).

Each patient has one annotation JSON whose ``annotations[*].blocks[*]`` entries are the
annotated teeth ("blocks"): a polygon (``points``), an FDI tooth number (``code``) and a list of
state codes (``states``): ``H`` sound, ``Im`` implant, ``M3i``/``I`` impacted, ``Te``/``TeM``
endodontically treated, ``R`` restored, ``M3f`` developing third molar.

Every crop file is named ``lyria__<patient_id>__b<idx>.png``, where ``idx`` is a per-class
counter over the blocks in the iteration order of the annotation JSON. All functions here
enumerate blocks in exactly that order, so names, boxes and FDI codes always line up.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from ..constants import PATHOLOGY, PRIORITY

#: State codes of the two classes that the four-class pool discarded.
RESTORED = {"R"}
DEVELOPING = {"M3f"}
_ALL_PATHOLOGY = {"Im", "M3i", "I", "TeM", "Te"}

#: Pixel box ``(x_min, y_min, x_max, y_max)``.
Box = tuple[int, int, int, int]


def class_for_states(states: list[str]) -> str | None:
    """Map the states of a block to one of the four classes.

    Pathological classes follow :data:`~toothfindings.constants.PRIORITY`
    (Implant > Impacted > Endodontics); ``Healthy`` only when the block is strictly ``{"H"}``.

    Parameters
    ----------
    states : list of str
        State codes of the block.

    Returns
    -------
    str or None
        Class name, or None when the block is outside the four-class label space.
    """
    st = set(states)
    matched = [c for c in PRIORITY if st & PATHOLOGY[c]]
    if matched:
        return matched[0]
    if st == {"H"}:
        return "Healthy"
    return None


def class_and_multi(states: list[str]) -> tuple[str | None, bool]:
    """Map the states of a block to a class, also telling whether several pathologies matched.

    Returns
    -------
    cls : str or None
        Same as :func:`class_for_states`.
    is_multi : bool
        True when more than one pathological class matched.
    """
    st = set(states)
    matched = [c for c in PRIORITY if st & PATHOLOGY[c]]
    if matched:
        return matched[0], len(matched) > 1
    if st == {"H"}:
        return "Healthy", False
    return None, False


def extra_class_for_states(states: list[str]) -> str | None:
    """Map the states of a block to ``Restored`` (R) or ``DevelopingM3`` (M3f), if any.

    Follows the canonical priority extended by two steps
    (Implant > Impacted > Endodontics > Restored > DevelopingM3): a block already covered by
    the four-class pool, or strictly sound, returns None so no crop changes meaning.
    """
    st = set(states)
    if st & _ALL_PATHOLOGY or st == {"H"}:
        return None
    if st & RESTORED:
        return "Restored"
    if st & DEVELOPING:
        return "DevelopingM3"
    return None


def bbox_from_points(points: list[dict], img_w: int, img_h: int) -> Box | None:
    """Return the axis-aligned box of a polygon clipped to the image, or None when it is empty.

    Parameters
    ----------
    points : list of dict
        Polygon vertices with ``x`` and ``y`` keys.
    img_w, img_h : int
        Image width and height.

    Returns
    -------
    Box or None
        ``(x_min, y_min, x_max, y_max)``, or None when the clipped box has zero width or height.
    """
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    x_min, y_min = max(0, min(xs)), max(0, min(ys))
    x_max, y_max = min(img_w, max(xs)), min(img_h, max(ys))
    if x_max <= x_min or y_max <= y_min:
        return None
    return (x_min, y_min, x_max, y_max)


def scaled_box(box: Box, W: int, H: int, s: float) -> Box:
    """Scale a box by ``s`` about its centre, rounding and clipping to a ``W x H`` image."""
    x1, y1, x2, y2 = box
    w, h = (x2 - x1) * s, (y2 - y1) * s
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (
        max(0, int(round(cx - w / 2))),
        max(0, int(round(cy - h / 2))),
        min(W, int(round(cx + w / 2))),
        min(H, int(round(cy + h / 2))),
    )


def parse_crop_name(path: str) -> tuple[str, int]:
    """Return ``(patient_id, block index)`` of a ``lyria__<pid>__b<idx>.png`` path."""
    base = Path(path).name[:-4]
    _, pid, b = base.split("__")
    return pid, int(b[1:])


def load_annotation(json_dir: Path, pid: str) -> dict:
    """Load the annotation JSON of one patient."""
    with open(Path(json_dir) / f"{pid}.json") as fh:
        return json.load(fh)


def iter_blocks(
    json_dir: Path, image_dir: Path, pid: str, img_size: tuple[int, int] | None = None
) -> Iterator[tuple[str, int, Box]]:
    """Yield ``(class, idx, box)`` for every block of the four classes.

    Parameters
    ----------
    json_dir, image_dir : Path
        Annotation and radiograph folders.
    pid : str
        Patient id.
    img_size : tuple, optional
        ``(W, H)``; read from the radiograph when omitted.

    Yields
    ------
    tuple
        ``(class, idx, box)``; nothing when the patient has no annotation file.
    """
    jf = Path(json_dir) / f"{pid}.json"
    if not jf.exists():
        return
    if img_size is None:
        img_size = Image.open(Path(image_dir) / f"{pid}.jpg").size
    W, H = img_size
    data = load_annotation(json_dir, pid)
    idx_per_class: Counter = Counter()
    for ann in data.get("annotations", []):
        for block in ann.get("blocks", []):
            cls = class_for_states(block.get("states", []))
            if cls is None:
                continue
            box = bbox_from_points(block.get("points", []), W, H)
            if box is None:
                continue
            idx = idx_per_class[cls]
            idx_per_class[cls] += 1
            yield cls, idx, box


def block_index(json_dir: Path, image_dir: Path, pid: str) -> dict[tuple[str, int], tuple[Box, int, int, int]]:
    """Map ``(class, idx)`` to ``(box, fdi_code, W, H)`` for the four-class blocks of one patient."""
    out = {}
    data = load_annotation(json_dir, pid)
    W, H = Image.open(Path(image_dir) / f"{pid}.jpg").size
    idx_per_class: Counter = Counter()
    for ann in data.get("annotations", []):
        for block in ann.get("blocks", []):
            cls = class_for_states(block.get("states", []))
            if cls is None:
                continue
            box = bbox_from_points(block.get("points", []), W, H)
            if box is None:
                continue
            out[(cls, idx_per_class[cls])] = (box, block["code"], W, H)
            idx_per_class[cls] += 1
    return out


# Small cache of block_index5 results, keyed by "<json_dir>|<pid>".
_BI5: dict[str, dict] = {}


def block_index5(json_dir: Path, image_dir: Path, pid: str) -> dict[tuple[str, int], tuple[Box, int | None, int, int]]:
    """Map ``(class, idx)`` to ``(box, fdi_code, W, H)``, also covering ``Restored``/``DevelopingM3``.

    The four canonical classes keep the indices of :func:`~toothfindings.data.crops.generate_crops`
    and the extra classes those of :func:`~toothfindings.data.crops.generate_extra_crops`, so the
    indices match the crop file names. Results are cached; ``fdi_code`` is None when absent.
    """
    key = f"{json_dir}|{pid}"
    if key in _BI5:
        return _BI5[key]
    data = load_annotation(json_dir, pid)
    W, H = Image.open(Path(image_dir) / f"{pid}.jpg").size
    out, idx_canonical, idx_extra = {}, Counter(), Counter()
    for ann in data.get("annotations", []):
        for block in ann.get("blocks", []):
            states = block.get("states", [])
            cls = class_for_states(states)
            extra_cls = extra_class_for_states(states) if cls is None else None
            if cls is None and extra_cls is None:
                continue
            box = bbox_from_points(block.get("points", []), W, H)
            if box is None:
                continue
            if cls is not None:
                out[(cls, idx_canonical[cls])] = (box, block.get("code"), W, H)
                idx_canonical[cls] += 1
            else:
                out[(extra_cls, idx_extra[extra_cls])] = (box, block.get("code"), W, H)
                idx_extra[extra_cls] += 1
    if len(_BI5) > 64:
        _BI5.clear()
    _BI5[key] = out
    return out


def block_codes(json_dir: Path, pid: str) -> dict[tuple[str, int], int]:
    """Map ``(class, idx)`` to the FDI code without opening the radiograph.

    Degenerate boxes are detected against a very large virtual image (no clipping), which is
    how the false-flags-by-tooth-position analysis indexes blocks.
    """
    data = load_annotation(json_dir, pid)
    idx_per_class: Counter = Counter()
    out = {}
    for ann in data.get("annotations", []):
        for block in ann.get("blocks", []):
            cls = class_for_states(block.get("states", []))
            if cls is None:
                continue
            if bbox_from_points(block.get("points", []), 10**6, 10**6) is None:
                continue
            out[(cls, idx_per_class[cls])] = block["code"]
            idx_per_class[cls] += 1
    return out
