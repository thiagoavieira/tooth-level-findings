"""CLAHE on top of MSTHGR and the materialisation of the three enhancement variants.

CLAHE is contrast-limited adaptive histogram equalisation. For every crop of a manifest three
grayscale-in-RGB variants are written once, under ``crops_enhanced/<variant>/<Class>/``:

``none``          plain grayscale;
``msthgr``        MSTHGR, then grayscale;
``msthgr-clahe``  CLAHE (clip 2.0, 8x8 tiles) of the MSTHGR result.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import Paths
from ..utils import read_csv
from .msthgr import apply_msthgr

VARIANTS = ["none", "msthgr", "msthgr-clahe"]
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)


def to_rgb3(gray_u8: np.ndarray) -> np.ndarray:
    """Replicate a single-channel uint8 array across three channels."""
    return np.stack([gray_u8] * 3, axis=-1)


def clahe(gray_u8: np.ndarray, clip: float = CLAHE_CLIP, tile: tuple[int, int] = CLAHE_TILE) -> np.ndarray:
    """Apply contrast-limited adaptive histogram equalisation (OpenCV) to a uint8 image.

    Parameters
    ----------
    gray_u8 : numpy.ndarray
        Single-channel uint8 image.
    clip : float
        Contrast limit (``clipLimit``).
    tile : tuple of int
        Grid of tiles (``tileGridSize``).

    Returns
    -------
    numpy.ndarray
        Equalised single-channel uint8 image.
    """
    import cv2

    return cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(gray_u8)


def enhance_variants(rgb: Image.Image, variants: list[str]) -> dict[str, np.ndarray]:
    """Return the requested enhancement variants of one RGB crop.

    Parameters
    ----------
    rgb : PIL.Image.Image
        RGB crop.
    variants : list of str
        Names from :data:`VARIANTS`; MSTHGR is computed once when either MSTHGR variant is asked for.

    Returns
    -------
    dict
        Variant name -> ``HxWx3`` uint8 array (grayscale replicated over three channels).
    """
    out = {}
    if "none" in variants:
        out["none"] = to_rgb3(np.array(rgb.convert("L")))
    if {"msthgr", "msthgr-clahe"} & set(variants):
        msthgr_l = np.array(apply_msthgr(rgb))
        if "msthgr" in variants:
            out["msthgr"] = to_rgb3(msthgr_l)
        if "msthgr-clahe" in variants:
            out["msthgr-clahe"] = to_rgb3(clahe(msthgr_l))
    return out


def _process_one(args: tuple) -> str:
    """Write the variants of one crop; ``args`` is ``(src, cls, out_root, variants)``."""
    src, cls, out_root, variants = args
    rgb = Image.open(src).convert("RGB")
    for variant, arr in enhance_variants(rgb, variants).items():
        Image.fromarray(arr).save(Path(out_root) / variant / cls / Path(src).name)
    return cls


def apply_enhancement(
    paths: Paths, manifest: Path | None = None, variants: list[str] | None = None, workers: int | None = None
) -> None:
    """Materialise the enhancement variants of every crop of a manifest under ``crops_enhanced``.

    Parameters
    ----------
    paths : Paths
        Crop locations; the variants are written under ``paths.crops_enhanced``.
    manifest : Path, optional
        Crop manifest; ``results/splits/crops_source/manifest.csv`` by default.
    variants : list of str, optional
        Subset of :data:`VARIANTS`.
    workers : int, optional
        Worker processes (CPU count minus two by default).
    """
    variants = list(variants or VARIANTS)
    rows = read_csv(manifest or paths.splits / "crops_source" / "manifest.csv")
    out_root = paths.crops_enhanced
    for variant in variants:
        for c in sorted({r["class"] for r in rows}):
            (out_root / variant / c).mkdir(parents=True, exist_ok=True)
    tasks = [(str(paths.crop_file(r["filepath"], r["source"])), r["class"], str(out_root), variants) for r in rows]
    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    print(f"Enhancing {len(tasks)} crops x {len(variants)} variants with {workers} workers...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if i % 200 == 0:
                print(f"  {i}/{len(tasks)}")
