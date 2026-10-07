"""Sound teeth flagged as findings, by tooth position, on the deployment-prevalence set.

Counts, for each tooth type, the sound teeth and the predictions they received, and the
impacted teeth and how many were missed. Reads a prevalence ``predictions.csv``; ``fp`` stands
for false positives.
"""

from __future__ import annotations

import collections
import os
from pathlib import Path

from ..constants import TOOTH_TYPES_FINE, tooth_type
from ..data.lyria_blocks import block_codes, parse_crop_name
from ..utils import read_csv, write_json

STAT_KEYS = [
    "gt_impacted",
    "impacted_missed",
    "sound_total",
    "sound_pred_impacted",
    "sound_pred_endo",
    "sound_pred_implant",
]


def tally(rows: list[dict], type_of) -> dict:
    """Count sound and impacted teeth and their predictions by tooth type.

    Only primary-source rows (file names starting with ``lyria__``) are counted.

    Parameters
    ----------
    rows : list of dict
        Prevalence prediction rows (``filepath``, ``class``, ``pred``).
    type_of : callable
        Function ``row -> tooth type``.

    Returns
    -------
    dict
        ``{key: {tooth type: count}}`` for every key of ``STAT_KEYS``, plus
        ``sound_pred_impacted_rate`` (``sound_pred_impacted / sound_total`` per type).
    """
    stats = {k: collections.Counter() for k in STAT_KEYS}
    for r in rows:
        if not os.path.basename(r["filepath"]).startswith("lyria__"):
            continue
        ttype = type_of(r)
        if r["class"] == "Healthy":
            stats["sound_total"][ttype] += 1
            if r["pred"] == "Impacted":
                stats["sound_pred_impacted"][ttype] += 1
            if r["pred"] == "Endodontics":
                stats["sound_pred_endo"][ttype] += 1
            if r["pred"] == "Implant":
                stats["sound_pred_implant"][ttype] += 1
        elif r["class"] == "Impacted":
            stats["gt_impacted"][ttype] += 1
            if r["pred"] != "Impacted":
                stats["impacted_missed"][ttype] += 1
    out = {k: dict(v) for k, v in stats.items()}
    out["sound_pred_impacted_rate"] = {t: stats["sound_pred_impacted"][t] / n for t, n in stats["sound_total"].items()}
    return out


def fp_by_tooth_from_annotations(predictions_csv: Path, lyria_json: Path, out: Path | None = None) -> dict:
    """Tally by fine tooth type read from the annotation JSON, as published (needs the restricted data).

    Parameters
    ----------
    predictions_csv : Path
        Prevalence ``predictions.csv``.
    lyria_json : Path
        Folder of the primary-source annotation JSON files.
    out : Path, optional
        JSON file to write the tally to.

    Returns
    -------
    dict
        See :func:`tally`.
    """
    cache: dict = {}

    def type_of(r):
        pid, idx = parse_crop_name(r["filepath"])
        if pid not in cache:
            cache[pid] = block_codes(lyria_json, pid)
        code = cache[pid].get((r["class"], idx), 0)
        return tooth_type(code, TOOTH_TYPES_FINE)

    res = tally(read_csv(predictions_csv), type_of)
    if out is not None:
        write_json(out, res)
    return res


def fp_by_tooth_from_predictions(predictions_csv: Path, typed_predictions_csv: Path) -> dict:
    """Tally by the coarse tooth type stored in another prevalence run of the same crops.

    Runs evaluated with the prior or with rescaled crops store a ``tooth_type`` column; joining
    by file path gives the coarse types (premolars merged, incisors merged), which is all the
    false-flags-by-tooth table needs, without the restricted annotations.
    """
    types = {r["filepath"]: r["tooth_type"] for r in read_csv(typed_predictions_csv)}
    return tally(read_csv(predictions_csv), lambda r: types[r["filepath"]])
