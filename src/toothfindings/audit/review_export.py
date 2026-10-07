"""Join the Label Studio export of the expert review with the blind map.

Produces ``review_long.csv`` (one row per crop and reader) and ``review_wide.csv`` (one row per
crop with both readers). The exports identify the readers' accounts and are not distributed;
the released ``review_wide.csv`` is the starting point of every audit analysis.

Inputs under ``<data>/review/``: ``export_<ds>.json`` and ``<ds>/blind_map_<ds>_<tag>.csv``;
optional ``results/review_analysis/exclusions.csv`` (``dataset, blind_id, reason``).
"""

from __future__ import annotations

from collections import defaultdict

from ..config import Paths
from ..utils import read_csv, read_json, write_dict_csv
from .core import CLIN, TAG


def load_annotations(paths: Paths, ds: str, exclusions: dict, tag: str = TAG) -> list[dict]:
    """Return one row per (crop, reader) of dataset ``ds``.

    Cancelled answers and answers with other than exactly one choice are dropped, as are
    tasks not in the blind map.

    Parameters
    ----------
    paths : Paths
        Configured paths (``export_<ds>.json`` and the blind map under ``paths.review_exports``).
    ds : str
        Dataset.
    exclusions : dict
        ``{(dataset, blind_id): reason}`` of crops declared out of scope.
    tag : str
        Tag of the blind map.

    Returns
    -------
    list of dict
        Rows of ``review_long.csv``: crop ids, model outputs from the blind map, exclusion flag
        and reason, the reader's ``label`` and ``annotator`` id.
    """
    blind_map = {r["blind_id"]: r for r in read_csv(paths.review_exports / ds / f"blind_map_{ds}_{tag}.csv")}
    rows = []
    for t in read_json(paths.review_exports / f"export_{ds}.json"):
        bid = t["data"].get("blind_id")
        m = blind_map.get(bid)
        if m is None:
            continue
        for a in t["annotations"]:
            if a.get("was_cancelled"):
                continue
            choices = a["choices"]
            if len(choices) != 1:
                continue
            annotator = a["by"]["id"] if isinstance(a["by"], dict) else a["by"]
            rows.append(
                {
                    "dataset": ds,
                    "blind_id": bid,
                    "task_id": t["task_id"],
                    "excluded": int((ds, bid) in exclusions),
                    "exclusion_reason": exclusions.get((ds, bid), ""),
                    "pan_id": m["pan_id"],
                    "tooth_idx": m["tooth_idx"],
                    "model_pred": m["pred"],
                    "label": choices[0],
                    "annotator": annotator,
                    "lead_time": a.get("lead_time"),
                    "created_at": a.get("created_at"),
                    "pred_prob": float(m["pred_prob"]),
                    "seed_agreement": int(m["seed_agreement"]),
                    "margin": float(m["margin"]),
                    "bucket": m["bucket"],
                    **{f"p_{c}": float(m[f"p_{c}"]) for c in CLIN},
                }
            )
    return rows


def to_wide(rows: list[dict]) -> list[dict]:
    """Pivot the per-reader rows to one row per crop.

    Readers are sorted by id and numbered (``reader1``/``label1``, ``reader2``/``label2``);
    ``n_readers`` counts them.
    """
    by_crop: dict = defaultdict(list)
    for r in rows:
        by_crop[(r["dataset"], r["blind_id"])].append(r)
    wide = []
    for rs in by_crop.values():
        rs = sorted(rs, key=lambda r: r["annotator"])
        base = {
            k: rs[0][k]
            for k in [
                "dataset",
                "blind_id",
                "pan_id",
                "tooth_idx",
                "model_pred",
                "excluded",
                "exclusion_reason",
                "pred_prob",
                "seed_agreement",
                "margin",
                "bucket",
            ]
            + [f"p_{c}" for c in CLIN]
        }
        base["n_readers"] = len(rs)
        for i, r in enumerate(rs, start=1):
            base[f"reader{i}"] = r["annotator"]
            base[f"label{i}"] = r["label"]
        wide.append(base)
    return wide


def build_review_tables(paths: Paths) -> tuple[int, int]:
    """Write ``review_long.csv`` and ``review_wide.csv``; returns their row counts ``(long, wide)``."""
    exclusions_csv = paths.review / "exclusions.csv"
    exclusions = (
        {(r["dataset"], r["blind_id"]): r.get("reason", "excluded") for r in read_csv(exclusions_csv)}
        if exclusions_csv.exists()
        else {}
    )
    rows = load_annotations(paths, "tufts", exclusions) + load_annotations(paths, "dentex", exclusions)
    write_dict_csv(paths.review / "review_long.csv", rows)
    wide = to_wide(rows)
    write_dict_csv(paths.review / "review_wide.csv", wide, sorted({k for r in wide for k in r}))
    return len(rows), len(wide)
