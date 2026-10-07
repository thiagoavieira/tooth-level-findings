"""Blind review tasks: the reviewer sees the class the model indicated, never its confidence.

The review selection (``keep_<ds>_<tag>.csv``) is shuffled with ``Random(0)`` and given neutral
sequential ids (``00001``, ...), so the id does not encode confidence. Writes ``blind_map_<ds>_<tag>.csv`` (id ->
source crop and model outputs) and Label Studio tasks ``tasks_<ds>_<tag>_blind.json`` under
``<data>/review/<ds>/``; the image URL is a placeholder of the annotation server.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import Paths
from ..constants import CLASSES4
from ..utils import read_csv, write_dict_csv


def build_blind_tasks(paths: Paths, ds: str, tag: str = "deit_posneg", batches_index: Path | None = None) -> int:
    """Write the blind map and the Label Studio tasks of one dataset.

    Parameters
    ----------
    paths : Paths
        Configured repository and data paths.
    ds : {"tufts", "dentex"}
        External dataset.
    tag : str
        Tag of the review selection (``keep_<ds>_<tag>.csv``); ``deit_posneg`` is the audited
        model (DeiT-S trained with position-aware negatives).
    batches_index : Path, optional
        ``index.csv`` of the review image batches (columns ``file``, ``pred``, ``batch``);
        ``<data>/review/labelstudio_batches/<ds>/index.csv`` by default.

    Returns
    -------
    int
        Number of tasks written.
    """
    # Batch image names are ``<prefix>__<crop stem>.png``; index them by crop stem.
    batch_of_crop = {}
    for r in read_csv(batches_index or paths.review_exports / "labelstudio_batches" / ds / "index.csv"):
        batch_of_crop[r["file"].split("__", 1)[1][:-4]] = (r["pred"], r["batch"], r["file"])
    keep = read_csv(paths.external / ds / f"keep_{ds}_{tag}.csv")
    random.Random(0).shuffle(keep)
    rows, tasks = [], []
    for i, r in enumerate(keep, start=1):
        pred_dir, batch, fname = batch_of_crop[Path(r["crop_file"]).name[:-4]]
        blind_id = f"{i:05d}"
        rows.append(
            {
                "blind_id": blind_id,
                "dataset": ds,
                "pred": r["pred"],
                "pan_id": r["pan_id"],
                "tooth_idx": r["tooth_idx"],
                "src_rel": f"{pred_dir}/{batch}/{fname}",
                "pred_prob": r["pred_prob"],
                "seed_agreement": r["seed_agreement"],
                "margin": r["margin"],
                "bucket": r["bucket"],
                **{f"p_{c}": r[f"p_{c}"] for c in CLASSES4},
            }
        )
        tasks.append(
            {
                "data": {
                    "image": f"/data/local-files/?d=iei_external/{ds}/review_blind/{blind_id}.png",
                    "dataset": ds,
                    "blind_id": blind_id,
                    "pred": r["pred"],
                },
                "predictions": [
                    {
                        "model_version": "blind",
                        "result": [
                            {
                                "from_name": "choice",
                                "to_name": "image",
                                "type": "choices",
                                "value": {"choices": [r["pred"]]},
                            }
                        ],
                    }
                ],
            }
        )
    out_dir = paths.review_exports / ds
    write_dict_csv(out_dir / f"blind_map_{ds}_{tag}.csv", rows)
    with open(out_dir / f"tasks_{ds}_{tag}_blind.json", "w") as fh:
        json.dump(tasks, fh)
    return len(tasks)
