"""How much of a real radiograph the deployment-prevalence set leaves out.

Counts every annotated tooth of the primary test patients (source ``lyria``, the InReDD-PAN924
annotations) that falls outside the four-class label space (restored, carious, ...), using the
same state-to-class mapping as the crops. Needs the annotations; writes
``label_space_exclusion.json``.
"""

from __future__ import annotations

from collections import Counter

from ..config import Paths
from ..data.lyria_blocks import class_for_states, load_annotation
from ..utils import read_csv, write_json


def label_space_exclusion(paths: Paths, write: bool = True) -> dict:
    """Count the teeth of the primary test patients inside and outside the label space.

    Parameters
    ----------
    paths : Paths
        Configured paths (canonical split and annotations).
    write : bool
        Also write ``label_space_exclusion.json`` under ``paths.review``.

    Returns
    -------
    dict
        Number of test patients and annotated teeth, teeth per class in the label space, the
        number and share of excluded teeth, and how many excluded teeth carry each state.
    """
    patient_ids = sorted(
        {r["patient_id"] for r in read_csv(paths.split_dir("splits") / "test.csv") if r["source"] == "lyria"}
    )
    total, kept, excluded, states = 0, Counter(), 0, Counter()
    for pid in patient_ids:
        for a in load_annotation(paths.lyria_json, pid).get("annotations", []):
            for b in a.get("blocks", []):
                total += 1
                c = class_for_states(b.get("states", []))
                if c:
                    kept[c] += 1
                else:
                    excluded += 1
                    states.update(set(b.get("states", [])))
    out = {
        "test_patients": len(patient_ids),
        "annotated_teeth": total,
        "in_label_space": dict(kept),
        "excluded": excluded,
        "excluded_share": round(excluded / total, 4),
        "excluded_with_state": dict(states.most_common()),
    }
    if write:
        write_json(paths.review / "label_space_exclusion.json", out, indent=1)
    return out
