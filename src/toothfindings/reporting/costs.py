"""Training wall-clock of the tooth-level grid, recovered from the run registry.

The duration of every completed run in ``experiments/registry.csv`` is its stored duration or
``end_ts - start_ts``; runs are grouped by ``enhancement|aug|arch``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..utils import read_csv

#: Registry columns joined with ``|`` into the group label of a run.
GROUP_COLS = ["enhancement", "aug", "arch"]


def run_duration(row: dict) -> float | None:
    """Return the duration in seconds of one registry row, or None when it cannot be recovered.

    The first non-empty of ``dur_s`` and ``total_train_s`` is used; otherwise the difference of
    the ISO timestamps ``end_ts - start_ts``.
    """
    for k in ("dur_s", "total_train_s"):
        if row.get(k) not in (None, "", "nan"):
            return float(row[k])
    start, end = row.get("start_ts"), row.get("end_ts")
    if start and end:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    return None


def training_times(registry: Path) -> list[dict]:
    """List the completed runs of a registry with their durations.

    Parameters
    ----------
    registry : Path
        Run registry CSV (``experiments/registry.csv``).

    Returns
    -------
    list of dict
        One row per completed run with a positive duration: ``run_id``, ``group``
        (``enhancement|aug|arch``) and ``dur_s`` (seconds rounded to an integer, as text).
    """
    out = []
    for r in read_csv(registry):
        if r.get("status") not in ("done", "completed", "ok"):
            continue
        duration = run_duration(r)
        if duration is None or duration <= 0:
            continue
        out.append(
            {
                "run_id": r.get("run_id"),
                "group": "|".join(r.get(c, "?") for c in GROUP_COLS if c in r),
                "dur_s": f"{duration:.0f}",
            }
        )
    return out
