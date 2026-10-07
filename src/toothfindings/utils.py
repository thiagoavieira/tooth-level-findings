"""Small CSV and JSON helpers used across the package."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of dictionaries."""
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: str | Path, rows: Iterable[Sequence[Any]], header: Sequence[str]) -> None:
    """Write rows (sequences) under a header, creating the parent folder."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def write_dict_csv(path: str | Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    """Write dictionaries with ``csv.DictWriter`` (field names from the first row by default)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames or rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with open(path) as fh:
        return json.load(fh)


def write_json(path: str | Path, obj: Any, indent: int | None = 2, **kw: Any) -> None:
    """Dump ``obj`` as JSON, creating the parent folder; ``kw`` is passed to ``json.dump``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=indent, **kw)
