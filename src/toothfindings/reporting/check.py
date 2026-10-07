"""Numeric comparison of generated tables with the published ones.

The published numbers are stored in ``reference/published_tables.json``: for every table, one
list per tabular and one entry per data row, with the row labels and the published numbers as
printed (``"93.0"``, ``"<0.001"`` for an upper bound). Only numbers are compared, because the
published notes and wording were edited by hand. A value matches when it rounds to the
published one: ``|ours - ref| <= 0.5 * 10**-decimals(ref)`` (plus a tiny epsilon).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .latex import Table


@dataclass
class Mismatch:
    """One failed comparison (``row`` is the 1-based data row, -1 for a structural mismatch)."""

    table: str
    row: int
    detail: str

    def __str__(self) -> str:
        return f"{self.table} row {self.row}: {self.detail}"


def parse_number(text: str) -> tuple[float, int, str]:
    """Parse a published number such as ``"93.0"`` or ``"<0.001"``.

    Returns
    -------
    tuple
        ``(value, decimals, bound)``, where ``decimals`` is the number of printed decimals and
        ``bound`` is ``"<"`` for an upper bound, else ``""``.
    """
    bound = "<" if text.startswith("<") else ""
    digits = text.lstrip("<")
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    return float(digits), decimals, bound


def compare_table(table: Table, reference: list) -> list[Mismatch]:
    """Compare a generated table with its published numbers.

    A value matches when it rounds to the published one, ``|ours - ref| <= 0.5 * 10**-decimals``
    (plus 1e-9); an upper bound ``<x`` matches when the generated value is below ``x`` or is
    itself shown as the same bound.

    Parameters
    ----------
    table : Table
        Generated table.
    reference : list
        Published rows of the table, one list per tabular (an entry of
        ``reference/published_tables.json``).

    Returns
    -------
    list of Mismatch
        Empty when every number matches.
    """
    mismatches: list[Mismatch] = []
    refs = reference
    if len(refs) != len(table.tabulars):
        return [Mismatch(table.name, -1, f"{len(refs)} tabulars in the reference, {len(table.tabulars)} generated")]
    i_row = 0
    for ref_rows, tab in zip(refs, table.tabulars):
        ours = tab.data_rows()
        if len(ref_rows) != len(ours):
            mismatches.append(
                Mismatch(table.name, -1, f"{len(ref_rows)} data rows in the reference, {len(ours)} generated")
            )
            continue
        for ref, row in zip(ref_rows, ours):
            i_row += 1
            ref_numbers = [parse_number(n) for n in ref["numbers"]]
            values = row.values
            bounds = [b for c in row.cells for b in (c.bounds or [""] * len(c.values))]
            if len(ref_numbers) != len(values):
                mismatches.append(
                    Mismatch(
                        table.name,
                        i_row,
                        f"{len(ref_numbers)} numbers in the reference {ref_numbers}, {len(values)} generated {values}",
                    )
                )
                continue
            for (ref_v, dec, ref_b), v, b in zip(ref_numbers, values, bounds):
                if ref_b == "<" or b == "<":
                    ok = v < ref_v + 1e-12 if b != "<" else ref_b == "<"
                else:
                    ok = abs(v - ref_v) <= 0.5 * 10 ** (-dec) + 1e-9
                if not ok:
                    mismatches.append(Mismatch(table.name, i_row, f"{v!r} does not round to {ref_v} ({dec} decimals)"))
    return mismatches


def check_tables(tables: dict[str, Table], reference_file: Path) -> list[Mismatch]:
    """Compare every generated table with its published numbers.

    Parameters
    ----------
    tables : dict
        Generated tables keyed by name (as returned by ``build_tables``).
    reference_file : Path
        JSON file of the published numbers (``reference/published_tables.json``).

    Returns
    -------
    list of Mismatch
        Mismatches of all tables, in table order.
    """
    reference = json.loads(Path(reference_file).read_text())
    mismatches = []
    for name, t in tables.items():
        mismatches += compare_table(t, reference[name])
    return mismatches
