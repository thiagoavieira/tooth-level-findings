"""Minimal LaTeX table model: each cell keeps the unrounded values it displays.

The regression test compares those values with the published numbers of each table,
so a generator and its check can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Cell:
    """A table cell: its LaTeX text and the values it shows, in display units and reading order."""

    tex: str
    values: list[float] = field(default_factory=list)
    #: Per value, ``"<"`` when the cell shows an upper bound (e.g. ``$<$0.0001``), else ``""``;
    #: may be left empty when no value is a bound.
    bounds: list[str] = field(default_factory=list)


@dataclass
class Row:
    """A table row: label columns (plain text, not compared) followed by data cells."""

    labels: list[str]
    cells: list[Cell]

    @property
    def values(self) -> list[float]:
        """All data values of the row, in reading order."""
        return [v for c in self.cells for v in c.values]


#: Marker placed in :attr:`Tabular.rows` to draw a ``\midrule`` between data rows.
MIDRULE = "midrule"


@dataclass
class Tabular:
    r"""One ``tabular`` environment.

    Attributes
    ----------
    colspec : str
        Column specification, e.g. ``"lccc"``.
    header : list of str
        Header lines, placed between ``\toprule`` and ``\midrule``.
    rows : list
        :class:`Row` objects and :data:`MIDRULE` markers.
    resize : str or None
        Maximum width; a wider tabular is scaled down with ``\resizebox``. ``None`` disables it.
    """

    colspec: str
    header: list[str]
    rows: list
    resize: str | None = r"\linewidth"

    def data_rows(self) -> list[Row]:
        """Return the rows that carry data (without the rules)."""
        return [r for r in self.rows if isinstance(r, Row)]

    def render(self) -> list[str]:
        """Return the LaTeX lines of the environment."""
        lines = []
        if self.resize:
            lines.append(rf"\resizebox{{\ifdim\width>{self.resize}{self.resize}\else\width\fi}}{{!}}{{%")
        lines += [rf"\begin{{tabular}}{{{self.colspec}}}", r"\toprule", *self.header, r"\midrule"]
        for r in self.rows:
            if r == MIDRULE:
                lines.append(r"\midrule")
            else:
                lines.append(" & ".join([*r.labels, *(c.tex for c in r.cells)]) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}" + ("%" if self.resize else "")]
        if self.resize:
            lines.append("}")
        return lines


@dataclass
class Table:
    """A table file: one or more tabulars and an optional note.

    Attributes
    ----------
    name : str
        File stem, e.g. ``tab_prevalence``.
    tabulars : list of Tabular
        Tabulars in order, separated by a small vertical space.
    note : str
        Text of the note under the table (omitted when empty).
    sources : list of str
        Result files the table is computed from (informative only).
    """

    name: str
    tabulars: list[Tabular]
    note: str = ""
    sources: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Return the full LaTeX content of ``<name>.tex``."""
        lines: list[str] = []
        for i, t in enumerate(self.tabulars):
            if i:
                lines += ["", r"\vspace{3mm}"]
            lines += t.render()
        if self.note:
            lines += [
                "",
                r"\fonte{Author's collection.}",
                r"\vspace{2mm}",
                r"\noindent\footnotesize Note: " + self.note,
            ]
        return "\n".join(lines) + "\n"

    def write(self, out_dir: Path) -> Path:
        """Write ``<out_dir>/<name>.tex`` and return its path."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{self.name}.tex"
        p.write_text(self.render())
        return p


# --------------------------------------------------------------------------- cell factories
def fmt_int(n: int) -> str:
    """Format an integer with the LaTeX thousands separator, ``1795 -> 1{,}795``."""
    return f"{n:,}".replace(",", "{,}")


def num(x: float, nd: int = 1, thousands: bool = False) -> Cell:
    """Build a cell with a plain number with ``nd`` decimals.

    With ``thousands`` and ``nd == 0`` the rounded value is printed with a thousands separator.
    """
    if thousands and nd == 0:
        return Cell(fmt_int(int(round(x))), [x])
    return Cell(f"{x:.{nd}f}", [x])


def integer(n: int, thousands: bool = True) -> Cell:
    """Build a cell with an integer count, with a thousands separator unless ``thousands`` is False."""
    return Cell(fmt_int(n) if thousands else str(n), [n])


def pct(x: float, nd: int = 1, suffix: str = "") -> Cell:
    """Build a cell showing a proportion as a percentage with ``nd`` decimals, followed by ``suffix``."""
    return Cell(f"{x * 100:.{nd}f}{suffix}", [x * 100])


def meansd(m: float, s: float, nd: int = 1) -> Cell:
    r"""Build a ``\meansd{mean}{sd}`` cell from proportions shown as percentages."""
    return Cell(rf"\meansd{{{m * 100:.{nd}f}}}{{{s * 100:.{nd}f}}}", [m * 100, s * 100])


def ci_range(lo: float, hi: float, nd: int = 1) -> Cell:
    """Build a ``lo to hi`` interval cell from proportions shown as percentages."""
    return Cell(f"{lo * 100:.{nd}f} to {hi * 100:.{nd}f}", [lo * 100, hi * 100])


def text(s: str) -> Cell:
    """Build a cell without numbers."""
    return Cell(s, [])


def join(*cells: Cell, sep: str = " ") -> Cell:
    """Concatenate cells with ``sep``, keeping their values and bounds in order."""
    return Cell(
        sep.join(c.tex for c in cells),
        [v for c in cells for v in c.values],
        [b for c in cells for b in (c.bounds or [""] * len(c.values))],
    )


def paren(c: Cell) -> Cell:
    """Wrap a cell in parentheses."""
    return Cell(f"({c.tex})", c.values, c.bounds)


def bold(c: Cell) -> Cell:
    r"""Wrap a cell in ``\textbf``."""
    return Cell(rf"\textbf{{{c.tex}}}", c.values, c.bounds)


def multicol(n: int, c: Cell, align: str = "c") -> Cell:
    r"""Wrap a cell in ``\multicolumn`` spanning ``n`` columns."""
    return Cell(rf"\multicolumn{{{n}}}{{{align}}}{{{c.tex}}}", c.values, c.bounds)


def signed(x: float, nd: int = 2) -> Cell:
    """Build a cell with a number with an explicit sign and ``nd`` decimals."""
    return Cell(f"{x:+.{nd}f}", [x])


def pvalue(p: float) -> Cell:
    """Build a p-value cell: ``$<$0.0001`` (an upper bound) below 1e-4, ``1.0`` from 0.9995, else three decimals."""
    if p < 1e-4:
        return Cell("$<$0.0001", [1e-4], ["<"])
    if p >= 0.9995:
        return Cell("1.0", [1.0])
    return Cell(f"{p:.3f}", [p])
