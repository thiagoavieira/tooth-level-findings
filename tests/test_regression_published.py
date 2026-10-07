"""Numeric regression against the published results.

Every table is regenerated from results/ and its numbers are compared with the published ones,
stored in reference/published_tables.json, allowing only for rounding. The numbers quoted in the published
text are compared with tests/expected/published_text_numbers.json. docs/REPRODUCE.md lists the
tables and their inputs.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from toothfindings.reporting.check import compare_table, parse_number
from toothfindings.reporting.figures import confusion_matrix_test, mitigation_data
from toothfindings.reporting.tables import GENERATORS
from toothfindings.reporting.text_numbers import text_numbers

HERE = Path(__file__).parent
PUBLISHED = json.loads((HERE.parent / "reference" / "published_tables.json").read_text())


@pytest.mark.parametrize("name", list(GENERATORS))
def test_table_matches_published_numbers(sources, name):
    table = GENERATORS[name](sources)
    n_values = sum(len(r.values) for t in table.tabulars for r in t.data_rows())
    assert n_values > 0
    bad = compare_table(table, PUBLISHED[name])
    assert not bad, "\n".join(map(str, bad))


def test_every_published_table_is_covered():
    assert set(PUBLISHED) == set(GENERATORS)


EXPECTED = json.loads((HERE / "expected" / "published_text_numbers.json").read_text())
TEXT_KEYS = [k for k in EXPECTED if not k.startswith("_")]


@pytest.fixture(scope="module")
def numbers(sources):
    return text_numbers(sources)


@pytest.mark.parametrize("key", TEXT_KEYS)
def test_text_number(numbers, key):
    ref = EXPECTED[key]["value"]
    value = numbers[key]
    if "e" in ref:
        mant, exp = ref.split("e")
        dec = len(mant.split(".")[1]) if "." in mant else 0
        assert abs(value / 10 ** int(exp) - float(mant)) <= 0.5 * 10**-dec + 1e-9, (value, ref)
    else:
        dec = len(ref.split(".")[1]) if "." in ref else 0
        assert abs(value - float(ref)) <= 0.5 * 10**-dec + 1e-9, (value, ref)


def test_confusion_figure_equals_the_published_matrix(paths):
    # published confusion matrix: sum over the 3 seeds of the reference model
    published = np.array([[2623, 74, 24, 15], [50, 2738, 104, 3], [4, 11, 384, 0], [0, 0, 0, 165]])
    assert (confusion_matrix_test(paths) == published).all()


def test_mitigation_figure_uses_the_table_values(paths):
    by_label = {r["label"]: r for r in mitigation_data(paths)}
    assert len(by_label) == 16
    frozen = by_label["InceptionV3, frozen"]
    row = next(r for t in PUBLISHED["tab_mitigation"] for r in t if "InceptionV3, frozen" in r["labels"])
    p, _sd, recall, *_ = [v for v, _, _ in map(parse_number, row["numbers"])]
    assert round(frozen["precision"], 1) == p and round(frozen["recall"], 1) == recall
