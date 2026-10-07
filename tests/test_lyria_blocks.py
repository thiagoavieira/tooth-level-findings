import json

import pytest
from PIL import Image

from toothfindings.constants import TOOTH_TYPES_FINE, tooth_type
from toothfindings.data import lyria_blocks as lb


@pytest.mark.parametrize(
    "states, expected",
    [
        (["Im"], "Implant"),
        (["Im", "M3i", "Te"], "Implant"),
        (["I", "TeM"], "Impacted"),
        (["M3i"], "Impacted"),
        (["Te", "R"], "Endodontics"),
        (["H"], "Healthy"),
        (["H", "R"], None),
        (["R"], None),
        ([], None),
    ],
)
def test_class_for_states_follows_the_priority(states, expected):
    assert lb.class_for_states(states) == expected


def test_class_and_multi_flags_several_pathologies():
    assert lb.class_and_multi(["Im", "Te"]) == ("Implant", True)
    assert lb.class_and_multi(["Te"]) == ("Endodontics", False)
    assert lb.class_and_multi(["H"]) == ("Healthy", False)
    assert lb.class_and_multi(["M3f"]) == (None, False)


@pytest.mark.parametrize(
    "states, expected",
    [
        (["R"], "Restored"),
        (["R", "M3f"], "Restored"),
        (["M3f"], "DevelopingM3"),
        (["R", "Te"], None),  # already an endodontic crop of the four-class pool
        (["H"], None),
        (["X"], None),
    ],
)
def test_extra_class_never_relabels_a_four_class_block(states, expected):
    assert lb.extra_class_for_states(states) == expected


def test_bbox_is_clipped_and_degenerate_boxes_are_dropped():
    pts = [{"x": -5, "y": 10}, {"x": 50, "y": 40}, {"x": 120, "y": 25}]
    assert lb.bbox_from_points(pts, 100, 100) == (0, 10, 100, 40)
    assert lb.bbox_from_points([{"x": 3, "y": 1}, {"x": 3, "y": 9}], 100, 100) is None
    assert lb.bbox_from_points([{"x": 150, "y": 1}, {"x": 160, "y": 9}], 100, 100) is None


def test_scaled_box_keeps_the_centre_and_clips():
    assert lb.scaled_box((40, 40, 60, 80), 200, 200, 1.0) == (40, 40, 60, 80)
    assert lb.scaled_box((40, 40, 60, 80), 200, 200, 1.5) == (35, 30, 65, 90)
    assert lb.scaled_box((0, 0, 20, 20), 25, 25, 2.0) == (0, 0, 25, 25)


def test_parse_crop_name():
    assert lb.parse_crop_name("Healthy/lyria__P0042__b17.png") == ("P0042", 17)


def _block(states, x, code):
    return {"states": states, "code": code, "points": [{"x": x, "y": 10}, {"x": x + 20, "y": 60}]}


@pytest.fixture
def annotation_dir(tmp_path):
    blocks = [
        _block(["H"], 0, 11),
        _block(["Te"], 30, 16),
        _block(["H"], 60, 18),
        _block(["R"], 90, 26),  # restored: outside the four classes
        {"states": ["H"], "code": 21, "points": [{"x": 5, "y": 5}, {"x": 5, "y": 9}]},  # degenerate
        _block(["M3f"], 120, 38),
        _block(["H"], 150, 47),
    ]
    (tmp_path / "json").mkdir()
    (tmp_path / "img").mkdir()
    (tmp_path / "json" / "P0001.json").write_text(
        json.dumps({"annotations": [{"blocks": blocks[:4]}, {"blocks": blocks[4:]}]})
    )
    Image.new("L", (200, 100)).save(tmp_path / "img" / "P0001.jpg")
    return tmp_path


def test_block_indices_count_per_class_in_annotation_order(annotation_dir):
    j, i = annotation_dir / "json", annotation_dir / "img"
    blocks = list(lb.iter_blocks(j, i, "P0001"))
    assert [(c, k) for c, k, _ in blocks] == [("Healthy", 0), ("Endodontics", 0), ("Healthy", 1), ("Healthy", 2)]
    assert blocks[1][2] == (30, 10, 50, 60)
    assert list(lb.iter_blocks(j, i, "P9999")) == []

    index = lb.block_index(j, i, "P0001")
    assert index[("Healthy", 2)] == ((150, 10, 170, 60), 47, 200, 100)
    assert lb.block_codes(j, "P0001") == {k: v[1] for k, v in index.items()}

    index5 = lb.block_index5(j, i, "P0001")
    assert {k: v for k, v in index5.items() if k in index} == index
    assert index5[("Restored", 0)][1] == 26
    assert index5[("DevelopingM3", 0)][1] == 38


@pytest.mark.parametrize("code, expected", [(18, "3rd molar"), (47, "2nd molar"), (11, "incisor"), (None, "unknown")])
def test_tooth_type_uses_the_fdi_unit_digit(code, expected):
    assert tooth_type(code) == expected


def test_fine_tooth_types():
    assert tooth_type(12, TOOTH_TYPES_FINE) == "lateral incisor"
