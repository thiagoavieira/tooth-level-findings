import numpy as np
import pytest
from PIL import Image

pytest.importorskip("skimage")
pytest.importorskip("cv2")

from toothfindings.enhance.clahe import VARIANTS, clahe, enhance_variants  # noqa: E402
from toothfindings.enhance.msthgr import apply_msthgr  # noqa: E402


@pytest.fixture
def crop():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (48, 32, 3), dtype=np.uint8))


def test_msthgr_returns_an_8_bit_gray_image_of_the_same_size(crop):
    out = apply_msthgr(crop, iterations=3)
    assert out.mode == "L" and out.size == crop.size


def test_msthgr_leaves_a_flat_image_flat():
    flat = Image.new("RGB", (24, 24), (90, 90, 90))
    # the conversion back to 8 bit truncates: 90 / 255 * 255 -> 89
    assert np.array_equal(np.array(apply_msthgr(flat, iterations=3)), np.full((24, 24), 89))


def test_msthgr_returns_non_rgb_input_unchanged():
    gray = Image.new("L", (8, 8))
    assert apply_msthgr(gray) is gray


def test_clahe_keeps_shape_and_dtype(crop):
    g = np.array(crop.convert("L"))
    out = clahe(g)
    assert out.shape == g.shape and out.dtype == np.uint8


def test_enhancement_variants_are_gray_replicated_over_three_channels(crop):
    out = enhance_variants(crop, VARIANTS)
    assert list(out) == VARIANTS
    for arr in out.values():
        assert arr.shape == (48, 32, 3) and arr.dtype == np.uint8
        assert (arr == arr[..., :1]).all()
    assert np.array_equal(out["none"][..., 0], np.array(crop.convert("L")))
    assert set(enhance_variants(crop, ["msthgr"])) == {"msthgr"}
