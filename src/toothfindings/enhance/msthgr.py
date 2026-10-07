"""MSTHGR enhancement of panoramic crops.

MSTHGR is a multi-scale white/black top-hat by geodesic reconstruction. For disk radii
``rad .. rad + iterations - 1`` the reconstructed white and black top-hats and their
scale-to-scale differences are computed; their maxima over scales are added (white) to and
subtracted (black) from the input, which sharpens small bright and dark structures such as
root canal fillings, implant threads and crowns.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _subtract(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """Return ``f1 - f2`` with negative values set to zero."""
    res = f1 - f2
    res[res < 0] = 0
    return res


def _top_hats(ip_arr: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the reconstructed white and black top-hats of ``ip_arr`` for a disk of radius ``r``."""
    from skimage.morphology import black_tophat, dilation, disk, erosion, reconstruction, white_tophat

    footprint = disk(r)
    wth = white_tophat(ip_arr, footprint=footprint)
    bth = black_tophat(ip_arr, footprint=footprint)
    opening_by_reconstruction = reconstruction(erosion(ip_arr, footprint=footprint), ip_arr, method="dilation")
    rwth = _subtract(ip_arr, opening_by_reconstruction)
    closing_by_reconstruction = reconstruction(dilation(ip_arr, footprint=footprint), ip_arr, method="erosion")
    rbth = _subtract(closing_by_reconstruction, ip_arr)
    rwth = reconstruction(rwth, wth, method="dilation")
    rbth = reconstruction(rbth, bth, method="dilation")
    return rwth, rbth


def apply_msthgr(image: Image.Image, rad: int = 1, iterations: int = 9) -> Image.Image:
    """Apply MSTHGR to an RGB image and return the enhanced grayscale image.

    Parameters
    ----------
    image : PIL.Image.Image
        RGB input; converted to gray as the mean of the three channels (truncated to uint8).
    rad : int
        Initial radius of the disk structuring element.
    iterations : int
        Number of scales (radii ``rad`` to ``rad + iterations - 1``); must be at least 2.

    Returns
    -------
    PIL.Image.Image
        Enhanced 8-bit grayscale (mode ``L``) image. A non-RGB input is returned unchanged
        after printing an error message.
    """
    from skimage import img_as_float

    ip_arr = np.array(image)
    if not (ip_arr.ndim == 3 and ip_arr.shape[2] == 3):
        print("ERROR: apply_msthgr: the input must be an RGB image with 3 channels")
        return image
    gray_array = np.mean(ip_arr, axis=2).astype(np.uint8)
    ip_arr = img_as_float(gray_array)

    # Top-hats per scale and their differences between consecutive scales.
    wths, bths, d_wths, d_bths = [], [], [], []
    r = rad
    rwth, rbth = _top_hats(ip_arr, r)
    wths.append(rwth)
    bths.append(rbth)
    for k in range(1, iterations):
        r += 1
        rwth, rbth = _top_hats(ip_arr, r)
        wths.append(rwth)
        bths.append(rbth)
        d_wths.append(_subtract(wths[k], wths[k - 1]))
        d_bths.append(_subtract(bths[k], bths[k - 1]))

    # Pixel-wise maxima over scales.
    m_wth, m_bth = wths[0], bths[0]
    m_dwth, m_dbth = d_wths[0], d_bths[0]
    for w in range(1, iterations):
        m_wth = np.maximum(wths[w], m_wth)
        m_bth = np.maximum(bths[w], m_bth)
    for w in range(1, iterations - 1):
        m_dwth = np.maximum(d_wths[w], m_dwth)
        m_dbth = np.maximum(d_bths[w], m_dbth)

    enhanced = np.clip(ip_arr + m_wth + m_dwth - m_bth - m_dbth, 0, 255)
    arr8 = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr8)
