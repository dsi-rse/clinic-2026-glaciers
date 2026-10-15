"""Image augmentations that may make crevasses easier to see in NISAR tiles.

Both functions take a tile as uint8 gray levels (0 = nodata, 1..255 = backscatter) and return
a float image between 0 and 1, with nodata pixels set back to 0.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.exposure import equalize_adapthist

NODATA = 0
LOW_PERCENTILE = 1
HIGH_PERCENTILE = 99


def clahe(tile: np.ndarray, clip_limit: float = 0.01) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Boosts local contrast, so faint features get brighter against their surroundings.

    Args:
        tile: uint8 tile, 0 = nodata.
        clip_limit: Higher values give more contrast (skimage default is 0.01).

    Returns:
        Float image between 0 and 1, nodata set to 0.
    """
    result = equalize_adapthist(tile, clip_limit=clip_limit)
    result[tile == NODATA] = 0
    return result


def high_pass(tile: np.ndarray, sigma: float = 50) -> np.ndarray:
    """High-pass filter: the tile minus a blurred copy of itself.

    The blur keeps only slow, large-scale brightness changes. Subtracting it removes them and
    leaves small features such as crevasses.

    Args:
        tile: uint8 tile, 0 = nodata.
        sigma: Blur size in pixels. Features much smaller than this are kept.

    Returns:
        Float image between 0 and 1 (stretched from the 1st to 99th percentile of valid
        pixels), nodata set to 0.
    """
    valid = tile != NODATA
    image = tile.astype(float)
    difference = image - gaussian_filter(image, sigma)
    low, high = np.percentile(difference[valid], [LOW_PERCENTILE, HIGH_PERCENTILE])
    result = np.clip((difference - low) / (high - low), 0, 1)
    result[~valid] = 0
    return result
