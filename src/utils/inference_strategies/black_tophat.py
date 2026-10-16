"""Morphological black top-hat enhancement for grayscale images."""

import numpy as np
from skimage.morphology import black_tophat, disk

EXPECTED_NDIM = 2


def apply_black_tophat(image: np.ndarray, radius: int = 5) -> np.ndarray:
    """Highlight dark features using morphological black top-hat.

    Args:
        image: A 2D grayscale image.
        radius: Radius of the disk-shaped structuring element.

    Returns:
        Black top-hat response image.
    """
    if image.ndim != EXPECTED_NDIM:
        raise ValueError("Input image must be 2D.")

    if radius < 1:
        raise ValueError("Radius must be positive.")

    return black_tophat(image, footprint=disk(radius))
