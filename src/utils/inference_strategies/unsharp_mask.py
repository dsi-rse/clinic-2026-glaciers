
"""Unsharp masking enhancement for grayscale images."""

import numpy as np
from skimage.filters import unsharp_mask


def apply_unsharp_mask(
    image: np.ndarray,
    radius: float = 1.0,
    amount: float = 1.0,
) -> np.ndarray:
    """Sharpen image details using unsharp masking."""

    if image.ndim != 2:
        raise ValueError("Input image must be 2D.")

    if radius <= 0:
        raise ValueError("Radius must be positive.")

    if amount < 0:
        raise ValueError("Amount cannot be negative.")

    return unsharp_mask(
        image,
        radius=radius,
        amount=amount,
        preserve_range=True,
    )
