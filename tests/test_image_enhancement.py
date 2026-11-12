"""Regression tests for image enhancement methods."""

from collections.abc import Callable

import numpy as np
import pytest

from utils.inference_strategies.black_tophat import apply_black_tophat
from utils.inference_strategies.unsharp_mask import apply_unsharp_mask


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a small grayscale test image."""
    return np.ones((10, 10), dtype=np.float32)


def test_black_tophat_output_shape(sample_image: np.ndarray) -> None:
    """Black top-hat should preserve image dimensions."""
    result = apply_black_tophat(sample_image, radius=2)

    assert result.shape == sample_image.shape


def test_unsharp_mask_output_shape(sample_image: np.ndarray) -> None:
    """Unsharp masking should preserve image dimensions."""
    result = apply_unsharp_mask(sample_image, radius=2, amount=1)

    assert result.shape == sample_image.shape


@pytest.mark.parametrize("radius", [0, -1])
def test_black_tophat_invalid_radius(sample_image: np.ndarray, radius: int) -> None:
    """Black top-hat should reject invalid radii."""
    with pytest.raises(ValueError, match="Radius must be positive"):
        apply_black_tophat(sample_image, radius=radius)


@pytest.mark.parametrize("radius", [0, -1])
def test_unsharp_mask_invalid_radius(sample_image: np.ndarray, radius: int) -> None:
    """Unsharp masking should reject invalid radii."""
    with pytest.raises(ValueError, match="Radius must be positive"):
        apply_unsharp_mask(sample_image, radius=radius, amount=1)


def test_unsharp_mask_negative_amount(sample_image: np.ndarray) -> None:
    """Unsharp masking should reject negative amounts."""
    with pytest.raises(ValueError, match="Amount cannot be negative"):
        apply_unsharp_mask(sample_image, radius=2, amount=-1)


@pytest.mark.parametrize(
    "enhancement",
    [apply_black_tophat, apply_unsharp_mask],
)
def test_invalid_image_dimensions(
    enhancement: Callable[..., np.ndarray],
) -> None:
    """Both methods should reject non-2D images."""
    image = np.ones((10, 10, 3))

    with pytest.raises(ValueError, match="Input image must be 2D"):
        enhancement(image)
