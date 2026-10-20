"""Regression tests for the image augmentations (nodata and constant tiles)."""

import numpy as np

from utils.augmentation import NODATA, clahe, high_pass

SIZE = 64
SIGMA = 5


def check_output(tile: np.ndarray, result: np.ndarray) -> None:
    """Output must be finite, the same shape, in [0, 1], with nodata kept as 0."""
    assert result.shape == tile.shape
    assert np.isfinite(result).all()
    assert result.min() >= 0
    assert result.max() <= 1
    assert (result[tile == NODATA] == 0).all()


def test_high_pass_all_nodata() -> None:
    tile = np.zeros((SIZE, SIZE), dtype=np.uint8)
    result = high_pass(tile, SIGMA)
    check_output(tile, result)
    assert (result == 0).all()


def test_high_pass_constant_valid_tile() -> None:
    tile = np.full((SIZE, SIZE), 100, dtype=np.uint8)
    check_output(tile, high_pass(tile, SIGMA))


def test_high_pass_constant_tile_next_to_nodata() -> None:
    tile = np.full((SIZE, SIZE), 100, dtype=np.uint8)
    tile[:, : SIZE // 2] = NODATA  # left half is nodata
    check_output(tile, high_pass(tile, SIGMA))


def test_high_pass_blur_ignores_nodata() -> None:
    """A flat valid area next to nodata must look flat, not like an edge."""
    tile = np.full((SIZE, SIZE), 100, dtype=np.uint8)
    tile[:, : SIZE // 2] = NODATA
    tile[10, 50] = 200  # one bright pixel so the tile is not constant
    result = high_pass(tile, SIGMA)
    check_output(tile, result)
    # Valid pixels at the nodata boundary and far from it should get the same value.
    assert result[40, SIZE // 2] == result[40, SIZE - 5]


def test_high_pass_random_tile_with_nodata() -> None:
    rng = np.random.default_rng(0)
    tile = rng.integers(1, 256, size=(SIZE, SIZE), dtype=np.uint8)
    tile[:10, :] = NODATA
    check_output(tile, high_pass(tile, SIGMA))


def test_clahe_keeps_nodata_zero() -> None:
    rng = np.random.default_rng(0)
    tile = rng.integers(1, 256, size=(SIZE, SIZE), dtype=np.uint8)
    tile[:10, :] = NODATA
    check_output(tile, clahe(tile))
