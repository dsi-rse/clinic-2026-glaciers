"""Tests for NISAR HDF5 loading and preprocessing."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from utils.inference_strategies.nisar_preprocessing import (
    DATASET_PATH,
    MASK_PATH,
    load_nisar_tile,
)


@pytest.fixture
def synthetic_nisar_file(tmp_path: Path) -> Path:
    """Create a small synthetic NISAR HDF5 file."""
    file_path = tmp_path / "synthetic_nisar.h5"

    image = np.array(
        [
            [1.0, 10.0, 100.0],
            [0.0, -1.0, np.nan],
            [5.0, np.inf, 20.0],
        ],
        dtype=np.float32,
    )

    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[2, 2] = 1

    with h5py.File(file_path, "w") as f:
        f.create_dataset(DATASET_PATH, data=image)
        f.create_dataset(MASK_PATH, data=mask)

    return file_path


def test_valid_backscatter_to_db(synthetic_nisar_file: Path) -> None:
    """Positive finite unmasked values should convert to dB."""
    image_db, valid = load_nisar_tile(
        synthetic_nisar_file,
        row=0,
        col=0,
        tile_size=3,
    )

    assert valid[0, 0]
    assert valid[0, 1]
    assert valid[0, 2]

    np.testing.assert_allclose(
        image_db[0],
        [0.0, 10.0, 20.0],
        atol=1e-5,
    )


def test_invalid_pixels_are_nan(synthetic_nisar_file: Path) -> None:
    """Invalid backscatter and masked pixels should become NaN."""
    image_db, valid = load_nisar_tile(
        synthetic_nisar_file,
        row=0,
        col=0,
        tile_size=3,
    )

    assert not valid[1, 0]  # Zero
    assert not valid[1, 1]  # Negative
    assert not valid[1, 2]  # NaN
    assert not valid[2, 1]  # Infinity
    assert not valid[2, 2]  # Exception mask

    assert np.all(np.isnan(image_db[~valid]))


@pytest.mark.parametrize(
    ("row", "col"),
    [(-1, 0), (0, -1), (1, 0), (0, 1)],
)
def test_invalid_tile_coordinates(
    synthetic_nisar_file: Path,
    row: int,
    col: int,
) -> None:
    """Reject negative or out-of-bounds tile coordinates."""
    with pytest.raises(ValueError):
        load_nisar_tile(
            synthetic_nisar_file,
            row=row,
            col=col,
            tile_size=3,
        )


def test_invalid_tile_size(synthetic_nisar_file: Path) -> None:
    """Reject non-positive tile sizes."""
    with pytest.raises(ValueError, match="Tile size must be positive"):
        load_nisar_tile(
            synthetic_nisar_file,
            row=0,
            col=0,
            tile_size=0,
        )
