"""Load and preprocess NISAR GCOV image tiles."""

from pathlib import Path

import h5py
import numpy as np

GRID_PATH = "science/LSAR/GCOV/grids/frequencyA"
DATASET_PATH = f"{GRID_PATH}/HHHH"
MASK_PATH = f"{GRID_PATH}/inputDataExceptionMask"

TILE_SIZE = 4096


def load_nisar_tile(
    file_path: Path,
    row: int,
    col: int,
    tile_size: int = TILE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a NISAR tile and convert valid backscatter values to dB."""
    if row < 0 or col < 0:
        raise ValueError("Tile coordinates must be non-negative.")

    if tile_size < 1:
        raise ValueError("Tile size must be positive.")

    row_start = row * tile_size
    col_start = col * tile_size

    with h5py.File(file_path, "r") as f:
        dataset = f[DATASET_PATH]
        height, width = dataset.shape

        if row_start >= height or col_start >= width:
            raise ValueError(f"Tile ({row}, {col}) is outside image bounds.")

        row_end = min(row_start + tile_size, height)
        col_end = min(col_start + tile_size, width)

        image = dataset[row_start:row_end, col_start:col_end]

        valid = np.isfinite(image) & (image > 0)

        if MASK_PATH in f:
            exception = f[MASK_PATH][
                row_start:row_end,
                col_start:col_end,
            ]
            valid &= exception == 0

    image_db = np.full(image.shape, np.nan, dtype=np.float32)

    if np.any(valid):
        image_db[valid] = 10 * np.log10(image[valid])

    return image_db, valid
