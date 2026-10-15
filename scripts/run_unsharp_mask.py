"""Apply unsharp masking to selected NISAR tiles."""

# ruff: noqa: E402

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.inference_strategies.unsharp_mask import (
    apply_unsharp_mask,  # pylint: disable=import-error,wrong-import-position
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BOX_ROOT = Path.home() / "Library/CloudStorage/Box-Box/2026-glaciers"

DATA_DIR = BOX_ROOT / "nisar_data/nisar_gcov_pig"
OUTPUT_DIR = BOX_ROOT / "unsharp_mask_augmentation"

GRID_PATH = "science/LSAR/GCOV/grids/frequencyA"
DATASET_PATH = f"{GRID_PATH}/HHHH"
MASK_PATH = f"{GRID_PATH}/inputDataExceptionMask"

TILE_SIZE = 4096
TILES = [(4, 2), (5, 3)]

RADIUS = 4
AMOUNT = 1.0


# --------------------------------------------------
# Processing
# --------------------------------------------------


def process_tile(
    file_path: Path,
    row: int,
    col: int,
) -> None:
    """Process and save one unsharp masking tile."""
    tile_dir = OUTPUT_DIR / f"tile_{row}_{col}"
    tile_dir.mkdir(parents=True, exist_ok=True)

    row_start = row * TILE_SIZE
    col_start = col * TILE_SIZE

    print(f"Processing tile ({row}, {col})...", flush=True)

    with h5py.File(file_path, "r") as f:
        dataset = f[DATASET_PATH]

        height, width = dataset.shape

        if row_start >= height or col_start >= width:
            raise ValueError(f"Tile ({row}, {col}) is outside image bounds.")

        row_end = min(row_start + TILE_SIZE, height)
        col_end = min(col_start + TILE_SIZE, width)

        image = dataset[row_start:row_end, col_start:col_end]

        exception = None

        if MASK_PATH in f:
            exception = f[MASK_PATH][row_start:row_end, col_start:col_end]

    # Identify valid pixels
    valid = np.isfinite(image) & (image > 0)

    if exception is not None:
        valid &= exception == 0

    if not np.any(valid):
        print(f"Skipping tile ({row}, {col}): no valid pixels.")
        return

    # Convert backscatter to decibels
    image_db = np.full(image.shape, np.nan, dtype=np.float32)
    image_db[valid] = 10 * np.log10(image[valid])

    # Fill invalid pixels temporarily
    median = np.nanmedian(image_db)
    image_clean = np.where(valid, image_db, median)

    # Apply unsharp masking
    enhanced = apply_unsharp_mask(
        image_clean,
        radius=RADIUS,
        amount=AMOUNT,
    )

    # Restore invalid pixels
    enhanced[~valid] = np.nan

    # Consistent contrast for original and enhanced
    vmin, vmax = np.percentile(image_db[valid], [1, 99])

    # Save original
    plt.imsave(
        tile_dir / "original.png",
        image_db,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    # Save enhanced
    plt.imsave(
        tile_dir / "unsharp_mask_enhanced.png",
        enhanced,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    # Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(image_db, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(enhanced, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Unsharp Mask (Radius={RADIUS}, Amount={AMOUNT})")
    axes[1].axis("off")

    plt.tight_layout()

    fig.savefig(
        tile_dir / "comparison.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved tile ({row}, {col}) to {tile_dir}", flush=True)


def main() -> None:
    """Run unsharp masking on selected tiles."""
    files = sorted(DATA_DIR.glob("*.h5"))

    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {DATA_DIR}")

    file_path = files[0]

    print(f"Using NISAR file: {file_path.name}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for row, col in TILES:
        process_tile(file_path, row, col)

    print("Unsharp masking processing complete!")


if __name__ == "__main__":
    main()
