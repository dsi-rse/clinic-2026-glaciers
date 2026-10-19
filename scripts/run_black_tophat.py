"""Apply black top-hat enhancement to selected NISAR tiles."""

# ruff: noqa: E402
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.inference_strategies.nisar_preprocessing import load_nisar_tile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.inference_strategies.black_tophat import (
    apply_black_tophat,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_ROOT = Path(os.environ.get("DATA_DIR", "/data"))

DATA_DIR = DATA_ROOT / "nisar_data" / "nisar_gcov_pig"
OUTPUT_DIR = DATA_ROOT / "black_tophat_augmentation"

FRAME_NAME = (
    "NISAR_L2_PR_GCOV_024_077_A_140_4005_SHSH_A_"
    "20260701T135211_20260701T135247_P05023_N_F_J_001.h5"
)

GRID_PATH = "science/LSAR/GCOV/grids/frequencyA"
DATASET_PATH = f"{GRID_PATH}/HHHH"
MASK_PATH = f"{GRID_PATH}/inputDataExceptionMask"

TILE_SIZE = 4096
TILES = [(4, 2), (5, 3)]

RADIUS = 4


# --------------------------------------------------
# Processing
# --------------------------------------------------


def process_tile(
    file_path: Path,
    row: int,
    col: int,
) -> None:
    """Process and save one black top-hat tile."""
    print(f"Processing tile ({row}, {col})...", flush=True)

    # Load NISAR tile and convert valid backscatter values to dB
    image_db, valid = load_nisar_tile(file_path, row, col)

    # Skip tiles with no valid pixels
    if not np.any(valid):
        print(f"Skipping tile ({row}, {col}): no valid pixels.")
        return

    # Create output directory
    tile_dir = OUTPUT_DIR / f"tile_{row}_{col}"
    tile_dir.mkdir(parents=True, exist_ok=True)

    # Fill invalid pixels temporarily with the median
    median = np.nanmedian(image_db)
    image_clean = np.where(valid, image_db, median)

    # Apply morphological black top-hat
    response = apply_black_tophat(image_clean, radius=RADIUS)

    # Subtract response to enhance dark features
    enhanced = image_clean - response

    # Restore invalid pixels
    response[~valid] = np.nan
    enhanced[~valid] = np.nan

    # Use consistent contrast for original and enhanced images
    vmin, vmax = np.percentile(image_db[valid], [1, 99])

    # Determine contrast range for the response image
    response_max = np.percentile(response[valid], 99)

    if response_max <= 0:
        response_max = 1.0

    # Save original image
    plt.imsave(
        tile_dir / "original.png",
        image_db,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    # Save enhanced image
    plt.imsave(
        tile_dir / "black_tophat_enhanced.png",
        enhanced,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    # Save black top-hat response
    plt.imsave(
        tile_dir / "black_tophat_response.png",
        response,
        cmap="gray",
        vmin=0,
        vmax=response_max,
    )

    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(
        image_db,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(
        enhanced,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(f"Black Top-Hat (Radius={RADIUS})")
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
    """Run black top-hat enhancement on selected tiles."""
    file_path = DATA_DIR / FRAME_NAME

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required NISAR frame not found: {file_path}\n"
            "Make sure the file is downloaded and DATA_DIR is configured."
        )

    print(f"Using NISAR file: {file_path.name}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for row, col in TILES:
        process_tile(file_path, row, col)

    print("Black top-hat processing complete!")


if __name__ == "__main__":
    main()
