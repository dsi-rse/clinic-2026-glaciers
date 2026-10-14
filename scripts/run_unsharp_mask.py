
"""Apply unsharp masking to one NISAR satellite image."""

from pathlib import Path
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Import our unsharp masking function
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from utils.inference_strategies.unsharp_mask import apply_unsharp_mask  # pylint: disable=import-error



# --------------------------------------------------
# Configuration
# --------------------------------------------------

BOX_ROOT = Path.home() / "Library/CloudStorage/Box-Box/2026-glaciers"

DATA_DIR = BOX_ROOT / "nisar_data/nisar_gcov_pig"
OUTPUT_DIR = BOX_ROOT / "unsharp_mask_augmentation"

DATASET_PATH = "science/LSAR/GCOV/grids/frequencyB/HHHH"

RADIUS = 3
AMOUNT = 1.0

# Same 1000 x 1000 pixel crop as black top-hat
ROW_START, ROW_END = 2500, 3500
COL_START, COL_END = 1000, 2000


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    """Apply unsharp masking to the first NISAR image."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("*.h5"))

    if not files:
        raise FileNotFoundError(
            f"No HDF5 files found in {DATA_DIR}"
        )

    # Process only the first file
    file_path = files[0]

    print(f"Processing: {file_path.name}", flush=True)

    # Load the image crop
    with h5py.File(file_path, "r") as f:
        dataset = f[DATASET_PATH]
        image = np.asarray(
            dataset[ROW_START:ROW_END, COL_START:COL_END]
    )

    # Identify valid radar backscatter pixels
    valid = np.isfinite(image) & (image > 0)

    if not np.any(valid):
        raise ValueError("No valid pixels found in the crop.")

    # Convert backscatter to decibels
    image_db = np.full(image.shape, np.nan, dtype=np.float32)
    image_db[valid] = 10 * np.log10(image[valid])

    # Temporarily fill invalid pixels for filtering
    median = np.nanmedian(image_db)
    image_clean = np.where(valid, image_db, median)

    # Apply unsharp masking
    enhanced = apply_unsharp_mask(
        image_clean,
        radius=RADIUS,
        amount=AMOUNT
    )

    # Restore invalid pixels
    enhanced[~valid] = np.nan

    # Use the same display scale for comparison
    vmin, vmax = np.percentile(image_db[valid], [2, 98])

    # Save original
    plt.imsave(
        OUTPUT_DIR / "original.png",
        image_db,
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )

    # Save enhanced image
    plt.imsave(
        OUTPUT_DIR / "unsharp_mask_enhanced.png",
        enhanced,
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )

    # Save side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(
        image_db,
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(
        enhanced,
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )
    axes[1].set_title(
        f"Unsharp Mask (Radius={RADIUS}, Amount={AMOUNT})"
    )
    axes[1].axis("off")

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "comparison.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Finished! Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
