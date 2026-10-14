
"""Apply morphological black top-hat enhancement to five NISAR images."""

from pathlib import Path
import sys
from utils.inference_strategies.black_tophat import apply_black_tophat
import h5py
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# Import the reusable black top-hat function
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BOX_ROOT = Path.home() / "Library/CloudStorage/Box-Box/2026-glaciers"

DATA_DIR = BOX_ROOT / "nisar_data/nisar_gcov_pig"
OUTPUT_DIR = BOX_ROOT / "black_tophat_augmentation"

DATASET_PATH = "science/LSAR/GCOV/grids/frequencyB/HHHH"

NUM_IMAGES = 5
RADIUS = 5

# Crop coordinates (1000 x 1000 pixels)
ROW_START, ROW_END = 2500, 3500
COL_START, COL_END = 1000, 2000


# --------------------------------------------------
# Image processing
# --------------------------------------------------

def process_image(file_path, output_dir):
    """Apply black top-hat enhancement to one NISAR image."""

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {file_path.name}")

    # Load radar image
    with h5py.File(file_path, "r") as f:
        dataset = f[DATASET_PATH]

        image = dataset[
            ROW_START:ROW_END,
            COL_START:COL_END
        ]

    # Identify valid positive backscatter values
    valid = np.isfinite(image) & (image > 0)

    if not np.any(valid):
        print("Skipping: no valid pixels found.")
        return False

    # Convert radar backscatter to decibels
    image_db = np.full(image.shape, np.nan, dtype=np.float32)
    image_db[valid] = 10 * np.log10(image[valid])

    # Fill invalid pixels temporarily for filtering
    median = np.nanmedian(image_db)
    image_clean = np.where(valid, image_db, median)

    # Apply morphological black top-hat
    response = apply_black_tophat(
        image_clean,
        radius=RADIUS
    )

    # Enhance dark features in the original image
    enhanced = image_clean - response

    # Restore invalid pixels
    response[~valid] = np.nan
    enhanced[~valid] = np.nan

    # Original and enhanced images share the same contrast scale
    vmin, vmax = np.percentile(image_db[valid], [2, 98])

    # Contrast scale for black top-hat response
    response_max = np.percentile(response[valid], 99)
    if response_max <= 0:
        response_max = 1.0

    # Save results
    images = {
        "original.png": (image_db, vmin, vmax),
        "black_tophat_response.png": (response, 0, response_max),
        "black_tophat_enhanced.png": (enhanced, vmin, vmax),
    }

    for filename, (img, low, high) in images.items():
        save_path = output_dir / filename

        plt.imsave(
            save_path,
            img,
            cmap="gray",
            vmin=low,
            vmax=high
        )

        print(f"  Saved: {filename}")

    return True

def is_available_locally(file_path):
    """Check whether a Box file is fully downloaded on macOS."""
    result = subprocess.run(
        ["ls", "-lO", str(file_path)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return False

    # macOS marks some cloud-only files as dataless.
    return "dataless" not in result.stdout.lower()

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    """Process up to five locally available NISAR images."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("*.h5"))

    print(f"Found {len(files)} NISAR files.")
    print("Checking which files are available locally...")

    successful = 0

    for file_path in files:
        if successful >= NUM_IMAGES:
            break

        if not is_available_locally(file_path):
            print(f"Skipping cloud-only file: {file_path.name}")
            continue

        image_output_dir = OUTPUT_DIR / f"image_{successful + 1:02d}"

        print(f"\n--- Image {successful + 1}/{NUM_IMAGES} ---")

        if process_image(file_path, image_output_dir):
            successful += 1

    print(f"\nFinished! Processed {successful} locally available images.")

if __name__ == "__main__":
    main()