"""Integration tests for NISAR enhancement scripts."""

from pathlib import Path
from types import ModuleType

import h5py
import matplotlib
import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch

from scripts import run_black_tophat, run_unsharp_mask
from utils.inference_strategies.nisar_preprocessing import (
    DATASET_PATH,
    MASK_PATH,
    load_nisar_tile,
)

matplotlib.use("Agg")

RUNNERS = [run_black_tophat, run_unsharp_mask]


@pytest.fixture
def synthetic_nisar_file(tmp_path: Path) -> Path:
    """Create a small NISAR-like HDF5 file with invalid pixels."""
    file_path = tmp_path / "synthetic_nisar.h5"

    image = np.full((12, 12), 10.0, dtype=np.float32)
    image[4:8, 5] = 1.0
    image[0, 0] = 0.0
    image[0, 1] = np.nan

    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[0, 2] = 1

    with h5py.File(file_path, "w") as f:
        f.create_dataset(DATASET_PATH, data=image)
        f.create_dataset(MASK_PATH, data=mask)

    return file_path


@pytest.fixture
def all_invalid_file(tmp_path: Path) -> Path:
    """Create an HDF5 file containing no valid backscatter."""
    file_path = tmp_path / "all_invalid.h5"

    with h5py.File(file_path, "w") as f:
        f.create_dataset(
            DATASET_PATH,
            data=np.zeros((12, 12), dtype=np.float32),
        )

    return file_path


def test_all_invalid_tile_loading(all_invalid_file: Path) -> None:
    """An all-invalid tile should contain only NaNs."""
    image_db, valid = load_nisar_tile(all_invalid_file, row=0, col=0, tile_size=12)

    assert not np.any(valid)
    assert np.all(np.isnan(image_db))


@pytest.mark.parametrize("runner", RUNNERS)
def test_all_invalid_tile_is_skipped(
    runner: ModuleType,
    all_invalid_file: Path,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Neither enhancement should save an all-invalid tile."""
    output_dir = tmp_path / runner.__name__
    monkeypatch.setattr(runner, "OUTPUT_DIR", output_dir)

    runner.process_tile(all_invalid_file, 0, 0)

    assert not (output_dir / "tile_0_0").exists()


@pytest.mark.parametrize(
    ("runner", "enhanced_filename"),
    [
        (run_black_tophat, "black_tophat_enhanced.png"),
        (run_unsharp_mask, "unsharp_mask_enhanced.png"),
    ],
)
def test_output_files_created(
    runner: ModuleType,
    enhanced_filename: str,
    synthetic_nisar_file: Path,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Each script should save its expected PNG files."""
    output_dir = tmp_path / runner.__name__
    monkeypatch.setattr(runner, "OUTPUT_DIR", output_dir)

    # Both runners call the shared loader with its default tile size.
    # Tile (0, 0) still loads the entire small synthetic image.
    runner.process_tile(synthetic_nisar_file, 0, 0)

    tile_dir = output_dir / "tile_0_0"

    assert (tile_dir / "original.png").is_file()
    assert (tile_dir / enhanced_filename).is_file()
    assert (tile_dir / "comparison.png").is_file()

    if runner is run_black_tophat:
        assert (tile_dir / "black_tophat_response.png").is_file()


@pytest.mark.parametrize(
    ("runner", "function_name"),
    [
        (run_black_tophat, "apply_black_tophat"),
        (run_unsharp_mask, "apply_unsharp_mask"),
    ],
)
def test_invalid_pixels_restored(
    runner: ModuleType,
    function_name: str,
    synthetic_nisar_file: Path,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Enhanced arrays should restore NaNs at invalid pixels."""
    output_dir = tmp_path / runner.__name__
    monkeypatch.setattr(runner, "OUTPUT_DIR", output_dir)

    _, valid = load_nisar_tile(synthetic_nisar_file, row=0, col=0, tile_size=12)
    captured = {}

    def capture_enhanced(
        path: str | Path,
        image: np.ndarray,
        **kwargs: object,
    ) -> None:
        if "enhanced" in str(path):
            captured["image"] = np.asarray(image).copy()

    # Capture the array passed to the enhanced PNG writer.
    monkeypatch.setattr(runner.plt, "imsave", capture_enhanced)

    runner.process_tile(synthetic_nisar_file, 0, 0)

    assert "image" in captured
    assert np.all(np.isnan(captured["image"][~valid]))
    assert np.all(np.isfinite(captured["image"][valid]))
