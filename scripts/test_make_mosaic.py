"""Tests for make_mosaic.py, on tiny synthetic arrays and frames."""

from pathlib import Path

import h5py
import make_mosaic
import numpy as np
import pytest
import rasterio

FACTOR = 2
SIZE = 8
PIXEL = 10.0
RESOLUTION = PIXEL * FACTOR
EPSG = 3031
FREQUENCY = "frequencyA"
ABSURD = 1e6  # a value that would wreck any mean it leaked into
POWER_MEAN = 25.75  # mean of 1, 1, 1 and 100


def test_average_is_taken_in_linear_power() -> None:
    """Values are averaged as power and then converted to dB, not averaged in dB."""
    block = np.array([[1, 1], [1, 100]], dtype=np.float32)
    in_db = make_mosaic.block_mean(block, FACTOR, linear=False)
    in_power = make_mosaic.block_mean(block, FACTOR, linear=True)
    assert in_power.shape == (1, 1)
    assert in_power[0, 0] == pytest.approx(POWER_MEAN)
    assert in_db[0, 0] == pytest.approx(10 * np.log10(POWER_MEAN))


def test_nodata_is_skipped() -> None:
    """NaN and non-positive pixels do not count towards the mean."""
    block = np.array([[4, np.nan], [0, 4]], dtype=np.float32)
    result = make_mosaic.block_mean(block, FACTOR, linear=True)
    assert result[0, 0] == pytest.approx(4.0)


def test_block_with_too_little_data_is_nodata() -> None:
    """One valid pixel out of four is below the minimum valid fraction."""
    block = np.array([[4, np.nan], [0, np.nan]], dtype=np.float32)
    assert np.isnan(make_mosaic.block_mean(block, FACTOR, linear=True)[0, 0])


def test_flagged_pixels_are_excluded() -> None:
    """A flagged pixel with an absurd value does not affect the mean."""
    block = np.array([[1, 1], [1, ABSURD]], dtype=np.float32)
    exception = np.array([[0, 0], [0, 2]], dtype=np.uint8)
    masked = make_mosaic.block_mean(block, FACTOR, linear=True, exception=exception)
    unmasked = make_mosaic.block_mean(block, FACTOR, linear=True)
    assert masked[0, 0] == pytest.approx(1.0)
    assert unmasked[0, 0] == pytest.approx((3 + ABSURD) / 4)


def test_mostly_or_fully_flagged_blocks_are_nodata() -> None:
    """Flagged pixels count as invalid when deciding whether a block has enough data."""
    block = np.ones((2, 2), dtype=np.float32)
    mostly = np.array([[0, 2], [2, 2]], dtype=np.uint8)
    fully = np.full((2, 2), 2, dtype=np.uint8)
    for exception in (mostly, fully):
        result = make_mosaic.block_mean(block, FACTOR, linear=True, exception=exception)
        assert np.isnan(result[0, 0])


def test_blocks_are_masked_independently() -> None:
    """A fully flagged block does not affect its clean neighbour."""
    block = np.ones((2, 4), dtype=np.float32)
    exception = np.zeros((2, 4), dtype=np.uint8)
    exception[:, :2] = 2
    result = make_mosaic.block_mean(block, FACTOR, linear=True, exception=exception)
    assert result.shape == (1, 2)
    assert np.isnan(result[0, 0])
    assert result[0, 1] == pytest.approx(1.0)


def write_frame(path: Path) -> None:
    """An 8 x 8 frame of ones with a fully flagged block and a partly flagged one."""
    values = np.ones((SIZE, SIZE), dtype=np.float32)
    exception = np.zeros((SIZE, SIZE), dtype=np.uint8)
    values[:2, :2] = ABSURD  # output pixel (0, 0): all four inputs flagged
    exception[:2, :2] = 2
    values[0, 2] = ABSURD  # output pixel (0, 1): one of four inputs flagged
    exception[0, 2] = 2
    with h5py.File(path, "w") as handle:
        grid = handle.create_group(make_mosaic.GRID_GROUP.format(frequency=FREQUENCY))
        grid.create_dataset("HHHH", data=values)
        grid.create_dataset(make_mosaic.EXCEPTION_MASK, data=exception)
        grid.create_dataset("xCoordinates", data=PIXEL / 2 + PIXEL * np.arange(SIZE))
        grid.create_dataset(
            "yCoordinates", data=SIZE * PIXEL - PIXEL / 2 - PIXEL * np.arange(SIZE)
        )
        projection = grid.create_dataset("projection", data=np.uint32(EPSG))
        projection.attrs["epsg_code"] = EPSG


def test_decimate_frame_masks_and_georeferences(tmp_path: Path) -> None:
    """A frame becomes a half-size GeoTIFF in the right place, with flagged pixels excluded."""
    source = tmp_path / "frame_20260723T015731_001.h5"
    write_frame(source)
    settings = make_mosaic.Settings("HHHH", FREQUENCY, RESOLUTION, linear=True)
    out = tmp_path / "out" / "frame.tif"

    result = make_mosaic.decimate_frame(source, out, settings)

    assert result.date == "20260723"
    assert result.crs.to_epsg() == EPSG
    with rasterio.open(out) as tif:
        data = tif.read(1)
        assert tif.crs.to_epsg() == EPSG
        assert tuple(tif.transform)[:6] == pytest.approx(
            (RESOLUTION, 0, 0, 0, -RESOLUTION, SIZE * PIXEL)
        )
    assert data.shape == (SIZE // FACTOR, SIZE // FACTOR)
    assert np.isnan(data[0, 0])
    keep = np.ones(data.shape, dtype=bool)
    keep[0, 0] = False
    np.testing.assert_allclose(data[keep], 1.0)
