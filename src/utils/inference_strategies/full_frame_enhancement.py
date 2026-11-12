"""Enhance full NISAR frequency A HHHH frames in memory-bounded chunks."""

import time
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np

from utils.inference_strategies.black_tophat import apply_black_tophat
from utils.inference_strategies.unsharp_mask import apply_unsharp_mask

GRID = "science/LSAR/GCOV/grids/frequencyA"
STORAGE_CHUNK_SIZE = 512
PROGRESS_INTERVAL = 10
EXPECTED_NDIM = 2


def windows(shape: tuple[int, int], size: int) -> Iterator[tuple[int, int, int, int]]:
    """Yield row and column bounds for each chunk, including edge chunks."""
    for y in range(0, shape[0], size):
        for x in range(0, shape[1], size):
            yield y, min(y + size, shape[0]), x, min(x + size, shape[1])


def read_db(
    grid: h5py.Group, rows: slice, cols: slice
) -> tuple[np.ndarray, np.ndarray]:
    """Read positive, finite, unflagged backscatter and convert it to dB."""
    values = grid["HHHH"][rows, cols]
    valid = np.isfinite(values) & (values > 0)
    if "inputDataExceptionMask" in grid:
        valid &= grid["inputDataExceptionMask"][rows, cols] == 0
    db = np.full(values.shape, np.nan, dtype=np.float32)
    db[valid] = 10 * np.log10(values[valid])
    return db, valid


def enhance_frame(
    input_path: Path,
    output_path: Path,
    method: str = "black-tophat",
    chunk_size: int = 2048,
) -> None:
    """Write a derived enhanced HDF5 frame, preserving coordinates and invalid pixels.

    Uses a global approximate median to fill invalid pixels during filtering and
    a 32-pixel overlap to avoid artificial chunk boundaries. Existing outputs
    and partial files are never intentionally overwritten.
    """
    if method not in {"black-tophat", "unsharp"}:
        raise ValueError("method must be black-tophat or unsharp")
    if chunk_size < STORAGE_CHUNK_SIZE or chunk_size % STORAGE_CHUNK_SIZE:
        raise ValueError("chunk-size must be a positive multiple of 512")
    if output_path.exists():
        raise FileExistsError("Output already exists; choose a new output path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(".partial")
    if partial.exists():
        raise FileExistsError("Partial output exists; inspect it before retrying")
    started = time.monotonic()
    with h5py.File(input_path, "r") as source:
        grid = source[GRID]
        shape = grid["HHHH"].shape
        if len(shape) != EXPECTED_NDIM or min(shape) == 0:
            raise ValueError("HHHH must be a nonempty 2D image")
        # A bounded 0.01 dB histogram; validate coverage rather than clip values.
        edges = np.linspace(-500, 500, 100001)
        histogram = np.zeros(100000, dtype=np.int64)
        valid_count = 0
        for y0, y1, x0, x1 in windows(shape, chunk_size):
            db, valid = read_db(grid, slice(y0, y1), slice(x0, x1))
            samples = db[valid]
            histogram += np.histogram(samples, bins=edges)[0]
            valid_count += int(valid.sum())
        if valid_count == 0 or histogram.sum() != valid_count:
            raise ValueError(
                "No valid pixels or histogram does not cover source values"
            )
        index = np.searchsorted(np.cumsum(histogram), (valid_count + 1) / 2)
        median = float((edges[index] + edges[index + 1]) / 2)
        print(
            f"Shape={shape}; valid={valid_count}; fill median={median:.3f} dB",
            flush=True,
        )
        with h5py.File(partial, "x") as output:
            dest = output.require_group(GRID)
            for name in [
                "xCoordinates",
                "yCoordinates",
                "xCoordinateSpacing",
                "yCoordinateSpacing",
                "projection",
            ]:
                source.copy(grid[name], dest, name=name, without_attrs=True)
                for key, value in grid[name].attrs.items():
                    if key not in {"DIMENSION_LIST", "REFERENCE_LIST", "CLASS", "NAME"}:
                        dest[name].attrs[key] = value
            for name in ["xCoordinates", "yCoordinates"]:
                dest[name].make_scale(name)
            image = dest.create_dataset(
                "HHHH",
                shape=shape,
                dtype="f4",
                chunks=tuple(min(STORAGE_CHUNK_SIZE, length) for length in shape),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fillvalue=np.nan,
            )
            image.dims[0].attach_scale(dest["yCoordinates"])
            image.dims[1].attach_scale(dest["xCoordinates"])
            image.attrs["units"] = "1"
            image.attrs["grid_mapping"] = "projection"
            image.attrs["_FillValue"] = np.float32(np.nan)
            image.attrs["description"] = (
                "Derived enhanced image; linear encoding of enhanced dB, not calibrated original backscatter"
            )
            output.attrs.update(
                source_file=str(input_path.resolve()),
                enhancement=method,
                radius=4,
                amount=1.0,
                halo=32,
                invalid_fill_db=median,
                median_histogram_bin_db=0.01,
                valid_pixels=valid_count,
            )
            total = ((shape[0] + chunk_size - 1) // chunk_size) * (
                (shape[1] + chunk_size - 1) // chunk_size
            )
            written_valid = 0
            for i, (y0, y1, x0, x1) in enumerate(windows(shape, chunk_size), 1):
                ay, by = max(0, y0 - 32), min(shape[0], y1 + 32)
                ax, bx = max(0, x0 - 32), min(shape[1], x1 + 32)
                db, valid = read_db(grid, slice(ay, by), slice(ax, bx))
                core = (slice(y0 - ay, y1 - ay), slice(x0 - ax, x1 - ax))
                core_valid = valid[core]
                if core_valid.any():
                    clean = np.where(valid, db, np.float32(median))
                    if method == "black-tophat":
                        enhanced = clean - apply_black_tophat(clean, radius=4)
                    else:
                        enhanced = apply_unsharp_mask(clean, radius=4, amount=1.0)
                    result = np.full(core_valid.shape, np.nan, dtype=np.float32)
                    with np.errstate(over="raise", under="raise", invalid="raise"):
                        result[core_valid] = np.power(
                            np.float32(10), enhanced[core][core_valid] / np.float32(10)
                        )
                    if not np.all(
                        np.isfinite(result[core_valid]) & (result[core_valid] > 0)
                    ):
                        raise ValueError(
                            "Enhanced values cannot be encoded as positive float32"
                        )
                    image[y0:y1, x0:x1] = result
                written_valid += int(core_valid.sum())
                if i % PROGRESS_INTERVAL == 0 or i == total:
                    print(
                        f"{i}/{total} chunks; elapsed {time.monotonic() - started:.0f}s",
                        flush=True,
                    )
            if written_valid != valid_count:
                raise ValueError("Validity count changed")
            output.attrs["complete"] = True
    partial.rename(output_path)
    print(f"Completed: {output_path}", flush=True)
