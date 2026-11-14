#!/usr/bin/env python3
r"""Stitch NISAR GCOV frames into a reduced-resolution GeoTIFF mosaic.

For each frame this reads the backscatter layer (default HHHH, frequencyA), averages
it down to a coarser pixel size in small blocks (so a 30000 x 30000 pixel frame never
has to fit in memory), converts to decibels, and saves a georeferenced GeoTIFF. The
per-frame GeoTIFFs are then merged into one mosaic you can open in QGIS.

Requires: h5py, numpy, rasterio (pip install h5py numpy rasterio).

Examples:
    python3 make_mosaic.py /mnt/Box/2026-glaciers/nisar_data --month 202607 --list
    python3 make_mosaic.py /mnt/Box/2026-glaciers/nisar_data --month 202607 \
        --resolution 200 --out-dir july_mosaic --name july_mosaic

Notes:
- --month YYYYMM keeps files whose name has an acquisition date in that month
  (for example 202607 matches ..._20260705T044409_...).
- Pixel size is in the units of the frame's projection (metres for UTM and polar
  stereographic). Smaller values make bigger files; 200 is plenty for finding a
  bounding box. The averaging is done on linear power, then converted to dB.
- Overlapping frames: --method decides which value wins (first, last, min, max).
- Coordinates in the file are treated as pixel centres. A half-pixel shift is
  negligible at the coarse pixel sizes used here.
- Frames in different projections are reprojected to the most common one (or --crs).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import DatasetWriter
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from rasterio.windows import Window

GRID_GROUP = "science/LSAR/GCOV/grids/{frequency}"
BLOCK_PIXELS = 16_000_000  # input pixels read at once (about 64 MB as float32)
MIN_VALID_FRACTION = (
    0.5  # an output pixel needs at least this share of valid input pixels
)
MAX_MOSAIC_PIXELS = 400_000_000  # refuse to merge anything bigger than this
MONTH_DIGITS = 6
DATE_RE = re.compile(r"_(\d{8})T\d{6}_")
MERGE_METHODS = ("first", "last", "min", "max")


@dataclass
class Settings:
    """Options that apply to every frame."""

    polarization: str
    frequency: str
    resolution: float
    linear: bool


@dataclass
class FrameResult:
    """What was learned about one frame, and where its GeoTIFF is."""

    source: Path
    date: str
    crs: CRS
    spacing: float
    tif: Path


def find_frames(inputs: list[str], month: str | None) -> list[Path]:
    """Return the .h5 files in the given files/folders, optionally for one month."""
    found: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            found.extend(sorted(path.rglob("*.h5")))
        elif path.is_file():
            found.append(path)
        else:
            sys.exit(f"Not a file or folder: {path}")
    if month:
        pattern = re.compile(rf"_{month}\d{{2}}T")
        found = [p for p in found if pattern.search(p.name)]
    return found


def frame_date(path: Path) -> str:
    """Return the acquisition date (YYYYMMDD) from a NISAR file name, or ''."""
    match = DATE_RE.search(path.name)
    return match.group(1) if match else ""


def _as_text(value: object) -> str:
    """Turn an HDF5 attribute (bytes, str, or array of either) into a string."""
    scalar = np.asarray(value).reshape(-1)[0]
    return scalar.decode() if isinstance(scalar, bytes) else str(scalar)


def read_crs(grid: h5py.Group) -> CRS:
    """Read the projection from a grid group (EPSG attribute, WKT, or dataset value)."""
    proj = grid["projection"]
    for key in ("epsg_code", "epsg"):
        if key in proj.attrs:
            return CRS.from_epsg(int(float(_as_text(proj.attrs[key]))))
    if "spatial_ref" in proj.attrs:
        return CRS.from_wkt(_as_text(proj.attrs["spatial_ref"]))
    return CRS.from_epsg(int(np.asarray(proj[()]).reshape(-1)[0]))


def block_mean(block: np.ndarray, factor: int, *, linear: bool) -> np.ndarray:
    """Average factor x factor pixel groups, skipping no-data; dB unless linear."""
    rows, cols = block.shape
    valid = np.isfinite(block) & (block > 0)
    cleaned = np.where(valid, block, 0.0).astype(np.float32)
    shape = (rows // factor, factor, cols // factor, factor)
    sums = cleaned.reshape(shape).sum(axis=(1, 3), dtype=np.float64)
    counts = valid.reshape(shape).sum(axis=(1, 3))
    enough = counts >= MIN_VALID_FRACTION * factor * factor
    mean = np.full(sums.shape, np.nan, dtype=np.float64)
    mean[enough] = sums[enough] / counts[enough]
    if not linear:
        mean[enough] = 10 * np.log10(mean[enough])
    return mean.astype(np.float32)


def _write_blocks(
    dset: h5py.Dataset, dst: DatasetWriter, factor: int, *, linear: bool
) -> None:
    """Read the frame in row blocks, average each block, and write it to the GeoTIFF."""
    n_cols = dst.width * factor
    out_rows_per_block = max(1, BLOCK_PIXELS // (n_cols * factor))
    for out_row in range(0, dst.height, out_rows_per_block):
        n_out = min(out_rows_per_block, dst.height - out_row)
        r0 = out_row * factor
        block = np.asarray(dset[r0 : r0 + n_out * factor, :n_cols], dtype=np.float32)
        result = block_mean(block, factor, linear=linear)
        dst.write(result, 1, window=Window(0, out_row, dst.width, n_out))


def decimate_frame(path: Path, out_path: Path, settings: Settings) -> FrameResult:
    """Write a reduced-resolution, georeferenced GeoTIFF for one frame."""
    with h5py.File(path, "r") as h5:
        grid = h5[GRID_GROUP.format(frequency=settings.frequency)]
        if settings.polarization not in grid:
            sys.exit(f"{path.name} has no {settings.polarization} layer.")
        dset = grid[settings.polarization]
        xs = np.asarray(grid["xCoordinates"][:2], dtype=np.float64)
        ys = np.asarray(grid["yCoordinates"][:2], dtype=np.float64)
        x_first, y_first = float(xs[0]), float(ys[0])
        dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
        if dx <= 0 or dy >= 0:
            sys.exit(
                f"{path.name}: expected x increasing and y decreasing; got {dx}, {dy}."
            )
        crs = read_crs(grid)
        factor = max(1, round(settings.resolution / dx))
        height, width = dset.shape[0] // factor, dset.shape[1] // factor
        transform = Affine(
            factor * dx, 0, x_first - dx / 2, 0, factor * dy, y_first - dy / 2
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "width": width,
            "height": height,
            "crs": crs,
            "transform": transform,
            "nodata": np.nan,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            _write_blocks(dset, dst, factor, linear=settings.linear)
    return FrameResult(path, frame_date(path), crs, dx, out_path)


def choose_crs(results: list[FrameResult], requested: str | None) -> CRS:
    """Use the requested CRS, or the one most frames already share."""
    if requested:
        return CRS.from_user_input(requested)
    counts: dict[str, int] = {}
    for r in results:
        counts[r.crs.to_wkt()] = counts.get(r.crs.to_wkt(), 0) + 1
    best = max(counts, key=lambda k: counts[k])
    return CRS.from_wkt(best)


def reproject_to(src_path: Path, dst_crs: CRS, dst_path: Path) -> Path:
    """Reproject a small GeoTIFF into dst_crs."""
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(crs=dst_crs, transform=transform, width=width, height=height)
        with rasterio.open(dst_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    return dst_path


def build_mosaic(
    tifs: list[Path], out_path: Path, resolution: float, method: str
) -> Path:
    """Merge GeoTIFFs that share a CRS into one mosaic GeoTIFF."""
    datasets = [rasterio.open(t) for t in tifs]
    try:
        left = min(d.bounds.left for d in datasets)
        right = max(d.bounds.right for d in datasets)
        bottom = min(d.bounds.bottom for d in datasets)
        top = max(d.bounds.top for d in datasets)
        n_pixels = ((right - left) / resolution) * ((top - bottom) / resolution)
        if n_pixels > MAX_MOSAIC_PIXELS:
            sys.exit(
                f"The mosaic would have about {n_pixels:,.0f} pixels. "
                "Use a larger --resolution."
            )
        array, transform = merge(datasets, res=resolution, method=method, nodata=np.nan)
        crs = datasets[0].crs
    finally:
        for d in datasets:
            d.close()
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": array.shape[1],
        "width": array.shape[2],
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype(np.float32))
    return out_path


def write_frames_csv(results: list[FrameResult], csv_path: Path) -> None:
    """Write a table of frames: file, date, CRS, native pixel size, bounds."""
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["file", "date", "crs", "native_pixel_m", "left", "bottom", "right", "top"]
        )
        for r in results:
            with rasterio.open(r.tif) as src:
                b = src.bounds
            w.writerow(
                [
                    r.source.name,
                    r.date,
                    r.crs.to_string(),
                    r.spacing,
                    b.left,
                    b.bottom,
                    b.right,
                    b.top,
                ]
            )


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "inputs", nargs="+", help="NISAR .h5 files and/or folders to search"
    )
    ap.add_argument(
        "--month", default=None, help="Only files acquired in this month, YYYYMM"
    )
    ap.add_argument(
        "--list", action="store_true", help="Just list the matching frames and exit"
    )
    ap.add_argument("--polarization", default="HHHH")
    ap.add_argument("--frequency", default="frequencyA")
    ap.add_argument(
        "--resolution",
        type=float,
        default=200.0,
        help="Output pixel size (default 200)",
    )
    ap.add_argument(
        "--linear", action="store_true", help="Keep linear power instead of dB"
    )
    ap.add_argument(
        "--crs", default=None, help="Mosaic CRS, e.g. EPSG:3031 (default: most common)"
    )
    ap.add_argument(
        "--method", choices=MERGE_METHODS, default="first", help="Overlap rule"
    )
    ap.add_argument("--out-dir", default="mosaic_out")
    ap.add_argument("--name", default="mosaic", help="Mosaic file name, without .tif")
    return ap.parse_args()


def main() -> None:
    """Find frames, make reduced-resolution GeoTIFFs, and stitch them together."""
    args = parse_args()
    if args.month and not (args.month.isdigit() and len(args.month) == MONTH_DIGITS):
        sys.exit("--month must look like 202607 (year then month).")
    frames = find_frames(args.inputs, args.month)
    if not frames:
        sys.exit("No matching .h5 frames found.")
    print(f"{len(frames)} frame(s):")
    for p in frames:
        print(f"  {frame_date(p) or '????????'}  {p.name}")
    if args.list:
        return

    out_dir = Path(args.out_dir)
    settings = Settings(args.polarization, args.frequency, args.resolution, args.linear)
    results = []
    for i, path in enumerate(frames, 1):
        tif = (
            out_dir
            / "frames"
            / f"{path.stem}_{args.polarization}_{int(args.resolution)}m.tif"
        )
        print(f"[{i}/{len(frames)}] {path.name} ...", flush=True)
        results.append(decimate_frame(path, tif, settings))
    write_frames_csv(results, out_dir / "frames.csv")

    target = choose_crs(results, args.crs)
    tifs = []
    for r in results:
        if r.crs == target:
            tifs.append(r.tif)
        else:
            print(
                f"Reprojecting {r.source.name} from {r.crs.to_string()} "
                f"to {target.to_string()}"
            )
            tifs.append(reproject_to(r.tif, target, r.tif.with_suffix(".reproj.tif")))
    mosaic = build_mosaic(
        tifs, out_dir / f"{args.name}.tif", args.resolution, args.method
    )

    with rasterio.open(mosaic) as src:
        b = src.bounds
        lon_lat = transform_bounds(src.crs, "EPSG:4326", *b)
        print(
            f"\nMosaic: {mosaic}  ({src.width} x {src.height} pixels, "
            f"CRS {src.crs.to_string()})"
        )
    print(
        f"  Bounds in the mosaic CRS: left {b.left:.0f}, bottom {b.bottom:.0f}, "
        f"right {b.right:.0f}, top {b.top:.0f}"
    )
    print(
        f"  Bounds in lon/lat: west {lon_lat[0]:.4f}, south {lon_lat[1]:.4f}, "
        f"east {lon_lat[2]:.4f}, north {lon_lat[3]:.4f}"
    )
    print(f"Frame table: {out_dir / 'frames.csv'}")


if __name__ == "__main__":
    main()
