"""Serve NISAR GCOV HDF5 frames to Label Studio as PNG tiles, rendered on the fly.

A frame is ~30000 x 30000 float32 pixels, far too big for a browser canvas, so it is cut into
TILE_SIZE x TILE_SIZE tiles and each tile is one Label Studio task. Nothing is written to disk
except a few-KB stats file per frame (the 1%/99% display range and which tiles have data).

Display scaling: 10*log10(HHHH), stretched so that the frame's 1st..99th percentile of valid
pixels maps to gray levels 1..255. Gray level 0 is reserved for nodata (NaN, non-positive, or
flagged in inputDataExceptionMask). The percentiles are per frame, not per tile, so that the
same backscatter looks the same in every tile of a frame.

Endpoints:
    GET /health
    GET /index                                 every tile with data, for every *.h5 in IMAGE_DIR
    GET /tile/<file>/<size>/<row>/<col>.png    one rendered size x size tile

The tile size is part of each tile's URL, so changing TILE_SIZE never changes what an existing
Label Studio task shows: it only changes which tiles /index offers for new tasks.

Configured by environment variables; see the constants below.
"""

import io
import json
import os
import re
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

IMAGE_DIR = Path(os.environ.get("IMAGE_DIR", "/data/images"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/cache"))
PORT = int(os.environ.get("PORT", "8081"))
# Valid pixels are counted per BLOCK x BLOCK block, which can be summed into tiles of any size
# that is a multiple of BLOCK. 512 matches the files' HDF5 chunks, so a tile never decompresses
# a chunk it only partly uses.
BLOCK = 512
MAX_TILE_SIZE = 8192
# The size of tiles offered by /index. A multiple of BLOCK.
TILE_SIZE = int(os.environ.get("TILE_SIZE", "4096"))
# Tile URLs from before the size was part of the URL meant this size.
LEGACY_TILE_SIZE = 2048
GRID = os.environ.get("HDF5_GRID", "/science/LSAR/GCOV/grids/frequencyA")
LAYER = os.environ.get("HDF5_LAYER", "HHHH")
EXCEPTION_MASK = "inputDataExceptionMask"
# Tiles with less valid data than this (mostly swath edges) are not offered for labeling.
MIN_VALID_FRACTION = float(os.environ.get("MIN_VALID_FRACTION", "0.05"))
LOW_PERCENTILE, HIGH_PERCENTILE = 1.0, 99.0

# Histogram used to find the percentiles without holding a whole frame in memory.
DB_MIN, DB_MAX, DB_BIN = -80.0, 40.0, 0.01
NUM_BINS = round((DB_MAX - DB_MIN) / DB_BIN)
# Rows read at a time in the stats pass: one chunk row. ~75 MB for a 30000-wide frame.
STRIP_ROWS = 512
# Bump when the stats format or meaning changes, to invalidate old cache files.
STATS_VERSION = 2

_stats_lock = threading.Lock()
_stats_memo: dict[str, dict] = {}


def valid_mask(values: np.ndarray, exception: np.ndarray | None) -> np.ndarray:
    """Pixels that count as data: finite, positive (so log10 works), and not flagged."""
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(values) & (values > 0)
    if exception is not None:
        valid &= exception == 0
    return valid


def _read(
    grid: h5py.Group, rows: slice, cols: slice
) -> tuple[np.ndarray, np.ndarray | None]:
    values = grid[LAYER][rows, cols]
    exception = grid[EXCEPTION_MASK][rows, cols] if EXCEPTION_MASK in grid else None
    return values, exception


def _percentile_from_histogram(counts: np.ndarray, percentile: float) -> float:
    cumulative = np.cumsum(counts)
    index = int(np.searchsorted(cumulative, cumulative[-1] * percentile / 100.0))
    return DB_MIN + (index + 0.5) * DB_BIN


def compute_stats(path: Path) -> dict:
    """One streaming pass over a frame: display range and valid-pixel count for every block."""
    block = BLOCK
    with h5py.File(path, "r") as handle:
        grid = handle[GRID]
        height, width = grid[LAYER].shape
        col_starts = np.arange(0, width, block)
        counts = np.zeros(NUM_BINS, dtype=np.int64)
        block_valid = np.zeros((-(-height // block), len(col_starts)), dtype=np.int64)

        y = 0
        while y < height:
            # Never let a strip straddle a block boundary, so each strip is in one block row.
            end = min(y + STRIP_ROWS, height, (y // block + 1) * block)
            values, exception = _read(grid, slice(y, end), slice(None))
            valid = valid_mask(values, exception)
            block_valid[y // block] += np.add.reduceat(valid.sum(axis=0), col_starts)
            db = 10.0 * np.log10(values[valid])
            bins = np.clip(((db - DB_MIN) / DB_BIN).astype(np.int64), 0, NUM_BINS - 1)
            counts += np.bincount(bins, minlength=NUM_BINS)
            y = end

    if counts.sum() == 0:
        low, high = DB_MIN, DB_MAX
    else:
        low = _percentile_from_histogram(counts, LOW_PERCENTILE)
        high = _percentile_from_histogram(counts, HIGH_PERCENTILE)
    return {
        "height": int(height),
        "width": int(width),
        "block": block,
        "low_db": low,
        "high_db": high,
        "block_valid": block_valid.tolist(),
    }


def get_stats(path: Path) -> dict:
    """Stats for a frame, from memory, then the cache directory, then a fresh pass."""
    path = path.resolve()  # the same frame must get the same key however it was named
    info = path.stat()
    key = f"{path}|{info.st_size}|{info.st_mtime_ns}|{BLOCK}|{GRID}/{LAYER}|{STATS_VERSION}"
    cache_file = CACHE_DIR / f"{path.stem}.json"
    with _stats_lock:  # also stops two requests from computing the same frame at once
        if key in _stats_memo:
            return _stats_memo[key]
        stats = None
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            if cached.get("key") == key:
                stats = cached
        if stats is None:
            print(
                f"Computing display range for {path.name} (one-time) ...",
                file=sys.stderr,
            )
            stats = {"key": key, **compute_stats(path)}
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(stats))
            except OSError as exc:
                print(f"Could not cache stats in {CACHE_DIR}: {exc}", file=sys.stderr)
        _stats_memo[key] = stats
        return stats


def check_tile_size(stats: dict, size: int) -> None:
    """Tiles must be whole blocks, so their valid-pixel counts can be summed from blocks."""
    block = stats["block"]
    if size % block or not block <= size <= MAX_TILE_SIZE:
        raise ValueError(
            f"tile size must be a multiple of {block} up to {MAX_TILE_SIZE}"
        )


def tile_bounds(
    stats: dict, size: int, row: int, col: int
) -> tuple[int, int, int, int]:
    """(x0, y0, width, height) of a tile. Tiles on the right and bottom edges are smaller."""
    x0, y0 = col * size, row * size
    return x0, y0, min(size, stats["width"] - x0), min(size, stats["height"] - y0)


def list_tiles(
    stats: dict, size: int | None = None, min_valid_fraction: float | None = None
) -> list[dict]:
    """Tiles with enough data to be worth labeling."""
    size = size or TILE_SIZE
    if min_valid_fraction is None:
        min_valid_fraction = MIN_VALID_FRACTION
    check_tile_size(stats, size)
    per_tile = size // stats["block"]
    block_valid = np.array(stats["block_valid"], dtype=np.int64)
    tiles = []
    for row in range(-(-stats["height"] // size)):
        for col in range(-(-stats["width"] // size)):
            rows = slice(row * per_tile, (row + 1) * per_tile)
            count = int(block_valid[rows, col * per_tile : (col + 1) * per_tile].sum())
            x0, y0, width, height = tile_bounds(stats, size, row, col)
            if count > 0 and count >= min_valid_fraction * width * height:
                tiles.append(
                    {
                        "tile_size": size,
                        "row": row,
                        "col": col,
                        "x0": x0,
                        "y0": y0,
                        "width": width,
                        "height": height,
                    }
                )
    return tiles


def render_tile(path: Path, stats: dict, size: int, row: int, col: int) -> np.ndarray:
    """A tile as uint8 gray levels: 0 for nodata, 1..255 for the low_db..high_db range."""
    check_tile_size(stats, size)
    x0, y0, width, height = tile_bounds(stats, size, row, col)
    if width <= 0 or height <= 0 or row < 0 or col < 0:
        raise IndexError(f"tile ({row}, {col}) is outside the frame")
    with h5py.File(path, "r") as handle:
        values, exception = _read(
            handle[GRID], slice(y0, y0 + height), slice(x0, x0 + width)
        )
    valid = valid_mask(values, exception)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(values)
    scaled = (db - stats["low_db"]) / (stats["high_db"] - stats["low_db"])
    gray = 1.0 + 254.0 * np.clip(np.nan_to_num(scaled), 0.0, 1.0)
    return np.where(valid, np.rint(gray), 0).astype(np.uint8)


def encode_png(gray: np.ndarray) -> bytes:
    """8-bit grayscale PNG. Fast compression: tiles are made per request, never stored."""
    buffer = io.BytesIO()
    Image.fromarray(gray).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def resolve(relative: str, root: Path | None = None) -> Path:
    """The file under root that a URL names, refusing anything that escapes root."""
    root = root or IMAGE_DIR
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path.suffix.lower() not in (
        ".h5",
        ".hdf5",
    ):
        raise FileNotFoundError(relative)
    if not path.is_file():
        raise FileNotFoundError(relative)
    return path


def build_index(root: Path | None = None) -> list[dict]:
    """Every labelable tile of every frame under root."""
    root = root or IMAGE_DIR
    index = []
    for path in sorted(
        p for p in root.rglob("*") if p.suffix.lower() in (".h5", ".hdf5")
    ):
        try:
            stats = get_stats(path)
        except (OSError, KeyError) as exc:
            print(f"Skipping {path}: {exc}", file=sys.stderr)
            continue
        relative = path.relative_to(root).as_posix()
        index.extend({"file": relative, **tile} for tile in list_tiles(stats))
    return index


TILE_URL = re.compile(
    r"^/tile/(?P<file>.+?\.(?:h5|hdf5))/(?:(?P<size>\d+)/)?(?P<row>\d+)/(?P<col>\d+)\.png$",
    re.IGNORECASE,
)


class Handler(BaseHTTPRequestHandler):
    """HTTP front end for the functions above."""

    def _send(
        self, status: int, body: bytes, content_type: str, cache: bool = False
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Label Studio's brush and magic wand read the image back off a canvas, which the
        # browser only allows for cross-origin images that opt in.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: object) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802 -- name required by BaseHTTPRequestHandler
        """Route a GET request."""
        path = urllib.parse.urlsplit(self.path).path
        try:
            if path == "/health":
                self._send_json(200, {"status": "ok"})
            elif path == "/index":
                self._send_json(200, build_index())
            elif match := TILE_URL.match(path):
                file = resolve(urllib.parse.unquote(match["file"]))
                size = int(match["size"]) if match["size"] else LEGACY_TILE_SIZE
                gray = render_tile(
                    file, get_stats(file), size, int(match["row"]), int(match["col"])
                )
                self._send(200, encode_png(gray), "image/png", cache=True)
            else:
                self._send_json(404, {"error": f"no such endpoint: {path}"})
        except (FileNotFoundError, IndexError) as exc:
            self._send_json(404, {"error": f"not found: {exc}"})
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:  # report, but keep serving other requests
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
            raise


def main() -> None:
    """Run the server until interrupted."""
    if not IMAGE_DIR.is_dir():
        sys.exit(f"Image directory {IMAGE_DIR} does not exist.")
    print(f"Serving tiles of {IMAGE_DIR} on port {PORT}", file=sys.stderr)
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
