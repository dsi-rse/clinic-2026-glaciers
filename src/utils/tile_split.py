"""Split NISAR tiles into train and test with a checkerboard of square blocks.

The frames overlap a lot (they all cover the same ice), so splitting by frame or by tile
number would still put the same ground in train and in test. Instead we split the ground,
following spatial block cross-validation (Kattenborn et al. 2022, Stock 2025):

1. Find the tiles of every frame and the position of each tile center on the map (metres).
2. Cover the map with a grid of square blocks. Blocks next to each other have opposite
   colors, like a checkerboard: one color is train, the other is test.
3. A tile goes to the block that holds its center, whatever frame it comes from, so
   overlapping frames of the same place land on the same side.
4. Drop the tiles near the block edges (a gap), so train and test tiles never touch.
5. Check that no train tile is closer than the gap to a test tile.

All NISAR frames use the same map projection (EPSG:3031, metres), so positions from
different frames can be compared directly.
"""

import importlib.util
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO_DIR = Path(__file__).resolve().parent.parent.parent
TILESERVER_FILE = REPO_DIR / "labelstudio" / "tileserver" / "tileserver.py"
EXPECTED_EPSG = 3031
METRES_PER_KM = 1000
# macOS file flag set on files that Box keeps online-only (not downloaded to this computer)
DATALESS_FLAG = 0x40000000
DEFAULT_TILE_SIZE = 4096  # pixels, the same as the Label Studio tiles


def load_tileserver(image_dir: Path, cache_dir: Path) -> object:
    """Import tileserver.py without changing it.

    It reads its folders from environment variables when imported, so we set them first.
    """
    os.environ["IMAGE_DIR"] = str(image_dir)
    os.environ["CACHE_DIR"] = str(cache_dir)
    spec = importlib.util.spec_from_file_location("tileserver", TILESERVER_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tileserver"] = module
    spec.loader.exec_module(module)
    return module


def find_frames(image_dir: Path) -> list[Path]:
    """Return all .h5 frames in image_dir (and its subfolders), sorted by path."""
    frames = sorted(
        path for path in image_dir.rglob("*") if path.suffix.lower() in (".h5", ".hdf5")
    )
    if not frames:
        msg = f"No .h5 files found in {image_dir}"
        raise FileNotFoundError(msg)
    return frames


def is_dataless(flags: int) -> bool:
    """True if these macOS file flags mark the file as online-only."""
    return bool(flags & DATALESS_FLAG)


def is_online_only(path: Path) -> bool:
    """True if the file is online-only (Box has not downloaded it), so reading it would wait.

    Only macOS marks files this way. On other systems every file counts as downloaded.
    """
    return is_dataless(getattr(path.stat(), "st_flags", 0))


def get_frame_tiles(
    frame: Path, tileserver: object, tile_size: int
) -> tuple[list[dict], float]:
    """Find the tiles with data in one frame and where each tile center is on the map.

    Returns the tiles (frame, row, col, x_m, y_m) and the tile width in metres.
    """
    # The same tiles that Label Studio offers: tiles with enough valid data
    stats = tileserver.get_stats(frame)  # slow the first time, then cached
    tiles = tileserver.list_tiles(stats, tile_size)

    with h5py.File(frame, "r") as handle:
        grid = handle[tileserver.GRID]
        epsg = int(grid["projection"].attrs["epsg_code"])
        if epsg != EXPECTED_EPSG:
            msg = f"{frame.name} uses EPSG:{epsg}, expected {EXPECTED_EPSG}"
            raise ValueError(msg)
        x_coordinates = grid["xCoordinates"][:]  # one value per pixel column, metres
        y_coordinates = grid["yCoordinates"][:]  # one value per pixel row, metres
        pixel_m = float(grid["xCoordinateSpacing"][()])

    rows = []
    for tile in tiles:
        # Position of the tile's middle pixel
        middle_col = tile["x0"] + tile["width"] // 2
        middle_row = tile["y0"] + tile["height"] // 2
        rows.append(
            {
                "frame": frame.name,
                "row": tile["row"],
                "col": tile["col"],
                "x_m": float(x_coordinates[middle_col]),
                "y_m": float(y_coordinates[middle_row]),
            }
        )
    return rows, tile_size * abs(pixel_m)


def get_all_tiles(
    frames: list[Path], tileserver: object, tile_size: int
) -> tuple[pd.DataFrame, float]:
    """Tiles of every frame in one table, and the tile width in metres."""
    all_rows = []
    tile_m = None
    for frame in frames:
        rows, frame_tile_m = get_frame_tiles(frame, tileserver, tile_size)
        if tile_m is not None and frame_tile_m != tile_m:
            msg = f"{frame.name} has a different pixel size from the other frames"
            raise ValueError(msg)
        tile_m = frame_tile_m
        all_rows.extend(rows)
    return pd.DataFrame(all_rows), tile_m


def check_no_leakage(
    tiles: pd.DataFrame, labels: pd.Series, tile_m: float, gap_km: float
) -> None:
    """Raise an error if a train tile is closer than the gap to a test tile.

    Two tiles are too close if both their x and y centers are less than
    (tile width + gap) apart: then the squares overlap or are inside the gap.
    """
    min_distance = tile_m + gap_km * METRES_PER_KM
    train_centers = tiles.loc[labels == "train", ["x_m", "y_m"]].to_numpy()
    test_centers = tiles.loc[labels == "test", ["x_m", "y_m"]].to_numpy()

    # Distance from each test tile to its nearest train tile. p=inf means "the larger of
    # the x and y distances", which is the right distance for squares.
    nearest, _ = cKDTree(train_centers).query(test_centers, p=np.inf)
    n_too_close = int((nearest < min_distance).sum())
    if n_too_close > 0:
        msg = f"Leakage: {n_too_close} test tiles are too close to a train tile"
        raise ValueError(msg)


def locate_blocks(
    tiles: pd.DataFrame, block_km: float, swap: bool = False
) -> pd.DataFrame:
    """Find the block of each tile and the color of that block (train or test).

    The grid of square blocks starts at the lower left corner of the tile centers. Returns
    block_x and block_y (the block's column and row) and block_color ("train" or "test").
    Blocks that share an edge always have opposite colors, like a checkerboard.
    """
    block_m = block_km * METRES_PER_KM
    block_x = ((tiles["x_m"] - tiles["x_m"].min()) // block_m).astype(int)
    block_y = ((tiles["y_m"] - tiles["y_m"].min()) // block_m).astype(int)

    # The color is the parity of the block's column plus its row
    is_first_color = (block_x + block_y) % 2 == 0
    is_train_color = ~is_first_color if swap else is_first_color
    block_color = pd.Series("test", index=tiles.index)
    block_color[is_train_color] = "train"
    return pd.DataFrame(
        {"block_x": block_x, "block_y": block_y, "block_color": block_color}
    )


def assign_blocks(
    tiles: pd.DataFrame,
    tile_m: float,
    block_km: float,
    gap_km: float = 0.0,
    swap: bool = False,
) -> pd.Series:
    """Label each tile "train", "test" or "gap" using square blocks in a checkerboard.

    A grid of square blocks (block_km on each side) covers the map. Blocks next to each
    other always have opposite colors, like a checkerboard: one color is train, the other
    is test. A tile belongs to the block that holds its center, whatever frame it comes
    from. Tiles whose center is closer to a block edge than half a tile (plus half of
    gap_km) are dropped ("gap"), so no train tile touches a test tile.
    """
    block_m = block_km * METRES_PER_KM
    margin = tile_m / 2 + gap_km * METRES_PER_KM / 2
    if block_m <= 2 * margin:
        msg = (
            f"A {block_km:g} km block is too small for {tile_m / METRES_PER_KM:.1f} km "
            f"tiles with a {gap_km:g} km gap: every tile would be dropped"
        )
        raise ValueError(msg)

    block_color = locate_blocks(tiles, block_km, swap)["block_color"]
    is_train_color = block_color == "train"

    # Position of each tile center in the grid, which starts at the corner of the data
    x_in_grid = tiles["x_m"] - tiles["x_m"].min()
    y_in_grid = tiles["y_m"] - tiles["y_m"].min()

    # Distance from each tile center to the nearest block edge, along x and along y
    x_left = x_in_grid % block_m
    y_below = y_in_grid % block_m
    x_to_edge = np.minimum(x_left, block_m - x_left)
    y_to_edge = np.minimum(y_below, block_m - y_below)
    near_edge = (x_to_edge < margin) | (y_to_edge < margin)

    labels = pd.Series("gap", index=tiles.index)
    labels[is_train_color & ~near_edge] = "train"
    labels[~is_train_color & ~near_edge] = "test"

    for name in ("train", "test"):
        if not (labels == name).any():
            msg = f"No {name} tiles with {block_km:g} km blocks: try a smaller gap"
            raise ValueError(msg)
    return labels


def load_tiles(
    image_dir: Path,
    cache_dir: Path,
    tile_size: int = DEFAULT_TILE_SIZE,
    include_online_only: bool = False,
) -> tuple[pd.DataFrame, float]:
    """Find the tiles of every frame in image_dir. Returns the table and the tile width (m).

    The table has one row per tile: frame, row, col, x_m and y_m (the tile center).
    Frames that Box keeps online-only are skipped unless include_online_only is True,
    because reading them means waiting for a large download.
    """
    tileserver = load_tileserver(image_dir, cache_dir)
    frames = find_frames(image_dir)

    if not include_online_only:
        online_only = [frame for frame in frames if is_online_only(frame)]
        frames = [frame for frame in frames if frame not in online_only]
        if online_only:
            print(f"Skipping {len(online_only)} online-only frames (not downloaded):")
            for frame in online_only:
                print(f"  {frame.name}")
        if not frames:
            msg = "No downloaded frames: make them available offline, or use --include-online-only"
            raise FileNotFoundError(msg)
    return get_all_tiles(frames, tileserver, tile_size)


def split_by_blocks(
    tiles: pd.DataFrame,
    tile_m: float,
    block_km: float,
    gap_km: float = 0.0,
    swap: bool = False,
) -> pd.DataFrame:
    """Add a split column (train, test or gap) with square blocks in a checkerboard."""
    labels = assign_blocks(tiles, tile_m, block_km, gap_km, swap)
    check_no_leakage(tiles, labels, tile_m, gap_km)
    blocks = locate_blocks(tiles, block_km, swap)
    return tiles.assign(split=labels, **blocks)
