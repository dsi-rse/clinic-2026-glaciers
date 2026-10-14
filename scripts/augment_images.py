"""Test image augmentations (CLAHE, high-pass) on random NISAR tiles.

For each randomly chosen tile and each method, saves a side-by-side figure of the input and
the augmented output. Tiles are cut and scaled with the same code as the Label Studio tile
server (labelstudio/tileserver/tileserver.py), so they look like the tiles you label.

Example:
TO-DO: Change file name and organize the script and the functions in utils.
    uv run python scripts/augment_images.py --n-frames 1 --n-tiles 3

Box keeps some frames online-only. If reading a frame times out, download it first (right-click
in Finder > Make Available Offline) or pass a downloaded frame with --frame.
"""

import argparse
import importlib.util
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from utils.augmentation import clahe, high_pass  # Important functions
from utils.settings import DATA_DIR  # settings variable which is the important file

REPO_DIR = Path(__file__).resolve().parent.parent
TILESERVER_FILE = REPO_DIR / "labelstudio" / "tileserver" / "tileserver.py"
DEFAULT_IMAGE_DIR = DATA_DIR / os.environ.get("LABELSTUDIO_IMAGE_SUBDIR", "")
DEFAULT_OUTPUT_DIR = DATA_DIR / "augmentation"
TILE_SIZE = 4096


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


def choose_tiles(
    tileserver: object,
    frames: list[Path],
    n_tiles: int,
    seed: int,
) -> list[tuple[Path, dict, dict]]:
    """Randomly choose n_tiles tiles with data from each frame.

    Returns a list of (frame path, frame stats, tile) for each chosen tile.
    """
    rng = np.random.default_rng(seed)
    chosen = []
    for frame in frames:
        stats = tileserver.get_stats(frame)  # slow the first time, then cached
        tiles = tileserver.list_tiles(stats, TILE_SIZE)
        for tile_number in rng.choice(
            len(tiles), min(n_tiles, len(tiles)), replace=False
        ):
            chosen.append((frame, stats, tiles[tile_number]))
    return chosen


def choose_frames(image_dir: Path, n_frames: int, seed: int) -> list[Path]:
    """Randomly choose n_frames .h5 frames from image_dir."""
    frames = sorted(
        path for path in image_dir.rglob("*") if path.suffix.lower() in (".h5", ".hdf5")
    )
    if not frames:
        sys.exit(f"No .h5 files found in {image_dir}")
    rng = np.random.default_rng(seed)
    numbers = rng.choice(len(frames), min(n_frames, len(frames)), replace=False)
    return [frames[number] for number in numbers]


def save_comparison(
    tile: np.ndarray, result: np.ndarray, method: str, title: str, path: Path
) -> None:
    """Save the input and the augmented output side by side."""
    figure = Figure(figsize=(14, 7))
    left, right = figure.subplots(1, 2)
    left.imshow(tile, cmap="gray", vmin=0, vmax=255)
    left.set_title("Input")
    right.imshow(result, cmap="gray", vmin=0, vmax=1)
    right.set_title(f"Output: {method}")
    for axis in (left, right):
        axis.axis("off")
    figure.suptitle(title, fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=100)


def main() -> None: # Main orchestrator
    """Make an input/output figure for every chosen tile and method."""
    # 1) This allows an interactive with the command line (nothing new to create)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--frame", type=Path, help="use this .h5 frame instead of random ones"
    )
    parser.add_argument(
        "--n-frames", type=int, default=1, help="random frames to sample"
    )
    parser.add_argument("--n-tiles", type=int, default=3, help="tiles per frame")
    parser.add_argument(
        "--seed", type=int, default=0, help="makes the sample repeatable"
    )
    parser.add_argument("--clip-limit", type=float, default=0.01, help="CLAHE strength")
    parser.add_argument(
        "--sigma", type=float, default=50, help="high-pass blur, pixels"
    )
    args = parser.parse_args() # Sets all the information into args

    # -----------------------------------------------------------------

    # 2) Load tile server

    # Methods as classes
    methods: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "clahe": lambda tile: clahe(tile, args.clip_limit),
        "high_pass": lambda tile: high_pass(tile, args.sigma),
    }
    # Validates the existance of mkdir otherwise it creates it
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load tileserver, this is an function from the original repository
    # I think it imports or store the images in the output path, it also
    # receives the images path
    tileserver = load_tileserver(args.image_dir, args.output_dir / "stats_cache")

    # Select the frames of the images. You can specify the frame you would like to 
    # work or choose randomly a number of frames.
    # TO-DO: Apply process to all the images? Or a sample?
    frames = (
        [args.frame]
        if args.frame
        else choose_frames(args.image_dir, args.n_frames, args.seed)
    )
    # For each frame: (function already created in the repository)
    # get the stats: size, valid-pixel count, display range?
    # List the tiles: for each tile (4096*4096 grid) lets store or save the tiles
    # that has a 5% of valid pixels. 
    # Randomly selected tiles (seed)
    chosen = choose_tiles(tileserver, frames, args.n_tiles, args.seed)

    log_lines = [f"seed={args.seed} clip_limit={args.clip_limit} sigma={args.sigma}"]

    # -----------------------------------------------------------------
    # For each choosen tiles (frame, row and col)
    for frame, stats, tile_info in chosen:
        row, col = tile_info["row"], tile_info["col"]
        # Read the tile and makes it uint8 (0 is no data)
        tile = tileserver.render_tile(frame, stats, TILE_SIZE, row, col)
        name = f"{frame.stem}_r{row}_c{col}"
        log_lines.append(f"{frame.name} row={row} col={col}")
        # Apply each method here (clahe and high pass)
        for method, function in methods.items():
            path = args.output_dir / f"{name}_{method}.png"
            # Input output png
            # Probably one image with the original, clahe, high pass outputs 
            # to make easy the comparison
            save_comparison(tile, function(tile), method, name, path) # Save side by side outputs
            print(f"Saved {path}")
    # Finally we write the tiles that were selected
    (args.output_dir / "tiles_used.txt").write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
