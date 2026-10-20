"""Test contrast enhancement (CLAHE, high-pass) on NISAR tiles.

This scripts loads frames or set of frames an applies two contrast enhancement
methods (CLAHE and high-pass). Since NISAR frames are big to work, they are
divided into tiles of 4096 * 4096 pixels. Then both enhancement functions are applied
and each output is save in their respective files.

Since we are comparing the results of both approaches,
a third image (original + clahe + high-pass) in saved in the comparison folder.

Finally, the current script allow us to run randomly a number of frames and tiles
as we are trying to compare which enhancement approach would work better in
the NISAR frames.

Which frames to use (pick one):
    default       a random sample of --n-frames frames
    --frames      a list of frames you choose
    --all-frames  every frame in --image-dir

Examples:
    uv run python scripts/enhance_contrast.py --n-frames 1 --n-tiles 3
    uv run python scripts/enhance_contrast.py --frames frame_a.h5 frame_b.h5 --n-tiles 5
    uv run python scripts/enhance_contrast.py --all-frames --n-tiles 2
    uv run python scripts/enhance_contrast.py --all-frames --all-tiles

Which tiles to use: --n-tiles random tiles per frame (default 3), or --all-tiles for
every tile with data. --all-tiles is slow and writes a lot of files.

Instructions:
    Before running the script verify that the images to run are available offline.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.image import imsave

from utils.augmentation import clahe, high_pass  # Important functions
from utils.settings import DATA_DIR  # settings variable which is the important file

REPO_DIR = Path(__file__).resolve().parent.parent
TILESERVER_FILE = REPO_DIR / "labelstudio" / "tileserver" / "tileserver.py"
DEFAULT_IMAGE_DIR = DATA_DIR / os.environ.get("LABELSTUDIO_IMAGE_SUBDIR", "")
DEFAULT_OUTPUT_DIR = DATA_DIR / "contrast_enhancement"
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
    all_tiles: bool = False,
) -> list[tuple[Path, dict, dict]]:
    """Choose tiles with data from each frame: every tile, or n_tiles random ones.

    Returns a list of (frame path, frame stats, tile) for each chosen tile.
    """
    rng = np.random.default_rng(seed)
    chosen = []
    for frame in frames:
        stats = tileserver.get_stats(frame)  # slow the first time, then cached
        tiles = tileserver.list_tiles(stats, TILE_SIZE)
        if all_tiles:
            tile_numbers = range(len(tiles))
        else:
            tile_numbers = rng.choice(
                len(tiles), min(n_tiles, len(tiles)), replace=False
            )
        for tile_number in tile_numbers:
            chosen.append((frame, stats, tiles[tile_number]))
    return chosen


def find_frames(image_dir: Path) -> list[Path]:
    """Return all .h5 frames in image_dir, sorted by path."""
    frames = sorted(
        path for path in image_dir.rglob("*") if path.suffix.lower() in (".h5", ".hdf5")
    )
    if not frames:
        sys.exit(f"No .h5 files found in {image_dir}")
    return frames


def choose_frames(frames: list[Path], n_frames: int, seed: int) -> list[Path]:
    """Randomly choose n_frames from frames."""
    rng = np.random.default_rng(seed)
    numbers = rng.choice(len(frames), min(n_frames, len(frames)), replace=False)
    return [frames[number] for number in numbers]


def save_comparison(
    tile: np.ndarray, results: dict[str, np.ndarray], title: str, path: Path
) -> None:
    """Save the input (left) and one panel per method (right), side by side.

    results maps a panel title (method and parameters) to its output image. Each panel is
    12 x 12 inches at 200 dpi, so the tile keeps most of its detail.
    """
    figure = Figure(figsize=(12 * (1 + len(results)), 12))
    axes = figure.subplots(1, 1 + len(results))
    axes[0].imshow(tile, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Input", fontsize=20, pad=20)
    for axis, (label, result) in zip(axes[1:], results.items(), strict=True):
        axis.imshow(result, cmap="gray", vmin=0, vmax=1)
        axis.set_title(label, fontsize=20, pad=20)
    for axis in axes:
        axis.axis("off")
    # Fixed margins: tight_layout ignores titles on axes that are switched off
    figure.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.92, wspace=0.02)
    figure.savefig(path, dpi=200)


def main() -> None:
    """Make one input/CLAHE/high-pass figure for every chosen tile."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--frames", type=Path, nargs="+", help="use these .h5 frames (a list)"
    )
    group.add_argument(
        "--all-frames", action="store_true", help="use every frame in --image-dir"
    )
    parser.add_argument(
        "--n-frames", type=int, default=1, help="random frames to sample (default mode)"
    )
    parser.add_argument("--n-tiles", type=int, default=3, help="tiles per frame")
    parser.add_argument(
        "--all-tiles",
        action="store_true",
        help="use every tile with data in each frame (ignores --n-tiles)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="makes the sample repeatable"
    )
    parser.add_argument("--clip-limit", type=float, default=0.01, help="CLAHE strength")
    parser.add_argument(
        "--sigma", type=float, default=50, help="high-pass blur, pixels"
    )
    args = parser.parse_args()

    for folder in ("clahe", "high_pass", "comparison"):
        (args.output_dir / folder).mkdir(parents=True, exist_ok=True)
    tileserver = load_tileserver(args.image_dir, args.output_dir / "stats_cache")

    # Choose the frames: a given list, all of them, or a random sample
    if args.frames:
        frames = args.frames
    elif args.all_frames:
        frames = find_frames(args.image_dir)
    else:
        frames = choose_frames(find_frames(args.image_dir), args.n_frames, args.seed)

    # Choose random tiles with data in each frame
    chosen = choose_tiles(
        tileserver, frames, args.n_tiles, args.seed, all_tiles=args.all_tiles
    )

    log_lines = [f"seed={args.seed} clip_limit={args.clip_limit} sigma={args.sigma}"]
    for frame, stats, tile_info in chosen:
        row, col = tile_info["row"], tile_info["col"]
        # Read the tile as uint8 (0 is no data)
        tile = tileserver.render_tile(frame, stats, TILE_SIZE, row, col)
        name = f"{frame.stem}_r{row}_c{col}"
        log_lines.append(f"{frame.name} row={row} col={col}")
        # Folder name -> (panel title with its parameter, output image)
        outputs = {
            "clahe": (
                f"CLAHE (clip_limit={args.clip_limit})",
                clahe(tile, args.clip_limit),
            ),
            "high_pass": (
                f"High-pass (sigma={args.sigma})",
                high_pass(tile, args.sigma),
            ),
        }
        # Only the processed tile, full size, one folder per method
        for folder, (_, image) in outputs.items():
            path = args.output_dir / folder / f"{name}.png"
            imsave(path, image, cmap="gray", vmin=0, vmax=1)
            # print(f"Saved {path}")
        # Input and all outputs side by side
        path = args.output_dir / "comparison" / f"{name}.png"
        save_comparison(tile, dict(outputs.values()), name, path)
        # print(f"Saved {path}")
    (args.output_dir / "tiles_used.txt").write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
