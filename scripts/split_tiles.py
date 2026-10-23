"""Split NISAR tiles into train and test with a checkerboard of square blocks.

Frames overlap a lot, so we split the ground and not the frames. The map is covered with
square blocks. Blocks next to each other have opposite colors, like a checkerboard: one
color is train, the other is test. A tile goes to the block that holds its center,
whatever frame it comes from. Tiles near the block edges are dropped (the gap), so train
and test tiles never touch. The result is one CSV per block size, with one row per tile.

The first run reads every frame once to find which tiles have data (slow, about 30 s for a
1.4 GB frame). The result is cached, so later runs are quick.

Examples:
    uv run python scripts/split_tiles.py --block-km 30 60 120 --tile-size 1024 --plot
    uv run python scripts/split_tiles.py --block-km 60 --swap --gap-km 5
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from utils.plot_split import plot_tile_split
from utils.settings import DATA_DIR
from utils.tile_split import DEFAULT_TILE_SIZE, load_tiles, split_by_blocks

DEFAULT_IMAGE_DIR = DATA_DIR / os.environ.get("LABELSTUDIO_IMAGE_SUBDIR", "")
# Shared with enhance_contrast.py, so frames it already read are not read again
DEFAULT_CACHE_DIR = DATA_DIR / "contrast_enhancement" / "stats_cache"
DEFAULT_OUTPUT = DATA_DIR / "splits" / "tile_split.csv"


def parse_args() -> argparse.Namespace:
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV to write. The block size is added to the name.",
    )
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument(
        "--block-km",
        type=float,
        nargs="+",
        required=True,
        help="Side of the square blocks in km. Several values make one split each.",
    )
    parser.add_argument(
        "--gap-km",
        type=float,
        default=0.0,
        help="Extra distance in km kept free between train and test tiles.",
    )
    parser.add_argument(
        "--swap",
        action="store_true",
        help="Swap the colors: the color that was test becomes train.",
    )
    parser.add_argument(
        "--include-online-only",
        action="store_true",
        help="Also read frames that Box keeps online-only (slow: each one is downloaded).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also save a map (PNG) next to each CSV.",
    )
    return parser.parse_args()


def save_split(
    table: pd.DataFrame, csv_path: Path, tile_m: float, block_km: float, plot: bool
) -> None:
    """Print a summary, save the CSV and, if asked, the map."""
    counts = table["split"].value_counts()
    print(
        f"{len(table)} tiles: " + ", ".join(f"{n} {name}" for name, n in counts.items())
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    if plot:
        png_path = csv_path.with_suffix(".png")
        plot_tile_split(table, tile_m, png_path, block_km)
        print(f"Saved {png_path}")


def main() -> None:
    """Split the tiles for each block size and save the results."""
    args = parse_args()
    try:
        tiles, tile_m = load_tiles(
            args.image_dir, args.cache_dir, args.tile_size, args.include_online_only
        )
        print(f"{len(tiles)} tiles from {tiles['frame'].nunique()} frames")

        for block_km in args.block_km:
            print(f"--- {block_km:g} km blocks")
            table = split_by_blocks(tiles, tile_m, block_km, args.gap_km, args.swap)
            name = f"{args.output.stem}_{args.tile_size}px_blocks_{block_km:g}km{args.output.suffix}"
            save_split(table, args.output.with_name(name), tile_m, block_km, args.plot)
    except (ValueError, FileNotFoundError) as error:
        sys.exit(f"Cannot split: {error}")


if __name__ == "__main__":
    main()
