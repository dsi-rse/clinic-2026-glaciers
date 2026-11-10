"""Split NISAR tiles into train and test with a checkerboard of square blocks.

Frames overlap a lot, so we split the ground and not the frames. The map is covered with
square blocks. Blocks next to each other have opposite colors, like a checkerboard: one
color is train, the other is test. A tile goes to the block that holds its center,
whatever frame it comes from. Tiles near the block edges are dropped (the gap), so train
and test tiles never touch. The result is one CSV per block size, with one row per tile.

With --region, only the tiles that lie completely inside a rectangle (the area of analysis)
are kept first. Then the same process runs on them (the checkerboard starts at their lower
left corner, as without --region). With several
tile sizes and block sizes, the script runs every combination and writes a summary table:
how many tiles were found, kept for train, kept for test and dropped.

The first run reads every frame once to find which tiles have data (slow, about 30 s for a
1.4 GB frame). The result is cached, so later runs are quick.

Examples:
    uv run python scripts/split_tiles.py --block-km 30 60 120 --tile-size 1024 --plot
    uv run python scripts/split_tiles.py --block-km 60 --swap --gap-km 5
    uv run python scripts/split_tiles.py --region -1840 -490 -1440 -90 --tile-size 1024 4096 --block-km 81.92 122.88 163.84 --gap-km 5 --plot
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.plot_split import plot_tile_split
from utils.settings import DATA_DIR
from utils.tile_split import (
    DEFAULT_TILE_SIZE,
    filter_region,
    load_tiles,
    split_by_blocks,
)

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
        help="CSV to write. The settings of the run (tile size, block size, gap, swap, "
        "region) are added to the name, so runs with other settings never overwrite it.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        nargs="+",
        default=[DEFAULT_TILE_SIZE],
        help="Tile size in pixels. Several values are run one after the other.",
    )
    parser.add_argument(
        "--region",
        type=float,
        nargs=4,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        help="Area of analysis: a rectangle in km, in the map system of the tiles "
        "(Antarctic polar stereographic). Only tiles completely inside are used.",
    )
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
        "--date-prefix",
        default="",
        help="Only use frames whose start date begins with this, e.g. 202607 for July "
        "2026 or 20260723 for one day. Default: all frames.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also save a map (PNG) next to each CSV.",
    )
    return parser.parse_args()


def region_tag(region: tuple[float, float, float, float] | None) -> str:
    """Short name part for the bounding box: "" without a box, else "_region_" and a code.

    The code comes from the coordinates, so another box gives another name. The exact
    coordinates are in the settings file (.json) next to the CSV.
    """
    if region is None:
        return ""
    code = hashlib.sha256(",".join(f"{x:g}" for x in region).encode()).hexdigest()
    return f"_region_{code[:6]}"


def gap_swap_tag(gap_km: float, swap: bool) -> str:
    """Name part for the gap and the swap, for example "_gap_5km_swap"."""
    return f"_gap_{gap_km:g}km" + ("_swap" if swap else "")


def save_split(
    table: pd.DataFrame,
    csv_path: Path,
    tile_m: float,
    block_km: float,
    plot: bool,
    region: tuple[float, float, float, float] | None,
    settings: dict,
) -> None:
    """Print a summary, save the CSV, the settings (.json) and, if asked, the map.

    The CSV, the settings and the map share one name, which holds the settings of the run.
    """
    counts = table["split"].value_counts()
    print(
        f"{len(table)} tiles: " + ", ".join(f"{n} {name}" for name, n in counts.items())
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    png_path = csv_path.with_suffix(".png")
    if plot:
        plot_tile_split(table, tile_m, png_path, block_km, region)
        print(f"Saved {png_path}")
    elif png_path.exists():
        # A map with the same name comes from an older run (for example with other frames
        # because of --include-online-only), so it could disagree with the new CSV.
        png_path.unlink()
        print(f"Removed the old map {png_path.name}: it did not come from this run")

    settings = {
        **settings,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "csv": csv_path.name,
        "map": png_path.name if plot else None,
        "n_tiles": len(table),
        **{f"n_{name}": int(n) for name, n in counts.items()},
    }
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Saved {json_path}")


def summarize(
    table: pd.DataFrame,
    tile_size: int,
    tile_m: float,
    block_km: float,
    gap_km: float,
    swap: bool,
    found: int,
) -> dict:
    """One row of the summary: how many tiles were found, kept for train and test, dropped."""
    counts = table["split"].value_counts()
    in_region = len(table)
    return {
        "tile_px": tile_size,
        "tile_km": round(tile_m / 1000, 2),
        "block_km": block_km,
        "gap_km": gap_km,
        "swap": swap,
        "tiles_found": found,
        "outside_region": found - in_region,
        "in_region": in_region,
        "train": int(counts.get("train", 0)),
        "test": int(counts.get("test", 0)),
        "dropped": int(counts.get("gap", 0)),
        "dropped_pct": round(100 * counts.get("gap", 0) / in_region, 1),
    }


def main() -> None:
    """Split the tiles for each tile size and block size, and save the results."""
    args = parse_args()
    # Region (Bounding Box)
    # The bounding box takes kilometres
    # The Antarctic polar stereographic one (EPSG:3031) works in meters.
    # Spatial reference system / coordinate reference system
    region = tuple(args.region) if args.region else None
    if region is not None and not (region[0] < region[2] and region[1] < region[3]):
        sys.exit("--region needs X_MIN < X_MAX and Y_MIN < Y_MAX")

    summary = []
    try:
        for tile_size in args.tile_size:
            print(f"=== tiles of {tile_size} px")
            # Load tiles using the repository function
            tiles, tile_m = load_tiles(
                args.image_dir,
                args.cache_dir,
                tile_size,
                args.include_online_only,
                args.date_prefix,
            )
            found = len(tiles)
            print(f"{found} tiles from {tiles['frame'].nunique()} frames")

            # Filter the tiles that lies outside the bounding box
            if region is not None:
                tiles = filter_region(tiles, tile_m, region)
                print(f"{len(tiles)} tiles lie completely inside the region")
            # For each block in the checkerboard
            for block_km in args.block_km:
                print(f"--- {block_km:g} km blocks")
                try:
                    table = split_by_blocks(
                        tiles, tile_m, block_km, args.gap_km, args.swap
                    )
                except ValueError as error:
                    print(f"Skipped: {error}")
                    continue
                name = (
                    f"{args.output.stem}{region_tag(region)}_{tile_size}px"
                    f"_blocks_{block_km:g}km{gap_swap_tag(args.gap_km, args.swap)}"
                    f"{args.output.suffix}"
                )
                settings = {
                    "tile_size_px": tile_size,
                    "tile_km": round(tile_m / 1000, 2),
                    "block_km": block_km,
                    "gap_km": args.gap_km,
                    "swap": args.swap,
                    "region_km": list(region) if region else None,
                    "include_online_only": args.include_online_only,
                    "image_dir": str(args.image_dir),
                }
                save_split(
                    table,
                    args.output.with_name(name),
                    tile_m,
                    block_km,
                    args.plot,
                    region,
                    settings,
                )
                summary.append(
                    summarize(
                        table,
                        tile_size,
                        tile_m,
                        block_km,
                        args.gap_km,
                        args.swap,
                        found,
                    )
                )
    except (ValueError, FileNotFoundError) as error:
        sys.exit(f"Cannot split: {error}")

    if summary:
        table = pd.DataFrame(summary)
        # The name holds every setting of the run, so another run never overwrites it
        sizes = "-".join(str(size) for size in args.tile_size)
        blocks = "-".join(f"{block:g}" for block in args.block_km)
        summary_path = args.output.with_name(
            f"{args.output.stem}{region_tag(region)}_{sizes}px_blocks_{blocks}km"
            f"{gap_swap_tag(args.gap_km, args.swap)}_summary.csv"
        )
        table.to_csv(summary_path, index=False)
        print()
        print(table.to_string(index=False))
        print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
