"""Plot all the frames of one stack to see how the ice moves between dates.

A stack id is the track, the direction and the frame, for example 077_A_140. Frames with the
same stack id show the same ground on different dates (12 days apart). The script cuts every
frame of the stack into tiles, draws the tiles at their position on the map (one panel per
date, same map limits), and adds a zoomed overlay of the first and the last date: red is the
first date, cyan is the last date. One PNG is written per stack id.

You can give the stack id, or the name of any frame of the stack.

Examples:
    uv run python scripts/plot_stack_tiles.py --stack-id 077_A_140
    uv run python scripts/plot_stack_tiles.py --stack-id NISAR_L2_PR_GCOV_024_077_A_140_4005_SHSH_A_20260701T135211_20260701T135247_P05023_N_F_J_001
"""

import argparse
import os
import sys
from pathlib import Path

from utils.settings import DATA_DIR
from utils.stack_plot import find_stack_frames, get_stack_id, plot_stack
from utils.tile_split import DEFAULT_TILE_SIZE, load_tileserver

DEFAULT_IMAGE_DIR = DATA_DIR / os.environ.get("LABELSTUDIO_IMAGE_SUBDIR", "")
# Shared with enhance_contrast.py and split_tiles.py
DEFAULT_CACHE_DIR = DATA_DIR / "contrast_enhancement" / "stats_cache"
DEFAULT_OUTPUT_DIR = DATA_DIR / "stack_plots"
DEFAULT_ZOOM_KM = 30


def parse_args() -> argparse.Namespace:
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--stack-id",
        required=True,
        help="Stack id like 077_A_140 (track, direction, frame), or a frame name.",
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument(
        "--zoom-km",
        type=float,
        default=DEFAULT_ZOOM_KM,
        help="Side of the zoomed overlay in km.",
    )
    parser.add_argument(
        "--include-online-only",
        action="store_true",
        help="Also read frames that Box keeps online-only (slow: each one is downloaded).",
    )
    return parser.parse_args()


def main() -> None:
    """Find the frames of the stack, plot them and save the picture."""
    args = parse_args()
    try:
        stack_id = get_stack_id(args.stack_id)
        frames, skipped = find_stack_frames(
            args.image_dir, stack_id, args.include_online_only
        )
    except (ValueError, FileNotFoundError) as error:
        sys.exit(f"Cannot plot: {error}")

    for frame in skipped:
        print(f"Skipping online-only frame (not downloaded): {frame.name}")
    if not frames:
        sys.exit(f"No downloaded frames found for stack {stack_id} in {args.image_dir}")
    print(f"Stack {stack_id}: {len(frames)} frames")
    for frame in frames:
        print(f"  {frame.name}")
    if len(frames) == 1:
        print("Only one date: the picture will not show any movement.")

    tileserver = load_tileserver(args.image_dir, args.cache_dir)
    output = args.output_dir / f"stack_{stack_id}_{args.tile_size}px.png"
    plot_stack(stack_id, frames, tileserver, args.tile_size, output, args.zoom_km)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
