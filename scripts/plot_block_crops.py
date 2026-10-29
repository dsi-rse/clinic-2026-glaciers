"""Study area, checkerboard of blocks, and tiles cropped at the block borders (three pictures).

1. The study area is the border of all the frames (read from the metadata JSON files). It is
   covered with a checkerboard of square blocks: one color is train, the other is test.
2. Tiles of a few sample stacks are placed at their position on the map. A stack id is the
   track, the direction and the frame, for example 077_A_140: the same place on different dates.
3. Every tile is cropped at the borders of the blocks. A tile that crosses a border is cut into
   pieces, and each piece belongs to the block it lies in.

Pictures (in DATA_DIR/block_crops):
    1_study_area_blocks.png   the border of the study area and the checkerboard
    2_sample_tiles.png        the tiles of the sample stacks, before cropping
    3_sample_cropped.png      the cropped pieces of the sample stacks

Only a few stacks are drawn, so the pictures do not carry every frame and tile.

Examples:
    uv run python scripts/plot_block_crops.py
    uv run python scripts/plot_block_crops.py --block-km 200 --stack-ids 077_A_140 083_D_126
"""

import argparse
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from utils.footprints import read_footprint
from utils.plot_split import SPLIT_COLORS
from utils.settings import DATA_DIR
from utils.stack_plot import (
    find_stack_frames,
    get_cycle,
    get_date,
    get_stack_id,
    get_tile_boxes,
    shrink,
)
from utils.tile_split import DEFAULT_TILE_SIZE, load_tileserver

DEFAULT_METADATA_DIR = DATA_DIR / "nisar_data" / "nisar_gcov_pig" / "cmr_metadata"
DEFAULT_IMAGE_DIR = DATA_DIR / os.environ.get("LABELSTUDIO_IMAGE_SUBDIR", "")
DEFAULT_CACHE_DIR = DATA_DIR / "contrast_enhancement" / "stats_cache"
DEFAULT_OUTPUT_DIR = DATA_DIR / "block_crops"
METRES_PER_KM = 1000
PIXEL_M = 10  # NISAR GCOV pixel size
SHRINK = 8  # pieces are drawn at 8 x 8 pixels per cell (80 m), enough for a map
STUDY_FILL = "#eeede9"  # very light gray: the study area
LINE_COLOR = "#3d3c39"  # borders and outlines
STACK_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#e87ba4",
    "#eda100",
]  # one color per sample stack
EDGE_EPS = 1e-6


def parts(shape: object) -> list:
    """The polygons of a shapely shape (one polygon, or several)."""
    return list(getattr(shape, "geoms", [shape]))


def draw_border(ax: object, shape: object, width: float = 1.4) -> None:
    """Fill the study area in light gray and draw its border."""
    for part in parts(shape):
        x_edge, y_edge = part.exterior.xy
        ax.fill(x_edge, y_edge, color=STUDY_FILL, zorder=0)
        ax.plot(x_edge, y_edge, color=LINE_COLOR, linewidth=width, zorder=6)


def make_blocks(
    study: object, block_km: float
) -> tuple[list[dict], tuple[float, float]]:
    """Square blocks over the study area, in a checkerboard. Returns the blocks and the grid origin.

    The grid starts at the lower left corner of the study area. Only blocks that touch the study
    area are kept. A block is train when its column plus its row is even, and test when it is odd.
    """
    x_origin, y_origin, x_end, y_end = study.bounds
    blocks = []
    for block_x in range(math.ceil((x_end - x_origin) / block_km)):
        for block_y in range(math.ceil((y_end - y_origin) / block_km)):
            left = x_origin + block_x * block_km
            bottom = y_origin + block_y * block_km
            if not box(left, bottom, left + block_km, bottom + block_km).intersects(
                study
            ):
                continue
            blocks.append(
                {
                    "block_x": block_x,
                    "block_y": block_y,
                    "left": left,
                    "bottom": bottom,
                    "color": "train" if (block_x + block_y) % 2 == 0 else "test",
                }
            )
    return blocks, (x_origin, y_origin)


def set_limits(ax: object, blocks: list[dict], block_km: float) -> None:
    """Fit the axes to the grid of blocks and use the same scale in x and y."""
    ax.set_xlim(
        min(b["left"] for b in blocks), max(b["left"] for b in blocks) + block_km
    )
    ax.set_ylim(
        min(b["bottom"] for b in blocks), max(b["bottom"] for b in blocks) + block_km
    )
    ax.set_aspect("equal")


def block_rectangles(blocks: list[dict], block_km: float) -> list[Rectangle]:
    """One square patch per block."""
    return [Rectangle((b["left"], b["bottom"]), block_km, block_km) for b in blocks]


def draw_block_borders(ax: object, blocks: list[dict], block_km: float) -> None:
    """Draw only the borders of the blocks."""
    ax.add_collection(
        PatchCollection(
            block_rectangles(blocks, block_km),
            facecolor="none",
            edgecolor=LINE_COLOR,
            linewidth=0.8,
            zorder=5,
        )
    )


def plot_study_area(
    study: object, blocks: list[dict], block_km: float, n_frames: int, output: Path
) -> None:
    """Picture 1: the border of the study area and the checkerboard."""
    fig = Figure(figsize=(8.5, 8.5))
    ax = fig.subplots()
    draw_border(ax, study)
    for color in ("train", "test"):
        of_color = [b for b in blocks if b["color"] == color]
        ax.add_collection(
            PatchCollection(
                block_rectangles(of_color, block_km),
                facecolor=SPLIT_COLORS[color],
                edgecolor="white",
                linewidth=1.5,
                alpha=0.55,
                zorder=1,
            )
        )
    set_limits(ax, blocks, block_km)
    n_train = sum(b["color"] == "train" for b in blocks)
    ax.legend(
        handles=[
            Patch(
                facecolor=SPLIT_COLORS["train"],
                alpha=0.55,
                label=f"train blocks ({n_train})",
            ),
            Patch(
                facecolor=SPLIT_COLORS["test"],
                alpha=0.55,
                label=f"test blocks ({len(blocks) - n_train})",
            ),
            Line2D(
                [],
                [],
                color=LINE_COLOR,
                linewidth=1.4,
                label=f"study area ({study.area:,.0f} km2)",
            ),
        ],
        loc="lower left",
    )
    ax.set_xlabel("x (km, Antarctic polar stereographic)")
    ax.set_ylabel("y (km)")
    ax.set_title(
        f"Border of all {n_frames} frames and a checkerboard of {block_km:g} km blocks"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=120)


def crop_tile(
    tile: dict, gray: np.ndarray, grid_origin: tuple[float, float], block_km: float
) -> list[dict]:
    """Cut one tile along the block borders. Returns one piece for every block the tile touches.

    tile has its box in km (left, right, bottom, top), and gray is the picture of the tile, drawn
    with cells of cell_km. Each piece has its block, its box and its part of the picture.
    """
    x_origin, y_origin = grid_origin
    cell_km = SHRINK * PIXEL_M / METRES_PER_KM
    first_x = math.floor((tile["left"] - x_origin) / block_km)
    last_x = math.floor((tile["right"] - x_origin - EDGE_EPS) / block_km)
    first_y = math.floor((tile["bottom"] - y_origin) / block_km)
    last_y = math.floor((tile["top"] - y_origin - EDGE_EPS) / block_km)

    pieces = []
    for block_x in range(first_x, last_x + 1):
        for block_y in range(first_y, last_y + 1):
            left = max(tile["left"], x_origin + block_x * block_km)
            right = min(tile["right"], x_origin + (block_x + 1) * block_km)
            bottom = max(tile["bottom"], y_origin + block_y * block_km)
            top = min(tile["top"], y_origin + (block_y + 1) * block_km)
            # Rows count from the top of the tile, columns from its left side
            col_0, col_1 = (
                round((left - tile["left"]) / cell_km),
                round((right - tile["left"]) / cell_km),
            )
            row_0, row_1 = (
                round((tile["top"] - top) / cell_km),
                round((tile["top"] - bottom) / cell_km),
            )
            part = gray[row_0:row_1, col_0:col_1]
            if part.size == 0 or not part.any():
                continue  # nothing to show: no pixels, or no data
            pieces.append(
                {
                    "block_x": block_x,
                    "block_y": block_y,
                    "color": "train" if (block_x + block_y) % 2 == 0 else "test",
                    "extent": (
                        tile["left"] + col_0 * cell_km,
                        tile["left"] + col_1 * cell_km,
                        tile["top"] - row_1 * cell_km,
                        tile["top"] - row_0 * cell_km,
                    ),
                    "image": part,
                }
            )
    return pieces


def choose_sample(args: argparse.Namespace) -> dict[str, list[Path]]:
    """The sample stacks and their downloaded frames, oldest first (at most --max-dates)."""
    if args.stack_ids:
        wanted = [get_stack_id(text) for text in args.stack_ids]
    else:
        # the stacks with the most downloaded frames
        counts = defaultdict(int)
        for json_path in sorted(args.metadata_dir.rglob("*.json")):
            counts[get_stack_id(json_path.name)] += 1
        wanted = sorted(counts, key=lambda stack: (-counts[stack], stack))

    sample = {}
    for stack_id in wanted:
        frames, skipped = find_stack_frames(args.image_dir, stack_id)
        for frame in skipped:
            print(f"Skipping online-only frame (not downloaded): {frame.name}")
        if frames:
            sample[stack_id] = frames[: args.max_dates]
        if len(sample) == args.n_stacks:
            break
    return sample


def main() -> None:
    """Draw the three pictures."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--block-km", type=float, default=200)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument(
        "--stack-ids", nargs="+", default=None, help="Sample stacks, like 077_A_140."
    )
    parser.add_argument(
        "--n-stacks",
        type=int,
        default=2,
        help="How many stacks to draw when --stack-ids is not given.",
    )
    parser.add_argument(
        "--max-dates", type=int, default=3, help="Most dates drawn per stack."
    )
    args = parser.parse_args()
    block_km = args.block_km
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. study area and checkerboard
    json_files = sorted(args.metadata_dir.rglob("*.json"))
    if not json_files:
        sys.exit(f"No metadata JSON files found in {args.metadata_dir}")
    study = unary_union([Polygon(read_footprint(path)) for path in json_files])
    blocks, grid_origin = make_blocks(study, block_km)
    out_1 = args.output_dir / "1_study_area_blocks.png"
    plot_study_area(study, blocks, block_km, len(json_files), out_1)
    print(
        f"Study area: {study.area:,.0f} km2 from {len(json_files)} frames, {len(blocks)} blocks of {block_km:g} km"
    )
    print(f"Saved {out_1}")

    # 2. tiles of the sample stacks
    try:
        sample = choose_sample(args)
    except (ValueError, FileNotFoundError) as error:
        sys.exit(f"Cannot plot: {error}")
    if not sample:
        sys.exit("No downloaded frames found for the sample stacks")
    tileserver = load_tileserver(args.image_dir, args.cache_dir)
    tile_boxes = {}  # frame -> tiles with their box in km
    for stack_id, frames in sample.items():
        for frame in frames:
            tile_boxes[frame] = [
                {
                    **tile,
                    **{
                        key: tile[key] / METRES_PER_KM
                        for key in ("left", "right", "bottom", "top")
                    },
                }
                for tile in get_tile_boxes(frame, tileserver, args.tile_size)
            ]
        print(
            f"Sample stack {stack_id}: {len(frames)} dates, {sum(len(tile_boxes[f]) for f in frames)} tiles"
        )

    fig = Figure(figsize=(8.5, 8.5))
    ax = fig.subplots()
    draw_border(ax, study)
    draw_block_borders(ax, blocks, block_km)
    handles = []
    for (stack_id, frames), color in zip(sample.items(), STACK_COLORS, strict=False):
        outlines = [
            Rectangle(
                (t["left"], t["bottom"]), t["right"] - t["left"], t["top"] - t["bottom"]
            )
            for frame in frames
            for t in tile_boxes[frame]
        ]
        ax.add_collection(
            PatchCollection(
                outlines, facecolor=color, alpha=0.06, edgecolor="none", zorder=2
            )
        )
        ax.add_collection(
            PatchCollection(
                outlines,
                facecolor="none",
                edgecolor=color,
                linewidth=0.9,
                alpha=0.8,
                zorder=3,
            )
        )
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                linewidth=2,
                label=f"stack {stack_id} ({len(frames)} dates, {len(outlines)} tiles)",
            )
        )
    handles.append(
        Line2D(
            [],
            [],
            color=LINE_COLOR,
            linewidth=0.8,
            label=f"block border ({block_km:g} km)",
        )
    )
    set_limits(ax, blocks, block_km)
    ax.legend(handles=handles, loc="lower left")
    ax.set_xlabel("x (km, Antarctic polar stereographic)")
    ax.set_ylabel("y (km)")
    ax.set_title(
        f"Tiles of {args.tile_size} px of the sample stacks, over the blocks (before cropping)"
    )
    fig.tight_layout()
    out_2 = args.output_dir / "2_sample_tiles.png"
    fig.savefig(out_2, dpi=120)
    print(f"Saved {out_2}")

    # 3. crop the tiles at the block borders
    n_columns = max(len(frames) for frames in sample.values())
    fig = Figure(figsize=(6.6 * n_columns, 6.9 * len(sample)))
    axes = fig.subplots(len(sample), n_columns, squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for row, (stack_id, frames) in enumerate(sample.items()):
        for column, frame in enumerate(frames):
            ax = axes[row][column]
            ax.set_visible(True)
            stats = tileserver.get_stats(frame)
            n_cut = n_pieces = 0
            for tile in tile_boxes[frame]:
                gray = tileserver.render_tile(
                    frame, stats, args.tile_size, tile["row"], tile["col"]
                )
                pieces = crop_tile(tile, shrink(gray, SHRINK), grid_origin, block_km)
                n_cut += len(pieces) > 1
                n_pieces += len(pieces)
                for piece in pieces:
                    left, right, bottom, top = piece["extent"]
                    ax.imshow(
                        piece["image"],
                        cmap="gray",
                        vmin=0,
                        vmax=255,
                        extent=(left, right, bottom, top),
                        zorder=2,
                    )
                    ax.add_patch(
                        Rectangle(
                            (left, bottom),
                            right - left,
                            top - bottom,
                            facecolor="none",
                            edgecolor=SPLIT_COLORS[piece["color"]],
                            linewidth=1.6,
                            zorder=4,
                        )
                    )
            draw_border(ax, study, width=0.9)
            draw_block_borders(ax, blocks, block_km)
            set_limits(ax, blocks, block_km)
            ax.set_title(
                f"{stack_id}  cycle {get_cycle(frame.name)}  ({get_date(frame.name)})\n"
                f"{len(tile_boxes[frame])} tiles, {n_cut} cut by a border, {n_pieces} pieces",
                fontsize=12,
            )
            ax.set_xlabel("x (km)")
            print(
                f"  {stack_id} {get_date(frame.name)}: {len(tile_boxes[frame])} tiles, {n_cut} cut by a border, {n_pieces} pieces"
            )
    axes[0][0].legend(
        handles=[
            Line2D(
                [],
                [],
                color=SPLIT_COLORS["train"],
                linewidth=2.5,
                label="piece in a train block",
            ),
            Line2D(
                [],
                [],
                color=SPLIT_COLORS["test"],
                linewidth=2.5,
                label="piece in a test block",
            ),
            Line2D([], [], color=LINE_COLOR, linewidth=0.8, label="block border"),
        ],
        loc="lower left",
    )
    fig.suptitle(
        f"Sample tiles cropped at the borders of {block_km:g} km blocks", fontsize=15
    )
    fig.tight_layout()
    out_3 = args.output_dir / "3_sample_cropped.png"
    fig.savefig(out_3, dpi=100)
    print(f"Saved {out_3}")


if __name__ == "__main__":
    main()
