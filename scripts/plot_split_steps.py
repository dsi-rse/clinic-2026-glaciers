"""Show how the train and test split looks on the data, in four pictures.

The pictures follow the steps of scripts/split_tiles.py (same options, one tile size and one
block size at a time). Tiles are never cropped: a tile is kept whole or dropped.

Pictures (in DATA_DIR/split_steps):
    1_study_area.png        the border of the study area and the bounding box (if --region)
    2_blocks.png            the study area left after the box, with the checkerboard of blocks
    3_sample_tiles.png      the tiles of a few sample stacks, over the blocks
    4_sample_split.png      the same tiles after the split: train, test or dropped

A stack id is the track, the direction and the frame, for example 077_A_140: the same place on
different dates. Only a few stacks are drawn in pictures 3 and 4.

Examples:
    uv run python scripts/plot_split_steps.py --tile-size 1024 --block-km 40.96 --gap-km 5
    uv run python scripts/plot_split_steps.py --region -1686.068 -362.187 -1561.229 -255.680 --tile-size 1024 --block-km 40.96 --gap-km 5
"""

import argparse
import os
import sys
from pathlib import Path

from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import box

from utils.plot_split import (
    DROPPED_COLOR,
    LINE_COLOR,
    SPLIT_COLORS,
    draw_blocks,
    draw_region,
    get_blocks,
    get_region,
    make_squares,
)
from utils.settings import DATA_DIR
from utils.stack_plot import get_date, get_stack_id, get_tile_boxes, shrink
from utils.tile_split import (
    DEFAULT_TILE_SIZE,
    filter_region,
    find_frames,
    load_tiles,
    load_tileserver,
    split_by_blocks,
)

DEFAULT_IMAGE_DIR = DATA_DIR / os.environ.get("LABELSTUDIO_IMAGE_SUBDIR", "")
DEFAULT_CACHE_DIR = DATA_DIR / "contrast_enhancement" / "stats_cache"
DEFAULT_OUTPUT_DIR = DATA_DIR / "split_steps"
METRES_PER_KM = 1000
SHRINK = (
    8  # tile pictures are drawn with 8 x 8 pixels averaged into one, enough for a map
)
BOX_COLOR = "#e34948"  # red: the bounding box
STACK_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#e87ba4",
    "#eda100",
]  # one color per sample stack


def parse_args() -> argparse.Namespace:
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--block-km", type=float, required=True)
    parser.add_argument("--gap-km", type=float, default=0.0)
    parser.add_argument("--swap", action="store_true")
    parser.add_argument(
        "--region",
        type=float,
        nargs=4,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        help="Bounding box in km, like in split_tiles.py. Without it, no tiles are removed.",
    )
    parser.add_argument(
        "--stack-ids",
        nargs="+",
        help="Sample stacks to draw, like 077_A_140. Default: the stacks with most frames.",
    )
    parser.add_argument(
        "--n-stacks", type=int, default=2, help="How many sample stacks to draw."
    )
    parser.add_argument(
        "--max-dates", type=int, default=3, help="Most dates drawn per stack."
    )
    return parser.parse_args()


def set_limits(ax: object, area: object) -> None:
    """Fit the axes to a shape (in km), with a small margin and the same scale in x and y."""
    left, bottom, right, top = area.bounds
    margin = 0.03 * max(right - left, top - bottom)
    ax.set_xlim(left - margin, right + margin)
    ax.set_ylim(bottom - margin, top + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km, Antarctic polar stereographic)")


def draw_box(ax: object, region: tuple | None) -> None:
    """Draw the bounding box as a red outline."""
    if region is None:
        return
    x_min, y_min, x_max, y_max = region
    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor="none",
            edgecolor=BOX_COLOR,
            linewidth=2,
            zorder=7,
        )
    )


def clip_to_box(ax: object, region: tuple | None) -> None:
    """Cut the blocks and tiles drawn so far at the bounding box, so the area is the box."""
    if region is None:
        return
    x_min, y_min, x_max, y_max = region
    clip = Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min, transform=ax.transData
    )
    for collection in ax.collections:
        collection.set_clip_path(clip)


def legend_below(ax: object, handles: list, ncol: int | None = None) -> None:
    """Put the legend outside the plot, right after the x axis title."""
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=ncol or len(handles),
        frameon=False,
    )


def block_handles() -> list:
    """Legend entries for the train and test blocks."""
    return [
        Patch(facecolor=color, alpha=0.6, label=f"{name} blocks")
        for name, color in SPLIT_COLORS.items()
    ]


def plot_study_area(
    all_tiles: object, tile_km: float, region: tuple | None, output: Path
) -> None:
    """Picture 1: the border of the study area, with the bounding box."""
    area = get_region(all_tiles, tile_km)
    fig = Figure(figsize=(8.5, 8.5))
    ax = fig.subplots()
    draw_region(ax, area)
    draw_box(ax, region)
    set_limits(ax, area)
    handles = [Line2D([], [], color=LINE_COLOR, linewidth=0.8, label="study area")]
    if region is not None:
        handles.append(Line2D([], [], color=BOX_COLOR, linewidth=2, label="box"))
    ax.legend(handles=handles, loc="lower left")
    ax.set_ylabel("y (km)")
    ax.set_title(f"Border of the study area ({len(all_tiles)} tiles)")
    fig.tight_layout()
    fig.savefig(output, dpi=120, bbox_inches="tight")


def plot_blocks(
    table: object, tile_km: float, block_km: float, region: tuple | None, output: Path
) -> None:
    """Picture 2: the study area left after the box, with the checkerboard of blocks."""
    area = get_region(table, tile_km)
    blocks = get_blocks(table, block_km, tile_km * METRES_PER_KM)
    fig = Figure(figsize=(8.5, 8.5))
    ax = fig.subplots()
    draw_region(ax, area)
    draw_blocks(ax, blocks, block_km, alpha=0.7, filled=True)
    clip_to_box(ax, region)
    set_limits(ax, area if region is None else box(*region))
    legend_below(ax, block_handles())
    ax.set_ylabel("y (km)")
    ax.set_title(f"Study area after the box, with {block_km:g} km blocks")
    fig.tight_layout()
    fig.savefig(output, dpi=120, bbox_inches="tight")


def choose_sample(
    table: object, n_stacks: int, max_dates: int, stack_ids: list[str] | None = None
) -> dict[str, list[str]]:
    """The sample stacks: stack id -> frame names, oldest first.

    Without stack_ids, the stacks with the most frames in the table.
    """
    stacks: dict[str, set[str]] = {}
    for name in table["frame"].unique():
        try:
            stacks.setdefault(get_stack_id(name), set()).add(name)
        except ValueError:
            continue
    if stack_ids:
        chosen = [get_stack_id(text) for text in stack_ids]
        for stack in chosen:
            if stack not in stacks:
                sys.exit(f"No tiles of stack {stack} are left after the bounding box")
    else:
        chosen = sorted(stacks, key=lambda stack: (-len(stacks[stack]), stack))[
            :n_stacks
        ]
    return {
        stack: sorted(stacks[stack], key=lambda name: (get_date(name), name))[
            :max_dates
        ]
        for stack in chosen
    }


def plot_sample_tiles(
    table: object,
    sample: dict[str, list[str]],
    tile_km: float,
    block_km: float,
    region: tuple | None,
    output: Path,
) -> None:
    """Picture 3: the tiles of the sample stacks as outlines, over the blocks."""
    area = get_region(table, tile_km)
    fig = Figure(figsize=(8.5, 8.5))
    ax = fig.subplots()
    draw_region(ax, area)
    draw_blocks(
        ax,
        get_blocks(table, block_km, tile_km * METRES_PER_KM),
        block_km,
        alpha=0.25,
        filled=True,
    )
    handles = block_handles()
    for (stack_id, names), color in zip(sample.items(), STACK_COLORS, strict=False):
        tiles = table[table["frame"].isin(names)]
        squares = make_squares(
            tiles["x_m"] / METRES_PER_KM, tiles["y_m"] / METRES_PER_KM, tile_km, True
        )
        ax.add_collection(
            PatchCollection(
                squares, facecolor=color, alpha=0.08, edgecolor="none", zorder=2
            )
        )
        ax.add_collection(
            PatchCollection(
                squares, facecolor="none", edgecolor=color, linewidth=1, zorder=3
            )
        )
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                linewidth=2,
                label=f"stack {stack_id} ({len(names)} dates, {len(tiles)} tiles)",
            )
        )
    clip_to_box(ax, region)
    set_limits(ax, area if region is None else box(*region))
    legend_below(ax, handles, ncol=1)
    ax.set_ylabel("y (km)")
    ax.set_title(f"Tiles of the sample stacks ({tile_km:.1f} km), over the blocks")
    fig.tight_layout()
    fig.savefig(output, dpi=120, bbox_inches="tight")


def plot_sample_split(
    table: object,
    sample: dict[str, list[str]],
    args: argparse.Namespace,
    tile_km: float,
    region: tuple | None,
    output: Path,
) -> None:
    """Picture 4: the sample tiles with their image, kept whole or dropped.

    The edge of a tile is violet (train) or green (test). Dropped tiles are faded and gray.
    """
    area = get_region(table, tile_km)
    blocks = get_blocks(table, args.block_km, tile_km * METRES_PER_KM)
    tileserver = load_tileserver(args.image_dir, args.cache_dir)
    paths = {path.name: path for path in find_frames(args.image_dir)}
    n_columns = max(len(names) for names in sample.values())
    fig = Figure(figsize=(6.6 * n_columns, 6.9 * len(sample)))
    axes = fig.subplots(len(sample), n_columns, squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for row, (stack_id, names) in enumerate(sample.items()):
        for column, name in enumerate(names):
            ax = axes[row][column]
            ax.set_visible(True)
            frame = paths[name]
            stats = tileserver.get_stats(frame)
            split_of = {
                (t.row, t.col): t.split
                for t in table[table["frame"] == name].itertuples()
            }
            counts = {"train": 0, "test": 0, "gap": 0}
            for tile in get_tile_boxes(frame, tileserver, args.tile_size):
                split = split_of.get((tile["row"], tile["col"]))
                if split is None:
                    continue  # outside the bounding box
                counts[split] += 1
                gray = shrink(
                    tileserver.render_tile(
                        frame, stats, args.tile_size, tile["row"], tile["col"]
                    ),
                    SHRINK,
                )
                left, right = (
                    tile["left"] / METRES_PER_KM,
                    tile["right"] / METRES_PER_KM,
                )
                bottom, top = (
                    tile["bottom"] / METRES_PER_KM,
                    tile["top"] / METRES_PER_KM,
                )
                ax.imshow(
                    gray,
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                    extent=(left, right, bottom, top),
                    alpha=0.3 if split == "gap" else 1,
                    zorder=2,
                )
                ax.add_patch(
                    Rectangle(
                        (left, bottom),
                        right - left,
                        top - bottom,
                        facecolor="none",
                        edgecolor=DROPPED_COLOR
                        if split == "gap"
                        else SPLIT_COLORS[split],
                        linewidth=1.4,
                        zorder=4,
                    )
                )
            draw_region(ax, area)
            draw_blocks(ax, blocks, args.block_km, alpha=1, filled=False)
            clip_to_box(ax, region)
            set_limits(ax, area if region is None else box(*region))
            ax.set_title(
                f"{stack_id}  ({get_date(name)})\n"
                f"{counts['train']} train, {counts['test']} test, {counts['gap']} dropped",
                fontsize=12,
            )
            print(f"  {stack_id} {get_date(name)}: {counts}")

    fig.legend(
        handles=[
            Line2D([], [], color=SPLIT_COLORS["train"], linewidth=2.5, label="train"),
            Line2D([], [], color=SPLIT_COLORS["test"], linewidth=2.5, label="test"),
            Line2D([], [], color=DROPPED_COLOR, linewidth=2.5, label="dropped"),
            Line2D([], [], color=LINE_COLOR, linewidth=0.8, label="block border"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        f"Sample tiles after the split: {args.block_km:g} km blocks, {args.gap_km:g} km gap",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=3)
    fig.savefig(output, dpi=100, bbox_inches="tight")


def main() -> None:
    """Run the split and draw the four pictures."""
    args = parse_args()
    region = tuple(args.region) if args.region else None
    if region is not None and not (region[0] < region[2] and region[1] < region[3]):
        sys.exit("--region needs X_MIN < X_MAX and Y_MIN < Y_MAX")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        all_tiles, tile_m = load_tiles(args.image_dir, args.cache_dir, args.tile_size)
        tile_km = tile_m / METRES_PER_KM
        plot_study_area(
            all_tiles, tile_km, region, args.output_dir / "1_study_area.png"
        )
        print(f"{len(all_tiles)} tiles in the study area")

        tiles = (
            all_tiles if region is None else filter_region(all_tiles, tile_m, region)
        )
        print(f"{len(tiles)} tiles after the bounding box")
        table = split_by_blocks(tiles, tile_m, args.block_km, args.gap_km, args.swap)
    except (ValueError, FileNotFoundError) as error:
        sys.exit(f"Cannot split: {error}")
    print(table["split"].value_counts().to_string())

    plot_blocks(table, tile_km, args.block_km, region, args.output_dir / "2_blocks.png")
    sample = choose_sample(table, args.n_stacks, args.max_dates, args.stack_ids)
    for stack_id, names in sample.items():
        print(f"Sample stack {stack_id}: {len(names)} dates")
    plot_sample_tiles(
        table,
        sample,
        tile_km,
        args.block_km,
        region,
        args.output_dir / "3_sample_tiles.png",
    )
    plot_sample_split(
        table,
        sample,
        args,
        tile_km,
        region,
        args.output_dir / "4_sample_split.png",
    )
    print(f"Saved the four pictures in {args.output_dir}")


if __name__ == "__main__":
    main()
