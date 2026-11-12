"""Draw the block split on a map, in three panels side by side.

A. The blocks (train or test) over the area of analysis.
B. Where the tiles land inside the blocks (tiles drawn as plain outlines).
C. The tiles colored by their split, with only the block borders drawn.
"""

from pathlib import Path

import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import box
from shapely.ops import unary_union

from utils.tile_split import get_tile_corner

# Violet and green were checked with the dataviz palette validator (very distinct, also for
# color-blind readers). Green is light on white, so every panel has a labeled legend.
SPLIT_COLORS = {"train": "#6250d6", "test": "#1baf7a"}
DROPPED_COLOR = "#c4c2ba"  # light gray: dropped tiles
REGION_COLOR = "#eeede9"  # very light gray: the area of analysis
LINE_COLOR = "#3d3c39"  # block borders, tile outlines and the region outline
METRES_PER_KM = 1000
TEXT_COLOR = "#0b0b0b"
BLOCK_COLUMNS = ["block_x", "block_y", "block_color"]


def get_blocks(table: pd.DataFrame, block_km: float, tile_m: float) -> pd.DataFrame:
    """One row per block that holds a tile: its lower left corner (km) and its color.

    The grid starts at the lower left corner of the tiles, as in split_by_blocks.
    """
    block_m = block_km * METRES_PER_KM
    blocks = table[BLOCK_COLUMNS].drop_duplicates().copy()
    x_start, y_start = get_tile_corner(table, tile_m)
    blocks["x_km"] = (x_start + blocks["block_x"] * block_m) / METRES_PER_KM
    blocks["y_km"] = (y_start + blocks["block_y"] * block_m) / METRES_PER_KM
    return blocks


def make_squares(
    x_km: pd.Series, y_km: pd.Series, side_km: float, centered: bool
) -> list[Rectangle]:
    """Square patches. centered: x, y are the center of each square, else its lower left."""
    shift = side_km / 2 if centered else 0
    return [
        Rectangle((x - shift, y - shift), side_km, side_km)
        for x, y in zip(x_km, y_km, strict=True)
    ]


def get_region(table: pd.DataFrame, tile_km: float) -> object:
    """The area covered by the tiles (the area of analysis), as a smooth shape in km."""
    half = tile_km / 2
    squares = [
        box(x - half, y - half, x + half, y + half)
        for x, y in zip(
            table["x_m"] / METRES_PER_KM, table["y_m"] / METRES_PER_KM, strict=True
        )
    ]
    region = unary_union(squares)
    # Tiles from different frames do not line up, which leaves hairline gaps and spikes in the
    # outline. Growing the shape by half a tile and shrinking it back closes them.
    return region.buffer(half).buffer(-half)


def draw_region(ax: object, region: object) -> None:
    """Fill the area of analysis in light gray and outline it."""
    for part in getattr(region, "geoms", [region]):
        x_edge, y_edge = part.exterior.xy
        ax.fill(x_edge, y_edge, color=REGION_COLOR, zorder=0)
        ax.plot(x_edge, y_edge, color=LINE_COLOR, linewidth=0.8, zorder=5)


def draw_blocks(
    ax: object, blocks: pd.DataFrame, block_km: float, alpha: float, filled: bool
) -> None:
    """Draw the blocks. filled: tinted with their color. Otherwise only the borders."""
    if filled:
        for name, color in SPLIT_COLORS.items():
            of_color = blocks[blocks["block_color"] == name]
            squares = make_squares(of_color["x_km"], of_color["y_km"], block_km, False)
            ax.add_collection(
                PatchCollection(
                    squares,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=1,
                    alpha=alpha,
                    zorder=1,
                )
            )
    else:
        squares = make_squares(blocks["x_km"], blocks["y_km"], block_km, False)
        ax.add_collection(
            PatchCollection(
                squares,
                facecolor="none",
                edgecolor=LINE_COLOR,
                linewidth=0.9,
                zorder=4,
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


def plot_tile_split(
    table: pd.DataFrame,
    tile_m: float,
    output: Path,
    block_km: float,
    region: tuple[float, float, float, float] | None = None,
) -> None:
    """Save the three-panel map. table is the output of split_by_blocks.

    region is the rectangle (x_min, y_min, x_max, y_max in km) used as the area of analysis.
    Without it, the area is the outline of the tiles.
    """
    missing = [name for name in BLOCK_COLUMNS if name not in table.columns]
    if missing:
        msg = f"The table has no {missing} columns: run split_tiles.py again"
        raise ValueError(msg)

    tile_km = tile_m / METRES_PER_KM
    blocks = get_blocks(table, block_km, tile_m)
    if region is None:
        area = get_region(table, tile_km)
    else:
        area = box(*region)
    x_km = table["x_m"] / METRES_PER_KM
    y_km = table["y_m"] / METRES_PER_KM

    fig = Figure(figsize=(21, 7.8))
    axes = fig.subplots(1, 3, sharex=True, sharey=True)
    panel_a, panel_b, panel_c = axes
    for ax in axes:
        draw_region(ax, area)

    # A: the blocks over the area of analysis
    draw_blocks(panel_a, blocks, block_km, alpha=0.9, filled=True)

    # B: the tiles as plain outlines, over lightly tinted blocks
    draw_blocks(panel_b, blocks, block_km, alpha=0.3, filled=True)
    panel_b.add_collection(
        PatchCollection(
            make_squares(x_km, y_km, tile_km, True),
            facecolor="none",
            edgecolor=LINE_COLOR,
            linewidth=0.3,
            alpha=0.25,
            zorder=3,
        )
    )

    # C: the block borders, and the tiles colored by split (dropped ones light gray)
    draw_blocks(panel_c, blocks, block_km, alpha=1, filled=False)
    colors = {**SPLIT_COLORS, "gap": DROPPED_COLOR}
    for name in ("gap", "train", "test"):
        keep = table["split"] == name
        panel_c.add_collection(
            PatchCollection(
                make_squares(x_km[keep], y_km[keep], tile_km, True),
                facecolor=colors[name],
                edgecolor="none",
                alpha=0.3 if name != "gap" else 0.6,
                zorder=2,
            )
        )

    # Axes limits: the whole grid of blocks, or the region
    if region is None:
        left = blocks["x_km"].min()
        right = blocks["x_km"].max() + block_km
        bottom = blocks["y_km"].min()
        top = blocks["y_km"].max() + block_km
    else:
        left, bottom, right, top = region
        margin = 0.03 * max(right - left, top - bottom)
        left, bottom, right, top = (
            left - margin,
            bottom - margin,
            right + margin,
            top + margin,
        )
    for ax in axes:
        clip_to_box(ax, region)
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_aspect("equal")
        ax.set_xlabel("x (km, Antarctic polar stereographic)")
    panel_a.set_ylabel("y (km, Antarctic polar stereographic)")

    counts = table["split"].value_counts()
    region_handle = Line2D(
        [], [], color=LINE_COLOR, linewidth=0.8, label="area of analysis"
    )
    block_handles = [
        Patch(facecolor=color, label=f"{name} blocks")
        for name, color in SPLIT_COLORS.items()
    ]
    panel_a.set_title("A. Blocks over the area of analysis", color=TEXT_COLOR)
    legend_below(panel_a, [*block_handles, region_handle])

    panel_b.set_title("B. Where the tiles land (outlines only)", color=TEXT_COLOR)
    tile_handle = Line2D([], [], color=LINE_COLOR, linewidth=0.5, label="tile outline")
    legend_below(panel_b, [*block_handles, tile_handle])

    panel_c.set_title("C. Tiles colored by split, block borders only", color=TEXT_COLOR)
    tile_handles = [
        Patch(
            facecolor=colors[name],
            label=f"{label} tiles ({counts.get(name, 0)})",
        )
        for name, label in (("train", "train"), ("test", "test"), ("gap", "dropped"))
    ]
    border_handle = Line2D(
        [], [], color=LINE_COLOR, linewidth=0.9, label="block border"
    )
    legend_below(panel_c, [*tile_handles, border_handle], ncol=2)

    fig.suptitle(
        f"{block_km:g} km blocks, {tile_km:.1f} km tiles: "
        f"{len(table)} tiles from {table['frame'].nunique()} frames",
        color=TEXT_COLOR,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, bbox_inches="tight")
