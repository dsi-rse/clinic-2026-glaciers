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

# Violet and green were checked with the dataviz palette validator (very distinct, also for
# color-blind readers). Green is light on white, so every panel has a labeled legend.
SPLIT_COLORS = {"train": "#6250d6", "test": "#1baf7a"}
DROPPED_COLOR = "#c4c2ba"  # light gray: dropped tiles
REGION_COLOR = "#eeede9"  # very light gray: the area of analysis
LINE_COLOR = "#3d3c39"  # block borders, tile outlines and the region outline
METRES_PER_KM = 1000
TEXT_COLOR = "#0b0b0b"
BLOCK_COLUMNS = ["block_x", "block_y", "block_color"]


def get_blocks(table: pd.DataFrame, block_km: float) -> pd.DataFrame:
    """One row per block that holds a tile: its lower left corner (km) and its color."""
    blocks = table[BLOCK_COLUMNS].drop_duplicates().copy()
    block_m = block_km * METRES_PER_KM
    blocks["x_km"] = (table["x_m"].min() + blocks["block_x"] * block_m) / METRES_PER_KM
    blocks["y_km"] = (table["y_m"].min() + blocks["block_y"] * block_m) / METRES_PER_KM
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


def plot_tile_split(
    table: pd.DataFrame, tile_m: float, output: Path, block_km: float
) -> None:
    """Save the three-panel map. table is the output of split_by_blocks."""
    missing = [name for name in BLOCK_COLUMNS if name not in table.columns]
    if missing:
        msg = f"The table has no {missing} columns: run split_tiles.py again"
        raise ValueError(msg)

    tile_km = tile_m / METRES_PER_KM
    blocks = get_blocks(table, block_km)
    region = get_region(table, tile_km)
    x_km = table["x_m"] / METRES_PER_KM
    y_km = table["y_m"] / METRES_PER_KM

    fig = Figure(figsize=(21, 7.8))
    axes = fig.subplots(1, 3, sharex=True, sharey=True)
    panel_a, panel_b, panel_c = axes
    for ax in axes:
        draw_region(ax, region)

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

    # Axes limits: the whole grid of blocks
    left = blocks["x_km"].min()
    right = blocks["x_km"].max() + block_km
    bottom = blocks["y_km"].min()
    top = blocks["y_km"].max() + block_km
    for ax in axes:
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
    panel_a.legend(handles=[*block_handles, region_handle], loc="lower left")

    panel_b.set_title("B. Where the tiles land (outlines only)", color=TEXT_COLOR)
    tile_handle = Line2D([], [], color=LINE_COLOR, linewidth=0.5, label="tile outline")
    panel_b.legend(handles=[*block_handles, tile_handle], loc="lower left")

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
    panel_c.legend(handles=[*tile_handles, border_handle], loc="lower left")

    fig.suptitle(
        f"{block_km:g} km blocks, {tile_km:.1f} km tiles: "
        f"{len(table)} tiles from {table['frame'].nunique()} frames",
        color=TEXT_COLOR,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
