"""Plot all the frames of one stack (the same place on different dates) to see the movement.

A stack id is the track, the direction and the frame, for example 077_A_140. Frames with the
same stack id show the same ground on different dates (12 days apart), so features of the ice
(crevasses, for example) appear shifted from one date to the next.

For each date we cut the frame into tiles, draw the tiles at their position on the map, and
show one panel per date with the same map limits. A last panel zooms in on the most textured
spot that both the first and the last date cover, and overlays the two dates: red is the
first date and cyan is the last date. Where the ice did not move, the two dates agree and the
overlay looks gray. Where it moved, red and cyan fringes appear.
"""

import re
from datetime import date
from pathlib import Path

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter, uniform_filter

from utils.tile_split import find_frames, get_frame_tiles, is_online_only

# Frame names look like ..._GCOV_<cycle>_<track>_<A or D>_<frame>_..._<yyyymmdd>T<hhmmss>_...
NAME_PATTERN = re.compile(r"_(\d+)_(\d+)_([AD])_(\d+)_")
DATE_PATTERN = re.compile(r"_(\d{4})(\d{2})(\d{2})T\d{6}_")
STACK_ID_PATTERN = re.compile(r"\d+_[AD]_\d+")

METRES_PER_KM = 1000
MIN_DATES = 2  # an overlay needs at least two dates
DECIMATE = 4  # every canvas cell is 4 x 4 pixels (40 m), to keep the picture light
DISPLAY_STEP = 4  # extra step for the full-area panels (160 m per cell is enough there)
SEARCH_STEP = 8  # coarse step used to look for the most textured spot
SEARCH_WINDOW = 16  # window (in coarse cells) used to measure texture
COVER_MIN = 0.99  # share of a window that both dates must cover
SMOOTH_SIGMA = (
    1.5  # light smoothing of the zoom, so the speckle does not hide the shapes
)
LOW_PERCENT, HIGH_PERCENT = 2, 98
OUTLINE_COLOR = "#eda100"
ZOOM_COLOR = "#d93a39"


def get_stack_id(text: str) -> str:
    """Return the stack id (for example 077_A_140) from a stack id or a full frame name."""
    match = NAME_PATTERN.search(text)
    if match is not None:
        _cycle, track, direction, frame = match.groups()
        return f"{track}_{direction}_{frame}"
    if STACK_ID_PATTERN.fullmatch(text):
        return text
    msg = f"Cannot read a stack id (like 077_A_140) from: {text}"
    raise ValueError(msg)


def get_cycle(name: str) -> str:
    """Return the cycle number of a frame name (the pass in time)."""
    match = NAME_PATTERN.search(name)
    if match is None:
        msg = f"Cannot find the cycle in the name: {name}"
        raise ValueError(msg)
    return match.group(1)


def get_date(name: str) -> date:
    """Return the acquisition date of a frame, read from its name."""
    match = DATE_PATTERN.search(name)
    if match is None:
        msg = f"Cannot find the date in the name: {name}"
        raise ValueError(msg)
    year, month, day = (int(part) for part in match.groups())
    return date(year, month, day)


def find_stack_frames(
    image_dir: Path, stack_id: str, include_online_only: bool = False
) -> tuple[list[Path], list[Path]]:
    """Find the frames of one stack, oldest first.

    Returns the frames to use and the online-only frames that were skipped. A frame that Box
    keeps online-only is skipped unless include_online_only is True, because reading it means
    waiting for a large download.
    """
    in_stack = []
    for frame in find_frames(image_dir):
        try:
            same_stack = get_stack_id(frame.name) == stack_id
        except ValueError:
            same_stack = False
        if same_stack:
            in_stack.append(frame)
    in_stack.sort(key=lambda frame: (get_date(frame.name), frame.name))

    if include_online_only:
        return in_stack, []
    skipped = [frame for frame in in_stack if is_online_only(frame)]
    return [frame for frame in in_stack if frame not in skipped], skipped


def get_tile_boxes(frame: Path, tileserver: object, tile_size: int) -> list[dict]:
    """Tiles of one frame with their box on the map, in metres.

    Each box has the tile row and col, and left, right, bottom and top. Tiles on the right
    and bottom edges of a frame are smaller than the others.
    """
    rows, tile_m = get_frame_tiles(frame, tileserver, tile_size)
    stats = tileserver.get_stats(frame)
    pixel_m = tile_m / tile_size

    boxes = []
    for tile in rows:
        width = min(tile_size, stats["width"] - tile["col"] * tile_size)
        height = min(tile_size, stats["height"] - tile["row"] * tile_size)
        half_width = width * pixel_m / 2
        half_height = height * pixel_m / 2
        boxes.append(
            {
                "row": tile["row"],
                "col": tile["col"],
                "left": tile["x_m"] - half_width,
                "right": tile["x_m"] + half_width,
                "bottom": tile["y_m"] - half_height,
                "top": tile["y_m"] + half_height,
            }
        )
    return boxes


def shrink(gray: np.ndarray, factor: int) -> np.ndarray:
    """Average blocks of factor x factor pixels, ignoring the pixels with no data (0)."""
    height = gray.shape[0] // factor * factor
    width = gray.shape[1] // factor * factor
    blocks = gray[:height, :width].reshape(
        height // factor, factor, width // factor, factor
    )
    valid = blocks > 0
    counts = valid.sum(axis=(1, 3))
    totals = (blocks * valid).sum(axis=(1, 3), dtype=np.float32)
    return np.where(counts > 0, totals / np.maximum(counts, 1), 0).astype(np.uint8)


def make_canvas(
    frame: Path,
    boxes: list[dict],
    tileserver: object,
    tile_size: int,
    bounds: tuple[float, float, float, float],
    cell_m: float,
) -> np.ndarray:
    """Draw the tiles of one frame at their map position on a canvas shared by all dates."""
    x_min, y_min, x_max, y_max = bounds
    canvas = np.zeros(
        (
            int(np.ceil((y_max - y_min) / cell_m)),
            int(np.ceil((x_max - x_min) / cell_m)),
        ),
        dtype=np.uint8,
    )
    stats = tileserver.get_stats(frame)
    for box in boxes:
        gray = tileserver.render_tile(frame, stats, tile_size, box["row"], box["col"])
        small = shrink(gray, DECIMATE)
        left = round((box["left"] - x_min) / cell_m)
        top = round((y_max - box["top"]) / cell_m)
        canvas[top : top + small.shape[0], left : left + small.shape[1]] = small
    return canvas


def find_textured_spot(first: np.ndarray, last: np.ndarray) -> tuple[int, int]:
    """Row and column (in canvas cells) of the most textured spot covered by both dates."""
    coarse_first = first[::SEARCH_STEP, ::SEARCH_STEP].astype(np.float32)
    covered = (first[::SEARCH_STEP, ::SEARCH_STEP] > 0) & (
        last[::SEARCH_STEP, ::SEARCH_STEP] > 0
    )
    mean = uniform_filter(coarse_first, SEARCH_WINDOW)
    texture = uniform_filter(coarse_first**2, SEARCH_WINDOW) - mean**2
    cover = uniform_filter(covered.astype(np.float32), SEARCH_WINDOW)
    texture[cover < COVER_MIN] = -1
    row, col = np.unravel_index(np.argmax(texture), texture.shape)
    return int(row) * SEARCH_STEP, int(col) * SEARCH_STEP


def stretch(image: np.ndarray) -> np.ndarray:
    """Scale an image to 0..1 with the 2nd and 98th percentiles of its valid pixels."""
    valid = image[image > 0]
    low, high = np.percentile(valid, [LOW_PERCENT, HIGH_PERCENT])
    return np.clip((image - low) / max(high - low, 1), 0, 1)


def plot_stack(
    stack_id: str,
    frames: list[Path],
    tileserver: object,
    tile_size: int,
    output: Path,
    zoom_km: float,
) -> None:
    """Save one picture of the stack: a panel per date and a zoomed overlay."""
    boxes = [get_tile_boxes(frame, tileserver, tile_size) for frame in frames]
    all_boxes = [box for frame_boxes in boxes for box in frame_boxes]
    bounds = (
        min(box["left"] for box in all_boxes),
        min(box["bottom"] for box in all_boxes),
        max(box["right"] for box in all_boxes),
        max(box["top"] for box in all_boxes),
    )
    x_min, y_min, x_max, y_max = bounds
    cell_m = DECIMATE * 10.0
    canvases = [
        make_canvas(frame, frame_boxes, tileserver, tile_size, bounds, cell_m)
        for frame, frame_boxes in zip(frames, boxes, strict=True)
    ]

    has_overlay = len(frames) >= MIN_DATES
    n_panels = len(frames) + (1 if has_overlay else 0)
    fig = Figure(figsize=(6.6 * n_panels, 7.2))
    axes = fig.subplots(1, n_panels, squeeze=False)[0]
    km_extent = (
        x_min / METRES_PER_KM,
        x_max / METRES_PER_KM,
        y_min / METRES_PER_KM,
        y_max / METRES_PER_KM,
    )

    # the zoom box, found from the first and the last date
    zoom_box = None
    if has_overlay:
        row, col = find_textured_spot(canvases[0], canvases[-1])
        half = int(zoom_km * METRES_PER_KM / cell_m / 2)
        top_row = min(max(row - half, 0), canvases[0].shape[0] - 2 * half)
        left_col = min(max(col - half, 0), canvases[0].shape[1] - 2 * half)
        zoom_box = (top_row, left_col, 2 * half)

    # one panel per date
    for ax, frame, frame_boxes, canvas in zip(
        axes, frames, boxes, canvases, strict=False
    ):
        ax.imshow(
            canvas[::DISPLAY_STEP, ::DISPLAY_STEP],
            cmap="gray",
            vmin=0,
            vmax=255,
            extent=km_extent,
        )
        outlines = [
            Rectangle(
                (box["left"] / METRES_PER_KM, box["bottom"] / METRES_PER_KM),
                (box["right"] - box["left"]) / METRES_PER_KM,
                (box["top"] - box["bottom"]) / METRES_PER_KM,
            )
            for box in frame_boxes
        ]
        ax.add_collection(
            PatchCollection(
                outlines, facecolor="none", edgecolor=OUTLINE_COLOR, linewidth=0.8
            )
        )
        if zoom_box is not None:
            top_row, left_col, size = zoom_box
            ax.add_patch(
                Rectangle(
                    (
                        (x_min + left_col * cell_m) / METRES_PER_KM,
                        (y_max - (top_row + size) * cell_m) / METRES_PER_KM,
                    ),
                    size * cell_m / METRES_PER_KM,
                    size * cell_m / METRES_PER_KM,
                    facecolor="none",
                    edgecolor=ZOOM_COLOR,
                    linewidth=2,
                )
            )
        ax.set_title(
            f"Cycle {get_cycle(frame.name)}  |  {get_date(frame.name)}  |  {len(frame_boxes)} tiles",
            fontsize=14,
        )
        ax.set_xlabel("x (km, Antarctic polar stereographic)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y (km)")
    axes[0].legend(
        handles=[
            Line2D([], [], color=OUTLINE_COLOR, linewidth=1.5, label="tile outline"),
            Line2D([], [], color=ZOOM_COLOR, linewidth=2, label="zoom area"),
        ],
        loc="lower left",
    )

    # the overlay of the first and the last date, zoomed
    if zoom_box is not None:
        top_row, left_col, size = zoom_box
        window = (slice(top_row, top_row + size), slice(left_col, left_col + size))
        first = stretch(
            gaussian_filter(canvases[0][window].astype(np.float32), SMOOTH_SIGMA)
        )
        last = stretch(
            gaussian_filter(canvases[-1][window].astype(np.float32), SMOOTH_SIGMA)
        )
        overlay = np.dstack([first, last, last])
        ax = axes[-1]
        ax.imshow(
            overlay,
            extent=(
                (x_min + left_col * cell_m) / METRES_PER_KM,
                (x_min + (left_col + size) * cell_m) / METRES_PER_KM,
                (y_max - (top_row + size) * cell_m) / METRES_PER_KM,
                (y_max - top_row * cell_m) / METRES_PER_KM,
            ),
        )
        days = (get_date(frames[-1].name) - get_date(frames[0].name)).days
        ax.set_title(
            f"Overlay, zoomed: red = {get_date(frames[0].name)}, cyan = {get_date(frames[-1].name)}\n"
            f"({days} days apart; gray = did not move, red/cyan fringes = moved)",
            fontsize=12,
        )
        ax.set_xlabel("x (km)")
        ax.set_aspect("equal")

    tile_km = tile_size * 10 / METRES_PER_KM
    fig.suptitle(
        f"Stack {stack_id}: {len(frames)} dates, tiles of {tile_size} px ({tile_km:.0f} km)",
        fontsize=16,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100)
