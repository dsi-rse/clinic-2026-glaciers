"""Plot the outlines (footprints) of all the frames of one stack, overlapping on one map.

A stack id is the track, the direction and the frame, for example 077_A_140. Frames with the
same stack id show the same place on different dates. We draw only the border of each frame
(read from the metadata JSON files, no images needed) with a very light fill, so you can see
where the frames overlap and how much the outline changes between dates.

By default the stack with the most frames is used.

Note: the outline of a frame is where the satellite took the picture on that date. It changes
a little between passes because the satellite does not take the photo at exactly the same
place every time. It does not show how the ice moves: that is only visible inside the images.

Examples:
    uv run python scripts/plot_stack_footprints.py
    uv run python scripts/plot_stack_footprints.py --stack-id 077_A_140
"""

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as PolygonPatch
from shapely.geometry import Polygon

from utils.footprints import read_footprint
from utils.settings import DATA_DIR
from utils.stack_plot import get_cycle, get_date, get_stack_id

DEFAULT_METADATA_DIR = DATA_DIR / "nisar_data" / "nisar_gcov_pig" / "cmr_metadata"
DEFAULT_OUTPUT_DIR = DATA_DIR / "stack_plots"
METRES_PER_KM = 1000
FILL_ALPHA = 0.12  # very light fill, so overlaps can be seen
# Colors of the dataviz palette, in its fixed order (blue, orange, aqua, violet, yellow, red)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100", "#e34948"]


def parse_args() -> argparse.Namespace:
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--stack-id",
        default=None,
        help="Stack id like 077_A_140, or a frame name. Default: the stack with most frames.",
    )
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Choose the stack, draw the outlines of its frames and save the picture."""
    args = parse_args()
    json_files = sorted(args.metadata_dir.rglob("*.json"))
    if not json_files:
        sys.exit(f"No metadata JSON files found in {args.metadata_dir}")

    # Frames per stack
    stacks: dict[str, list[Path]] = {}
    for json_path in json_files:
        try:
            stacks.setdefault(get_stack_id(json_path.name), []).append(json_path)
        except ValueError:
            continue
    counts = Counter({stack: len(files) for stack, files in stacks.items()})
    print(
        "Frames per stack (top 5):",
        ", ".join(f"{s} ({n})" for s, n in counts.most_common(5)),
    )

    try:
        stack_id = (
            get_stack_id(args.stack_id)
            if args.stack_id
            else counts.most_common(1)[0][0]
        )
    except ValueError as error:
        sys.exit(f"Cannot plot: {error}")
    if stack_id not in stacks:
        sys.exit(f"No frames found for stack {stack_id}")
    files = sorted(stacks[stack_id], key=lambda path: (get_date(path.name), path.name))
    print(f"Using stack {stack_id} with {len(files)} frames")

    footprints = [read_footprint(path) for path in files]
    labels = [
        f"cycle {get_cycle(path.name)}  ({get_date(path.name)})" for path in files
    ]

    # How much the outline changes between dates
    first = Polygon(footprints[0])
    for label, corners in zip(labels, footprints, strict=True):
        polygon = Polygon(corners)
        overlap = polygon.intersection(first).area / first.area
        shifts = [math.dist(a, b) for a, b in zip(corners, footprints[0], strict=True)]
        print(
            f"  {label}: overlap with the first date {overlap:.1%}, corner shift {min(shifts):.2f} to {max(shifts):.2f} km"
        )

    fig = Figure(figsize=(8.5, 8.5))
    ax = fig.subplots()
    for index, corners in enumerate(footprints):
        color = COLORS[index % len(COLORS)]
        ax.add_patch(
            PolygonPatch(corners, facecolor=color, alpha=FILL_ALPHA, edgecolor="none")
        )
        ax.add_patch(
            PolygonPatch(corners, facecolor="none", edgecolor=color, linewidth=1.8)
        )
    all_x = [x for corners in footprints for x, _ in corners]
    all_y = [y for corners in footprints for _, y in corners]
    margin = 15
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect("equal")
    ax.grid(color="#e1e0d9", linewidth=0.6)
    ax.set_xlabel("x (km, Antarctic polar stereographic)")
    ax.set_ylabel("y (km)")
    ax.legend(
        handles=[
            Patch(
                facecolor=COLORS[i % len(COLORS)],
                edgecolor=COLORS[i % len(COLORS)],
                alpha=0.6,
                label=label,
            )
            for i, label in enumerate(labels)
        ],
        loc="lower left",
    )
    ax.set_title(
        f"Stack {stack_id}: outlines of {len(files)} frames (the same place on different dates)"
    )
    fig.tight_layout()

    output = args.output_dir / f"footprints_{stack_id}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
