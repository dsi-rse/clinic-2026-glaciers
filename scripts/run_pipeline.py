"""Prepare the tiles for labeling, from the bounding box to the assignment, step by step.

The steps run in this order. Each one uses a script that already exists in the repo:

    1. box          Check the bounding box (the area of analysis).
    2. split        Train and test split of the tiles          (scripts/split_tiles.py)
    3. tiles        List of the tiles to label, from the split CSV (the gap tiles are left out)
    4. assign       Assign the tiles to the labelers           (labelstudio/assign_tiles.py)

With --augmentation (high_pass, black_tophat, clahe or unsharp_mask) the assignment uses the
augmented frames <name>_<method>.h5, which must already be in the same folder as the original
frames. The split always uses the originals, so it is the same for every method.

Label Studio is not part of the pipeline: each person loads their own tiles by hand when it
finishes (the pipeline prints how, and labelstudio/ASSIGNMENTS.md has the details).

Everything one run makes goes in one folder of the repo (data/pipeline/<run name>, ignored by
git), the same with Docker and with uv. The frames are still read from DATA_DIR. It has a
pipeline_settings.json that says how it was made. Use --until to stop after a step.

Examples:
    uv run python scripts/run_pipeline.py
    uv run python scripts/run_pipeline.py --until tiles
    uv run python scripts/run_pipeline.py --augmentation black_tophat
    uv run python scripts/run_pipeline.py --region 1700 620 2200 800 --block-km 122.88
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from utils.settings import DATA_DIR
from utils.tile_split import (
    AUGMENTATION_METHODS,
    filter_frames_by_date,
    find_frames,
    is_augmented,
)

REPO_DIR = Path(__file__).resolve().parent.parent
STEPS = ("box", "split", "tiles", "assign")
# Amery Ice Shelf box in km (EPSG:3031), from scripts/get_amery_outline.py. The team is
# choosing a tighter box around the ice shelf: pass it with --region when it is ready.
DEFAULT_REGION = (1681.978, 609.405, 2246.848, 840.133)
CODE_LENGTH = 6  # characters of the code in the run folder name
DEFAULT_PEOPLE = ["Jeanne", "Bailey", "Sophie", "Alexandra"]
# Same folder with Docker (the repo is /project there) and with uv, and git ignores data/
DEFAULT_RUN_ROOT = REPO_DIR / "data" / "pipeline"
DEFAULT_IMAGE_DIR = DATA_DIR / os.environ.get("LABELSTUDIO_IMAGE_SUBDIR", "")


def parse_args() -> argparse.Namespace:
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--until",
        choices=STEPS,
        default=STEPS[-1],
        help="Last step to run (default: all of them).",
    )
    parser.add_argument(
        "--region",
        type=float,
        nargs=4,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        default=DEFAULT_REGION,
        help="Bounding box in km (EPSG:3031). Only tiles completely inside are used.",
    )
    parser.add_argument(
        "--date-prefix",
        default="202607",
        help="Only frames whose start date begins with this (default: July 2026).",
    )
    parser.add_argument("--tile-size", type=int, default=2048, help="Tile size in px.")
    parser.add_argument("--block-km", type=float, default=163.84, help="Block size.")
    parser.add_argument("--gap-km", type=float, default=0.0)
    parser.add_argument("--swap", action="store_true", help="Swap train and test.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--people", nargs="+", default=DEFAULT_PEOPLE)
    parser.add_argument("--seed", type=int, default=42, help="Same seed, same result.")
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["10%", "30%", "100%"],
        help="Cumulative sizes of the small, medium and large batches.",
    )
    parser.add_argument(
        "--augmentation",
        choices=["none", *AUGMENTATION_METHODS],
        default="none",
        help="Read the augmented frames <name>_<method>.h5 (in --image-dir, next to the "
        "originals) for the assignment and Label Studio. The split always uses the originals.",
    )
    parser.add_argument(
        "--name",
        help="Name of the run folder in data/pipeline (default: run_ and a code made from "
        "the settings, so other settings give another folder).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Full folder for this run. Instead of --name, to write outside data/pipeline.",
    )
    return parser.parse_args()


def check_region(region: tuple[float, float, float, float]) -> None:
    """Stop with a message if the box is not X_MIN < X_MAX and Y_MIN < Y_MAX."""
    x_min, y_min, x_max, y_max = region
    if not (x_min < x_max and y_min < y_max):
        sys.exit("--region needs X_MIN < X_MAX and Y_MIN < Y_MAX")


def get_settings(args: argparse.Namespace) -> dict:
    """The settings that decide what a run makes.

    The frames folder is not in it: Docker and uv write the same folder, and they see the
    frames at different paths.
    """
    settings = {
        "date_prefix": args.date_prefix,
        "region_km": list(args.region),
        "tile_size_px": args.tile_size,
        "block_km": args.block_km,
        "gap_km": args.gap_km,
        "swap": args.swap,
        "people": args.people,
        "seed": args.seed,
        "tiers": args.tiers,
    }
    if args.augmentation != "none":
        settings["augmentation"] = (
            args.augmentation
        )  # not saved for none: same name as before
    return settings


def get_run_name(args: argparse.Namespace) -> str:
    """Name of the run folder: --name, or run_ and a short code made from the settings.

    The same settings give the same name, and other settings give another one, so two
    different runs never share a folder. The settings are in pipeline_settings.json.
    """
    if args.name:
        return args.name
    text = json.dumps(get_settings(args), sort_keys=True)
    return f"run_{hashlib.sha256(text.encode()).hexdigest()[:CODE_LENGTH]}"


def save_settings(run_dir: Path, args: argparse.Namespace) -> None:
    """Save the settings in the run folder, or stop if it holds a run with other settings."""
    path = run_dir / "pipeline_settings.json"
    settings = get_settings(args)
    if path.exists():
        old = json.loads(path.read_text())
        changed = [key for key, value in settings.items() if old.get(key) != value]
        if changed:
            sys.exit(
                f"{run_dir} already has a run with other settings ({', '.join(changed)}). "
                "Use another --name, or delete that folder."
            )
    path.write_text(
        json.dumps({**settings, "image_dir": str(args.image_dir)}, indent=2) + "\n"
    )


def check_augmented_frames(args: argparse.Namespace) -> None:
    """Stop if a frame of the chosen dates has no augmented copy (<name>_<method>.h5)."""
    if args.augmentation == "none":
        return
    try:
        frames = find_frames(args.image_dir)
        originals = [frame for frame in frames if not is_augmented(frame)]
        originals = filter_frames_by_date(originals, args.date_prefix, args.image_dir)
    except (FileNotFoundError, ValueError) as error:
        sys.exit(f"Cannot check the frames: {error}")
    missing = [
        f"{frame.stem}_{args.augmentation}{frame.suffix}"
        for frame in originals
        if not frame.with_name(
            f"{frame.stem}_{args.augmentation}{frame.suffix}"
        ).exists()
    ]
    if missing:
        sys.exit(
            f"{len(missing)} {args.augmentation} frame(s) not found next to the originals, "
            "for example:\n  " + missing[0] + "\nMake the augmented frames first."
        )


def split_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    """The command of step 2: split_tiles.py, writing in the run folder."""
    command = [
        sys.executable,
        str(REPO_DIR / "scripts" / "split_tiles.py"),
        "--image-dir",
        str(args.image_dir),
        "--output",
        str(run_dir / "tile_split.csv"),
        "--tile-size",
        str(args.tile_size),
        "--region",
        *(str(value) for value in args.region),
        "--block-km",
        str(args.block_km),
        "--gap-km",
        str(args.gap_km),
        "--plot",
    ]
    if args.date_prefix:
        command += ["--date-prefix", args.date_prefix]
    if args.swap:
        command.append("--swap")
    return command


def find_split_csv(run_dir: Path) -> Path:
    """The split CSV that step 2 wrote in the run folder (not the summary)."""
    found = [
        path
        for path in run_dir.glob("tile_split*_blocks_*km*.csv")
        if not path.name.endswith("_summary.csv")
    ]
    if len(found) != 1:
        sys.exit(
            f"Expected one split CSV in {run_dir}, found {len(found)}. Run step split."
        )
    return found[0]


def count_tiles(split_csv: Path) -> dict[str, int]:
    """Count the tiles of a split CSV: train, test and gap (dropped)."""
    counts = {"train": 0, "test": 0, "gap": 0}
    with split_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            counts[row["split"]] += 1
    return counts


def assign_command(
    args: argparse.Namespace, split_csv: Path, assignments: Path
) -> list[str]:
    """The command of step 4: assign_tiles.py, reading the split CSV."""
    command = [
        sys.executable,
        str(REPO_DIR / "labelstudio" / "assign_tiles.py"),
        "--split-csv",
        str(split_csv),
        "--tile-size",
        str(args.tile_size),
        "--image-dir",
        str(args.image_dir),
        "--people",
        *args.people,
        "--seed",
        str(args.seed),
        "--tiers",
        *args.tiers,
        "--out",
        str(assignments),
    ]
    if args.augmentation != "none":
        command += ["--augmentation", args.augmentation]
    return command


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    """Print a command, run it from the repo folder, and stop the pipeline if it fails."""
    print("$", " ".join(command))
    result = subprocess.run(command, cwd=REPO_DIR, env=env, check=False)  # noqa: S603
    if result.returncode != 0:
        sys.exit(f"Step failed (exit code {result.returncode}). Fix it and run again.")


def main() -> None:
    """Run the steps in order, up to --until."""
    args = parse_args()
    check_region(args.region)
    run_dir = args.run_dir or DEFAULT_RUN_ROOT / get_run_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    method = "" if args.augmentation == "none" else f"_{args.augmentation}"
    assignments = (
        run_dir / f"tile_assignments_{args.tile_size}_seed{args.seed}{method}.json"
    )
    last = STEPS.index(args.until)
    print("\n=== Pipeline")
    print(f"a) Pipeline folder: {run_dir}\n")
    print(
        f"b) Settings: dates {args.date_prefix or 'all'}, {args.tile_size} px tiles, "
        f"{args.block_km:g} km blocks, gap {args.gap_km:g} km, seed {args.seed}, "
        f"augmentation {args.augmentation}"
    )
    print("\n=== Step 1/4 box")
    print(f"Bounding box (km): {' '.join(str(v) for v in args.region)}")
    check_augmented_frames(args)
    save_settings(run_dir, args)

    if last >= STEPS.index("split"):
        print("\n=== Step 2/4 split")
        run(split_command(args, run_dir))

    if last >= STEPS.index("tiles"):
        print("\n=== Step 3/4 tiles to label")
        split_csv = find_split_csv(run_dir)
        counts = count_tiles(split_csv)
        print(f"{split_csv.name}")
        print(
            f"{counts['train'] + counts['test']} tiles to label "
            f"({counts['train']} train, {counts['test']} test); "
            f"{counts['gap']} dropped at block edges"
        )

    if last >= STEPS.index("assign"):
        print("\n=== Step 4/4 assign")
        run(assign_command(args, split_csv, assignments))

    print(f"\nDone. Files are in {run_dir}")
    if last == STEPS.index("assign"):
        # In Docker the repo is /project, which is a different path on your computer: show
        # the path from the repo folder on, to be written with the full path of the repo.
        in_docker = Path("/.dockerenv").exists()
        shown = (
            f"<full path of the repo>/{assignments.relative_to(REPO_DIR)}"
            if in_docker and assignments.is_relative_to(REPO_DIR)
            else str(assignments)
        )
        print(
            "\n=== Final Step",
            "\nPlease adds this to your .env\n"
            "and run make labelstudio:\n"
            "  LABELSTUDIO_ASSIGNEE=<your name>\n"
            f"  LABELSTUDIO_ASSIGNMENTS={shown}\n"
            "  LABELSTUDIO_MAX_TIER=small\n"
            f"  TILESERVER_TILE_SIZE={args.tile_size}\n"
            "Write the full path of the file on your computer, not ~."
            + (
                f"\nThe tiles are the {args.augmentation} frames "
                "(<name>_"
                + args.augmentation
                + ".h5) in the same folder as the originals."
                if args.augmentation != "none"
                else ""
            ),
            "\n===End",
        )


if __name__ == "__main__":
    main()
