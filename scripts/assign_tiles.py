#!/usr/bin/env python3
r"""Assign Label Studio tiles to team members. Standard library only.

Run this ONCE for the whole team (one person), then share the output file
(tile_assignments.json) with everyone, e.g. in the Box project folder. Each person
points LABELSTUDIO_ASSIGNEE / LABELSTUDIO_ASSIGNMENTS at it in their own .env and
re-runs `make labelstudio`; only their tiles are loaded.

- Every tile is assigned to `--copies` (default 2) DIFFERENT people.
- Workloads stay balanced: each person's count differs by at most 1, overall and at
  the end of every tier.
- Nested tiers: small is contained in medium, which is contained in large.
- By default the shuffle is spread across frames, so even the small tier covers all
  frames instead of clustering in a few (use --no-stratify for a plain shuffle).
- Each person works through the tiers in order, shuffled within a tier.

The tile list comes from the tile server (`make labelstudio` starts it), the same
/index endpoint the setup script uses. With --split-csv it comes from a train/test split
CSV instead (the output of split_tiles.py): the `gap` tiles are left out and --tile-size
gives the tile size of that split.

Examples:
    python3 assign_tiles.py --split-csv splits/tile_split_region_2048px_blocks_163.84km.csv \
        --tile-size 2048 --people Jeanne Bailey Sophie Alexandra --seed 42 \
        --tiers 50 150 405
    python3 assign_tiles.py --people Jeanne Bailey Sophie Alexandra --seed 42 \
        --tiers 10% 30% 100%
    python3 assign_tiles.py --people A B C D --seed 42 --tiers 40 120 400 \
        --tileserver http://localhost:8081 --out tile_assignments.json

--tiers takes CUMULATIVE sizes: a plain number is a count of tiles, a number with %
is a percentage of all tiles. "10% 30% 100%" means small = 10% of tiles, medium = 30%
in total (including small), large = everything. With three tiers they are named
small/medium/large; override with --tier-names. Keep the same --seed to reproduce an
assignment exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

Tile = dict[str, Any]
Assignments = dict[str, list[tuple[Tile, int]]]

DEFAULT_TIER_NAMES = ("small", "medium", "large")
URL_TIMEOUT_SECONDS = 3600  # the tile server's first scan of a big frame can be slow


def tile_key(tile: Tile) -> str:
    """Return a string that uniquely identifies a tile."""
    return f"{tile['file']}|{tile['tile_size']}|{tile['row']}|{tile['col']}"


def load_tiles(tileserver: str, index_file: str | None) -> list[Tile]:
    """Load the tile list from a saved index file or from the tile server, sorted."""
    if index_file:
        tiles = json.loads(Path(index_file).read_text())
    else:
        url = tileserver.rstrip("/") + "/index"
        try:
            with urllib.request.urlopen(url, timeout=URL_TIMEOUT_SECONDS) as response:  # noqa: S310
                tiles = json.loads(response.read())
        except urllib.error.URLError as exc:
            sys.exit(
                f"Could not reach the tile server at {url}: {exc}\n"
                "Start it with `make labelstudio`, or pass --tileserver / --index-file."
            )
    # Sorted, so the same data and seed always give the same assignment
    return sorted(tiles, key=lambda t: (t["file"], t["row"], t["col"]))


def load_tiles_from_split(csv_path: str, tile_size: int) -> list[Tile]:
    """Load the tiles to label from a split CSV written by split_tiles.py.

    Only the train and test tiles are kept: the `gap` tiles were dropped by the split.
    The CSV has no tile size column, so it is given with --tile-size.
    """
    tiles = []
    with Path(csv_path).open(newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "gap":
                continue
            tiles.append(
                {
                    "file": row["frame"],
                    "tile_size": tile_size,
                    "row": int(row["row"]),
                    "col": int(row["col"]),
                }
            )
    # Sorted, so the same data and seed always give the same assignment
    return sorted(tiles, key=lambda t: (t["file"], t["row"], t["col"]))


def default_tier_names(n_tiers: int) -> list[str]:
    """Return default tier names for the number of tiers."""
    if n_tiers == 1:
        return ["all"]
    if n_tiers == len(DEFAULT_TIER_NAMES):
        return list(DEFAULT_TIER_NAMES)
    return [f"tier{i}" for i in range(1, n_tiers + 1)]


def parse_tiers(specs: list[str], total: int) -> list[int]:
    """Turn counts or percentages like ["10%", "40"] into cumulative tile counts."""
    sizes = []
    for spec in specs:
        try:
            if spec.endswith("%"):
                sizes.append(round(total * float(spec[:-1]) / 100))
            else:
                sizes.append(int(spec))
        except ValueError:
            sys.exit(
                f"Can't read tier size {spec!r}; "
                "use a count (40) or a percentage (10%)."
            )
    if any(s <= 0 for s in sizes) or any(b <= a for a, b in zip(sizes, sizes[1:])):
        sys.exit(
            f"Tier sizes must be positive and strictly increasing; "
            f"got {sizes} for {total} tiles."
        )
    if sizes[-1] > total:
        sys.exit(f"Largest tier ({sizes[-1]}) exceeds the number of tiles ({total}).")
    return sizes


def spread_order(
    tiles: list[Tile], rng: random.Random, *, stratify: bool = True
) -> list[Tile]:
    """Return all tiles in random order (with stratify, prefixes span all frames)."""
    if not stratify:
        order = tiles[:]
        rng.shuffle(order)
        return order
    by_frame: dict[str, list[Tile]] = defaultdict(list)
    for tile in tiles:
        by_frame[tile["file"]].append(tile)
    for frame_tiles in by_frame.values():
        rng.shuffle(frame_tiles)
    order = []
    while any(by_frame.values()):
        round_frames = [f for f in sorted(by_frame) if by_frame[f]]
        rng.shuffle(round_frames)
        order.extend(by_frame[f].pop() for f in round_frames)
    return order


def assign(
    order: list[Tile],
    people: list[str],
    copies: int,
    sizes: list[int],
    rng: random.Random,
) -> Assignments:
    """Give each tile to `copies` different people; tier k = first sizes[k] tiles."""
    if len(people) < copies:
        sys.exit("Need at least as many people as reviewers per tile.")
    load = dict.fromkeys(people, 0)
    out: Assignments = {p: [] for p in people}
    start = 0
    for tier, end in enumerate(sizes):
        for tile in order[start:end]:
            pick = people[:]
            rng.shuffle(pick)
            pick.sort(key=lambda p: load[p])  # stable: random tie-break
            for person in pick[:copies]:
                out[person].append((tile, tier))
                load[person] += 1
        start = end
    for person in people:
        rng.shuffle(out[person])
        out[person].sort(key=lambda x: x[1])  # tiers in order, random within a tier
    return out


def check(condition: bool, message: str) -> None:  # noqa: FBT001
    """Raise AssertionError with a message if a sanity check fails."""
    if not condition:
        raise AssertionError(message)


def verify(out: Assignments, copies: int, sizes: list[int]) -> None:
    """Sanity-check the assignment: reviewers, nesting, tier sizes and balance."""
    seen: dict[str, int] = defaultdict(int)
    tier_of: dict[str, int] = {}
    for person, items in out.items():
        keys = [tile_key(t) for t, _ in items]
        check(len(keys) == len(set(keys)), f"Duplicate tile for {person}")
        for tile, tier in items:
            seen[tile_key(tile)] += 1
            check(tier_of.setdefault(tile_key(tile), tier) == tier, "Tile in two tiers")
    check(
        all(c == copies for c in seen.values()),
        "A tile has the wrong number of reviewers",
    )
    check(len(seen) == sizes[-1], "Wrong number of tiles assigned")
    for k, size in enumerate(sizes):
        check(
            sum(1 for v in tier_of.values() if v <= k) == size,
            f"Tier {k} has the wrong size",
        )
    counts = [len(v) for v in out.values()]
    check(max(counts) - min(counts) <= 1, "Workloads are not balanced")


def build_spec(
    out: Assignments,
    tiers: list[dict[str, Any]],
    seed: int,
    tile_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build the JSON document that setup.sh reads."""
    names = [t["name"] for t in tiers]
    return {
        "version": 1,
        "seed": seed,
        "copies": args.copies,
        "stratified_by_frame": not args.no_stratify,
        "tile_size": tile_size,
        "tiers": tiers,
        "assignments": {
            person: [
                {
                    "file": t["file"],
                    "tile_size": t["tile_size"],
                    "row": t["row"],
                    "col": t["col"],
                    "tier": names[tier],
                    "order": i,
                }
                for i, (t, tier) in enumerate(items, 1)
            ]
            for person, items in out.items()
        },
    }


def write_master_csv(out: Assignments, names: list[str], csv_path: Path) -> None:
    """Write one row per tile with its tier and both reviewers.

    The tier is the one a tile FIRST appears in. Tiers are cumulative, so the medium
    training set is the small + medium rows, and large is every row.
    """
    by_tile: dict[str, tuple[Tile, int, list[str]]] = {}
    for person, items in out.items():
        for tile, tier in items:
            by_tile.setdefault(tile_key(tile), (tile, tier, []))[2].append(person)
    rows = sorted(
        by_tile.values(), key=lambda v: (v[1], v[0]["file"], v[0]["row"], v[0]["col"])
    )
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["source_file", "tile_row", "tile_col", "tier", "reviewer_1", "reviewer_2"]
        )
        for tile, tier, who in rows:
            w.writerow(
                [tile["file"], tile["row"], tile["col"], names[tier], *sorted(who)]
            )


def print_summary(
    order: list[Tile], out: Assignments, names: list[str], sizes: list[int]
) -> None:
    """Print tier sizes, frame coverage, and each person's workload."""
    previous = 0
    for name, size in zip(names, sizes):
        frames = {t["file"] for t in order[:size]}
        print(
            f"  {name}: {size} tiles total (+{size - previous} new), "
            f"spread over {len(frames)} frame(s)"
        )
        previous = size
    for person, items in out.items():
        per_tier = ", ".join(
            f"{n} {sum(1 for _, x in items if x == k)}" for k, n in enumerate(names)
        )
        print(f"  {person}: {len(items)} tasks ({per_tier})")


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--people",
        nargs="+",
        required=True,
        help="Names, exactly as each person will put them in LABELSTUDIO_ASSIGNEE",
    )
    ap.add_argument(
        "--tiers",
        nargs="+",
        default=None,
        help="Cumulative sizes: counts or percentages, e.g. 10%% 30%% 100%%",
    )
    ap.add_argument("--tier-names", nargs="+", default=None)
    ap.add_argument(
        "--copies", type=int, default=2, help="Reviewers per tile (default 2)"
    )
    ap.add_argument(
        "--seed", type=int, default=None, help="Seed, for a reproducible assignment"
    )
    ap.add_argument(
        "--tileserver",
        default=os.environ.get("TILESERVER_URL", "http://localhost:8081"),
    )
    ap.add_argument(
        "--index-file",
        default=None,
        help="Read the tile list from a saved copy of the tile server's /index instead",
    )
    ap.add_argument(
        "--split-csv",
        default=None,
        help="Read the tiles from a split CSV (output of split_tiles.py) instead of the "
        "tile server. The gap tiles are left out. Needs --tile-size.",
    )
    ap.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Tile size in pixels of the split CSV, e.g. 2048 (only with --split-csv)",
    )
    ap.add_argument(
        "--no-stratify",
        action="store_true",
        help="Plain random shuffle instead of spreading tiers across frames",
    )
    ap.add_argument("--out", default="tile_assignments.json")
    return ap.parse_args()


def main() -> None:
    """Assign tiles and write the assignment file and master CSV."""
    args = parse_args()

    if len(set(args.people)) != len(args.people):
        sys.exit("Names in --people must be unique.")
    if args.split_csv:
        if not args.tile_size:
            sys.exit("--split-csv needs --tile-size (for example --tile-size 2048).")
        tiles = load_tiles_from_split(args.split_csv, args.tile_size)
    else:
        tiles = load_tiles(args.tileserver, args.index_file)
    if not tiles:
        sys.exit("The tile list is empty; no tiles with data were found.")
    tile_sizes = {t["tile_size"] for t in tiles}
    if len(tile_sizes) != 1:
        sys.exit(
            f"Tiles come in several sizes {sorted(tile_sizes)}; expected exactly one."
        )

    sizes = parse_tiers(args.tiers or [str(len(tiles))], len(tiles))
    names = args.tier_names or default_tier_names(len(sizes))
    if len(names) != len(sizes):
        sys.exit("--tier-names must have the same number of entries as --tiers.")

    seed = (
        args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
    )
    rng = random.Random(seed)  # noqa: S311 (reproducible shuffling, not cryptography)
    order = spread_order(tiles, rng, stratify=not args.no_stratify)
    out = assign(order, args.people, args.copies, sizes, rng)
    verify(out, args.copies, sizes)

    tiers = [{"name": n, "total_tiles": s} for n, s in zip(names, sizes)]
    spec = build_spec(out, tiers, seed, next(iter(tile_sizes)), args)
    Path(args.out).write_text(json.dumps(spec, indent=2))
    csv_path = Path(args.out).with_suffix(".csv")
    write_master_csv(out, names, csv_path)

    frames = {t["file"] for t in tiles}
    print(
        f"{len(tiles)} tiles in {len(frames)} frame(s); using {sizes[-1]}, "
        f"{args.copies} reviewers each; seed {seed}"
    )
    print_summary(order, out, names, sizes)
    print(f"Wrote {args.out} and {csv_path}")


if __name__ == "__main__":
    main()