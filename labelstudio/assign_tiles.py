#!/usr/bin/env python3
"""
Assign Label Studio tiles to team members. Standard library only.

Run this ONCE for the whole team (one person), then share the output file
(tile_assignments.json) with everyone, e.g. in the Box project folder. Each person points
LABELSTUDIO_ASSIGNEE / LABELSTUDIO_ASSIGNMENTS at it in their own .env and re-runs
`make labelstudio`; only their tiles are loaded.

- Every tile is assigned to `--copies` (default 2) DIFFERENT people.
- Workloads stay balanced: each person's count differs by at most 1, overall and at the
  end of every tier.
- Nested tiers: small is contained in medium, which is contained in large.
- By default the shuffle is spread across frames, so even the small tier covers all frames
  instead of clustering in a few (use --no-stratify for a plain random shuffle).
- Each person works through the tiers in order, shuffled within a tier.

The tile list comes from the tile server (`make labelstudio` starts it), the same /index
endpoint the setup script uses.

Examples:
    python3 assign_tiles.py --people Jeanne Bailey Sophie Alexandra --seed 42 \\
        --tiers 10% 30% 100%
    python3 assign_tiles.py --people A B C D --seed 42 --tiers 40 120 400 \\
        --tileserver http://localhost:8081 --out tile_assignments.json

--tiers takes CUMULATIVE sizes: a plain number is a count of tiles, a number with % is a
percentage of all tiles. "10% 30% 100%" means small = 10% of tiles, medium = 30% in total
(including small), large = everything. With three tiers they are named small/medium/large;
override with --tier-names. Keep the same --seed to reproduce an assignment exactly.
"""
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


def tile_key(t):
    return f"{t['file']}|{t['tile_size']}|{t['row']}|{t['col']}"


def load_tiles(tileserver, index_file):
    if index_file:
        tiles = json.loads(Path(index_file).read_text())
    else:
        url = tileserver.rstrip("/") + "/index"
        try:
            with urllib.request.urlopen(url, timeout=3600) as response:
                tiles = json.loads(response.read())
        except urllib.error.URLError as exc:
            sys.exit(f"Could not reach the tile server at {url}: {exc}\n"
                     "Start it with `make labelstudio`, or pass --tileserver / --index-file.")
    # Sorted, so the same data and seed always give the same assignment
    return sorted(tiles, key=lambda t: (t["file"], t["row"], t["col"]))


def default_tier_names(n):
    if n == 1:
        return ["all"]
    if n == 3:
        return ["small", "medium", "large"]
    return [f"tier{i}" for i in range(1, n + 1)]


def parse_tiers(specs, total):
    sizes = []
    for s in specs:
        try:
            sizes.append(round(total * float(s[:-1]) / 100) if s.endswith("%") else int(s))
        except ValueError:
            sys.exit(f"Can't read tier size {s!r}; use a count (40) or a percentage (10%).")
    if any(x <= 0 for x in sizes) or any(b <= a for a, b in zip(sizes, sizes[1:])):
        sys.exit(f"Tier sizes must be positive and strictly increasing; got {sizes} "
                 f"for {total} tiles.")
    if sizes[-1] > total:
        sys.exit(f"Largest tier ({sizes[-1]}) exceeds the number of tiles ({total}).")
    return sizes


def spread_order(tiles, rng, stratify=True):
    """Random order of all tiles. With stratify, any prefix is spread evenly over frames."""
    if not stratify:
        order = tiles[:]
        rng.shuffle(order)
        return order
    by_frame = defaultdict(list)
    for t in tiles:
        by_frame[t["file"]].append(t)
    for lst in by_frame.values():
        rng.shuffle(lst)
    order = []
    while any(by_frame.values()):
        round_frames = [f for f in sorted(by_frame) if by_frame[f]]
        rng.shuffle(round_frames)
        for f in round_frames:
            order.append(by_frame[f].pop())
    return order


def assign(order, people, copies, sizes, rng):
    """Return {person: [(tile, tier_index), ...]}; tier k = first sizes[k] tiles of `order`."""
    if len(people) < copies:
        sys.exit("Need at least as many people as reviewers per tile.")
    load = {p: 0 for p in people}
    out = {p: [] for p in people}
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


def verify(out, copies, sizes):
    seen, tier_of = defaultdict(int), {}
    for person, items in out.items():
        keys = [tile_key(t) for t, _ in items]
        assert len(keys) == len(set(keys)), f"Duplicate tile for {person}"
        for t, tier in items:
            seen[tile_key(t)] += 1
            assert tier_of.setdefault(tile_key(t), tier) == tier
    assert all(c == copies for c in seen.values()), "A tile does not have exactly the right reviewers"
    assert len(seen) == sizes[-1]
    for k, size in enumerate(sizes):
        assert sum(1 for v in tier_of.values() if v <= k) == size
    counts = [len(v) for v in out.values()]
    assert max(counts) - min(counts) <= 1, "Workloads are not balanced"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--people", nargs="+", required=True,
                    help="Names, exactly as each person will put them in LABELSTUDIO_ASSIGNEE")
    ap.add_argument("--tiers", nargs="+", default=None,
                    help="Cumulative sizes: counts or percentages, e.g. 10%% 30%% 100%%")
    ap.add_argument("--tier-names", nargs="+", default=None)
    ap.add_argument("--copies", type=int, default=2, help="Reviewers per tile (default 2)")
    ap.add_argument("--seed", type=int, default=None, help="Seed, for a reproducible assignment")
    ap.add_argument("--tileserver", default=os.environ.get("TILESERVER_URL", "http://localhost:8081"))
    ap.add_argument("--index-file", default=None,
                    help="Read the tile list from a saved copy of the tile server's /index instead")
    ap.add_argument("--no-stratify", action="store_true",
                    help="Plain random shuffle instead of spreading tiers across frames")
    ap.add_argument("--out", default="tile_assignments.json")
    args = ap.parse_args()

    if len(set(args.people)) != len(args.people):
        sys.exit("Names in --people must be unique.")
    tiles = load_tiles(args.tileserver, args.index_file)
    if not tiles:
        sys.exit("The tile list is empty; no tiles with data were found.")
    tile_sizes = {t["tile_size"] for t in tiles}
    if len(tile_sizes) != 1:
        sys.exit(f"Tiles come in several sizes {sorted(tile_sizes)}; expected exactly one.")

    sizes = parse_tiers(args.tiers or [str(len(tiles))], len(tiles))
    names = args.tier_names or default_tier_names(len(sizes))
    if len(names) != len(sizes):
        sys.exit("--tier-names must have the same number of entries as --tiers.")

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
    rng = random.Random(seed)
    order = spread_order(tiles, rng, stratify=not args.no_stratify)
    out = assign(order, args.people, args.copies, sizes, rng)
    verify(out, args.copies, sizes)

    spec = {
        "version": 1,
        "seed": seed,
        "copies": args.copies,
        "stratified_by_frame": not args.no_stratify,
        "tile_size": next(iter(tile_sizes)),
        "tiers": [{"name": n, "total_tiles": s} for n, s in zip(names, sizes)],
        "assignments": {
            person: [{"file": t["file"], "tile_size": t["tile_size"], "row": t["row"],
                      "col": t["col"], "tier": names[tier], "order": i}
                     for i, (t, tier) in enumerate(items, 1)]
            for person, items in out.items()
        },
    }
    Path(args.out).write_text(json.dumps(spec, indent=2))

    # Human-readable master sheet. 'tier' is the tier a tile FIRST appears in; tiers are
    # cumulative, so the medium training set = small + medium rows, large = all rows.
    by_tile = {}
    for person, items in out.items():
        for t, tier in items:
            by_tile.setdefault(tile_key(t), [t, tier, []])[2].append(person)
    csv_path = Path(args.out).with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_file", "tile_row", "tile_col", "tier", "reviewer_1", "reviewer_2"])
        for t, tier, who in sorted(by_tile.values(), key=lambda v: (v[1], v[0]["file"], v[0]["row"], v[0]["col"])):
            w.writerow([t["file"], t["row"], t["col"], names[tier], *sorted(who)])

    frames = {t["file"] for t in tiles}
    print(f"{len(tiles)} tiles in {len(frames)} frame(s); using {sizes[-1]}, "
          f"{args.copies} reviewers each; seed {seed}")
    prev = 0
    for name, size in zip(names, sizes):
        used = {t["file"] for t, tier in (v[:2] for v in by_tile.values()) if names[tier] in names[:names.index(name) + 1]}
        print(f"  {name}: {size} tiles total (+{size - prev} new), spread over {len(used)} frame(s)")
        prev = size
    for person, items in out.items():
        per = ", ".join(f"{n} {sum(1 for _, x in items if x == k)}" for k, n in enumerate(names))
        print(f"  {person}: {len(items)} tasks ({per})")
    print(f"Wrote {args.out} and {csv_path}")


if __name__ == "__main__":
    main()
