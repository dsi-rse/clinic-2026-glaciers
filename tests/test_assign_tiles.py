"""Tests for labelstudio/assign_tiles.py (made-up tiles, no real data, no tile server)."""

import argparse
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "labelstudio" / "assign_tiles.py"
spec = importlib.util.spec_from_file_location("assign_tiles", SCRIPT)
assign_tiles = importlib.util.module_from_spec(spec)
sys.modules["assign_tiles"] = assign_tiles
spec.loader.exec_module(assign_tiles)

PEOPLE = ["Jeanne", "Bailey", "Sophie", "Alexandra"]
TILE_SIZE = 2048
FRAMES = ["frame_a.h5", "frame_b.h5"]
TIER_SIZES = [4, 8, 12]
TIER_NAMES = ["small", "medium", "large"]
N_TILES = 12  # tiles that are not dropped
N_COLS = 6
N_TRAIN_COLS = 3
COPIES = 2


def write_split_csv(path: Path) -> None:
    """A split CSV with 12 usable tiles (train and test) and 3 dropped ones (gap)."""
    header = [
        "frame",
        "row",
        "col",
        "x_m",
        "y_m",
        "split",
        "block_x",
        "block_y",
        "block_color",
    ]
    rows = []
    for frame in FRAMES:
        for col in range(N_COLS):
            split = "train" if col < N_TRAIN_COLS else "test"
            rows.append([frame, 0, col, col * 1000, 0, split, 0, 0, split])
    rows += [
        ["frame_a.h5", 1, col, col * 1000, 1000, "gap", 0, 0, "train"]
        for col in range(3)
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def make_assignment(tiles: list, seed: int) -> dict:
    """Run the same steps as main(): shuffle, assign, check."""
    rng = random.Random(seed)  # noqa: S311 (a seeded shuffle, not cryptography)
    order = assign_tiles.spread_order(tiles, rng)
    out = assign_tiles.assign(order, PEOPLE, COPIES, TIER_SIZES, rng)
    assign_tiles.verify(out, COPIES, TIER_SIZES)
    return out


@pytest.fixture
def tiles(tmp_path: Path) -> list:
    """The 12 usable tiles of the made-up split CSV, with flat frame paths."""
    csv_path = tmp_path / "split.csv"
    write_split_csv(csv_path)
    return assign_tiles.load_tiles_from_split(
        str(csv_path), TILE_SIZE, {name: name for name in FRAMES}
    )


def test_dropped_tiles_are_left_out(tiles: list) -> None:
    """Only train and test tiles are assigned: the 3 gap tiles are not."""
    assert len(tiles) == N_TILES
    assert {t["row"] for t in tiles} == {0}  # the gap tiles are in row 1
    assert {t["tile_size"] for t in tiles} == {TILE_SIZE}


def test_same_seed_gives_the_same_assignment(tiles: list) -> None:
    """Running twice with one seed gives the same tiles in the same order for everyone."""
    assert make_assignment(tiles, seed=42) == make_assignment(tiles, seed=42)


def test_another_seed_gives_another_assignment(tiles: list) -> None:
    """A different seed changes the assignment."""
    assert make_assignment(tiles, seed=1) != make_assignment(tiles, seed=2)


def test_assignment_is_balanced_and_every_tile_has_two_different_people(
    tiles: list,
) -> None:
    """Each tile has 2 different people, and workloads differ by at most 1 in every tier."""
    out = make_assignment(tiles, seed=7)
    people_of_tile: dict[str, list[str]] = {}
    for person, items in out.items():
        for tile, _ in items:
            people_of_tile.setdefault(assign_tiles.tile_key(tile), []).append(person)
    assert len(people_of_tile) == N_TILES
    assert all(len(set(people)) == COPIES for people in people_of_tile.values())
    for tier in range(len(TIER_SIZES)):
        counts = [sum(1 for _, t in items if t <= tier) for items in out.values()]
        assert max(counts) - min(counts) <= 1


def test_tiers_are_nested_and_in_order(tiles: list) -> None:
    """Everyone labels small tiles before medium and large, and the tiers have the set sizes."""
    out = make_assignment(tiles, seed=3)
    for items in out.values():
        tiers = [tier for _, tier in items]
        assert tiers == sorted(tiers)
    first_tier = {}
    for items in out.values():
        for tile, tier in items:
            first_tier[assign_tiles.tile_key(tile)] = tier
    for tier, size in enumerate(TIER_SIZES):
        assert sum(1 for t in first_tier.values() if t <= tier) == size


def build_spec_for(out: dict) -> dict:
    """The JSON document, as main() makes it."""
    args = argparse.Namespace(copies=2, no_stratify=False)
    tiers = [{"name": n, "total_tiles": s} for n, s in zip(TIER_NAMES, TIER_SIZES)]
    return assign_tiles.build_spec(out, tiers, seed=7, tile_size=TILE_SIZE, args=args)


def load_assignment_like_setup_sh(spec: dict, assignee: str, index: list) -> dict:
    """The part of labelstudio/setup.sh (PR #26) that reads the JSON, copied from it.

    Returns {tile key: assignment} for the tiles of `assignee` that are in the tile
    server index. Keep this in step with load_assignment() in setup.sh.
    """

    def key(file: str, size: int, row: int, col: int) -> str:
        return f"{file}|{size}|{row}|{col}"

    names = [t["name"] for t in spec["tiers"]]
    assert {t["tile_size"] for t in index} == {spec["tile_size"]}
    available = {key(t["file"], t["tile_size"], t["row"], t["col"]) for t in index}
    chosen = {}
    for a in spec["assignments"][assignee]:
        assert a["tier"] in names
        k = key(a["file"], a["tile_size"], a["row"], a["col"])
        if k in available:
            chosen[k] = a
    return chosen


def fake_tile_server_index(frame_paths: list[str]) -> list[dict]:
    """What the tile server's /index lists: the same tiles, frames named by their path."""
    return [
        {"file": path, "tile_size": TILE_SIZE, "row": 0, "col": col, "x0": 0, "y0": 0}
        for path in frame_paths
        for col in range(N_COLS)
    ]


def test_json_works_with_the_setup_script_loader(tiles: list) -> None:
    """Every assigned tile is found in the tile server index, in order, with tier and order."""
    spec = build_spec_for(make_assignment(tiles, seed=7))
    index = fake_tile_server_index(FRAMES)
    for person in PEOPLE:
        chosen = load_assignment_like_setup_sh(spec, person, index)
        assert len(chosen) == len(spec["assignments"][person])
        assert [a["order"] for a in chosen.values()] == list(range(1, len(chosen) + 1))
    json.dumps(spec)  # the spec can be written as JSON


def test_frames_in_subfolders_use_the_path_the_tile_server_uses(tmp_path: Path) -> None:
    """With frames in a subfolder, assigned tiles still match the tile server's file names."""
    images = tmp_path / "images"
    (images / "july").mkdir(parents=True)
    for name in FRAMES:
        (images / "july" / name).touch()
    frame_paths = assign_tiles.find_frame_paths(str(images))
    assert frame_paths == {name: f"july/{name}" for name in FRAMES}

    csv_path = tmp_path / "split.csv"
    write_split_csv(csv_path)
    tiles = assign_tiles.load_tiles_from_split(str(csv_path), TILE_SIZE, frame_paths)
    spec = build_spec_for(make_assignment(tiles, seed=7))
    index = fake_tile_server_index([f"july/{name}" for name in FRAMES])
    for person in PEOPLE:
        chosen = load_assignment_like_setup_sh(spec, person, index)
        assert len(chosen) == len(spec["assignments"][person])


def test_frame_missing_from_the_image_folder_stops_with_a_message(
    tmp_path: Path,
) -> None:
    """A frame of the split CSV that is not in --image-dir gives a clear error."""
    csv_path = tmp_path / "split.csv"
    write_split_csv(csv_path)
    with pytest.raises(SystemExit, match="frame_a.h5"):
        assign_tiles.load_tiles_from_split(str(csv_path), TILE_SIZE, {})


def test_person_csv_has_the_order_of_each_person(tiles: list, tmp_path: Path) -> None:
    """The per-person CSV lists each person's tiles in the order of the JSON."""
    out = make_assignment(tiles, seed=7)
    spec = build_spec_for(out)
    csv_path = tmp_path / "by_person.csv"
    assign_tiles.write_person_csv(out, TIER_NAMES, csv_path)
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    for person in PEOPLE:
        mine = [r for r in rows if r["person"] == person]
        expected = spec["assignments"][person]
        assert [int(r["order"]) for r in mine] == [a["order"] for a in expected]
        assert [r["labelstudio_title"] for r in mine] == [
            a["labelstudio_title"] for a in expected
        ]
