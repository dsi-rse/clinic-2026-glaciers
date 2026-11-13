"""Tests for labelstudio/assign_tiles.py.

They cover reproducible assignments, distinct reviewers, balanced workloads, nested
batches, the master CSV, and input validation.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

PEOPLE = ["Alice", "Bob", "Cara", "Dev"]
TILE_SIZE = 4096
TILES_PER_FRAME = {
    "nisar_data/A.h5": 6,
    "nisar_data/B.h5": 8,
    "nisar_data/C.h5": 10,
    "nisar_data/D.h5": 16,
}
TILES_PER_ROW = 5
N_TILES = sum(TILES_PER_FRAME.values())
TIER_SIZES = [8, 20, N_TILES]
TIER_ARGS = [str(size) for size in TIER_SIZES]
TIER_NAMES = ["small", "medium", "large"]
DEFAULT_COPIES = 2
THREE_COPIES = 3
SEED = "42"
OTHER_SEED = "7"
TOO_MANY_COPIES = 5

TileId = tuple[str, int, int]
RunAssign = Callable[..., dict[str, Any]]


def find_labelstudio_dir() -> Path:
    """Find the labelstudio/ folder by walking up from this file."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "labelstudio" / "assign_tiles.py").is_file():
            return parent / "labelstudio"
    pytest.fail("Could not find labelstudio/assign_tiles.py above the tests folder.")


def make_tiles() -> list[dict[str, Any]]:
    """Build a small fake tile index: four frames of different sizes."""
    return [
        {
            "file": frame,
            "tile_size": TILE_SIZE,
            "row": i // TILES_PER_ROW,
            "col": i % TILES_PER_ROW,
            "x0": (i % TILES_PER_ROW) * TILE_SIZE,
            "y0": (i // TILES_PER_ROW) * TILE_SIZE,
        }
        for frame, count in TILES_PER_FRAME.items()
        for i in range(count)
    ]


def tile_id(record: dict[str, Any]) -> TileId:
    """Return the (file, row, col) identity of a tile record."""
    return (record["file"], record["row"], record["col"])


def reviewers_by_tile(spec: dict[str, Any]) -> dict[TileId, list[str]]:
    """Invert the assignment file: tile -> the people who review it."""
    reviewers: dict[TileId, list[str]] = defaultdict(list)
    for person, records in spec["assignments"].items():
        for record in records:
            reviewers[tile_id(record)].append(person)
    return reviewers


def tier_of_tile(spec: dict[str, Any]) -> dict[TileId, str]:
    """Return each tile's tier, failing if its reviewers disagree about it."""
    tiers: dict[TileId, str] = {}
    for records in spec["assignments"].values():
        for record in records:
            assert tiers.setdefault(tile_id(record), record["tier"]) == record["tier"]
    return tiers


@pytest.fixture(scope="module")
def assign_tiles() -> ModuleType:
    """Import labelstudio/assign_tiles.py as a module."""
    path = find_labelstudio_dir() / "assign_tiles.py"
    spec = importlib.util.spec_from_file_location("assign_tiles", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def index_file(tmp_path: Path) -> Path:
    """Write the fake tile index to a file for --index-file."""
    path = tmp_path / "index.json"
    path.write_text(json.dumps(make_tiles()))
    return path


@pytest.fixture
def run_assign(
    assign_tiles: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    index_file: Path,
    tmp_path: Path,
) -> RunAssign:
    """Return a function that runs assign_tiles.py main() and loads its JSON output."""

    def run(*extra: str, name: str = "assign.json") -> dict[str, Any]:
        out = tmp_path / name
        argv = ["assign_tiles.py", "--people", *PEOPLE]
        argv += ["--index-file", str(index_file), "--out", str(out), *extra]
        monkeypatch.setattr(sys, "argv", argv)
        assign_tiles.main()
        return json.loads(out.read_text())

    return run


# ---------------------------------------------------------------- reproducibility


def test_same_seed_gives_identical_assignment(run_assign: RunAssign) -> None:
    """Running twice with the same seed and settings gives the same assignment."""
    first = run_assign("--seed", SEED, "--tiers", *TIER_ARGS, name="a.json")
    second = run_assign("--seed", SEED, "--tiers", *TIER_ARGS, name="b.json")
    assert first == second


def test_different_seeds_give_different_assignments(run_assign: RunAssign) -> None:
    """A different seed changes who gets which tiles."""
    first = run_assign("--seed", SEED, "--tiers", *TIER_ARGS, name="a.json")
    second = run_assign("--seed", OTHER_SEED, "--tiers", *TIER_ARGS, name="b.json")
    assert first["assignments"] != second["assignments"]


def test_recorded_seed_reproduces_an_unseeded_run(run_assign: RunAssign) -> None:
    """Without --seed one is chosen and recorded; reusing it reproduces the run."""
    unseeded = run_assign("--tiers", *TIER_ARGS, name="a.json")
    replay = run_assign(
        "--seed", str(unseeded["seed"]), "--tiers", *TIER_ARGS, name="b.json"
    )
    assert unseeded["assignments"] == replay["assignments"]


# ---------------------------------------------------------------- reviewers


@pytest.mark.parametrize("copies", [DEFAULT_COPIES, THREE_COPIES])
def test_every_tile_has_distinct_reviewers(run_assign: RunAssign, copies: int) -> None:
    """Each tile goes to exactly `copies` different people."""
    spec = run_assign("--seed", SEED, "--tiers", *TIER_ARGS, "--copies", str(copies))
    reviewers = reviewers_by_tile(spec)
    assert len(reviewers) == N_TILES
    for people in reviewers.values():
        assert len(people) == copies
        assert len(set(people)) == copies


def test_nobody_gets_the_same_tile_twice(run_assign: RunAssign) -> None:
    """A person's own list never repeats a tile."""
    spec = run_assign("--seed", SEED, "--tiers", *TIER_ARGS)
    for records in spec["assignments"].values():
        ids = [tile_id(r) for r in records]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------- balance


def test_workloads_are_balanced_overall_and_at_every_tier(
    run_assign: RunAssign,
) -> None:
    """At the end of each tier, everyone's cumulative count is within one."""
    spec = run_assign("--seed", SEED, "--tiers", *TIER_ARGS)
    for last_tier in range(len(TIER_NAMES)):
        allowed = set(TIER_NAMES[: last_tier + 1])
        counts = [
            sum(1 for r in records if r["tier"] in allowed)
            for records in spec["assignments"].values()
        ]
        assert max(counts) - min(counts) <= 1


# ---------------------------------------------------------------- nested tiers


def test_tiers_have_the_requested_cumulative_sizes(run_assign: RunAssign) -> None:
    """Small, medium and large hold 8, 20 and all tiles in total."""
    spec = run_assign("--seed", SEED, "--tiers", *TIER_ARGS)
    tiers = tier_of_tile(spec)
    for last_tier, size in enumerate(TIER_SIZES):
        allowed = set(TIER_NAMES[: last_tier + 1])
        assert sum(1 for t in tiers.values() if t in allowed) == size
    assert [t["name"] for t in spec["tiers"]] == TIER_NAMES
    assert [t["total_tiles"] for t in spec["tiers"]] == TIER_SIZES


def test_both_reviewers_agree_on_a_tiles_tier(run_assign: RunAssign) -> None:
    """A tile sits in one tier for everyone who reviews it."""
    spec = run_assign("--seed", SEED, "--tiers", *TIER_ARGS)
    assert len(tier_of_tile(spec)) == N_TILES


def test_growing_later_tiers_does_not_change_earlier_ones(
    run_assign: RunAssign,
) -> None:
    """The small tier is the same tiles whatever the medium tier's size is."""
    first = run_assign("--seed", SEED, "--tiers", "8", "20", "40", name="a.json")
    second = run_assign("--seed", SEED, "--tiers", "8", "30", "40", name="b.json")
    small = [
        {t for t, tier in tier_of_tile(s).items() if tier == "small"}
        for s in (first, second)
    ]
    assert small[0] == small[1]


def test_each_person_works_through_tiers_in_order(run_assign: RunAssign) -> None:
    """Every person's list is small first, then medium, then large, numbered 1..n."""
    spec = run_assign("--seed", SEED, "--tiers", *TIER_ARGS)
    rank = {name: i for i, name in enumerate(TIER_NAMES)}
    for records in spec["assignments"].values():
        ranks = [rank[r["tier"]] for r in records]
        assert ranks == sorted(ranks)
        assert [r["order"] for r in records] == list(range(1, len(records) + 1))


def test_small_tier_covers_every_frame(run_assign: RunAssign) -> None:
    """By default the small tier is spread across all frames."""
    spec = run_assign("--seed", SEED, "--tiers", *TIER_ARGS)
    small_frames = {t[0] for t, tier in tier_of_tile(spec).items() if tier == "small"}
    assert small_frames == set(TILES_PER_FRAME)


def test_single_tier_is_used_when_no_tiers_given(run_assign: RunAssign) -> None:
    """Without --tiers there is one tier called 'all' containing every tile."""
    spec = run_assign("--seed", SEED)
    assert [t["name"] for t in spec["tiers"]] == ["all"]
    assert set(tier_of_tile(spec).values()) == {"all"}


# ---------------------------------------------------------------- master CSV


@pytest.mark.parametrize("copies", [DEFAULT_COPIES, THREE_COPIES])
def test_csv_reviewer_columns_match_copies(
    run_assign: RunAssign, tmp_path: Path, copies: int
) -> None:
    """The CSV header has one reviewer column per copy, and every row fits it."""
    run_assign("--seed", SEED, "--tiers", *TIER_ARGS, "--copies", str(copies))
    with (tmp_path / "assign.csv").open(newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    assert header[4:] == [f"reviewer_{i}" for i in range(1, copies + 1)]
    assert len(body) == N_TILES
    for row in body:
        assert len(row) == len(header)
        reviewers = row[4:]
        assert all(reviewers)
        assert len(set(reviewers)) == copies


# ---------------------------------------------------------------- validation


def test_duplicate_tier_names_are_rejected(
    run_assign: RunAssign, tmp_path: Path
) -> None:
    """Repeated names would let later tiers load under the 'small' limit."""
    with pytest.raises(SystemExit, match="unique"):
        run_assign("--tiers", *TIER_ARGS, "--tier-names", "small", "small", "large")
    assert not (tmp_path / "assign.json").exists()


def test_tier_names_must_match_number_of_tiers(run_assign: RunAssign) -> None:
    """Too few or too many names is an error."""
    with pytest.raises(SystemExit, match="same number"):
        run_assign("--tiers", *TIER_ARGS, "--tier-names", "small", "large")


def test_custom_unique_tier_names_are_used(run_assign: RunAssign) -> None:
    """Unique custom names are accepted and recorded."""
    spec = run_assign(
        "--seed", SEED, "--tiers", *TIER_ARGS, "--tier-names", "a", "b", "c"
    )
    assert [t["name"] for t in spec["tiers"]] == ["a", "b", "c"]


def test_tiers_must_be_increasing(run_assign: RunAssign) -> None:
    """Tier sizes that don't grow are rejected."""
    with pytest.raises(SystemExit, match="strictly increasing"):
        run_assign("--tiers", "20", "8")


def test_largest_tier_cannot_exceed_tile_count(run_assign: RunAssign) -> None:
    """Asking for more tiles than exist is rejected."""
    with pytest.raises(SystemExit, match="exceeds"):
        run_assign("--tiers", "8", str(N_TILES + 1))


def test_cannot_ask_for_more_copies_than_people(run_assign: RunAssign) -> None:
    """Each tile needs distinct reviewers, so --copies cannot exceed the people."""
    with pytest.raises(SystemExit, match="at least as many people"):
        run_assign("--copies", str(TOO_MANY_COPIES))
