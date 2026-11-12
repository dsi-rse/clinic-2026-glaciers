"""Tests for the checkerboard split of tiles (made-up positions, no real data)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils import tile_split
from utils.tile_split import (
    assign_blocks,
    check_no_leakage,
    filter_frames_by_date,
    filter_region,
    find_frames,
    get_start_date,
    locate_blocks,
    split_by_blocks,
)

TILE_M = 40_960
GAP_KM = 10


def test_leakage_is_detected() -> None:
    """A train tile next to a test tile raises an error."""
    tiles = pd.DataFrame({"x_m": [0.0, TILE_M], "y_m": [0.0, 0.0]})
    labels = pd.Series(["train", "test"])
    with pytest.raises(ValueError, match="Leakage"):
        check_no_leakage(tiles, labels, TILE_M, GAP_KM)


def test_tiles_far_apart_are_not_leakage() -> None:
    """Tiles farther apart than one tile width plus the gap pass the check."""
    far = TILE_M + GAP_KM * 1000 + 1
    tiles = pd.DataFrame({"x_m": [0.0, far], "y_m": [0.0, 0.0]})
    labels = pd.Series(["train", "test"])
    check_no_leakage(tiles, labels, TILE_M, GAP_KM)


# Tests for the checkerboard of square blocks
BLOCK_KM = 100
BLOCK_TILE_M = 10_000
BLOCK_M = BLOCK_KM * 1000


def make_block_centers() -> pd.DataFrame:
    """Row 0 is a corner tile at (0, 0) that fixes where the grid starts.

    Then comes one tile at the center of each block of a 4 x 4 grid.
    """
    centers = [(0.0, 0.0)]
    for column in range(4):
        for row in range(4):
            centers.append((BLOCK_M * (column + 0.5), BLOCK_M * (row + 0.5)))
    return pd.DataFrame(centers, columns=["x_m", "y_m"])


def test_blocks_next_to_each_other_have_opposite_labels() -> None:
    """Blocks that share an edge get different labels, diagonal blocks the same one."""
    tiles = make_block_centers()
    labels = assign_blocks(tiles, BLOCK_TILE_M, BLOCK_KM)
    # Row 1 is block (0, 0), row 2 is block (0, 1), row 5 is block (1, 0), row 6 is (1, 1)
    assert labels[1] != labels[2]  # neighbor above
    assert labels[1] != labels[5]  # neighbor to the right
    assert labels[1] == labels[6]  # diagonal


def test_blocks_swap_flips_the_labels() -> None:
    """With swap, every train block becomes a test block and the other way round."""
    tiles = make_block_centers()
    labels = assign_blocks(tiles, BLOCK_TILE_M, BLOCK_KM)
    swapped = assign_blocks(tiles, BLOCK_TILE_M, BLOCK_KM, swap=True)
    centers = tiles.index[1:]
    assert (labels[centers] != swapped[centers]).all()


def test_blocks_drop_tiles_near_an_edge() -> None:
    """A tile closer to a block edge than half a tile is dropped."""
    tiles = make_block_centers()
    edge_tile = pd.DataFrame({"x_m": [BLOCK_M - 4_000.0], "y_m": [BLOCK_M * 0.5]})
    tiles = pd.concat([tiles, edge_tile], ignore_index=True)
    labels = assign_blocks(tiles, BLOCK_TILE_M, BLOCK_KM)
    assert labels.iloc[-1] == "gap"
    assert labels[1] != "gap"


def test_blocks_gap_widens_the_dropped_strip() -> None:
    """A tile that is kept without a gap can be dropped once a gap is asked for."""
    tiles = make_block_centers()
    near_tile = pd.DataFrame({"x_m": [BLOCK_M - 8_000.0], "y_m": [BLOCK_M * 0.5]})
    tiles = pd.concat([tiles, near_tile], ignore_index=True)
    assert assign_blocks(tiles, BLOCK_TILE_M, BLOCK_KM, gap_km=0).iloc[-1] != "gap"
    assert assign_blocks(tiles, BLOCK_TILE_M, BLOCK_KM, gap_km=10).iloc[-1] == "gap"


def test_blocks_same_place_same_label() -> None:
    """Two tiles at the same position (two frames) get the same label."""
    tiles = make_block_centers()
    tiles = pd.concat([tiles, tiles.iloc[1:]], ignore_index=True)
    labels = assign_blocks(tiles, BLOCK_TILE_M, BLOCK_KM)
    first_pass = labels.iloc[1:17].to_numpy()
    second_pass = labels.iloc[17:].to_numpy()
    assert (first_pass == second_pass).all()


def test_blocks_keep_train_and_test_apart_for_random_tiles() -> None:
    """For many random tiles, no train tile is within one tile width of a test tile."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 4 * BLOCK_M, size=(2000, 2))
    tiles = pd.DataFrame(positions, columns=["x_m", "y_m"])
    labels = assign_blocks(tiles, BLOCK_TILE_M, BLOCK_KM)
    assert {"train", "test", "gap"} <= set(labels)
    check_no_leakage(tiles, labels, BLOCK_TILE_M, gap_km=0)


def test_blocks_too_small_for_the_tiles() -> None:
    """A block no bigger than a tile would drop every tile, so we get a clear error."""
    tiles = make_block_centers()
    with pytest.raises(ValueError, match="too small"):
        assign_blocks(tiles, BLOCK_TILE_M, block_km=10)


def test_dataless_flag_means_online_only() -> None:
    """The macOS dataless flag marks a file as online-only, other flags do not."""
    assert tile_split.is_dataless(tile_split.DATALESS_FLAG)
    assert tile_split.is_dataless(tile_split.DATALESS_FLAG | 1)
    assert not tile_split.is_dataless(0)
    assert not tile_split.is_dataless(1)


def test_a_normal_file_is_not_online_only(tmp_path: Path) -> None:
    """A file on disk is downloaded, so it is listed as a frame and not skipped."""
    frame = tmp_path / "frame.h5"
    frame.write_text("x")
    assert not tile_split.is_online_only(frame)
    assert find_frames(tmp_path) == [frame]


def test_region_keeps_only_tiles_completely_inside() -> None:
    """A tile is kept only if its whole square lies inside the rectangle."""
    tiles = pd.DataFrame(
        {
            "x_m": [5_000.0, 4_999.0, 95_000.0, 95_001.0, 50_000.0],
            "y_m": [5_000.0, 50_000.0, 50_000.0, 50_000.0, 50_000.0],
        }
    )
    kept = filter_region(tiles, BLOCK_TILE_M, (0, 0, 100, 100))  # a 100 x 100 km box
    assert kept["x_m"].tolist() == [5_000.0, 95_000.0, 50_000.0]


def test_grid_starts_at_the_origin_when_given() -> None:
    """With an origin, block columns count from the origin and not from the first tile."""
    tiles = pd.DataFrame({"x_m": [50_000.0, 150_000.0], "y_m": [0.0, 0.0]})
    default = locate_blocks(tiles, BLOCK_KM)
    shifted = locate_blocks(tiles, BLOCK_KM, origin=(100_000.0, 0.0))
    assert default["block_x"].tolist() == [0, 1]
    assert shifted["block_x"].tolist() == [-1, 0]


def test_split_starts_the_grid_at_the_corner_of_the_tiles() -> None:
    """The first block begins at the edge of the lowest tile, so blocks cover every tile."""
    tiles = pd.DataFrame({"x_m": [5_000.0, 150_000.0], "y_m": [5_000.0, 5_000.0]})
    table = split_by_blocks(tiles, BLOCK_TILE_M, BLOCK_KM)
    assert table["block_x"].tolist() == [0, 1]
    assert table["block_y"].tolist() == [0, 0]


def make_frames(day_by_name: dict[str, str]) -> list[Path]:
    """Frame paths named like NISAR files, with the given start day (no files are made)."""
    return [
        Path(
            f"NISAR_L2_PR_GCOV_026_{name}_A_142_4005_SHSH_A_{day}T015731_{day}T015807_P05023_N_F_J_001.h5"
        )
        for name, day in day_by_name.items()
    ]


def test_date_prefix_keeps_only_the_frames_of_that_month() -> None:
    """A month prefix keeps July frames and drops August frames."""
    frames = make_frames({"041": "20260723", "062": "20260724", "063": "20260816"})
    kept = filter_frames_by_date(frames, "202607", Path("images"))
    assert kept == frames[:2]


def test_date_prefix_can_pick_one_day() -> None:
    """A full day prefix keeps only the frames of that day."""
    frames = make_frames({"041": "20260723", "062": "20260724"})
    assert filter_frames_by_date(frames, "20260724", Path("images")) == frames[1:]


def test_no_date_prefix_keeps_every_frame() -> None:
    """An empty prefix is no filter, even for names that are not NISAR names."""
    frames = [Path("not_a_nisar_name.h5")]
    assert filter_frames_by_date(frames, "", Path("images")) == frames


def test_date_prefix_without_a_match_says_so() -> None:
    """A prefix that matches no frame raises an error that names the prefix."""
    frames = make_frames({"041": "20260723"})
    with pytest.raises(FileNotFoundError, match="202609"):
        filter_frames_by_date(frames, "202609", Path("images"))


@pytest.mark.parametrize(
    "name", ["short_name.h5", "a_b_c_d_e_f_g_h_i_j_k_notadate_x.h5"]
)
def test_unexpected_file_name_gives_a_clear_message(name: str) -> None:
    """A name without a start date gives a ValueError with the file name, not an IndexError."""
    with pytest.raises(ValueError, match=name):
        filter_frames_by_date([Path(name)], "202607", Path("images"))


def test_start_date_is_read_from_the_name() -> None:
    """The start date and time is the 12th part of the name."""
    assert get_start_date(make_frames({"041": "20260723"})[0]) == "20260723T015731"
