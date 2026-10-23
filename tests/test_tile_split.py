"""Tests for the checkerboard split of tiles (made-up positions, no real data)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils import tile_split
from utils.tile_split import assign_blocks, check_no_leakage, find_frames

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
