"""Tests for the stack plot helpers (made-up frame names, no real data)."""

from datetime import date
from pathlib import Path

import pytest

from utils.stack_plot import find_stack_frames, get_cycle, get_date, get_stack_id

FRAME = "NISAR_L2_PR_GCOV_024_077_A_140_4005_SHSH_A_20260701T135211_20260701T135247_P05023_N_F_J_001"


def make_frame(directory: Path, cycle: str, stack: str, day: str) -> Path:
    """Create an empty file named like a frame of the given cycle, stack and day."""
    track, direction, frame = stack.split("_")
    name = (
        f"NISAR_L2_PR_GCOV_{cycle}_{track}_{direction}_{frame}_4005_SHSH_A_"
        f"{day}T135211_{day}T135247_P05023_N_F_J_001.h5"
    )
    path = directory / name
    path.write_text("x")
    return path


def test_stack_id_from_a_frame_name() -> None:
    """The stack id is the track, direction and frame, whatever the cycle."""
    assert get_stack_id(FRAME) == "077_A_140"
    assert get_stack_id(FRAME.replace("GCOV_024", "GCOV_026")) == "077_A_140"


def test_stack_id_from_a_stack_id() -> None:
    """A stack id given directly is returned as it is."""
    assert get_stack_id("077_A_140") == "077_A_140"


def test_stack_id_bad_text() -> None:
    """Text that is not a stack id or a frame name gives a clear error."""
    with pytest.raises(ValueError, match="Cannot read a stack id"):
        get_stack_id("not a stack")


def test_cycle_and_date_from_a_frame_name() -> None:
    """The cycle and the acquisition date are read from the name."""
    assert get_cycle(FRAME) == "024"
    assert get_date(FRAME) == date(2026, 7, 1)


def test_find_stack_frames_picks_one_stack_oldest_first(tmp_path: Path) -> None:
    """Only the frames of the stack are returned, sorted by date."""
    later = make_frame(tmp_path, "026", "077_A_140", "20260725")
    earlier = make_frame(tmp_path, "024", "077_A_140", "20260701")
    make_frame(tmp_path, "024", "083_D_126", "20260701")  # another stack

    frames, skipped = find_stack_frames(tmp_path, "077_A_140")

    assert frames == [earlier, later]
    assert skipped == []


def test_find_stack_frames_ignores_files_that_are_not_frames(tmp_path: Path) -> None:
    """A .h5 file with another kind of name does not stop the search."""
    (tmp_path / "other.h5").write_text("x")
    frame = make_frame(tmp_path, "024", "077_A_140", "20260701")

    frames, _skipped = find_stack_frames(tmp_path, "077_A_140")

    assert frames == [frame]
