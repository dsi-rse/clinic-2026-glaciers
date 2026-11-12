"""Tests for scripts/run_pipeline.py (made-up files, no real data, nothing is run)."""

import csv
import importlib.util
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("DATA_DIR", "/tmp/glaciers-data")  # noqa: S108 (only read, never written)
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_pipeline.py"
spec = importlib.util.spec_from_file_location("run_pipeline", SCRIPT)
run_pipeline = importlib.util.module_from_spec(spec)
sys.modules["run_pipeline"] = run_pipeline
spec.loader.exec_module(run_pipeline)

SPLIT_NAME = "tile_split_region_abc123_2048px_blocks_163.84km_gap_0km_date_202607.csv"


def make_args(monkeypatch: pytest.MonkeyPatch, *options: str) -> object:
    """The options of the script, as if they were typed on the command line."""
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", *options])
    return run_pipeline.parse_args()


def write_split_csv(path: Path) -> None:
    """A split CSV with 2 train, 1 test and 1 gap tile."""
    rows = [("a.h5", "train"), ("a.h5", "train"), ("b.h5", "test"), ("b.h5", "gap")]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "row", "col", "split"])
        writer.writerows(
            (frame, 0, col, split) for col, (frame, split) in enumerate(rows)
        )


def test_the_steps_are_in_the_expected_order() -> None:
    """Box, split, tiles, assign. Label Studio is not a step: each person does it by hand."""
    assert run_pipeline.STEPS == ("box", "split", "tiles", "assign")


def test_box_must_have_min_below_max() -> None:
    """A box with X_MIN above X_MAX stops with a message."""
    with pytest.raises(SystemExit, match="X_MIN < X_MAX"):
        run_pipeline.check_region((10.0, 0.0, 5.0, 8.0))
    run_pipeline.check_region((0.0, 0.0, 5.0, 8.0))  # a good box does nothing


def test_run_name_is_short_and_changes_with_the_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same settings give the same short name. Other settings (box, swap) give another one."""
    default = run_pipeline.get_run_name(make_args(monkeypatch))
    again = run_pipeline.get_run_name(make_args(monkeypatch))
    swapped = run_pipeline.get_run_name(make_args(monkeypatch, "--swap"))
    other_box = run_pipeline.get_run_name(
        make_args(monkeypatch, "--region", "1", "2", "3", "4")
    )
    assert default == again
    assert default.startswith("run_")
    assert len(default) == len("run_") + run_pipeline.CODE_LENGTH
    assert len({default, swapped, other_box}) == 3  # noqa: PLR2004


def test_a_chosen_name_is_used_as_it_is(monkeypatch: pytest.MonkeyPatch) -> None:
    """With --name the folder is called like that."""
    assert (
        run_pipeline.get_run_name(make_args(monkeypatch, "--name", "amery_july"))
        == "amery_july"
    )


def test_the_frames_folder_does_not_change_the_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Docker and uv see the frames at different paths but must write the same folder."""
    one = run_pipeline.get_run_name(
        make_args(monkeypatch, "--image-dir", str(tmp_path / "a"))
    )
    other = run_pipeline.get_run_name(
        make_args(monkeypatch, "--image-dir", "/data/frames")
    )
    assert one == other


def test_a_folder_with_other_settings_is_not_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same --name with another box stops with a message. The same settings are fine."""
    first = make_args(monkeypatch, "--name", "x")
    run_pipeline.save_settings(tmp_path, first)
    run_pipeline.save_settings(tmp_path, first)  # same settings: no problem
    other = make_args(monkeypatch, "--name", "x", "--region", "1", "2", "3", "4")
    with pytest.raises(SystemExit, match="region_km"):
        run_pipeline.save_settings(tmp_path, other)


def test_split_command_uses_the_box_dates_and_run_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Step 2 passes the box, the dates and the output folder to split_tiles.py."""
    args = make_args(
        monkeypatch, "--region", "1", "2", "3", "4", "--date-prefix", "202608"
    )
    command = run_pipeline.split_command(args, tmp_path)
    assert command[1].endswith("split_tiles.py")
    assert command[command.index("--region") + 1 : command.index("--region") + 5] == [
        "1.0", "2.0", "3.0", "4.0",
    ]  # fmt: skip
    assert command[command.index("--date-prefix") + 1] == "202608"
    assert command[command.index("--output") + 1] == str(tmp_path / "tile_split.csv")


def test_find_split_csv_ignores_the_summary(tmp_path: Path) -> None:
    """The split CSV is found, and the summary CSV next to it is not mistaken for it."""
    (tmp_path / SPLIT_NAME).touch()
    (tmp_path / SPLIT_NAME.replace(".csv", "_summary.csv")).touch()
    assert run_pipeline.find_split_csv(tmp_path).name == SPLIT_NAME


def test_find_split_csv_stops_when_there_is_none(tmp_path: Path) -> None:
    """Without a split CSV the pipeline stops and says to run the split step."""
    with pytest.raises(SystemExit, match="Run step split"):
        run_pipeline.find_split_csv(tmp_path)


def test_tiles_to_label_do_not_include_the_dropped_ones(tmp_path: Path) -> None:
    """Step 3 counts train, test and gap tiles."""
    split_csv = tmp_path / SPLIT_NAME
    write_split_csv(split_csv)
    assert run_pipeline.count_tiles(split_csv) == {"train": 2, "test": 1, "gap": 1}


def test_assign_command_reads_the_split_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Step 4 gives Jeanne's script the split CSV, the tile size, the frames folder and the seed."""
    args = make_args(monkeypatch, "--seed", "7", "--image-dir", str(tmp_path))
    command = run_pipeline.assign_command(
        args, tmp_path / SPLIT_NAME, tmp_path / "a.json"
    )
    assert command[1].endswith("labelstudio/assign_tiles.py")
    assert command[command.index("--split-csv") + 1] == str(tmp_path / SPLIT_NAME)
    assert command[command.index("--tile-size") + 1] == "2048"
    assert command[command.index("--image-dir") + 1] == str(tmp_path)
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--out") + 1] == str(tmp_path / "a.json")


def test_augmentation_is_not_connected_yet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Asking for an augmentation stops before anything runs."""
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--augmentation", "clahe"])
    with pytest.raises(SystemExit, match="not connected"):
        run_pipeline.main()


def test_runs_are_written_in_the_repo_not_in_data_dir() -> None:
    """Docker and uv share one output folder: data/pipeline in the repo (git ignores data/)."""
    assert run_pipeline.DEFAULT_RUN_ROOT == run_pipeline.REPO_DIR / "data" / "pipeline"
    assert not run_pipeline.DEFAULT_RUN_ROOT.is_relative_to(run_pipeline.DATA_DIR)
