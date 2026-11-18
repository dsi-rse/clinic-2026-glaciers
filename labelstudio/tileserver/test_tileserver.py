"""Tests for the HDF5 -> PNG tile server, on small synthetic frames."""

import io
from pathlib import Path

import h5py
import numpy as np
import pytest
import tileserver
from PIL import Image

HEIGHT, WIDTH, BLOCK, TILE = 300, 250, 64, 128
MAX_GRAY = 255
# 3 x 2 tiles of TILE, minus the nearly-empty bottom-right one
TILES_WITH_DATA = 5
EPSG = 3031
OTHER_EPSG = 32633
PIXEL = 10.0
X_FIRST = PIXEL / 2
Y_FIRST = HEIGHT * PIXEL - PIXEL / 2


@pytest.fixture
def frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A 300 x 250 frame with NaNs, a zero, flagged pixels, and an almost-empty corner tile."""
    rng = np.random.default_rng(12345)
    values = rng.lognormal(mean=-3.0, sigma=1.5, size=(HEIGHT, WIDTH)).astype(
        np.float32
    )
    values[:, :40] = np.nan  # nodata strip, like the swath edge of a real frame
    values[5, 100] = 0.0  # log10 undefined
    values[2 * TILE :, TILE:] = np.nan  # bottom-right tile (2, 1) ...
    values[260, 210] = 0.5  # ... has a single valid pixel
    exception = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    exception[50:60, 50:150] = (
        2  # flagged, with absurd values that must not move the stretch
    )
    values[50:60, 50:150] = 1e6

    path = tmp_path / "images" / "frame.h5"
    path.parent.mkdir()
    with h5py.File(path, "w") as handle:
        grid = handle.create_group(tileserver.GRID)
        grid.create_dataset(tileserver.LAYER, data=values, chunks=(64, 64))
        grid.create_dataset(tileserver.EXCEPTION_MASK, data=exception, chunks=(64, 64))

    monkeypatch.setattr(tileserver, "BLOCK", BLOCK)
    monkeypatch.setattr(tileserver, "TILE_SIZE", TILE)
    monkeypatch.setattr(
        tileserver, "STRIP_ROWS", 50
    )  # does not divide BLOCK: exercises strips
    monkeypatch.setattr(tileserver, "IMAGE_DIR", path.parent)
    monkeypatch.setattr(tileserver, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(tileserver, "_stats_memo", {})
    return path


def expected_valid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Values and validity, computed directly from the whole frame in memory."""
    with h5py.File(path, "r") as handle:
        grid = handle[tileserver.GRID]
        values = grid[tileserver.LAYER][:]
        exception = grid[tileserver.EXCEPTION_MASK][:]
    with np.errstate(invalid="ignore"):
        return values, np.isfinite(values) & (values > 0) & (exception == 0)


def test_percentiles_ignore_nodata_and_flagged(frame: Path) -> None:
    """The stretch is the 1..99% range of valid pixels in dB, to within a histogram bin."""
    values, valid = expected_valid(frame)
    stats = tileserver.compute_stats(frame)
    low, high = np.percentile(10 * np.log10(values[valid]), [1, 99])
    assert stats["low_db"] == pytest.approx(low, abs=2 * tileserver.DB_BIN)
    assert stats["high_db"] == pytest.approx(high, abs=2 * tileserver.DB_BIN)
    assert stats["high_db"] < 10 * np.log10(1e6) - 10  # flagged pixels were excluded


def test_block_valid_counts(frame: Path) -> None:
    """Per-block valid counts match a direct count, despite strips not dividing blocks."""
    _, valid = expected_valid(frame)
    counts = np.array(tileserver.compute_stats(frame)["block_valid"])
    assert counts.shape == (5, 4)
    for row in range(5):
        for col in range(4):
            block = valid[
                row * BLOCK : (row + 1) * BLOCK, col * BLOCK : (col + 1) * BLOCK
            ]
            assert counts[row, col] == block.sum()


def test_list_tiles_skips_nearly_empty(frame: Path) -> None:
    """The bottom-right tile has one valid pixel and is dropped; edge tiles are smaller."""
    _, valid = expected_valid(frame)
    tiles = tileserver.list_tiles(tileserver.compute_stats(frame), TILE, 0.05)
    by_position = {(t["row"], t["col"]): t for t in tiles}
    assert set(by_position) == {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)}
    assert by_position[(2, 0)]["height"] == HEIGHT - 2 * TILE
    assert by_position[(0, 1)]["width"] == WIDTH - TILE
    assert by_position[(1, 1)]["x0"] == TILE and by_position[(1, 1)]["y0"] == TILE
    assert {t["tile_size"] for t in tiles} == {TILE}


def test_list_tiles_any_block_multiple(frame: Path) -> None:
    """The same stats give tiles of any size that is a whole number of blocks."""
    stats = tileserver.compute_stats(frame)
    assert (
        len(tileserver.list_tiles(stats, BLOCK, 0.0)) < 5 * 4
    )  # the empty corner is skipped
    big = tileserver.list_tiles(stats, 4 * BLOCK, 0.0)
    assert [(t["row"], t["col"], t["width"], t["height"]) for t in big] == [
        (0, 0, WIDTH, 4 * BLOCK),
        (1, 0, WIDTH, HEIGHT - 4 * BLOCK),
    ]
    for bad in [BLOCK + 1, BLOCK // 2, tileserver.MAX_TILE_SIZE + BLOCK]:
        with pytest.raises(ValueError):
            tileserver.list_tiles(stats, bad)


def test_render_tile(frame: Path) -> None:
    """Nodata is gray 0, valid pixels are 1..255, and the PNG round-trips."""
    values, valid = expected_valid(frame)
    stats = tileserver.get_stats(frame)
    gray = tileserver.render_tile(frame, stats, TILE, 0, 0)
    assert gray.shape == (TILE, TILE) and gray.dtype == np.uint8
    tile_valid = valid[:TILE, :TILE]
    assert (gray[~tile_valid] == 0).all()
    assert (gray[tile_valid] >= 1).all()
    assert (
        gray[tile_valid].min() == 1 and gray[tile_valid].max() == MAX_GRAY
    )  # clipped tails

    edge = tileserver.render_tile(frame, stats, TILE, 2, 1)
    assert edge.shape == (HEIGHT - 2 * TILE, WIDTH - TILE)

    decoded = np.array(Image.open(io.BytesIO(tileserver.encode_png(gray))))
    np.testing.assert_array_equal(decoded, gray)

    with pytest.raises(IndexError):
        tileserver.render_tile(frame, stats, TILE, 3, 0)

    # A different size renders the same pixels at the same frame coordinates.
    big = tileserver.render_tile(frame, stats, 2 * TILE, 0, 0)
    small = tileserver.render_tile(frame, stats, TILE, 1, 0)
    np.testing.assert_array_equal(big[TILE : 2 * TILE, :TILE], small)


def test_stats_are_cached(frame: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second server process reads the stats file instead of rescanning the frame."""
    first = tileserver.get_stats(frame)
    assert (tileserver.CACHE_DIR / "frame.json").exists()
    tileserver._stats_memo.clear()

    def rescan(*args: object) -> None:
        raise AssertionError("rescanned a frame whose stats were cached")

    monkeypatch.setattr(tileserver, "compute_stats", rescan)
    assert tileserver.get_stats(frame) == first


def test_resolve_rejects_escapes(frame: Path) -> None:
    """URLs cannot reach files outside the image directory, or non-HDF5 files."""
    root = frame.parent
    assert tileserver.resolve("frame.h5", root) == frame.resolve()
    (root.parent / "secret.h5").write_bytes(b"")
    (root / "notes.txt").write_text("hi")
    for bad in ["../secret.h5", "/etc/passwd", "notes.txt", "missing.h5"]:
        with pytest.raises(FileNotFoundError):
            tileserver.resolve(bad, root)


def test_index(frame: Path) -> None:
    """The index lists every tile with data, tagged with its file."""
    index = tileserver.build_index(frame.parent)
    assert {t["file"] for t in index} == {"frame.h5"}
    assert len(index) == TILES_WITH_DATA


def test_tile_url() -> None:
    """Tile URLs carry their size; older URLs without one still parse."""
    match = tileserver.TILE_URL.match("/tile/sub/dir/a%20b.h5/4096/3/12.png")
    assert match.group("file", "size", "row", "col") == (
        "sub/dir/a%20b.h5",
        "4096",
        "3",
        "12",
    )
    legacy = tileserver.TILE_URL.match("/tile/a.h5/3/12.png")
    assert legacy.group("file", "size", "row", "col") == ("a.h5", None, "3", "12")
    assert tileserver.TILE_URL.match("/tile/a.txt/3/12.png") is None


def add_coordinates(path: Path, epsg: int = EPSG) -> None:
    """Give the frame a pixel grid with its top-left corner at (0, HEIGHT * PIXEL)."""
    with h5py.File(path, "a") as handle:
        grid = handle[tileserver.GRID]
        grid.create_dataset("xCoordinates", data=X_FIRST + PIXEL * np.arange(WIDTH))
        grid.create_dataset("yCoordinates", data=Y_FIRST - PIXEL * np.arange(HEIGHT))
        projection = grid.create_dataset("projection", data=np.uint32(epsg))
        projection.attrs["epsg_code"] = epsg


def offered(
    stats: dict, monkeypatch: pytest.MonkeyPatch, box: str, crs: str = "EPSG:3031"
) -> set[tuple[int, int]]:
    """The (row, col) of every tile list_tiles offers with BBOX set to box."""
    monkeypatch.setattr(tileserver, "BBOX", box)
    monkeypatch.setattr(tileserver, "BBOX_CRS", crs)
    return {(t["row"], t["col"]) for t in tileserver.list_tiles(stats, TILE, 0.05)}


def test_compute_stats_reads_coordinates(frame: Path) -> None:
    """The frame's origin, pixel spacing and EPSG code are kept in the stats."""
    add_coordinates(frame)
    stats = tileserver.compute_stats(frame)
    assert stats["epsg"] == EPSG
    assert (stats["x_first"], stats["y_first"]) == (X_FIRST, Y_FIRST)
    assert (stats["dx"], stats["dy"]) == (PIXEL, -PIXEL)


def test_compute_stats_without_coordinates(frame: Path) -> None:
    """A frame with no coordinates still gets stats, just without georeferencing."""
    assert "x_first" not in tileserver.compute_stats(frame)


def test_box_selects_tiles(frame: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only tiles that overlap the box are offered."""
    add_coordinates(frame)
    stats = tileserver.compute_stats(frame)
    # Tile columns start at x = 0 and 1280; tile rows start at y = 3000, 1720 and 440.
    assert offered(stats, monkeypatch, "100,2000,1000,2900") == {(0, 0)}
    assert offered(stats, monkeypatch, "1000,2000,1500,2900") == {(0, 0), (0, 1)}
    assert offered(stats, monkeypatch, "100,500,1000,2900") == {(0, 0), (1, 0)}


def test_box_touching_a_tile_edge(frame: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A box that only shares an edge with a tile does not select it."""
    add_coordinates(frame)
    stats = tileserver.compute_stats(frame)
    # x = 1280 is the boundary between tile columns 0 and 1.
    assert offered(stats, monkeypatch, "1280,2000,2000,2900") == {(0, 1)}


def test_box_outside_the_frame(frame: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A box beside or above the frame selects no tiles."""
    add_coordinates(frame)
    stats = tileserver.compute_stats(frame)
    assert offered(stats, monkeypatch, "-5000,2000,-4000,2900") == set()
    assert offered(stats, monkeypatch, "100,4000,1000,5000") == set()


def test_box_covering_frame_matches_no_box(
    frame: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A box larger than the frame offers the same tiles as no box at all."""
    add_coordinates(frame)
    stats = tileserver.compute_stats(frame)
    monkeypatch.setattr(tileserver, "BBOX", "")
    unrestricted = {
        (t["row"], t["col"]) for t in tileserver.list_tiles(stats, TILE, 0.05)
    }
    assert len(unrestricted) == TILES_WITH_DATA
    assert offered(stats, monkeypatch, "-100000,-100000,100000,100000") == unrestricted


def test_box_still_drops_nearly_empty_tiles(
    frame: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A box over the one-valid-pixel tile (2, 1) does not bring that tile back."""
    add_coordinates(frame)
    stats = tileserver.compute_stats(frame)
    assert offered(stats, monkeypatch, "1300,10,2400,400") == set()


def test_box_in_another_crs_offers_nothing(
    frame: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frames in a different projection than BBOX_CRS get no tiles."""
    add_coordinates(frame, epsg=OTHER_EPSG)
    stats = tileserver.compute_stats(frame)
    assert offered(stats, monkeypatch, "100,2000,1000,2900") == set()


def test_box_in_unknown_crs_offers_nothing(
    frame: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A box given in the wrong CRS is not silently applied to a frame."""
    add_coordinates(frame)
    stats = tileserver.compute_stats(frame)
    assert offered(stats, monkeypatch, "100,2000,1000,2900", crs="EPSG:4326") == set()


def test_box_needs_coordinates(frame: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting BBOX on a frame with no coordinates raises, rather than guessing."""
    stats = tileserver.compute_stats(frame)
    monkeypatch.setattr(tileserver, "BBOX", "0,0,1,1")
    with pytest.raises(KeyError):
        tileserver.list_tiles(stats, TILE, 0.05)


def test_index_applies_box(frame: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The index only lists tiles inside the box."""
    add_coordinates(frame)
    monkeypatch.setattr(tileserver, "BBOX", "100,2000,1000,2900")
    index = tileserver.build_index(frame.parent)
    assert {(t["row"], t["col"]) for t in index} == {(0, 0)}
    assert {t["file"] for t in index} == {"frame.h5"}


def test_index_skips_frames_without_coordinates_when_box_is_set(
    frame: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frame the box cannot be applied to is skipped, not a server error."""
    monkeypatch.setattr(tileserver, "BBOX", "0,0,1,1")
    assert tileserver.build_index(frame.parent) == []
