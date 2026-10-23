# Scripts

| Script | What it does | Quick start (Docker) |
| --- | --- | --- |
| [`enhance_contrast.py`](#enhance_contrastpy) | Makes CLAHE and high-pass versions of NISAR tiles | `make enhance-contrast` |
| [`split_tiles.py`](#split_tilespy) | Splits tiles into train and test with a checkerboard | `make split-tiles` |

## Setup

You can run the scripts with **Docker + Make** or with **`uv`**.

| | Docker | `uv` |
| --- | --- | --- |
| Run a script | `make enhance-contrast` | `make enhance-contrast-local` |
| Add options | `make enhance-contrast ARGS="--n-tiles 5"` | `make enhance-contrast-local ARGS="--n-tiles 5"` |
| See all options | `docker compose run --rm clinic-2026-glaciers python scripts/<script>.py --help` | `uv run python scripts/<script>.py --help` |

(Same pattern for `split-tiles` and `split-tiles-local`.)

Before you start:

- Docker: Docker must be running, and `.env` must have `DATA_DIR` (see the main README).
- `uv`: run `uv sync` once.

---

## `enhance_contrast.py`

### What it does

Takes NISAR frames, cuts tiles (4096 x 4096 px, HHHH backscatter in dB, using the Label Studio
tile server code in `labelstudio/tileserver/tileserver.py`) and applies two enhancements:

| Method | Code | Effect |
| --- | --- | --- |
| CLAHE | `utils.augmentation.clahe` | Boosts local contrast. |
| High-pass | `utils.augmentation.high_pass` | Removes slow, large-scale brightness changes and keeps small features. |

### Run it

Pick **one option for frames** and **one for tiles**:

| | Option | What it does |
| --- | --- | --- |
| Frames | `--n-frames N` (default 1) | N random frames. |
| Frames | `--frames a.h5 b.h5` | Only the frames you list. |
| Frames | `--all-frames` | Every frame in the image folder. |
| Tiles | `--n-tiles N` (default 3) | N random tiles in each frame. |
| Tiles | `--all-tiles` | Every tile with data. Slow, writes many large files. Wins over `--n-tiles`. |

Examples:

```bash
# Default: 1 random frame, 3 random tiles
make enhance-contrast

# Your own frames, 5 random tiles each, high-pass sigma 30
make enhance-contrast ARGS="--frames frame_a.h5 frame_b.h5 --n-tiles 5 --sigma 30"

# Every frame, 2 random tiles each
make enhance-contrast ARGS="--all-frames --n-tiles 2"

# Every frame, every tile
make enhance-contrast ARGS="--all-frames --all-tiles"
```

For `uv`, use `make enhance-contrast-local` (same `ARGS`), or without Make:
`uv run python scripts/enhance_contrast.py --all-frames --n-tiles 2`.

### Other options

| Option | Default | Meaning |
| --- | --- | --- |
| `--clip-limit` | 0.01 | CLAHE contrast limit. Higher is stronger. |
| `--sigma` | 50 | High-pass blur size in pixels. |
| `--seed` | 0 | Same seed gives the same random tiles. |

### Where the files are

- **Input:** frames in `DATA_DIR/$LABELSTUDIO_IMAGE_SUBDIR` (see `.env`).
- **Output:** `DATA_DIR/contrast_enhancement/`

| Path | Content |
| --- | --- |
| `clahe/`, `high_pass/` | The processed tile only, full size (4096 x 4096 PNG). |
| `comparison/` | Input, CLAHE and high-pass side by side, with the parameters in the titles. |
| `tiles_used.txt` | Which tiles and parameters were used. |
| `stats_cache/` | What the tile server learned about each frame. |

Every PNG is named `<frame>_r<row>_c<col>.png`.

### Good to know

- The first run on a frame is slow (about 30 s for a 1.4 GB frame) while the display range
  is computed. Later runs are quick.
- Box keeps some frames online-only. If reading a frame times out, download it first
  (right-click in Finder, Make Available Offline), or pass a downloaded frame with `--frames`.

---

## `split_tiles.py`

### What it does

Splits the NISAR tiles into **train** and **test** with a checkerboard of square blocks. The
test score then reflects new places, not near-copies of the training data. This is spatial
block cross-validation (Kattenborn et al. 2022; Stock 2025). Code: `src/utils/tile_split.py`.

**Why split by position?** Our frames overlap a lot (they all cover the same ice). Splitting
by frame or by tile number would still put the same ground in both train and test.

### How it works

1. **Find tiles.** Takes the tiles with data in every frame (the same ones Label Studio
   offers) and the map position of each tile center in metres. Positions come from
   `xCoordinates` and `yCoordinates` in each `.h5` frame (EPSG:3031). The script stops if a
   frame uses another projection or pixel size.
2. **Make blocks.** Covers the map with square blocks (`--block-km` per side). Neighbors have
   opposite colors, like a checkerboard: one color is train, the other test (`--swap`
   exchanges them).
3. **Assign tiles.** A tile goes to the block that holds its center, whatever frame it comes
   from. Overlapping frames of the same place land on the same side.
4. **Drop border tiles.** Tiles whose center is closer to a block edge than half a tile (plus
   half of `--gap-km`) are labeled `gap`. No train tile then touches a test tile.
5. **Check.** Stops with an error if any train tile is closer than one tile width (plus the
   gap) to a test tile.

### Run it

```bash
# Default: 1024 px tiles (10 km) with 30, 60, 120 km blocks,
# and 4096 px tiles (41 km) with 100, 150, 200 km blocks, with maps
make split-tiles

# Your own options
make split-tiles ARGS="--block-km 60 --swap --gap-km 5 --plot"
```

Same with `uv`: `make split-tiles-local`, or
`uv run python scripts/split_tiles.py --block-km 60 --swap --gap-km 5 --plot`.

### Options

| Option | Default | What it does |
| --- | --- | --- |
| `--block-km` | required | Block sizes in km. Several sizes make one CSV (and map) each. |
| `--tile-size` | 4096 | Tile size in pixels (a multiple of 512). 1024 px is about 10 km. |
| `--gap-km` | 0 | Extra distance kept free between train and test tiles. |
| `--swap` | off | Swap train and test colors. |
| `--plot` | off | Also save a map (PNG) next to each CSV. |
| `--include-online-only` | off | Also read frames that Box keeps online-only (skipped and listed by default, because each one has to be downloaded). |
| `--image-dir` | `DATA_DIR/$LABELSTUDIO_IMAGE_SUBDIR` | Folder with the `.h5` frames (subfolders are searched too). |
| `--cache-dir` | `DATA_DIR/contrast_enhancement/stats_cache` | Tile server cache. Shared with `enhance_contrast.py`. |
| `--output` | `DATA_DIR/splits/tile_split.csv` | Tile and block size are added to the name, e.g. `tile_split_1024px_blocks_60km.csv`. |

### Output

**CSV:** one per block size, one row per tile.

| Column | Meaning |
| --- | --- |
| `frame`, `row`, `col` | The tile. `row` and `col` match `enhance_contrast.py` and Label Studio. |
| `x_m`, `y_m` | Tile center on the map. |
| `split` | `train`, `test` or `gap`. **Ignore `gap` rows when training.** |
| `block_x`, `block_y` | The tile's block. |
| `block_color` | Whether that block is train or test. |

**Map** (`--plot`): one PNG per block size, three panels side by side.

| Panel | Shows |
| --- | --- |
| A. Blocks over the area | Blocks colored train (violet) or test (green), with the area covered by tiles outlined and shaded. |
| B. Where the tiles land | Same blocks, lightly tinted, with every tile as a plain outline. |
| C. Tiles colored by split | Only block borders drawn. Tiles colored by split (dropped tiles in light gray). |

Tiles from different frames overlap, so squares in B and C are semi-transparent.

### Choosing the block size

- **Try several sizes and compare model scores.** If small blocks score much higher than
  large ones, they probably still leak. Stock (2025) found block size to be the most
  important choice; block shape and assignment matter little.
- **Blocks must be bigger than a tile plus the gap**, or the script stops. Small blocks lose
  more tiles to the gap:
  - 10 km tiles, 30 km blocks: only the central 20 x 20 km of each block is kept.
  - 41 km tiles (`--tile-size 4096`): blocks must be larger than 82 km. Use 100, 150 or
    200 km. 100 km blocks keep only about a third of the tiles.
- **Sizes in the papers (e.g. 30 to 200 km) fit their data, not ours.** Pick sizes for our
  data, ideally from how quickly similarity between tiles falls with distance.

### Good to know

- The first run reads every frame once and is slow. Later runs use the cache.
- Online-only Box frames are skipped (and listed), so a run never waits for a download. To
  include them, make them available offline in Finder, or use `--include-online-only`.
  Skipped frames mean fewer tiles, so check that the map still covers the whole area.
  Detecting online-only files works on macOS only.
- The grid starts at the lower left corner of the tiles.
- All tiles come from one area, so the test score measures performance in this area, not on
  other glaciers. Data from other places is needed to test that.
- There is no validation set yet, only train and test.
