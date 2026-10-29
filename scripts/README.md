# Scripts

| Script | What it does | Quick start (Docker) |
| --- | --- | --- |
| [`enhance_contrast.py`](#enhance_contrastpy) | Makes CLAHE and high-pass versions of NISAR tiles | `make enhance-contrast` |
| [`split_tiles.py`](#split_tilespy) | Splits tiles into train and test with a checkerboard | `make split-tiles` |
| [`plot_split_steps.py`](#plot_split_stepspy) | Draws the split in four pictures: study area, blocks, sample tiles, tiles after the split | `make plot-split-steps ARGS="--block-km 40.96"` |
| [`plot_ice_movement.py`](#plot_ice_movementpy) | Plots all the dates of one stack to see how the ice moves | `make plot-ice-movement ARGS="--stack-id 077_A_140"` |

## Setup

You can run the scripts with **Docker + Make** or with **`uv`**.

| | Docker | `uv` |
| --- | --- | --- |
| Run a script | `make enhance-contrast` | `make enhance-contrast-local` |
| Add options | `make enhance-contrast ARGS="--n-tiles 5"` | `make enhance-contrast-local ARGS="--n-tiles 5"` |
| See all options | `docker compose run --rm clinic-2026-glaciers python scripts/<script>.py --help` | `uv run python scripts/<script>.py --help` |

(Same pattern for `split-tiles`, `split-tiles-custom`, `split-tiles-pine-island` and `plot-split-steps`, each with a `-local` version.)

Before you start:

- Docker: Docker must be running, and `.env` must have `DATA_DIR` (see the main README).
- `uv`: run `uv sync` once.

Docker and Box on macOS: the container cannot read or overwrite Box files that are
online-only (error `Resource deadlock avoided`). Make the frames available offline, and send
the output to a local folder outside Box. `data/` in the repo is mounted in the container and
ignored by git:

```bash
make enhance-contrast ARGS="--output-dir /project/data/test_out --n-frames 1 --n-tiles 1"
```

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
2. **Bounding box (optional).** With `--region`, only the tiles that lie completely inside the
   box are kept (the area of analysis). Tiles that only partly overlap it are dropped.
3. **Make blocks.** Covers the tiles that are left with square blocks (`--block-km` per side),
   starting at their lower left corner. Neighbors have opposite colors, like a checkerboard:
   one color is train, the other test (`--swap` exchanges them).
4. **Assign tiles.** A tile goes to the block that holds its center, whatever frame it comes
   from. Overlapping frames of the same place land on the same side.
5. **Drop border tiles.** Tiles whose center is closer to a block edge than half a tile (plus
   half of `--gap-km`) are labeled `gap`. No train tile then touches a test tile.
6. **Check.** Stops with an error if any train tile is closer than one tile width (plus the
   gap) to a test tile.

### Run it

```bash
# Default: 1024 px tiles (10 km) with 30, 60, 120 km blocks,
# and 4096 px tiles (41 km) with 100, 150, 200 km blocks, with maps
make split-tiles

# Your own options
make split-tiles-custom ARGS="--block-km 60 --swap --gap-km 5 --plot"

# Pine Island comparison: original box, 1024 and 2048 px tiles, 4 block sizes, gap 0 km
make split-tiles-pine-island
```

Same with `uv`: add `-local` to the target (`make split-tiles-local`), or
`uv run python scripts/split_tiles.py --block-km 60 --swap --gap-km 5 --plot`.

With a bounding box, one run compares several tile sizes and block sizes and writes a
summary table:

```bash
uv run python scripts/split_tiles.py \
  --region -1686.068 -362.187 -1561.229 -255.680 \
  --tile-size 1024 2048 --block-km 30.72 40.96 53.33 61.44 --gap-km 0 --plot \
  --output "$DATA_DIR/splits/block_size_comparison/tile_split.csv"
```

### Options

| Option | Default | What it does |
| --- | --- | --- |
| `--block-km` | required | Block sizes in km. Several sizes make one CSV (and map) each. |
| `--tile-size` | 4096 | Tile size in pixels (a multiple of 512). 1024 px is about 10 km. Several sizes run one after the other. |
| `--region` | off | Bounding box `X_MIN Y_MIN X_MAX Y_MAX` in **km** (map x and y, EPSG:3031: not metres, not latitude and longitude). Only tiles completely inside are used. |
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

**Summary** (with `--region`): `<output name>_region_summary.csv`, one row per tile size and
block size: `tiles_found`, `outside_region`, `in_region`, `train`, `test`, `dropped` and
`dropped_pct` (the share of the tiles inside the box). Combinations where the block is too
small for the tile are skipped with a message.

**Map** (`--plot`): one PNG per block size, three panels side by side. Legends are below the
plots. With `--region`, the blocks are cut at the box.

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
- The grid starts at the lower left corner of the tiles that are left (inside the box, if
  `--region` is given). A box that is a whole number of blocks wastes fewer tiles.
- Counts include every date of every place, so the same ground is counted several times.
- Many kept tiles at the edge of a radar frame are mostly black (the tile server keeps a tile
  with at least 5% valid pixels). There is no filter for this yet.
- All tiles come from one area, so the test score measures performance in this area, not on
  other glaciers. Data from other places is needed to test that.
- There is no validation set yet, only train and test.

## `plot_split_steps.py`

### What it does

Shows how the split looks on the data, in four pictures. It runs the same steps as
`split_tiles.py` (same options), for **one tile size and one block size** per run. Tiles are
never cropped: a tile is kept whole or dropped. Code: `scripts/plot_split_steps.py`, with
helpers in `src/utils/plot_split.py`.

| Picture | Shows |
| --- | --- |
| `1_study_area.png` | The border of the study area (all frames) and the bounding box, if `--region` is given. |
| `2_blocks.png` | The area left after the box, with the checkerboard of train and test blocks. |
| `3_sample_tiles.png` | The tiles of a few sample stacks as outlines over the blocks. |
| `4_sample_split.png` | The same tiles with their radar image: violet edge = train, green = test, faded gray = dropped. One panel per stack and date. |

A stack id is the track, the direction and the frame (for example `077_A_140`): the same
place on different dates.

### Run it

```bash
make plot-split-steps ARGS="--region -1686.068 -362.187 -1561.229 -255.680 --tile-size 1024 --block-km 40.96 --gap-km 0 --stack-ids 077_A_140 019_A_141 091_A_140"
```

Same with `uv`: `make plot-split-steps-local ARGS="..."`. For several tile and block sizes,
run it in a loop (each run reads the images again, so it takes a few minutes):

```bash
for size in 1024 2048; do
  for block in 30.72 40.96 53.33 61.44; do
    uv run python scripts/plot_split_steps.py \
      --region -1686.068 -362.187 -1561.229 -255.680 \
      --tile-size $size --block-km $block --gap-km 0 \
      --output-dir "$DATA_DIR/split_steps/block_size_comparison/${size}px/blocks_${block}km"
  done
done
```

### Options

| Option | Default | What it does |
| --- | --- | --- |
| `--block-km` | required | Block size in km (one value). |
| `--tile-size` | 4096 | Tile size in pixels (one value). |
| `--region` | off | Bounding box in km, like in `split_tiles.py`. |
| `--gap-km` | 0 | Extra distance kept free between train and test tiles. |
| `--swap` | off | Swap train and test colors. |
| `--stack-ids` | the stacks with most frames | Sample stacks to draw in pictures 3 and 4. They need tiles inside the box. |
| `--n-stacks` | 2 | How many sample stacks when `--stack-ids` is not given. |
| `--max-dates` | 3 | Most dates drawn per stack. |
| `--image-dir`, `--cache-dir` | as in `split_tiles.py` | Frames and tile server cache. |
| `--output-dir` | `DATA_DIR/split_steps` | Folder for the four pictures. A new run overwrites files with the same name. |

### Good to know

- Pictures 3 and 4 read the images of the sample stacks, so a run takes a few minutes.
- In a small box, a stack that covers one corner falls in one block, so it can be all train
  or all test.

## `plot_ice_movement.py`

### What it does

Plots all the frames of one **stack** (the same place on different dates, 12 days apart) to see
how the ice moves. A stack id is the track, the direction and the frame, for example `077_A_140`.
You can also give the name of any frame of the stack.

For every date the script cuts the frame into tiles (4096 px, about 41 km, by default), draws the
tiles at their position on the map, and shows one panel per date with the same map limits. A last
panel zooms in on the most textured spot covered by the first and the last date and overlays
them: **red is the first date, cyan is the last date**. Where the ice did not move the overlay
looks gray. Where it moved, red and cyan fringes appear. The code is in `src/utils/stack_plot.py`.

### Run it

```bash
make plot-ice-movement ARGS="--stack-id 077_A_140"           # Docker
make plot-ice-movement-local ARGS="--stack-id 077_A_140"     # uv

uv run python scripts/plot_ice_movement.py --stack-id 077_A_140 --zoom-km 20
```

| Option | Default | What it does |
| --- | --- | --- |
| `--stack-id` | required | Stack id like `077_A_140`, or the name of a frame of the stack. |
| `--tile-size` | 4096 | Tile size in pixels (a multiple of 512). |
| `--zoom-km` | 30 | Side of the zoomed overlay in km. |
| `--include-online-only` | off | Also read frames that Box keeps online-only (slow). By default they are skipped and listed. |
| `--image-dir` | `DATA_DIR/$LABELSTUDIO_IMAGE_SUBDIR` | Folder with the `.h5` frames. |
| `--cache-dir` | `DATA_DIR/contrast_enhancement/stats_cache` | Shared with the other scripts. |
| `--output-dir` | `DATA_DIR/stack_plots` | Where the PNG is written. |

Output: one PNG per stack id, `stack_<stack id>_<tile size>px.png`.

### Good to know

- Only the dates that are downloaded are used. Online-only frames are skipped and listed, so a
  stack may have fewer dates than it has frames. With one date there is nothing to compare.
- The zoom goes to the most textured spot, which can be sea ice or icebergs at the ice front.
  They move with the ocean and the wind, so their movement is not the speed of the glacier.
- The picture only shows the movement. It does not measure it.
