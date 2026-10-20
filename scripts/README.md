# Documentation

### `enhance_contrast.py`

Applies contrast enhancements to NISAR frames.

| Method | Function | What it does |
| --- | --- | --- |
| CLAHE | `utils.augmentation.clahe` (`skimage.exposure.equalize_adapthist`) | Boosts local contrast. |
| High-pass | `utils.augmentation.high_pass` (original minus `scipy.ndimage.gaussian_filter` blur) | Removes slow, large-scale brightness changes and keeps small features. |

Tiles are cut and scaled with the Label Studio tile server code
(`labelstudio/tileserver/tileserver.py`) 4096 x 4096 pixels, HHHH backscatter in dB.

### Run it

Each run needs two choices: **which frames** to use, and **which tiles** in each frame.

| Choice | Option | What it does |
| --- | --- | --- |
| Frames | `--n-frames N` (default 1) | N random frames. |
| Frames | `--frames a.h5 b.h5` | Only the frames you list. |
| Frames | `--all-frames` | Every frame in the image folder. |
| Tiles | `--n-tiles N` (default 3) | N random tiles in each frame. |
| Tiles | `--all-tiles` | Every tile with data. Slow, and writes many large files. |

Use one option for the frames and one for the tiles. If you give both `--n-tiles` and
`--all-tiles`, `--all-tiles` wins.

Examples:

```bash
# Default: 1 random frame, 3 random tiles
make enhance-contrast

# Show all options
uv run python scripts/enhance_contrast.py --help

# Your own frames, 5 random tiles in each, high-pass sigma 30
uv run python scripts/enhance_contrast.py --frames frame_a.h5 frame_b.h5 --n-tiles 5 --sigma 30

# Every frame, 2 random tiles in each
uv run python scripts/enhance_contrast.py --all-frames --n-tiles 2

# Every frame, every tile
uv run python scripts/enhance_contrast.py --all-frames --all-tiles
```

Where the files are:

- Frames are read from `DATA_DIR/$LABELSTUDIO_IMAGE_SUBDIR` (see `.env`).
- Output goes to `DATA_DIR/contrast_enhancement/`:
  - `clahe/` and `high_pass/`: the processed tile only, full size (4096 x 4096 PNG).
  - `comparison/`: input | CLAHE | high-pass side by side, with the parameters in the titles.
  - `tiles_used.txt` (which tiles and parameters were used) and a `stats_cache/` folder.
  - Every PNG is named `<frame>_r<row>_c<col>.png`.

Good to know:

- The same `--seed` gives the same random tiles.
- The first run on a frame is slow (about 30 s for a 1.4 GB frame) while the display range is
  computed. Later runs are quick.
- Box keeps some frames online-only. If reading a frame times out, download it first
  (right-click in Finder, Make Available Offline), or pass a downloaded frame with `--frames`.

### Parameters used

| Parameter | Value | Meaning |
| --- | --- | --- |
| `--clip-limit` | 0.01 | CLAHE contrast limit. Higher is stronger. |
| `--sigma` | 50 | High-pass blur size in pixels. |
| `--seed` | 0 | Random sample of tiles. |

