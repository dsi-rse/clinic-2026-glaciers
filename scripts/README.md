# Scripts

## `augment_images.py`: do augmentations make crevasses easier to see?

Applies two image augmentations to random NISAR tiles and saves a side-by-side figure
(input | output) for each tile and method.

| Method | Function | What it does |
| --- | --- | --- |
| CLAHE | `utils.augmentation.clahe` (`skimage.exposure.equalize_adapthist`) | Boosts local contrast. |
| High-pass | `utils.augmentation.high_pass` (original minus `scipy.ndimage.gaussian_filter` blur) | Removes slow, large-scale brightness changes and keeps small features. |

Tiles are cut and scaled with the Label Studio tile server code
(`labelstudio/tileserver/tileserver.py`, imported and not modified), so they look like the tiles
you label: 4096 x 4096 pixels, HHHH backscatter in dB.

### Run it

```bash
make augment-images                                   # defaults: 1 random frame, 3 random tiles
uv run python scripts/augment_images.py --help        # all options
uv run python scripts/augment_images.py --frame path/to/frame.h5 --n-tiles 5 --sigma 30
```

- Frames are read from `DATA_DIR/$LABELSTUDIO_IMAGE_SUBDIR` (see `.env`).
- Figures, `tiles_used.txt` (which tiles and parameters were used) and a `stats_cache/` folder
  are written to `DATA_DIR/augmentation/`.
- The same `--seed` gives the same tiles.
- The first run on a frame is slow (about 30 s for a 1.4 GB frame) while the display range is
  computed. Later runs are quick.
- Box keeps some frames online-only. If reading a frame times out, download it first
  (right-click in Finder, Make Available Offline), or pass a downloaded frame with `--frame`.

### Parameters used

| Parameter | Value | Meaning |
| --- | --- | --- |
| `--clip-limit` | 0.01 | CLAHE contrast limit. Higher is stronger. |
| `--sigma` | 50 | High-pass blur size in pixels. |
| `--seed` | 0 | Random sample of tiles. |

### Tiles used

First run (frame `NISAR_L2_PR_GCOV_024_141_D_125_..._001.h5`, tile size 4096), rows and columns
as in `tiles_used.txt`: (3, 6), (3, 4), (5, 4).

### Results

TODO (student): look at the figures and summarize which method made crevasses more visible,
where each helped or hurt, and anything else important. Add the figures you want to show.
