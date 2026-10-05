# Hand-labeling glacier images with Label Studio

This directory runs a local [Label Studio](https://labelstud.io) instance for drawing
segmentation masks of crevasses. The mask itself records your confidence: you paint in one of
three shades of orange (light = low, medium = medium, dark = high), and anything left
unpainted means "not a crevasse". The masks you produce here are the training data for the
image-recognition model.

Everything is preconfigured. You should not have to click through any setup screens.

---

## 1. Put the images where Label Studio can find them

The images are NISAR L2 GCOV radar frames, as HDF5 (`.h5`) files. They go in an `images/`
subdirectory of your `DATA_DIR` (subdirectories below that are fine too):

```
$DATA_DIR/
└── images/          <- .h5 files here
```

`DATA_DIR` is set in your `.env` file (copy `.env.example` to `.env` if you haven't yet). If
your images live in a differently-named subdirectory, set `LABELSTUDIO_IMAGE_SUBDIR` in `.env`
instead of renaming anything.

### How a frame becomes something you can label

A frame is about 30000 × 30000 pixels, far too big for a browser. A small tile server
(`tileserver/`, run as its own container) cuts it into 4096 × 4096 **tiles** (41 km on a
side), and each tile is one labeling task. The tiles are rendered to PNG as your browser asks
for them, so there is no second copy of the data on disk. Tiles that are almost all empty
(outside the radar swath) are skipped.

What you see is the `HHHH` backscatter from `/science/LSAR/GCOV/grids/frequencyA`, in
decibels (`10·log10`), stretched so that the frame's 1st to 99th percentile runs from dark to
bright. The stretch is per frame, not per tile, so the same surface looks the same in every
tile of a frame. Pixels with no data — outside the swath, or flagged in
`inputDataExceptionMask` — are pure black.

## 2. Start it

You need Docker running, `make`, and `python3` on your machine. From the repository root:

```bash
make labelstudio
```

This starts the Label Studio and tile server containers, creates the project, and adds one
task per tile. It is safe to re-run at any time — re-running is how you pick up new `.h5`
files or changes to `label_config.xml`. (Label Studio's own "Sync Storage" button does not
apply here; re-run `make labelstudio` instead.)

The first run after you add a frame is slow: the tile server reads the whole frame once to
find its display range and which tiles have data. Expect roughly 10 seconds for a 1.4 GB frame
on a fast machine, and a few minutes for a 4 GB frame on a laptop. The result is remembered,
so later runs are quick.

Then open **<http://localhost:8080>** and log in:

| | |
|---|---|
| Username | `student@uchicago.edu` |
| Password | `glaciers2026` |

These are defaults for a server that only listens on your own machine; override them with
`LABELSTUDIO_USERNAME` and `LABELSTUDIO_PASSWORD` in `.env` if you like.

Open the **Glacier crevasse segmentation** project and click the first tile, or use **Label
All Tasks** to work through them back to back. The heading above each tile says which frame it
comes from and where in the frame it is.

When you're done, stop both servers with:

```bash
make labelstudio-stop
```

This removes the two containers but keeps your annotations, which live in a Docker volume;
`make labelstudio` brings everything back as you left it.

## 3. Label a tile

**Select a confidence label first** — click one of the orange chips above the image, or press
`1` (low), `2` (medium), or `3` (high). Until a label is selected the brush silently does
nothing, which looks exactly like a broken tool.

| Label | Key | Color |
|---|---|---|
| `low` | `1` | light orange |
| `medium` | `2` | medium orange |
| `high` | `3` | dark orange |

Then, in this order:

1. **Raise the contrast** with the slider above the image. Do this before anything else. Faint
   crevasses are close to invisible at native contrast, and no amount of careful drawing fixes
   a feature you can't see. Adjust brightness too if it helps.
2. **Zoom to 200–400%.** At 100% a crevasse can be one or two pixels wide, which is too small
   to trace accurately.
3. **Set a small brush** (`[` and `]`) — roughly 4–8 pixels.
4. **Paint one color at a time.** Press `1` and drag along every crevasse you'd call `low`,
   following its curve. Release between crevasses, but don't press anything else: each new
   drag joins the same `low` region. Then **press `U`, then `2`**, and paint all the `medium`
   ones; then **`U`, then `3`** for `high`. You never need to Submit to change colors.
5. **Submit** (`Ctrl`/`⌘` + `Enter`). Label Studio moves to the next tile. Submit tiles with
   no crevasses too, with nothing painted: "no crevasses here" is a label.

Painting by color keeps a tile to about three regions, which matters: Submit takes time per
region (roughly half a second per region on a 4096 tile on a fast computer, more on a
laptop). It doesn't matter how many strokes are in a region. If you forget something, go
back to a color by clicking that color's region in the **Regions** panel and painting more
(the label key would relabel the region, so don't press it in that case). Then press `U` when
you're done.

The contrast and brightness sliders only change what you see. They never alter your mask or
the image file.

### Hotkeys

| Action | Key |
|---|---|
| Select label: low / medium / high | `1` / `2` / `3` |
| Brush tool | `B` |
| Eraser (erases from the selected region) | `E` |
| Smaller / bigger brush | `[` / `]` |
| Magic wand | `W` |
| Pan the image | `H`, then drag |
| Zoom in / out | `Ctrl`/`⌘` + `+` / `-`, or `Ctrl`/`⌘` + scroll |
| Fit whole image / actual size | `Shift`+`1` / `Shift`+`2` |
| Undo / redo | `Ctrl`/`⌘` + `Z` / `Ctrl`/`⌘` + `Shift` + `Z` |
| Delete selected region | `Backspace` |
| Deselect region (press before switching colors) | `U` |
| Show/hide all masks | `Ctrl`/`⌘` + `H` |
| Submit | `Ctrl`/`⌘` + `Enter` |

Behaviors worth knowing:

- **Picking a label while a region is selected relabels that region.** If you just painted
  `low` strokes (the region stays selected) and press `3`, they all turn dark orange — it does
  *not* start a new `high` region. Painting while a region is selected extends that region in
  its own color. So **always press `U` (deselect) before switching colors.** If you relabel
  something by accident, `Ctrl`/`⌘` + `Z` undoes it. (This also means you can fix a region's
  confidence on purpose: click it, then press the right number.) There is no setting to turn
  this off; "Select region after creating it" in the labeling settings doesn't affect
  brushes.
- **The eraser (`E`) erases from the selected region.** Click the region you want to trim (or
  pick it in the Regions panel), then erase. Erased pixels go back to "not a crevasse".
  `Backspace` deletes the whole selected region — if you painted by color, that is *every*
  stroke of that color on the tile, so to remove one stroke, erase it or undo it instead.
- **The magic wand (`W`)** grows a selection from pixels of similar brightness. On
  high-contrast images it can fill a whole crevasse from one click; on noisy ones it bleeds.
  Try it, keep it if it helps.

### What to label

Trace **along** the crevasses. The goal is lines on the features, not filled polygons around a
whole crevasse field.

Do not label the black no-data areas, or open water.

If you think something might be a crevasse but can't tell even at high contrast, paint it
`low` rather than leaving it out.

### What the confidence means

Confidence is per crevasse, not per tile: it says how sure you are that *these pixels* are a
crevasse. Use the same standard as everyone else on the team, so the ratings are comparable:

| Label | Use it when |
|---|---|
| `high` | The crevasse is unambiguous. |
| `medium` | Mostly clear, but a judgment call, or a faint feature. |
| `low` | The image quality is poor here, or you are genuinely unsure whether it is a crevasse. |

A `low` label is useful information, not an admission of failure — it tells us which pixels to
weight less or revisit. Paint what you see and mark it `low`.

## 4. Export your labels

```bash
make labelstudio-export
```

This writes a timestamped file to `$DATA_DIR/labels/`, named with your username so team
members never overwrite each other.

**Export often.** Your annotations live inside a Docker volume, not in this repository.
Stopping with `make labelstudio-stop` is safe, but `docker compose down -v` deletes the volume
and every unexported label with it.

Only tiles you have submitted appear in the export.

### What the export looks like

One JSON object per labeled tile. `data` says which frame the tile came from and where its
top-left corner is in that frame. Each `annotations[].result` array holds one `brushlabels`
entry per region, and each region's label is its confidence:

```json
{
  "data": { "image": "http://localhost:8081/tile/NISAR_L2_PR_GCOV_....h5/4096/1/6.png",
            "source_file": "NISAR_L2_PR_GCOV_....h5", "tile_size": 4096,
            "tile_row": 1, "tile_col": 6, "x0": 24576, "y0": 4096 },
  "annotations": [{
    "result": [
      { "type": "brushlabels", "from_name": "mask",
        "original_width": 4096, "original_height": 4096,
        "value": { "format": "rle", "rle": [ ... ], "brushlabels": ["high"] } },
      { "type": "brushlabels", "from_name": "mask",
        "original_width": 4096, "original_height": 4096,
        "value": { "format": "rle", "rle": [ ... ], "brushlabels": ["low"] } }
    ]
  }]
}
```

Masks are run-length encoded. To get one numpy array per tile, with 0 = not a crevasse,
1 = low, 2 = medium, 3 = high:

```python
import json
import numpy as np
from label_studio_sdk.converter.brush import decode_rle

LEVEL = {"low": 1, "medium": 2, "high": 3}

task = json.load(open("labels-....json"))[0]
regions = [r for r in task["annotations"][0]["result"] if r["type"] == "brushlabels"]
h, w = regions[0]["original_height"], regions[0]["original_width"]
mask = np.zeros((h, w), dtype=np.uint8)
for region in regions:
    alpha = np.array(decode_rle(region["value"]["rle"]), dtype=np.uint8).reshape(h, w, 4)[:, :, 3]
    level = LEVEL[region["value"]["brushlabels"][0]]
    mask = np.maximum(mask, np.where(alpha > 0, level, 0).astype(np.uint8))
```

Each region arrives as a separate `brushlabels` entry, and there may be more than one region
per color. Where regions of different confidence overlap, `np.maximum` keeps the higher one.

The mask is in the tile's own pixel coordinates. Its pixel `[i, j]` is pixel
`[y0 + i, x0 + j]` of `HHHH` in `source_file`, so to line it up with the original data:

```python
import h5py

d = task["data"]
with h5py.File(f"{DATA_DIR}/images/{d['source_file']}") as f:
    hhhh = f["science/LSAR/GCOV/grids/frequencyA/HHHH"][d["y0"] : d["y0"] + h, d["x0"] : d["x0"] + w]
# hhhh.shape == mask.shape
```

Tiles on the right and bottom edges of a frame are smaller than `tile_size`;
`original_width` and `original_height` always give the true size.

### Changing the tile size

Set `TILESERVER_TILE_SIZE` in `.env` (a multiple of 512, up to 8192; default 4096) and run
`make labelstudio`. Smaller tiles mean more tasks, but each one loads and submits faster,
which helps on slow laptops. Tiles that nobody has started are replaced by tiles of the new
size. Tiles with labels or drafts are kept as they are, and so are their exports: every task
records its own `tile_size`, `x0`, and `y0`.

## 5. Changing the label classes

Edit `label_config.xml` (the `<Label>` list) and run `make labelstudio` again to push the
change.

Adding a class does not invalidate existing annotations, but it does not backfill them either:
images already submitted come back without the new class. Settle the class list before a
labeling push, not during one.

---

## Troubleshooting

**The brush doesn't draw anything.** No label is selected. Press `1`, `2`, or `3`.

**A whole region changed color.** You picked a label while that region was selected, which
relabels it. Undo with `Ctrl`/`⌘` + `Z`, press `U`, then pick the label.

**Submit takes a long time.** The tile has many regions; Submit costs time per region. Paint
one color at a time (§3) so that a tile has about three regions, or use a smaller
`TILESERVER_TILE_SIZE`.

**The eraser doesn't erase.** No region is selected. Click the region first, then erase.

**`make labelstudio` fails with an error about annotations being incompatible with the
labeling config.** You have annotations from an older version of `label_config.xml` (for
example the earlier single-`crevasse` label with a separate confidence rating). Export them if
you want them (`make labelstudio-export`), then `docker compose down -v` and
`make labelstudio` again.

**The project is empty.** Your `.h5` files aren't where the tile server is looking. They
must be in `$DATA_DIR/images/` (or whatever `LABELSTUDIO_IMAGE_SUBDIR` says). Fix the
location, then re-run `make labelstudio`. `docker compose logs tileserver` lists any file it
skipped, and why (for example, a file without `frequencyA/HHHH`).

**Tiles show as broken images.** The tile server isn't running, or isn't on port 8081. Check
with `docker compose ps` and `docker compose logs tileserver`, and start it with
`make labelstudio`. Opening <http://localhost:8081/health> in your browser should show
`{"status": "ok"}`.

**`make labelstudio` sits at "Indexing HDF5 frames" for a long time.** It is reading each new
frame once (see §2). `docker compose logs -f tileserver` shows which frame it is on.

**`make labelstudio` fails on the `DATA_DIR` mount.** Docker Compose does not expand a leading
`~` in `.env`, even though the Python code in `src/utils/settings.py` does. Use a full absolute
path (`/Users/you/...`) or a `./relative/` one.

**Port 8080 or 8081 is already in use.** Something else is on that port. Stop it, or change
the published port in `docker-compose.yaml` and set `LABELSTUDIO_URL` (8080) or
`TILESERVER_URL` (8081) in `.env` to match. Changing `TILESERVER_URL` after tiles are loaded
leaves the old tasks pointing at the old address; reset as below.

**`python3 is required`.** The two scripts in this directory use `python3` (standard library
only) to talk to the Label Studio API. Install it, or run the equivalent steps through the
Label Studio web UI.

**Nothing works and you want to start over.** `docker compose down -v` resets Label Studio
completely — new database, no projects, **no annotations**. Export first. (It also clears the
tile server's remembered display ranges, which are recomputed on the next run. Your `.h5` files
are never touched.)
