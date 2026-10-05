# Hand-labeling glacier images with Label Studio

This directory runs a local [Label Studio](https://labelstud.io) instance for drawing
segmentation masks of crevasses. The mask itself records your confidence: you paint in one of
three shades of orange (light = low, medium = medium, dark = high), and anything left
unpainted means "not a crevasse". The masks you produce here are the training data for the
image-recognition model.

Everything is preconfigured. You should not have to click through any setup screens.

---

## 1. Put the images where Label Studio can find them

Images go in an `images/` subdirectory of your `DATA_DIR`:

```
$DATA_DIR/
└── images/          <- PNGs here
```

`DATA_DIR` is set in your `.env` file (copy `.env.example` to `.env` if you haven't yet). If
your images live in a differently-named subdirectory, set `LABELSTUDIO_IMAGE_SUBDIR` in `.env`
instead of renaming anything.

The subdirectory is not optional — Label Studio refuses to serve files directly out of the
directory it treats as its root, for security reasons.

## 2. Start it

You need Docker running, `make`, and `python3` on your machine. From the repository root:

```bash
make labelstudio
```

This starts the Label Studio container, creates the project, points it at your images, and
loads them. It is safe to re-run at any time — re-running is how you pick up new images or
changes to `label_config.xml`.

Then open **<http://localhost:8080>** and log in:

| | |
|---|---|
| Username | `student@uchicago.edu` |
| Password | `glaciers2026` |

These are defaults for a server that only listens on your own machine; override them with
`LABELSTUDIO_USERNAME` and `LABELSTUDIO_PASSWORD` in `.env` if you like.

Open the **Glacier crevasse segmentation** project and click the first image, or use **Label
All Tasks** to work through them back to back.

To stop the server: `docker compose stop labelstudio`.

## 3. Label an image

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
4. **Drag along each crevasse**, following its curve, in the color that matches your
   confidence for *that* crevasse. Release and start a new drag for the next one. One image
   can (and usually will) mix all three colors.
5. **To switch confidence, press `U` first**, then `1`/`2`/`3`. See the warning below.
6. **Submit** (`Ctrl`/`⌘` + `Enter`). Label Studio moves to the next image.

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
| Deselect region | `U` |
| Show/hide all masks | `Ctrl`/`⌘` + `H` |
| Submit | `Ctrl`/`⌘` + `Enter` |

Behaviors worth knowing:

- **Picking a label while a region is selected relabels that region.** If you just painted a
  `low` stroke (it stays selected) and press `3`, the whole stroke turns dark orange — it does
  *not* start a new `high` stroke. Likewise, painting while a region is selected extends that
  region in its own color. So **always press `U` (deselect) before switching confidence.** If
  you relabel something by accident, `Ctrl`/`⌘` + `Z` undoes it. (This also means you can fix
  a stroke's confidence on purpose: click it, then press the right number.)
- **The eraser (`E`) erases from the selected region.** Click the stroke you want to trim (or
  pick it in the Regions panel), then erase. Erased pixels go back to "not a crevasse".
  To remove a whole stroke, select it and press `Backspace`.
- **The magic wand (`W`)** grows a selection from pixels of similar brightness. On
  high-contrast images it can fill a whole crevasse from one click; on noisy ones it bleeds.
  Try it, keep it if it helps.

### What to label

Trace **along** the crevasses. The goal is lines on the features, not filled polygons around a
whole crevasse field.

Do not label the background outside the ice, and do not label any text, borders, or color bars
that were burned into the image when it was generated.

If you think something might be a crevasse but can't tell even at high contrast, paint it
`low` rather than leaving it out.

### What the confidence means

Confidence is per stroke, not per image: it says how sure you are that *these pixels* are a
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
Stopping the container is safe, but `docker compose down -v` deletes the volume and every
unexported label with it.

Only images you have submitted appear in the export.

### What the export looks like

One JSON object per labeled image. Each `annotations[].result` array holds one `brushlabels`
entry per region, and each region's label is its confidence:

```json
{
  "data": { "image": "/data/local-files/?d=images/example.png" },
  "annotations": [{
    "result": [
      { "type": "brushlabels", "from_name": "mask",
        "original_width": 420, "original_height": 420,
        "value": { "format": "rle", "rle": [ ... ], "brushlabels": ["high"] } },
      { "type": "brushlabels", "from_name": "mask",
        "original_width": 420, "original_height": 420,
        "value": { "format": "rle", "rle": [ ... ], "brushlabels": ["low"] } }
    ]
  }]
}
```

Masks are run-length encoded. To get one numpy array per image, with 0 = not a crevasse,
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

Several brush strokes on one image arrive as several `brushlabels` entries. Where strokes of
different confidence overlap, `np.maximum` keeps the higher one.

## 5. Changing the label classes

Edit `label_config.xml` (the `<Label>` list) and run `make labelstudio` again to push the
change.

Adding a class does not invalidate existing annotations, but it does not backfill them either:
images already submitted come back without the new class. Settle the class list before a
labeling push, not during one.

---

## Troubleshooting

**The brush doesn't draw anything.** No label is selected. Press `1`, `2`, or `3`.

**A whole stroke changed color.** You picked a label while that stroke was selected, which
relabels it. Undo with `Ctrl`/`⌘` + `Z`, press `U`, then pick the label.

**The eraser doesn't erase.** No region is selected. Click the stroke first, then erase.

**`make labelstudio` fails with an error about annotations being incompatible with the
labeling config.** You have annotations from an older version of `label_config.xml` (for
example the earlier single-`crevasse` label with a separate confidence rating). Export them if
you want them (`make labelstudio-export`), then `docker compose down -v` and
`make labelstudio` again.

**Images are broken or the project is empty.** Your images aren't where Label Studio is
looking. They must be in `$DATA_DIR/images/` (or whatever `LABELSTUDIO_IMAGE_SUBDIR` says).
Fix the location, then re-run `make labelstudio`.

**`make labelstudio` fails on the `DATA_DIR` mount.** Docker Compose does not expand a leading
`~` in `.env`, even though the Python code in `src/utils/settings.py` does. Use a full absolute
path (`/Users/you/...`) or a `./relative/` one.

**Port 8080 is already in use.** Something else is on that port. Stop it, or set
`LABELSTUDIO_URL` and change the published port in the `labelstudio` service in
`docker-compose.yaml`.

**`python3 is required`.** The two scripts in this directory use `python3` (standard library
only) to talk to the Label Studio API. Install it, or run the equivalent steps through the
Label Studio web UI.

**Nothing works and you want to start over.** `docker compose down -v` resets Label Studio
completely — new database, no projects, **no annotations**. Export first.
