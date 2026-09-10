# Hand-labeling glacier images with Label Studio

This directory runs a local [Label Studio](https://labelstud.io) instance for drawing
segmentation masks of crevasses, plus a confidence rating for each image. The masks you
produce here are the training data for the image-recognition model.

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

**Select the `crevasse` label first** — click the red chip above the image or press `1`.
Until a label is selected the brush silently does nothing, which looks exactly like a broken
tool.

Then, in this order:

1. **Raise the contrast** with the slider above the image. Do this before anything else. Faint
   crevasses are close to invisible at native contrast, and no amount of careful drawing fixes
   a feature you can't see. Adjust brightness too if it helps.
2. **Zoom to 200–400%.** At 100% a crevasse can be one or two pixels wide, which is too small
   to trace accurately.
3. **Set a small brush** (`[` and `]`) — roughly 4–8 pixels.
4. **Drag along each crevasse**, following its curve. Release and start a new drag for the next
   one.
5. **Choose a confidence** (`7`, `8`, or `9`). This is required; Submit stays blocked until you
   pick one.
6. **Submit** (`Ctrl`/`⌘` + `Enter`). Label Studio moves to the next image.

The contrast and brightness sliders only change what you see. They never alter your mask or
the image file.

### Hotkeys

| Action | Key |
|---|---|
| Select the `crevasse` label | `1` |
| Confidence: low / medium / high | `7` / `8` / `9` |
| Brush tool | `B` |
| Eraser | `E` |
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

Two behaviors worth knowing:

- **Painting while a region is selected extends that region** instead of making a new one. With
  a single class this makes no difference to the result, so don't fight it. Press `U` first if
  you want each crevasse as a separate region.
- **The magic wand (`W`)** grows a selection from pixels of similar brightness. On
  high-contrast images it can fill a whole crevasse from one click; on noisy ones it bleeds.
  Try it, keep it if it helps.

### What to label

Trace **along** the crevasses. The goal is lines on the features, not filled polygons around a
whole crevasse field.

Do not label the background outside the ice, and do not label any text, borders, or color bars
that were burned into the image when it was generated.

If you cannot tell whether something is a crevasse even at high contrast, leave it out and drop
your confidence rating.

### What the confidence means

Use the same standard as everyone else on the team, so the ratings are comparable:

| Rating | Use it when |
|---|---|
| `high` | The crevasses are unambiguous and you believe your mask covers all of them. |
| `medium` | Mostly clear, but you made judgment calls, or there are faint features you may have missed. |
| `low` | The image quality is poor, or you are genuinely unsure what counts as a crevasse here. |

A `low` rating is useful information, not an admission of failure — it tells us which labels to
weight less or revisit. Label the image as best you can and mark it `low`.

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

One JSON object per labeled image. Each `annotations[].result` array holds your mask and your
confidence as two sibling entries:

```json
{
  "data": { "image": "/data/local-files/?d=images/example.png" },
  "annotations": [{
    "result": [
      { "type": "brushlabels", "from_name": "mask",
        "original_width": 420, "original_height": 420,
        "value": { "format": "rle", "rle": [ ... ], "brushlabels": ["crevasse"] } },
      { "type": "choices", "from_name": "confidence",
        "value": { "choices": ["high"] } }
    ]
  }]
}
```

Masks are run-length encoded. To get a numpy array back:

```python
import json
import numpy as np
from label_studio_sdk.converter.brush import decode_rle

task = json.load(open("labels-....json"))[0]
region = next(r for r in task["annotations"][0]["result"] if r["type"] == "brushlabels")
h, w = region["original_height"], region["original_width"]
mask = np.array(decode_rle(region["value"]["rle"]), dtype=np.uint8).reshape(h, w, 4)[:, :, 3]
# mask > 0 is the binary crevasse mask
```

Several brush strokes on one image arrive as several `brushlabels` entries — OR them together
for a single mask per image.

## 5. Changing the label classes

Edit `label_config.xml` (the `<Label>` list) and run `make labelstudio` again to push the
change.

Adding a class does not invalidate existing annotations, but it does not backfill them either:
images already submitted come back without the new class. Settle the class list before a
labeling push, not during one.

---

## Troubleshooting

**The brush doesn't draw anything.** No label is selected. Press `1`.

**Submit does nothing.** You haven't chosen a confidence rating. Press `7`, `8`, or `9`.

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
