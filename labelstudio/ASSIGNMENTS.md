# Labeling your assigned tiles

Each tile is labeled by two people, and you only see the tiles assigned to you, in a
random order. Tiles come in batches (small, then medium, then large) so we can test
whether more labeled data improves the model. The coordinator has already made the
assignment. This page covers what **you** need to do.

Run every command in a regular terminal (on Windows, the WSL2/Ubuntu terminal), from the
repository root. You do not need a "Docker terminal"; Docker just has to be running.

## 1. Before you start

- **Docker is running.** On Windows, open Docker Desktop and check that `docker ps`
  works in your WSL terminal.
- **You have the latest code:** `git pull` on `main`.
- **You have a `.env` file.** If not, run `cp .env.example .env`.
- **The Box data folder is synced**, so the `.h5` files are really there. In `.env`,
  `DATA_DIR` should be the full path to it (for example `/mnt/Box/2026-glaciers`).
  Use a full absolute path, not `~`.
- **You have the assignment file**, `tile_assignments.json`. It's in the shared Box
  folder. **Please let me know if you don't see it!**

## 2. Add your settings to `.env`

Open `.env` (`nano .env`) and add these four lines:

```
LABELSTUDIO_IMAGE_SUBDIR=nisar_data/nisar_gcov_amery
LABELSTUDIO_ASSIGNEE=YourName
LABELSTUDIO_ASSIGNMENTS=/mnt/Box/2026-glaciers/tile_assignments.json
LABELSTUDIO_MAX_TIER=small
```

- **`LABELSTUDIO_IMAGE_SUBDIR`** must be exactly the same as everyone else's. Copy it as
  shown.
- **`LABELSTUDIO_ASSIGNEE`** is your name, spelled and capitalized as in the assignment
  (Jeanne, Bailey, Sophie, or Alexandra).
- **`LABELSTUDIO_ASSIGNMENTS`** is the full path to `tile_assignments.json` on **your**
  machine. Your Box path may differ from the example.
- **`LABELSTUDIO_MAX_TIER`** stays `small` until the team tells you to move on
  (see step 5).

Don't change `TILESERVER_TILE_SIZE`. Everyone must use the same value. It must be the tile
size the assignment was made for: **2048** for the Amery July assignment (see the last section).

Save with `Ctrl+O`, `Enter`, then exit with `Ctrl+X`.

## 3. Load your tiles

```
make labelstudio
```

The first run can take a few minutes while it indexes the frames. When it finishes, look
for a line like:

```
Assignment: YourName, tier(s) small: 25 tile(s)
```

"Tiles to label" at the end should be about that number. If you see an error or a
warning instead, see Troubleshooting below.

## 4. Label

Open <http://localhost:8080>, log in, and open the project. Use **Label All Tasks** to go
through your tiles in order. Each tile's heading ends with its batch and position, like
`[small #3]`. For how to draw the masks, see the main labeling guide.

If the order ever looks off, sort the Data Manager by the `order` column.

**Export your labels regularly:**

```
make labelstudio-export
```

Your labels live inside Docker, so an export is your backup.

## 5. Moving to the next batch

Don't do this until the **whole team** has finished the current batch and we've all agreed to move on. Then:

1. In `.env`, change `LABELSTUDIO_MAX_TIER` to `medium`. Later, change it to `large`.
2. Run `make labelstudio` again.

It adds only the new tiles. Everything you've already labeled stays.

## What `make labelstudio` does to your project

- Unlabeled tasks for tiles that aren't yours, or are in a batch you haven't loaded
  yet, are removed from your project.
- Tiles with an annotation, a skip, or a draft are **never** removed.

## Troubleshooting

**`make: *** No rule to make target ...`**
You're not in the repository root. `cd` to the folder with the `Makefile`.

**There's no `Assignment:` line.**
`LABELSTUDIO_ASSIGNEE` isn't being read. Check `.env` for typos and make sure the line
doesn't start with `#`.

**`... the assignment file ... does not exist`**
`LABELSTUDIO_ASSIGNMENTS` points at the wrong place. Check the path with `ls` and use a
full absolute path.

**`'X' is not in ... Names there: ...`**
Your name doesn't match. Use one of the names listed, spelled exactly.

**`LABELSTUDIO_MAX_TIER is 'x'; the tiers ... are: ...`**
Use one of the batch names listed (`small`, `medium`, or `large`).

**`... was made for N px tiles, but the tile server is offering M px tiles`**
Your `TILESERVER_TILE_SIZE` doesn't match. Set it to N (or remove the line if N is the
default 4096).

**`WARNING: N assigned tile(s) are not in the tile server's index`**
Your data folder is missing frames, or your `LABELSTUDIO_IMAGE_SUBDIR` is different from
the others'. Check the setting and make sure Box has finished syncing.

**`WARNING: N tile(s) in your data are not in the assignment file`**
New frames were added after the assignment was made. Tell the coordinator; nobody will
label those tiles until a new assignment is made.

**`Kept N older task(s) that have labels or drafts`**
Normal. Those tiles were labeled before you switched to assignments, so they're left
alone.

**Anything else.**
Copy the full output of `make labelstudio` and send it to the coordinator.

## Amery July assignment: how it was made (coordinator)

The Amery assignment uses the tiles of the train/test split (`scripts/split_tiles.py`), not every
tile the tile server offers: tiles dropped at block edges (`gap`) are left out. Make it once:

```
python3 labelstudio/assign_tiles.py \
  --split-csv "$DATA_DIR/splits/amery_block_size_comparison/tile_split_region_2048px_blocks_163.84km.csv" \
  --tile-size 2048 \
  --image-dir "$DATA_DIR/nisar_data/nisar_gcov_amery" \
  --people Jeanne Bailey Sophie Alexandra --seed 42 --tiers 50 150 405 \
  --out "$DATA_DIR/tile_assignments_2048.json"
```

- `--image-dir` must be the same folder the tile server reads (`DATA_DIR` + `LABELSTUDIO_IMAGE_SUBDIR`).
  The split CSV only has file names; this option adds the folder part (for example `july/NISAR_...h5`) so the
  tiles match the ones Label Studio loads, also when the frames are in subfolders.
- It writes three files next to each other:

| File | Use it for |
| --- | --- |
| `tile_assignments_2048.json` | **The file everyone loads** (`LABELSTUDIO_ASSIGNMENTS`). It has each person's tiles and their `order`. |
| `tile_assignments_2048.csv` | Reference: one row per tile, with its tier and the two reviewers. No per-person order. |
| `tile_assignments_2048_by_person.csv` | Reference: one row per person and tile, in that person's order. |

- The order only comes from the JSON through `make labelstudio`. Do not label from the CSVs.
- `labelstudio_title` (in the JSON and the CSVs) is the task title without the `[small #n]` tag. Paste it in the
  Data Manager filter (Title contains ...) to find a tile.
- The tile server also offers the August frames and the tiles outside the bounding box, so
  `make labelstudio` prints `WARNING: N tile(s) in your data are not in the assignment file`. For Amery this is
  expected: those tiles are not assigned and are not loaded.
