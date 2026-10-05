#!/usr/bin/env bash
# Create (or update) the Label Studio project for glacier crevasse segmentation.
#
# Every tile of every HDF5 frame in $DATA_DIR (or $DATA_DIR/$LABELSTUDIO_IMAGE_SUBDIR, if set,
# and any subdirectories below it) becomes one task. The tiles themselves
# are rendered on request by the tileserver container (tileserver/tileserver.py); a task only
# holds the tile's URL.
#
# Safe to re-run: it reuses an existing project and only adds tiles it hasn't added before, so
# `make labelstudio` calls it on every start. Run it again after you add frames to
# $DATA_DIR or edit label_config.xml.
#
# Requires: docker (running), python3 (any version 3.8+; only the standard library is used).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"

# The LABELSTUDIO_* defaults below must match the labelstudio service in
# ../docker-compose.yaml. Override them in .env.
#
# .env supplies defaults, but variables already set in your environment win -- the same
# precedence python-dotenv uses in src/utils/settings.py.
if [ -f "$ROOT/.env" ]; then
  while IFS= read -r line; do
    case "$line" in "" | "#"*) continue ;; esac
    key=${line%%=*}
    [ "$key" = "$line" ] && continue      # line has no '='
    [ -n "${!key+x}" ] && continue        # already set in the environment
    export "$key=${line#*=}"
  done < "$ROOT/.env"
fi
export LS_URL="${LABELSTUDIO_URL:-http://localhost:8080}"
export LS_TOKEN="${LABELSTUDIO_TOKEN:-clinic2026glaciersdevtoken}"
export LS_IMAGE_SUBDIR="${LABELSTUDIO_IMAGE_SUBDIR:-}"
export LS_CONFIG_FILE="$HERE/label_config.xml"
# The tile server as your browser sees it. Tasks store image URLs under this address.
export LS_TILESERVER_URL="${TILESERVER_URL:-http://localhost:8081}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to run this script but was not found on your PATH." >&2
  exit 1
fi

exec python3 - <<'PYEOF'
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

URL = os.environ["LS_URL"].rstrip("/")
TOKEN = os.environ["LS_TOKEN"]
IMAGE_SUBDIR = os.environ["LS_IMAGE_SUBDIR"].strip("/")
IMAGE_PATH = f"$DATA_DIR/{IMAGE_SUBDIR}" if IMAGE_SUBDIR else "$DATA_DIR"
TILES = os.environ["LS_TILESERVER_URL"].rstrip("/")
TITLE = "Glacier crevasse segmentation"


def api(path, payload=None, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    verb = method or ("POST" if data else "GET")
    req = urllib.request.Request(f"{URL}{path}", data=data, method=verb)
    req.add_header("Authorization", f"Token {TOKEN}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        sys.exit(f"{verb} {path} failed with HTTP {exc.code}\n{detail}")
    except urllib.error.URLError as exc:
        sys.exit(f"Could not reach Label Studio at {URL}: {exc.reason}")
    return json.loads(body) if body else None


def wait_for(name, url, service):
    print(f"Waiting for {name} at {url} ...")
    for _ in range(90):
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5):
                return
        except Exception:
            time.sleep(2)
    sys.exit(
        f"{name} did not come up in 3 minutes.\n"
        f"Check it with:  docker compose logs {service}"
    )


def tile_index():
    """Every tile with data. Slow the first time a frame is seen: it is scanned once."""
    try:
        # An hour: a big frame takes a minute or two to scan on a laptop, and there may be many.
        with urllib.request.urlopen(f"{TILES}/index", timeout=3600) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        sys.exit(f"Tile server /index failed with HTTP {exc.code}\n{exc.read().decode(errors='replace')}")
    except urllib.error.URLError as exc:
        sys.exit(f"Could not reach the tile server at {TILES}: {exc.reason}")


def existing_tasks(pid):
    found, page = [], 1
    while True:
        # fields=all includes draft_exists, which (unlike "drafts") counts every user's drafts.
        response = api(f"/api/tasks/?project={pid}&fields=all&page_size=500&page={page}")
        tasks = response if isinstance(response, list) else response.get("tasks", [])
        found.extend(tasks)
        total = None if isinstance(response, list) else response.get("total")
        if not tasks or total is None or len(found) >= total:
            return found
        page += 1


def untouched(task):
    """No annotation, skip, or draft by anyone. Missing fields count as touched, to be safe."""
    return (
        task.get("total_annotations", 1) == 0
        and task.get("cancelled_annotations", 1) == 0
        and task.get("draft_exists", True) is False
    )


def as_list(response):
    """The API returns either a bare list or a paginated {'results': [...]} object."""
    if isinstance(response, list):
        return response
    return response.get("results", [])


wait_for("Label Studio", URL, "labelstudio")
wait_for("the tile server", TILES, "tileserver")

with open(os.environ["LS_CONFIG_FILE"]) as handle:
    label_config = handle.read()

project_payload = {
    "title": TITLE,
    "description": "Hand-drawn crevasse masks, painted in three shades of orange for low, medium, and high confidence.",
    "label_config": label_config,
}

project = next((p for p in as_list(api("/api/projects/?page_size=1000")) if p["title"] == TITLE), None)
if project:
    pid = project["id"]
    print(f"Reusing project {pid}: {TITLE}")
    api(f"/api/projects/{pid}/", project_payload, method="PATCH")
    print("Labeling interface updated from label_config.xml")
else:
    pid = api("/api/projects/", project_payload)["id"]
    print(f"Created project {pid}: {TITLE}")

print(f"Indexing HDF5 frames in {IMAGE_PATH} ...")
print("  (each new frame is scanned once to find its display range; ~10 s to a few minutes)")
tiles = tile_index()
frames = sorted({t["file"] for t in tiles})
print(f"Found {len(tiles)} tiles with data in {len(frames)} frame(s)")

wanted = {}
for t in tiles:
    image = f"{TILES}/tile/{urllib.parse.quote(t['file'])}/{t['tile_size']}/{t['row']}/{t['col']}.png"
    granule = t["file"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
    wanted[image] = {
        "image": image,
        "title": f"{granule}  ({t['tile_size']} px tile, row {t['row']}, col {t['col']})",
        "source_file": t["file"],
        "tile_size": t["tile_size"],
        "tile_row": t["row"],
        "tile_col": t["col"],
        "x0": t["x0"],
        "y0": t["y0"],
    }

# Tasks for tiles the tile server no longer offers: from a different TILESERVER_TILE_SIZE, or
# from a frame that was removed. Untouched ones are deleted; anything someone has worked on is
# kept, and still displays correctly, because its URL includes its own tile size.
existing = existing_tasks(pid)
have = {t["data"].get("image") for t in existing}
stale = [t for t in existing if t["data"].get("image") not in wanted]
removed = [t for t in stale if untouched(t)]
for t in removed:
    api(f"/api/tasks/{t['id']}/", method="DELETE")
if removed:
    print(f"Removed {len(removed)} unlabeled task(s) for tiles that are no longer offered")
if len(stale) > len(removed):
    print(f"Kept {len(stale) - len(removed)} older task(s) that have labels or drafts")

new_tasks = [{"data": data} for image, data in wanted.items() if image not in have]
for start in range(0, len(new_tasks), 500):
    api(f"/api/projects/{pid}/import", new_tasks[start : start + 500])
print(f"Added {len(new_tasks)} new tile(s); {len(wanted) - len(new_tasks)} were already loaded")

count = api(f"/api/projects/{pid}/")["task_number"]
print()
print(f"Tiles to label: {count}")
if not count:
    print()
    print(f"No HDF5 frames found. Put your .h5 files in {IMAGE_PATH} and run this again.")
print(f"Label here: {URL}/projects/{pid}/data")
PYEOF
