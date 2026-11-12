#!/usr/bin/env bash
# Create (or update) the Label Studio project for glacier crevasse segmentation.
#
# Every tile of every HDF5 frame in $DATA_DIR (or $DATA_DIR/$LABELSTUDIO_IMAGE_SUBDIR, if set,
# and any subdirectories below it) becomes one task. The tiles themselves
# are rendered on request by the tileserver container (tileserver/tileserver.py); a task only
# holds the tile's URL.
#
# Optional: if LABELSTUDIO_ASSIGNEE is set, only the tiles assigned to that person in the
# assignment file (made once for the team with assign_tiles.py) become tasks. Each task then
# also carries its `tier` (small/medium/large), `order`, and `reviewer`, and tasks are added
# in `order`, so Label Studio's default sequential sampling serves them in that order.
# LABELSTUDIO_MAX_TIER limits loading to that tier and the ones before it.
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
# Optional per-person assignment (see assign_tiles.py). Unset = load every tile, as before.
export LS_ASSIGNEE="${LABELSTUDIO_ASSIGNEE:-}"
export LS_ASSIGNMENTS="${LABELSTUDIO_ASSIGNMENTS:-$HERE/tile_assignments.json}"
export LS_MAX_TIER="${LABELSTUDIO_MAX_TIER:-}"

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
ASSIGNEE = os.environ["LS_ASSIGNEE"].strip()
ASSIGNMENTS = os.environ["LS_ASSIGNMENTS"]
MAX_TIER = os.environ["LS_MAX_TIER"].strip()
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


def tile_key(file, size, row, col):
    return f"{file}|{size}|{row}|{col}"


def load_assignment(assignee, path, max_tier, tiles):
    """This person's tiles from the team assignment file: {tile_key: {tier, order}}."""
    path = os.path.expanduser(os.path.expandvars(path))
    if not os.path.isfile(path):
        sys.exit(
            f"LABELSTUDIO_ASSIGNEE is {assignee!r}, but the assignment file {path} does not exist.\n"
            "Create it once for the team with assign_tiles.py and share it, then point\n"
            "LABELSTUDIO_ASSIGNMENTS at it in .env. Leave LABELSTUDIO_ASSIGNEE unset to load every tile."
        )
    with open(path) as handle:
        spec = json.load(handle)
    mine = spec["assignments"].get(assignee)
    if mine is None:
        sys.exit(f"{assignee!r} is not in {path}. Names there: {', '.join(spec['assignments'])}")

    names = [t["name"] for t in spec["tiers"]]
    if max_tier:
        if max_tier not in names:
            sys.exit(f"LABELSTUDIO_MAX_TIER is {max_tier!r}; the tiers in {path} are: {', '.join(names)}")
        allowed = set(names[: names.index(max_tier) + 1])
    else:
        allowed = set(names)

    sizes = {t["tile_size"] for t in tiles}
    if sizes != {spec["tile_size"]}:
        sys.exit(
            f"{path} was made for {spec['tile_size']} px tiles, but the tile server is offering "
            f"{', '.join(str(s) for s in sorted(sizes))} px tiles. Set TILESERVER_TILE_SIZE to "
            f"{spec['tile_size']}, or make a new assignment file."
        )

    available = {tile_key(t["file"], t["tile_size"], t["row"], t["col"]) for t in tiles}
    everyone = {
        tile_key(a["file"], a["tile_size"], a["row"], a["col"])
        for person_tiles in spec["assignments"].values()
        for a in person_tiles
    }
    chosen, missing = {}, 0
    for a in mine:
        if a["tier"] not in allowed:
            continue
        key = tile_key(a["file"], a["tile_size"], a["row"], a["col"])
        if key in available:
            chosen[key] = a
        else:
            missing += 1
    if missing:
        print(f"WARNING: {missing} assigned tile(s) are not in the tile server's index (frame missing "
              f"from your data folder?). They were skipped.")
    unassigned = len(available - everyone)
    if unassigned:
        print(f"WARNING: {unassigned} tile(s) in your data are not in the assignment file "
              f"(new frames since it was made?). Nobody will label them until it is re-made.")
    print(f"Assignment: {assignee}, tier(s) {', '.join(n for n in names if n in allowed)}: "
          f"{len(chosen)} tile(s)")
    return chosen


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

assigned = load_assignment(ASSIGNEE, ASSIGNMENTS, MAX_TIER, tiles) if ASSIGNEE else None

wanted = {}
for t in tiles:
    extra = {}
    if assigned is not None:
        a = assigned.get(tile_key(t["file"], t["tile_size"], t["row"], t["col"]))
        if a is None:
            continue
        extra = {"tier": a["tier"], "order": a["order"], "reviewer": ASSIGNEE}
    image = f"{TILES}/tile/{urllib.parse.quote(t['file'])}/{t['tile_size']}/{t['row']}/{t['col']}.png"
    granule = t["file"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
    title = f"{granule}  ({t['tile_size']} px tile, row {t['row']}, col {t['col']})"
    if extra:
        title += f"  [{extra['tier']} #{extra['order']}]"
    wanted[image] = {
        "image": image,
        "title": title,
        "source_file": t["file"],
        "tile_size": t["tile_size"],
        "tile_row": t["row"],
        "tile_col": t["col"],
        "x0": t["x0"],
        "y0": t["y0"],
        **extra,
    }

# Tasks for tiles the tile server no longer offers (or, with an assignment, tiles that are not
# yours or not in a tier you are loading): from a different TILESERVER_TILE_SIZE, a frame that
# was removed, or an earlier unassigned load. Untouched ones are deleted; anything someone has
# worked on is kept, and still displays correctly, because its URL includes its own tile size.
existing = existing_tasks(pid)
stale = [t for t in existing if t["data"].get("image") not in wanted]
stale_removed = [t for t in stale if untouched(t)]


def outdated(task):
    """Untouched task for a tile you should label, but with old or missing tier/order info."""
    if assigned is None or not untouched(task):
        return False
    new = wanted.get(task["data"].get("image"))
    return new is not None and any(task["data"].get(k) != new.get(k) for k in ("tier", "order", "reviewer"))


refresh = [t for t in existing if outdated(t)]
for t in stale_removed + refresh:
    api(f"/api/tasks/{t['id']}/", method="DELETE")
if stale_removed:
    print(f"Removed {len(stale_removed)} unlabeled task(s) for tiles that are no longer offered or assigned")
if refresh:
    print(f"Refreshed {len(refresh)} unlabeled task(s) with their tier and order")
if len(stale) > len(stale_removed):
    print(f"Kept {len(stale) - len(stale_removed)} older task(s) that have labels or drafts")

have = {t["data"].get("image") for t in existing} - {t["data"].get("image") for t in refresh}
new_tasks = [{"data": data} for image, data in wanted.items() if image not in have]
if assigned is not None:
    # Tasks get ascending IDs in the order they are added, which is the order they are served.
    new_tasks.sort(key=lambda task: task["data"]["order"])
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
