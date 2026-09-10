#!/usr/bin/env bash
# Create (or update) the Label Studio project for glacier crevasse segmentation.
#
# Safe to re-run: it reuses an existing project and storage instead of duplicating them, so
# `make labelstudio` calls it on every start. Run it again after you add images to
# $DATA_DIR/images/ or edit label_config.xml.
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
export LS_IMAGE_SUBDIR="${LABELSTUDIO_IMAGE_SUBDIR:-images}"
export LS_CONFIG_FILE="$HERE/label_config.xml"
# Where DATA_DIR is mounted inside the container, per docker-compose.yaml. Only worth
# changing if you are running Label Studio outside of Docker.
export LS_FILES_ROOT="${LABELSTUDIO_FILES_ROOT:-/label-studio/files}"

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
import urllib.request

URL = os.environ["LS_URL"].rstrip("/")
TOKEN = os.environ["LS_TOKEN"]
IMAGE_SUBDIR = os.environ["LS_IMAGE_SUBDIR"]
TITLE = "Glacier crevasse segmentation"

# Label Studio only serves a local file if some storage's path is a prefix of that file's
# directory, and it requires that path to be a strict subdirectory of the document root.
# The document root is DATA_DIR (mounted read-only at /label-studio/files), so the images
# have to live one level down, in $DATA_DIR/$LABELSTUDIO_IMAGE_SUBDIR.
STORAGE_PATH = f"{os.environ['LS_FILES_ROOT'].rstrip('/')}/{IMAGE_SUBDIR}"
IMAGE_REGEX = r".*\.(png|jpg|jpeg|tif|tiff|PNG|JPG|JPEG|TIF|TIFF)$"


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


def wait_for_server():
    print(f"Waiting for Label Studio at {URL} ...")
    for _ in range(90):
        try:
            with urllib.request.urlopen(f"{URL}/health", timeout=5):
                return
        except Exception:
            time.sleep(2)
    sys.exit(
        "Label Studio did not come up in 3 minutes.\n"
        "Check it with:  docker compose logs labelstudio"
    )


def as_list(response):
    """The API returns either a bare list or a paginated {'results': [...]} object."""
    if isinstance(response, list):
        return response
    return response.get("results", [])


wait_for_server()

with open(os.environ["LS_CONFIG_FILE"]) as handle:
    label_config = handle.read()

project_payload = {
    "title": TITLE,
    "description": "Hand-drawn crevasse masks with a per-image confidence rating.",
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

storages = as_list(api(f"/api/storages/localfiles/?project={pid}"))
storage = next((s for s in storages if s.get("path") == STORAGE_PATH), None)
if storage:
    sid = storage["id"]
    print(f"Reusing local storage {sid} -> {STORAGE_PATH}")
else:
    sid = api(
        "/api/storages/localfiles/",
        {
            "project": pid,
            "title": "images",
            "path": STORAGE_PATH,
            "use_blob_urls": True,
            "recursive_scan": True,
            "regex_filter": IMAGE_REGEX,
        },
    )["id"]
    print(f"Created local storage {sid} -> {STORAGE_PATH}")

print(f"Syncing images from $DATA_DIR/{IMAGE_SUBDIR} ...")
api(f"/api/storages/localfiles/{sid}/sync", {})
time.sleep(2)

count = api(f"/api/projects/{pid}/")["task_number"]
print()
print(f"Images loaded: {count}")
if not count:
    print()
    print(f"No images found. Put your PNGs in $DATA_DIR/{IMAGE_SUBDIR}/ and run this again,")
    print("or press 'Sync Storage' in the project's Cloud Storage settings.")
print(f"Label here: {URL}/projects/{pid}/data")
PYEOF
