#!/usr/bin/env bash
# Export the crevasse annotations to $DATA_DIR/labels/.
#
# The filename includes your username and a timestamp, so exports from different people (and
# from different days) never overwrite each other. Export often: the annotations themselves
# live inside a Docker volume, and `docker compose down -v` would delete them.
#
# Requires: docker (running), python3.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"

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

if [ -z "${DATA_DIR:-}" ]; then
  echo "DATA_DIR is not set. Copy .env.example to .env and set it." >&2
  exit 1
fi

# Resolve DATA_DIR the same way src/utils/settings.py does: expand a leading ~, and treat a
# relative path as relative to the repository root rather than the current directory.
out_dir="${DATA_DIR/#\~/$HOME}"
out_dir="${out_dir%/}"
case "$out_dir" in /*) ;; *) out_dir="$ROOT/${out_dir#./}" ;; esac
export LS_OUT_DIR="$out_dir/labels"
mkdir -p "$LS_OUT_DIR"
export LS_STAMP="$(date +%Y%m%d-%H%M%S)"
export LS_USER="${USER:-unknown}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to run this script but was not found on your PATH." >&2
  exit 1
fi

exec python3 - <<'PYEOF'
import json
import os
import sys
import urllib.error
import urllib.request

URL = os.environ["LS_URL"].rstrip("/")
TOKEN = os.environ["LS_TOKEN"]
TITLE = "Glacier crevasse segmentation"
OUT = os.path.join(
    os.environ["LS_OUT_DIR"], f"labels-{os.environ['LS_USER']}-{os.environ['LS_STAMP']}.json"
)


def api(path):
    req = urllib.request.Request(f"{URL}{path}")
    req.add_header("Authorization", f"Token {TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        sys.exit(f"GET {path} failed with HTTP {exc.code}\n{exc.read().decode(errors='replace')}")
    except urllib.error.URLError as exc:
        sys.exit(
            f"Could not reach Label Studio at {URL}: {exc.reason}\n"
            "Is it running? Start it with:  make labelstudio"
        )


projects = json.loads(api("/api/projects/?page_size=1000"))
projects = projects if isinstance(projects, list) else projects.get("results", [])
project = next((p for p in projects if p["title"] == TITLE), None)
if project is None:
    sys.exit(f'No project titled "{TITLE}". Run:  make labelstudio')

# Only annotated tasks are exported. Add &download_all_tasks=true for every task.
raw = api(f"/api/projects/{project['id']}/export?exportType=JSON")
with open(OUT, "wb") as handle:
    handle.write(raw)

tasks = json.loads(raw)
regions = sum(len(a.get("result", [])) for t in tasks for a in t.get("annotations", []))
print(f"Exported {len(tasks)} labeled image(s), {regions} region(s) total")
print(f"  -> {OUT}")
if not tasks:
    print("\nNothing was exported because no images have been submitted yet.")
PYEOF
