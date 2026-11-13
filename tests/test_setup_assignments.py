"""Tests for how labelstudio/setup.sh loads tasks when an assignment is configured.

setup.sh is run for real (through bash) against small in-process fakes of the Label
Studio API and the tile server, so no Docker or network is needed. The assignment file
comes from the real assign_tiles.py.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

PEOPLE = ["Alice", "Bob", "Cara", "Dev"]
TILE_SIZE = 4096
TILES_PER_FRAME = {
    "nisar_data/A.h5": 6,
    "nisar_data/B.h5": 8,
    "nisar_data/C.h5": 10,
    "nisar_data/D.h5": 16,
}
TILES_PER_ROW = 5
N_TILES = sum(TILES_PER_FRAME.values())
TIER_ARGS = ["8", "20", str(N_TILES)]
TIER_NAMES = ["small", "medium", "large"]
PROJECT_ID = 1
SCRIPT_TIMEOUT_SECONDS = 120
WRONG_TILE_SIZE = 2048
PROTECTED_KINDS = 3

TileId = tuple[str, int, int]


# ---------------------------------------------------------------- fakes


class FakeLabelStudio:
    """In-memory stand-in for the parts of the Label Studio API that setup.sh uses."""

    def __init__(self) -> None:
        """Start with no project and no tasks."""
        self.project: dict[str, Any] | None = None
        self.tasks: list[dict[str, Any]] = []
        self.next_id = 1

    def add_task(self, data: dict[str, Any]) -> None:
        """Add an untouched task, as Label Studio's import does."""
        self.tasks.append(
            {
                "id": self.next_id,
                "data": data,
                "total_annotations": 0,
                "cancelled_annotations": 0,
                "draft_exists": False,
            }
        )
        self.next_id += 1

    def by_id(self, task_id: int) -> dict[str, Any] | None:
        """Return the task with this ID, or None if it is gone."""
        return next((t for t in self.tasks if t["id"] == task_id), None)


def ls_handler(state: FakeLabelStudio) -> type[BaseHTTPRequestHandler]:
    """Build a request handler that serves the fake Label Studio API."""

    class Handler(BaseHTTPRequestHandler):
        """Serve just the endpoints setup.sh calls."""

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002, ARG002
            """Keep test output quiet."""

        def reply(self, status: HTTPStatus, payload: object = None) -> None:
            """Send a JSON reply (or an empty one when payload is None)."""
            body = b"" if payload is None else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def read_json(self) -> Any:  # noqa: ANN401
            """Read the JSON request body."""
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length)) if length else None

        def do_GET(self) -> None:  # noqa: N802
            """Serve health checks, project lookups, and the task list."""
            if self.path == "/health":
                self.reply(HTTPStatus.OK, {})
            elif self.path.startswith("/api/projects/?"):
                results = [state.project] if state.project else []
                self.reply(HTTPStatus.OK, {"results": results})
            elif self.path == f"/api/projects/{PROJECT_ID}/":
                self.reply(HTTPStatus.OK, {"task_number": len(state.tasks)})
            elif self.path.startswith("/api/tasks/"):
                self.reply(
                    HTTPStatus.OK, {"tasks": state.tasks, "total": len(state.tasks)}
                )
            else:
                self.reply(HTTPStatus.NOT_FOUND, {})

        def do_POST(self) -> None:  # noqa: N802
            """Create the project or import tasks."""
            body = self.read_json()
            if self.path == "/api/projects/":
                state.project = {"id": PROJECT_ID, "title": body["title"]}
                self.reply(HTTPStatus.OK, {"id": PROJECT_ID})
            elif self.path == f"/api/projects/{PROJECT_ID}/import":
                for task in body:
                    state.add_task(task["data"])
                self.reply(HTTPStatus.OK, {})
            else:
                self.reply(HTTPStatus.NOT_FOUND, {})

        def do_PATCH(self) -> None:  # noqa: N802
            """Accept project updates."""
            self.read_json()
            self.reply(HTTPStatus.OK, {})

        def do_DELETE(self) -> None:  # noqa: N802
            """Delete a task."""
            match = re.fullmatch(r"/api/tasks/(\d+)/", self.path)
            if not match:
                self.reply(HTTPStatus.NOT_FOUND, {})
                return
            task_id = int(match.group(1))
            state.tasks = [t for t in state.tasks if t["id"] != task_id]
            self.reply(HTTPStatus.NO_CONTENT)

    return Handler


def tile_server_handler(tiles: list[dict[str, Any]]) -> type[BaseHTTPRequestHandler]:
    """Build a request handler that serves the fake tile server (/health and /index)."""

    class Handler(BaseHTTPRequestHandler):
        """Serve the tile index."""

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002, ARG002
            """Keep test output quiet."""

        def do_GET(self) -> None:  # noqa: N802
            """Serve /health and /index."""
            ok = self.path in ("/health", "/index")
            body = json.dumps(
                tiles if self.path == "/index" else {"status": "ok"}
            ).encode()
            self.send_response(HTTPStatus.OK if ok else HTTPStatus.NOT_FOUND)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def make_tiles() -> list[dict[str, Any]]:
    """Build a small fake tile index: four frames of different sizes."""
    return [
        {
            "file": frame,
            "tile_size": TILE_SIZE,
            "row": i // TILES_PER_ROW,
            "col": i % TILES_PER_ROW,
            "x0": (i % TILES_PER_ROW) * TILE_SIZE,
            "y0": (i // TILES_PER_ROW) * TILE_SIZE,
        }
        for frame, count in TILES_PER_FRAME.items()
        for i in range(count)
    ]


# ---------------------------------------------------------------- the rig


@dataclass
class Rig:
    """Everything a test needs: fake servers, script copies, and the assignment."""

    script: Path
    env: dict[str, str]
    ls: FakeLabelStudio
    tiles: list[dict[str, Any]]
    spec_path: Path

    def run(self, **settings: str) -> subprocess.CompletedProcess[str]:
        """Run setup.sh with LABELSTUDIO_* settings, like `make labelstudio` would."""
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash is not available")
        env = {**self.env, **settings}
        return subprocess.run(  # noqa: S603
            [bash, str(self.script)],
            env=env,
            capture_output=True,
            text=True,
            timeout=SCRIPT_TIMEOUT_SECONDS,
            check=False,
        )

    def spec(self) -> dict[str, Any]:
        """Load the assignment file."""
        return json.loads(self.spec_path.read_text())

    def expected(self, person: str, tiers: list[str]) -> set[TileId]:
        """Tiles the assignment file gives `person` within the given tiers."""
        return {
            (r["file"], r["row"], r["col"])
            for r in self.spec()["assignments"][person]
            if r["tier"] in tiers
        }

    def loaded(self) -> set[TileId]:
        """Tiles currently loaded as tasks in the fake project."""
        return {
            (t["data"]["source_file"], t["data"]["tile_row"], t["data"]["tile_col"])
            for t in self.ls.tasks
        }

    def task_for(self, tile: TileId) -> dict[str, Any]:
        """Return the loaded task for a tile."""
        for task in self.ls.tasks:
            data = task["data"]
            if (data["source_file"], data["tile_row"], data["tile_col"]) == tile:
                return task
        pytest.fail(f"No task loaded for {tile}")


def find_labelstudio_dir() -> Path:
    """Find the labelstudio/ folder by walking up from this file."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "labelstudio" / "setup.sh").is_file():
            return parent / "labelstudio"
    pytest.fail("Could not find labelstudio/setup.sh above the tests folder.")


def start(server: ThreadingHTTPServer) -> str:
    """Serve in a background thread and return the server's URL."""
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    """Start the fakes, copy setup.sh somewhere with no .env, and make an assignment."""
    source = find_labelstudio_dir()
    work = tmp_path / "repo" / "labelstudio"
    work.mkdir(parents=True)
    for name in ("setup.sh", "label_config.xml"):
        shutil.copy(source / name, work / name)

    tiles = make_tiles()
    ls = FakeLabelStudio()
    ls_server = ThreadingHTTPServer(("127.0.0.1", 0), ls_handler(ls))
    tile_server = ThreadingHTTPServer(("127.0.0.1", 0), tile_server_handler(tiles))
    ls_url, tile_url = start(ls_server), start(tile_server)

    index = tmp_path / "index.json"
    index.write_text(json.dumps(tiles))
    spec_path = tmp_path / "tile_assignments.json"
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(source / "assign_tiles.py"),
            *["--people", *PEOPLE, "--seed", "42", "--tiers", *TIER_ARGS],
            *["--index-file", str(index), "--out", str(spec_path)],
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("LABELSTUDIO_", "TILESERVER_", "LS_"))
    }
    env = {
        **clean,
        "LABELSTUDIO_URL": ls_url,
        "TILESERVER_URL": tile_url,
        "LABELSTUDIO_TOKEN": "test-token",
        "LABELSTUDIO_ASSIGNMENTS": str(spec_path),
    }
    try:
        yield Rig(work / "setup.sh", env, ls, tiles, spec_path)
    finally:
        for server in (ls_server, tile_server):
            server.shutdown()
            server.server_close()


def assert_ok(result: subprocess.CompletedProcess[str]) -> None:
    """Fail with the script's output if it did not exit cleanly."""
    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------- default behaviour


def test_without_an_assignee_every_tile_is_loaded(rig: Rig) -> None:
    """With no LABELSTUDIO_ASSIGNEE, setup.sh loads all tiles with no extra fields."""
    assert_ok(rig.run())
    assert len(rig.ls.tasks) == N_TILES
    for task in rig.ls.tasks:
        assert "tier" not in task["data"]
        assert "reviewer" not in task["data"]


# ---------------------------------------------------------------- reruns


def test_rerunning_adds_no_duplicate_tasks(rig: Rig) -> None:
    """A second identical run changes nothing: same task IDs, no repeated tiles."""
    settings = {"LABELSTUDIO_ASSIGNEE": "Alice", "LABELSTUDIO_MAX_TIER": "medium"}
    assert_ok(rig.run(**settings))
    ids_before = [t["id"] for t in rig.ls.tasks]
    assert ids_before

    second = rig.run(**settings)
    assert_ok(second)
    assert [t["id"] for t in rig.ls.tasks] == ids_before
    images = [t["data"]["image"] for t in rig.ls.tasks]
    assert len(images) == len(set(images))
    assert "Added 0 new tile(s)" in second.stdout


def test_rerunning_without_an_assignee_adds_no_duplicates(rig: Rig) -> None:
    """The default (everything) mode is also safe to re-run."""
    assert_ok(rig.run())
    assert_ok(rig.run())
    assert len(rig.ls.tasks) == N_TILES


# ---------------------------------------------------------------- filtering


@pytest.mark.parametrize("person", PEOPLE)
def test_each_person_loads_only_their_own_tiles(rig: Rig, person: str) -> None:
    """Loading the largest tier gives exactly the tiles assigned to that person."""
    assert_ok(rig.run(LABELSTUDIO_ASSIGNEE=person, LABELSTUDIO_MAX_TIER="large"))
    assert rig.loaded() == rig.expected(person, TIER_NAMES)
    assert {t["data"]["reviewer"] for t in rig.ls.tasks} == {person}


def test_tier_limit_loads_only_that_tier_and_earlier(rig: Rig) -> None:
    """MAX_TIER=small loads only small tiles, and medium adds only the new ones."""
    alice = {"LABELSTUDIO_ASSIGNEE": "Alice"}
    assert_ok(rig.run(**alice, LABELSTUDIO_MAX_TIER="small"))
    assert rig.loaded() == rig.expected("Alice", ["small"])
    assert {t["data"]["tier"] for t in rig.ls.tasks} == {"small"}
    small_ids = {t["id"] for t in rig.ls.tasks}

    assert_ok(rig.run(**alice, LABELSTUDIO_MAX_TIER="medium"))
    assert rig.loaded() == rig.expected("Alice", ["small", "medium"])
    assert small_ids <= {t["id"] for t in rig.ls.tasks}
    assert "large" not in {t["data"]["tier"] for t in rig.ls.tasks}


def test_lowering_the_tier_limit_removes_untouched_later_tiles(rig: Rig) -> None:
    """Unlabeled tasks above a lowered MAX_TIER are removed."""
    alice = {"LABELSTUDIO_ASSIGNEE": "Alice"}
    assert_ok(rig.run(**alice, LABELSTUDIO_MAX_TIER="large"))
    assert_ok(rig.run(**alice, LABELSTUDIO_MAX_TIER="small"))
    assert rig.loaded() == rig.expected("Alice", ["small"])


def test_tasks_are_added_in_the_persons_order(rig: Rig) -> None:
    """Task IDs rise with `order`, so tiles are served in the intended sequence."""
    assert_ok(rig.run(LABELSTUDIO_ASSIGNEE="Bob", LABELSTUDIO_MAX_TIER="large"))
    by_id = sorted(rig.ls.tasks, key=lambda t: t["id"])
    orders = [t["data"]["order"] for t in by_id]
    assert orders == sorted(orders)
    tiers = [TIER_NAMES.index(t["data"]["tier"]) for t in by_id]
    assert tiers == sorted(tiers)


def test_unassigned_tiles_are_warned_about_and_not_loaded(rig: Rig) -> None:
    """A tile in the data but not in the assignment file is skipped with a warning."""
    extra = {**rig.tiles[0], "file": "nisar_data/NEW.h5"}
    rig.tiles.append(extra)
    result = rig.run(LABELSTUDIO_ASSIGNEE="Alice", LABELSTUDIO_MAX_TIER="large")
    assert_ok(result)
    assert "not in the assignment file" in result.stdout
    assert "nisar_data/NEW.h5" not in {t[0] for t in rig.loaded()}


# ---------------------------------------------------------------- work in progress


def protect(rig: Rig, tiles: list[TileId]) -> dict[int, dict[str, Any]]:
    """Mark three tasks as annotated, skipped, and drafted; return snapshots of them."""
    annotated, skipped, drafted = (rig.task_for(t) for t in tiles[:PROTECTED_KINDS])
    annotated["total_annotations"] = 1
    skipped["cancelled_annotations"] = 1
    drafted["draft_exists"] = True
    return {
        t["id"]: json.loads(json.dumps(t["data"]))
        for t in (annotated, skipped, drafted)
    }


def test_annotated_skipped_and_drafted_tasks_survive_a_change_of_assignee(
    rig: Rig,
) -> None:
    """Switching to another person never deletes tasks someone has worked on."""
    assert_ok(rig.run(LABELSTUDIO_ASSIGNEE="Alice", LABELSTUDIO_MAX_TIER="large"))
    protected = protect(rig, sorted(rig.expected("Alice", TIER_NAMES)))

    assert_ok(rig.run(LABELSTUDIO_ASSIGNEE="Bob", LABELSTUDIO_MAX_TIER="small"))

    for task_id, data in protected.items():
        task = rig.ls.by_id(task_id)
        assert task is not None, "a task with work on it was deleted"
        assert task["data"] == data
    protected_tiles = {
        (d["source_file"], d["tile_row"], d["tile_col"]) for d in protected.values()
    }
    assert rig.loaded() == rig.expected("Bob", ["small"]) | protected_tiles


def test_annotated_tasks_survive_lowering_the_tier_limit(rig: Rig) -> None:
    """A labeled task above a lowered tier limit is kept."""
    alice = {"LABELSTUDIO_ASSIGNEE": "Alice"}
    assert_ok(rig.run(**alice, LABELSTUDIO_MAX_TIER="large"))
    large_only = sorted(rig.expected("Alice", ["large"]))
    labeled = rig.task_for(large_only[0])
    labeled["total_annotations"] = 1
    assert_ok(rig.run(**alice, LABELSTUDIO_MAX_TIER="small"))
    assert rig.ls.by_id(labeled["id"]) is not None


def test_outdated_tasks_are_refreshed_only_if_untouched(rig: Rig) -> None:
    """Untouched tasks with stale tier info are recreated; labeled ones are kept."""
    settings = {"LABELSTUDIO_ASSIGNEE": "Alice", "LABELSTUDIO_MAX_TIER": "small"}
    assert_ok(rig.run(**settings))
    first, second, *_ = sorted(rig.ls.tasks, key=lambda t: t["id"])
    labeled_id, untouched_id = first["id"], second["id"]
    for task in (first, second):
        task["data"]["tier"] = "stale"
    first["total_annotations"] = 1

    result = rig.run(**settings)
    assert_ok(result)

    labeled = rig.ls.by_id(labeled_id)
    assert labeled is not None
    assert labeled["data"]["tier"] == "stale"
    assert rig.ls.by_id(untouched_id) is None
    assert "Refreshed 1 unlabeled task(s)" in result.stdout
    assert {t["data"]["tier"] for t in rig.ls.tasks if t["id"] != labeled_id} == {
        "small"
    }


# ---------------------------------------------------------------- bad settings


def run_expecting_failure(rig: Rig, expected: str, **settings: str) -> None:
    """Run setup.sh, check it fails with a message, and that no tasks were touched."""
    result = rig.run(**settings)
    assert result.returncode != 0
    assert expected in result.stderr
    assert rig.ls.tasks == []


def test_unknown_assignee_is_rejected(rig: Rig) -> None:
    """A name that isn't in the assignment file stops the run and lists valid names."""
    run_expecting_failure(rig, "Names there: Alice", LABELSTUDIO_ASSIGNEE="Nobody")


def test_unknown_tier_is_rejected(rig: Rig) -> None:
    """A tier limit that isn't in the file stops the run and lists valid tiers."""
    run_expecting_failure(
        rig,
        "small, medium, large",
        LABELSTUDIO_ASSIGNEE="Alice",
        LABELSTUDIO_MAX_TIER="huge",
    )


def test_missing_assignment_file_is_rejected(rig: Rig) -> None:
    """Pointing at a file that does not exist gives a clear error."""
    rig.spec_path.unlink()
    run_expecting_failure(rig, "does not exist", LABELSTUDIO_ASSIGNEE="Alice")


def test_tile_size_mismatch_is_rejected(rig: Rig) -> None:
    """An assignment made for a different tile size is refused."""
    spec = rig.spec()
    spec["tile_size"] = WRONG_TILE_SIZE
    rig.spec_path.write_text(json.dumps(spec))
    run_expecting_failure(
        rig, f"{WRONG_TILE_SIZE} px tiles", LABELSTUDIO_ASSIGNEE="Alice"
    )


def test_repeated_tier_names_in_the_file_are_rejected(rig: Rig) -> None:
    """A hand-edited file with repeated names can't make a limit load later tiers."""
    spec = rig.spec()
    spec["tiers"][1]["name"] = "small"
    rig.spec_path.write_text(json.dumps(spec))
    run_expecting_failure(
        rig,
        "repeated tier names",
        LABELSTUDIO_ASSIGNEE="Alice",
        LABELSTUDIO_MAX_TIER="small",
    )
