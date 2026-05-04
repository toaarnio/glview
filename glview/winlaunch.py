"""
Windows GUI launcher that batches shell activations into one viewer process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import suppress


SPOOL_DIRNAME = "glview-shell-open"
LOCK_FILENAME = "leader.lock"
DEBOUNCE_SECONDS = 0.35


class _SpawnBackend:

    def popen(self, args):
        kwargs = {
            "close_fds": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            kwargs["stdin"] = subprocess.DEVNULL
            kwargs["stdout"] = subprocess.DEVNULL
            kwargs["stderr"] = subprocess.DEVNULL
        return subprocess.Popen(args, **kwargs)  # noqa: S603


class _TimeBackend:

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def main() -> None:
    launch(sys.argv[1:])


def launch(filespecs, spool_root: Path | None = None, spawn_backend=None, time_backend=None) -> bool:
    filespecs = [str(Path(filespec)) for filespec in filespecs if filespec]
    if not filespecs:
        return False

    spool_dir = _spool_dir(spool_root)
    spool_dir.mkdir(parents=True, exist_ok=True)
    event_path = spool_dir / f"{time.time_ns()}-{uuid.uuid4().hex}.json"
    event_path.write_text(json.dumps(filespecs), encoding="utf-8")

    leader_lock = spool_dir / LOCK_FILENAME
    if not _try_become_leader(leader_lock):
        return True

    spawn_backend = _SpawnBackend() if spawn_backend is None else spawn_backend
    time_backend = _TimeBackend() if time_backend is None else time_backend
    try:
        time_backend.sleep(DEBOUNCE_SECONDS)
        pending_files = _collect_pending_files(spool_dir)
        if pending_files:
            spawn_backend.popen(_viewer_command(pending_files))
    finally:
        _release_leader_lock(leader_lock)
    return True


def _spool_dir(spool_root: Path | None) -> Path:
    root = Path(tempfile.gettempdir()) if spool_root is None else Path(spool_root)
    return root / SPOOL_DIRNAME


def _try_become_leader(lock_path: Path) -> bool:
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.close(fd)
    return True


def _release_leader_lock(lock_path: Path) -> None:
    with suppress(FileNotFoundError):
        lock_path.unlink()


def _collect_pending_files(spool_dir: Path):
    filespecs = []
    event_paths = sorted(path for path in spool_dir.glob("*.json") if path.is_file())
    for event_path in event_paths:
        try:
            payload = json.loads(event_path.read_text(encoding="utf-8"))
            filespecs.extend(str(Path(item)) for item in payload)
        finally:
            with suppress(FileNotFoundError):
                event_path.unlink()
    return filespecs


def _viewer_command(filespecs):
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable)
        if executable.stem.lower() == "glview-launcher":
            viewer = executable.with_name(f"glview{executable.suffix}")
            if viewer.exists():
                return [str(viewer), *filespecs]
        return [sys.executable, *filespecs]
    return [sys.executable, "-m", "glview.glview", *filespecs]
