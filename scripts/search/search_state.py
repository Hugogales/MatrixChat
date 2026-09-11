"""Atomic filesystem state and ticket queue for the month-long search.

The controller is the only writer of canonical ``search_state.json``.
Workers claim immutable ticket files with ``os.replace`` and publish immutable
result files. This is intentionally simpler and safer on a shared HPC
filesystem than concurrent SQLite writes.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expand_processed_dirs(config: dict) -> dict:
    """Resolve ``~`` in every ``processed_dirs`` entry to an absolute path."""
    processed_dirs = config.get("processed_dirs")
    if not processed_dirs:
        return config
    config["processed_dirs"] = {
        str(key): str(Path(value).expanduser())
        for key, value in processed_dirs.items()
    }
    return config


def sbatch_search_exports(search_dir: Path) -> str:
    """Build ``sbatch --export=...`` for a search dir, including launch config."""
    exports = f"ALL,SEARCH_DIR={search_dir}"
    launch = search_dir / "launch_config.json"
    if launch.exists():
        payload = read_json(launch, {})
        config_path = payload.get("path")
        if config_path:
            exports += f",SEARCH_CONFIG={config_path}"
    return exports


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


@dataclass(frozen=True)
class SearchPaths:
    root: Path

    @property
    def state(self) -> Path:
        return self.root / "search_state.json"

    @property
    def decisions(self) -> Path:
        return self.root / "decision_log.jsonl"

    @property
    def pending(self) -> Path:
        return self.root / "queue" / "pending"

    @property
    def claimed(self) -> Path:
        return self.root / "queue" / "claimed"

    @property
    def results(self) -> Path:
        return self.root / "queue" / "results"

    @property
    def done(self) -> Path:
        return self.root / "queue" / "done"

    @property
    def candidates(self) -> Path:
        return self.root / "candidates"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def best(self) -> Path:
        return self.root / "best"

    @property
    def lease(self) -> Path:
        return self.root / "controller_lease.json"

    @property
    def controller_lock(self) -> Path:
        return self.root / "controller.lock"

    def initialize(self) -> None:
        for path in (
            self.pending,
            self.claimed,
            self.results,
            self.done,
            self.candidates,
            self.reports / "plots",
            self.best,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def candidate_dir(self, candidate_id: str) -> Path:
        return self.candidates / candidate_id


def ticket_filename(priority: int, ticket_id: str) -> str:
    # Lexicographic sort gives highest numeric priority first.
    return f"{999999 - int(priority):06d}_{ticket_id}.json"


def enqueue_ticket(paths: SearchPaths, ticket: dict) -> Path:
    ticket = dict(ticket)
    ticket.setdefault("ticket_id", uuid.uuid4().hex)
    ticket.setdefault("created_at", utc_now())
    ticket.setdefault("priority", 100)
    destination = paths.pending / ticket_filename(ticket["priority"], ticket["ticket_id"])
    atomic_write_json(destination, ticket)
    return destination


def pending_tickets(paths: SearchPaths) -> list[Path]:
    return sorted(paths.pending.glob("*.json"))


def claim_ticket(paths: SearchPaths, worker_id: str) -> tuple[Path, dict] | None:
    """Atomically claim the highest-priority available ticket."""
    for source in pending_tickets(paths):
        destination = paths.claimed / source.name
        try:
            os.replace(source, destination)
        except FileNotFoundError:
            continue  # another worker won the race
        ticket = read_json(destination, {})
        ticket.update(
            claimed_by=worker_id,
            claimed_at=utc_now(),
            claimed_unix=time.time(),
        )
        atomic_write_json(destination, ticket)
        return destination, ticket
    return None


def publish_result(paths: SearchPaths, ticket: dict, result: dict) -> Path:
    payload = {
        **result,
        "ticket_id": ticket["ticket_id"],
        "candidate_id": ticket.get("candidate_id"),
        "ticket_type": ticket["type"],
        "completed_at": utc_now(),
    }
    destination = paths.results / f"{ticket['ticket_id']}.json"
    atomic_write_json(destination, payload)
    return destination


def consume_results(paths: SearchPaths, processed: Iterable[str]) -> list[tuple[Path, dict]]:
    processed = set(processed)
    rows = []
    for path in sorted(paths.results.glob("*.json")):
        if path.stem in processed:
            continue
        payload = read_json(path)
        if payload:
            rows.append((path, payload))
    return rows


def finish_ticket(paths: SearchPaths, ticket_id: str) -> None:
    for path in paths.claimed.glob(f"*_{ticket_id}.json"):
        os.replace(path, paths.done / path.name)


def reclaim_stale_claims(paths: SearchPaths, max_age_seconds: float) -> list[str]:
    now = time.time()
    reclaimed = []
    for path in paths.claimed.glob("*.json"):
        ticket = read_json(path, {})
        claimed_at = float(ticket.get("claimed_unix", 0.0) or 0.0)
        if claimed_at and now - claimed_at <= max_age_seconds:
            continue
        ticket.pop("claimed_by", None)
        ticket.pop("claimed_at", None)
        ticket.pop("claimed_unix", None)
        destination = paths.pending / path.name
        atomic_write_json(destination, ticket)
        path.unlink(missing_ok=True)
        reclaimed.append(ticket.get("ticket_id", path.stem))
    return reclaimed


def acquire_directory_lock(path: Path) -> bool:
    try:
        path.mkdir(parents=False)
        return True
    except FileExistsError:
        return False


def release_directory_lock(path: Path) -> None:
    try:
        path.rmdir()
    except FileNotFoundError:
        pass

