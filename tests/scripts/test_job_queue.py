"""Tests for scripts/ops/job_queue.py (2026-08-25): the manipulable priority
queue of not-yet-submitted job specs. Pure file-based logic, no Slurm/network
access -- each test uses a fresh tmp_path queue file."""

import pytest

from scripts.ops import job_queue


def _add(entries, name="run_a", partition="dgx", priority=100, gpu_count=1, env=None):
    return job_queue.add_job(
        entries, name=name, sbatch_script="scripts/training/x.sbatch",
        env=env or {}, gpu_count=gpu_count, partition=partition,
        priority=priority, notes="",
    )


def test_add_job_sets_defaults_and_queued_status():
    entries = []
    entry = _add(entries)
    assert entry["status"] == job_queue.STATUS_QUEUED
    assert entry["submitted_job_id"] is None
    assert entry["id"].startswith("q_")
    assert entries == [entry]


def test_sorted_for_processing_orders_by_priority_then_added_at():
    entries = []
    e1 = _add(entries, name="low_prio", priority=100)
    e2 = _add(entries, name="high_prio", priority=1)
    e3 = _add(entries, name="mid_prio", priority=50)
    ordered = job_queue.sorted_for_processing(entries)
    assert [e["id"] for e in ordered] == [e2["id"], e3["id"], e1["id"]]


def test_sorted_for_processing_excludes_non_queued():
    entries = []
    queued = _add(entries, name="still_queued")
    held = _add(entries, name="held_one")
    held["status"] = job_queue.STATUS_HELD
    submitted = _add(entries, name="already_submitted")
    submitted["status"] = job_queue.STATUS_SUBMITTED
    ordered = job_queue.sorted_for_processing(entries)
    assert [e["id"] for e in ordered] == [queued["id"]]


def test_find_entry_returns_none_when_missing():
    assert job_queue.find_entry([], "q_missing") is None


def test_save_and_load_queue_roundtrip(tmp_path):
    path = tmp_path / "queue.json"
    entries = []
    _add(entries, name="round_trip")
    job_queue.save_queue(entries, path)
    loaded = job_queue.load_queue(path)
    assert len(loaded) == 1
    assert loaded[0]["name"] == "round_trip"


def test_cli_add_then_list(tmp_path, capsys):
    path = tmp_path / "queue.json"
    args = _ns(
        command="add", queue_path=path, name="cli_added", sbatch="x.sbatch",
        env=["RUN_NAME=cli_added", "TURN_REWARD_MODE=balanced_ce"],
        gpu_count=2, partition="dgxh100", priority=10, notes="test note",
    )
    job_queue.cmd_add(args)
    entries = job_queue.load_queue(path)
    assert len(entries) == 1
    assert entries[0]["env"] == {"RUN_NAME": "cli_added", "TURN_REWARD_MODE": "balanced_ce"}
    assert entries[0]["gpu_count"] == 2
    assert entries[0]["partition"] == "dgxh100"

    list_args = _ns(queue_path=path, status=None)
    job_queue.cmd_list(list_args)
    out = capsys.readouterr().out
    assert "cli_added" in out
    assert "dgxh100" in out


def test_cli_bump_moves_entry_to_front(tmp_path):
    path = tmp_path / "queue.json"
    entries = []
    e1 = _add(entries, name="a", priority=100)
    e2 = _add(entries, name="b", priority=50)
    job_queue.save_queue(entries, path)

    job_queue.cmd_bump(_ns(queue_path=path, id=e1["id"]))
    reloaded = job_queue.load_queue(path)
    bumped = job_queue.find_entry(reloaded, e1["id"])
    other = job_queue.find_entry(reloaded, e2["id"])
    assert bumped["priority"] < other["priority"]
    ordered = job_queue.sorted_for_processing(reloaded)
    assert ordered[0]["id"] == e1["id"]


def test_cli_set_priority(tmp_path):
    path = tmp_path / "queue.json"
    entries = []
    e1 = _add(entries, name="a", priority=100)
    job_queue.save_queue(entries, path)
    job_queue.cmd_set_priority(_ns(queue_path=path, id=e1["id"], priority=3))
    reloaded = job_queue.find_entry(job_queue.load_queue(path), e1["id"])
    assert reloaded["priority"] == 3


def test_cli_pause_and_resume(tmp_path):
    path = tmp_path / "queue.json"
    entries = []
    e1 = _add(entries, name="a")
    job_queue.save_queue(entries, path)

    job_queue.cmd_pause(_ns(queue_path=path, id=e1["id"]))
    held = job_queue.find_entry(job_queue.load_queue(path), e1["id"])
    assert held["status"] == job_queue.STATUS_HELD
    assert job_queue.sorted_for_processing(job_queue.load_queue(path)) == []

    job_queue.cmd_resume(_ns(queue_path=path, id=e1["id"]))
    resumed = job_queue.find_entry(job_queue.load_queue(path), e1["id"])
    assert resumed["status"] == job_queue.STATUS_QUEUED


def test_cli_pause_rejects_non_queued_entry(tmp_path):
    path = tmp_path / "queue.json"
    entries = []
    e1 = _add(entries, name="a")
    e1["status"] = job_queue.STATUS_SUBMITTED
    job_queue.save_queue(entries, path)
    with pytest.raises(SystemExit):
        job_queue.cmd_pause(_ns(queue_path=path, id=e1["id"]))


def test_cli_remove_deletes_queued_entry(tmp_path):
    path = tmp_path / "queue.json"
    entries = []
    e1 = _add(entries, name="a")
    job_queue.save_queue(entries, path)
    job_queue.cmd_remove(_ns(queue_path=path, id=e1["id"]))
    assert job_queue.load_queue(path) == []


def test_cli_remove_rejects_submitted_entry(tmp_path):
    path = tmp_path / "queue.json"
    entries = []
    e1 = _add(entries, name="a")
    e1["status"] = job_queue.STATUS_SUBMITTED
    job_queue.save_queue(entries, path)
    with pytest.raises(SystemExit):
        job_queue.cmd_remove(_ns(queue_path=path, id=e1["id"]))
    assert len(job_queue.load_queue(path)) == 1


def test_cli_cancel_queued_entry_does_not_call_scancel(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a))
    path = tmp_path / "queue.json"
    entries = []
    e1 = _add(entries, name="a")
    job_queue.save_queue(entries, path)
    job_queue.cmd_cancel(_ns(queue_path=path, id=e1["id"]))
    assert calls == []
    cancelled = job_queue.find_entry(job_queue.load_queue(path), e1["id"])
    assert cancelled["status"] == job_queue.STATUS_CANCELLED


def test_cli_cancel_submitted_entry_calls_scancel(tmp_path, monkeypatch):
    calls = []

    class _Result:
        returncode = 0

    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a) or _Result())
    path = tmp_path / "queue.json"
    entries = []
    e1 = _add(entries, name="a")
    e1["status"] = job_queue.STATUS_SUBMITTED
    e1["submitted_job_id"] = "999888"
    job_queue.save_queue(entries, path)
    job_queue.cmd_cancel(_ns(queue_path=path, id=e1["id"]))
    assert calls and calls[0][0] == ["scancel", "999888"]
    cancelled = job_queue.find_entry(job_queue.load_queue(path), e1["id"])
    assert cancelled["status"] == job_queue.STATUS_CANCELLED


def test_parse_env_pairs_rejects_malformed_entry():
    with pytest.raises(SystemExit):
        job_queue._parse_env_pairs(["NOTKEYVALUE"])


class _ns:
    """Minimal argparse.Namespace-like stand-in for calling cmd_* functions
    directly with just the fields they need."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
