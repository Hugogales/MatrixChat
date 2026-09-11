"""Tests for scripts/ops/gpu_yield_daemon.py's pure logic (2026-08-24).

Slurm-calling functions (squeue_partition, scontrol_show_job, and anything
that shells out) are exercised indirectly by unit-testing the pure
functions they feed (victim selection, env-building, elapsed-time parsing)
on constructed in-memory data -- no real cluster access, matching this
project's existing search/worker test conventions."""

import argparse
import re
import subprocess
import sys

from scripts.ops import job_queue
from scripts.ops.gpu_yield_daemon import (
    DEFAULT_PROTECT_PATTERN,
    Daemon,
    _elapsed_seconds,
    build_resume_env,
    choose_gpu_victims,
    choose_node_to_vacate,
)


def test_main_creates_log_dir_before_locking(tmp_path):
    """Regression (2026-08-25): first real launch crashed with
    FileNotFoundError because acquire_directory_lock's mkdir(parents=False)
    assumed logs/gpu_yield_daemon/ already existed. --once with a fresh
    --project-root must not crash even when nothing has been created yet."""
    result = subprocess.run(
        [sys.executable, "scripts/ops/gpu_yield_daemon.py",
         "--project-root", str(tmp_path),
         "--our-user", "nobody-for-test",
         # A partition name that cannot exist on any real cluster, so this
         # test never depends on real squeue/cluster state.
         "--partitions", "no_such_partition_gpu_yield_test",
         "--once"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def _job(job_id, name, elapsed, nodelist="node1", user="me", state="RUNNING"):
    return {
        "job_id": job_id, "name": name, "elapsed": elapsed,
        "nodelist": nodelist, "user": user, "state": state, "partition": "dgxh100",
    }


def test_elapsed_seconds_parses_hms_and_dhms():
    assert _elapsed_seconds("0:05:00") == 300
    assert _elapsed_seconds("1:02:03") == 3723
    assert _elapsed_seconds("2-01:00:00") == 2 * 86400 + 3600
    assert _elapsed_seconds("05:00") == 300  # squeue can omit hours entirely


def test_elapsed_seconds_handles_garbage_gracefully():
    assert _elapsed_seconds("N/A") == 0.0


def test_choose_gpu_victims_prefers_shortest_elapsed(monkeypatch):
    jobs = [
        _job("1", "race_a", "2:00:00"),
        _job("2", "race_b", "0:05:00"),
        _job("3", "race_c", "1:00:00"),
    ]
    monkeypatch.setattr("scripts.ops.gpu_yield_daemon.requested_gpus", lambda job_id: 1)
    protect_re = re.compile(DEFAULT_PROTECT_PATTERN, re.IGNORECASE)
    chosen = choose_gpu_victims(jobs, need_gpus=1, protect_re=protect_re)
    assert [j["job_id"] for j in chosen] == ["2"]


def test_choose_gpu_victims_skips_protected_names(monkeypatch):
    jobs = [
        _job("1", "race_final_test_leader", "0:01:00"),
        _job("2", "race_exploration_arm", "5:00:00"),
    ]
    monkeypatch.setattr("scripts.ops.gpu_yield_daemon.requested_gpus", lambda job_id: 1)
    protect_re = re.compile(DEFAULT_PROTECT_PATTERN, re.IGNORECASE)
    chosen = choose_gpu_victims(jobs, need_gpus=1, protect_re=protect_re)
    assert [j["job_id"] for j in chosen] == ["2"]


def test_choose_gpu_victims_accumulates_across_multiple_jobs(monkeypatch):
    jobs = [
        _job("1", "race_a", "0:01:00"),
        _job("2", "race_b", "0:02:00"),
    ]
    monkeypatch.setattr("scripts.ops.gpu_yield_daemon.requested_gpus", lambda job_id: 1)
    protect_re = re.compile(DEFAULT_PROTECT_PATTERN, re.IGNORECASE)
    chosen = choose_gpu_victims(jobs, need_gpus=2, protect_re=protect_re)
    assert {j["job_id"] for j in chosen} == {"1", "2"}


def test_choose_node_to_vacate_picks_least_sunk_cost_node():
    jobs = [
        _job("1", "race_a", "5:00:00", nodelist="dh-dgx1-1"),
        _job("2", "race_b", "0:10:00", nodelist="dh-dgx1-2"),
        _job("3", "race_c", "0:20:00", nodelist="dh-dgx1-2"),
    ]
    protect_re = re.compile(DEFAULT_PROTECT_PATTERN, re.IGNORECASE)
    assert choose_node_to_vacate(jobs, protect_re) == "dh-dgx1-2"


def test_choose_node_to_vacate_returns_none_when_all_protected():
    jobs = [_job("1", "final_test_leader", "5:00:00", nodelist="dh-dgx1-1")]
    protect_re = re.compile(DEFAULT_PROTECT_PATTERN, re.IGNORECASE)
    assert choose_node_to_vacate(jobs, protect_re) is None


def _daemon(tmp_path, live=False, budget_dgx=16, budget_h100=4):
    args = argparse.Namespace(
        partitions=["dgx", "dgxh100"],
        our_user="me",
        project_root=str(tmp_path),
        interval_seconds=15.0,
        protect_pattern=DEFAULT_PROTECT_PATTERN,
        resume_step_increment=2000,
        state_file="logs/gpu_yield_daemon/state.json",
        audit_log="logs/gpu_yield_daemon/decisions.jsonl",
        queue_file="logs/job_queue/queue.json",
        budget_dgx_gpus=budget_dgx,
        budget_h100_gpus=budget_h100,
        live=live,
        max_runtime_hours=116.0,
        self_sbatch="scripts/ops/gpu_yield_daemon.sbatch",
        once=True,
    )
    return Daemon(args)


class _FakeSbatchResult:
    def __init__(self, job_id="424242"):
        self.returncode = 0
        self.stdout = f"Submitted batch job {job_id}\n"
        self.stderr = ""


def test_fill_from_queue_submits_highest_priority_fitting_entries(tmp_path, monkeypatch):
    daemon = _daemon(tmp_path, live=True)
    entries = []
    e_low = job_queue.add_job(entries, "low", "x.sbatch", {}, gpu_count=1, partition="dgx", priority=100, notes="")
    e_high = job_queue.add_job(entries, "high", "x.sbatch", {}, gpu_count=1, partition="dgx", priority=1, notes="")
    job_queue.save_queue(entries, daemon.queue_path)

    monkeypatch.setattr("scripts.ops.gpu_yield_daemon.subprocess.run", lambda *a, **k: _FakeSbatchResult())
    # Budget=1, only room for ONE entry -- must pick the higher-priority one.
    daemon.budget = {"dgx": 1, "dgxh100": 4}
    daemon.fill_from_queue({"dgx": [], "dgxh100": []})

    reloaded = job_queue.load_queue(daemon.queue_path)
    high = job_queue.find_entry(reloaded, e_high["id"])
    low = job_queue.find_entry(reloaded, e_low["id"])
    assert high["status"] == job_queue.STATUS_SUBMITTED
    assert high["submitted_job_id"] == "424242"
    assert low["status"] == job_queue.STATUS_QUEUED


def test_submit_queue_entry_passes_h100_gres_and_mem_for_dgxh100_partition(tmp_path, monkeypatch):
    """Regression: queue entries on dgxh100 must request h100 GPUs explicitly;
    relying on evaluate_final_checkpoint.sbatch's baked-in V100 #SBATCH lines
    silently lands on the wrong partition/GPU type."""
    daemon = _daemon(tmp_path, live=True)
    entries = []
    entry = job_queue.add_job(
        entries, "eval_adaptive", "scripts/final_runs/evaluate_final_checkpoint.sbatch",
        {"CONTRACT": "c.json"}, gpu_count=1, partition="dgxh100", priority=1, notes="",
    )
    job_queue.save_queue(entries, daemon.queue_path)

    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeSbatchResult()

    monkeypatch.setattr("scripts.ops.gpu_yield_daemon.subprocess.run", _fake_run)
    daemon.fill_from_queue({"dgx": [], "dgxh100": []})

    assert "--partition=dgxh100" in captured["cmd"]
    assert "--gres=gpu:h100:1" in captured["cmd"]
    assert "--mem=80G" in captured["cmd"]


def test_submit_queue_entry_passes_comma_containing_env_via_subprocess_env(tmp_path, monkeypatch):
    """Regression guard for .cursor/rules/sbatch-env-overrides.mdc: a queued
    entry's env dict legitimately contains comma-bearing values (e.g.
    DATASET_WEIGHTS), which Slurm's `--export=ALL,K=V,...` would silently
    corrupt by splitting on every comma. `submit_queue_entry` must pass such
    overrides via the subprocess's own `env=` kwarg (with a bare
    `--export=ALL`), never inline in the `--export=` string."""
    daemon = _daemon(tmp_path, live=True)
    entries = []
    tricky_env = {"DATASET_WEIGHTS": "ami=0.5,meld=0.15,werewolf=0.35", "RUN_NAME": "abl"}
    entry = job_queue.add_job(entries, "abl", "x.sbatch", tricky_env, gpu_count=1, partition="dgx", priority=1, notes="")
    job_queue.save_queue(entries, daemon.queue_path)

    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return _FakeSbatchResult()

    monkeypatch.setattr("scripts.ops.gpu_yield_daemon.subprocess.run", _fake_run)
    daemon.fill_from_queue({"dgx": [], "dgxh100": []})

    # No comma-joined KEY=VALUE pairs anywhere in the --export= argument.
    export_arg = next(a for a in captured["cmd"] if a.startswith("--export="))
    assert export_arg == "--export=ALL"
    assert captured["env"]["DATASET_WEIGHTS"] == "ami=0.5,meld=0.15,werewolf=0.35"
    assert captured["env"]["RUN_NAME"] == "abl"


def test_fill_from_queue_respects_already_running_usage(tmp_path, monkeypatch):
    daemon = _daemon(tmp_path, live=True)
    entries = []
    entry = job_queue.add_job(entries, "a", "x.sbatch", {}, gpu_count=1, partition="dgx", priority=1, notes="")
    job_queue.save_queue(entries, daemon.queue_path)

    called = []
    monkeypatch.setattr("scripts.ops.gpu_yield_daemon.subprocess.run", lambda *a, **k: called.append(a) or _FakeSbatchResult())
    monkeypatch.setattr("scripts.ops.gpu_yield_daemon.requested_gpus", lambda job_id: 16)
    daemon.budget = {"dgx": 16, "dgxh100": 4}
    # 16 GPUs already used by one of our own running jobs -> zero headroom.
    all_jobs = {"dgx": [{"job_id": "1", "user": "me", "state": "RUNNING"}], "dgxh100": []}
    daemon.fill_from_queue(all_jobs)

    assert called == []
    still_queued = job_queue.find_entry(job_queue.load_queue(daemon.queue_path), entry["id"])
    assert still_queued["status"] == job_queue.STATUS_QUEUED


def test_fill_from_queue_dry_run_does_not_mutate(tmp_path, monkeypatch):
    daemon = _daemon(tmp_path, live=False)
    entries = []
    entry = job_queue.add_job(entries, "a", "x.sbatch", {}, gpu_count=1, partition="dgx", priority=1, notes="")
    job_queue.save_queue(entries, daemon.queue_path)
    calls = []
    monkeypatch.setattr("scripts.ops.gpu_yield_daemon.subprocess.run", lambda *a, **k: calls.append(a))
    daemon.fill_from_queue({"dgx": [], "dgxh100": []})
    assert calls == []
    still_queued = job_queue.find_entry(job_queue.load_queue(daemon.queue_path), entry["id"])
    assert still_queued["status"] == job_queue.STATUS_QUEUED


def test_sync_queue_with_slurm_marks_completed_when_job_leaves_squeue(tmp_path):
    daemon = _daemon(tmp_path)
    entries = []
    entry = job_queue.add_job(entries, "a", "x.sbatch", {}, gpu_count=1, partition="dgx", priority=1, notes="")
    entry["status"] = job_queue.STATUS_SUBMITTED
    entry["submitted_job_id"] = "555"
    job_queue.save_queue(entries, daemon.queue_path)

    daemon.sync_queue_with_slurm({"dgx": [], "dgxh100": []})
    reloaded = job_queue.find_entry(job_queue.load_queue(daemon.queue_path), entry["id"])
    assert reloaded["status"] == job_queue.STATUS_COMPLETED


def test_sync_queue_with_slurm_leaves_still_running_job_alone(tmp_path):
    daemon = _daemon(tmp_path)
    entries = []
    entry = job_queue.add_job(entries, "a", "x.sbatch", {}, gpu_count=1, partition="dgx", priority=1, notes="")
    entry["status"] = job_queue.STATUS_SUBMITTED
    entry["submitted_job_id"] = "555"
    job_queue.save_queue(entries, daemon.queue_path)

    still_running = {"dgx": [{"job_id": "555", "user": "me", "state": "RUNNING"}], "dgxh100": []}
    daemon.sync_queue_with_slurm(still_running)
    reloaded = job_queue.find_entry(job_queue.load_queue(daemon.queue_path), entry["id"])
    assert reloaded["status"] == job_queue.STATUS_SUBMITTED


def test_find_queue_entry_by_job_id(tmp_path):
    daemon = _daemon(tmp_path)
    entries = []
    entry = job_queue.add_job(entries, "a", "x.sbatch", {}, gpu_count=1, partition="dgx", priority=1, notes="")
    entry["submitted_job_id"] = "777"
    job_queue.save_queue(entries, daemon.queue_path)
    found = daemon.find_queue_entry_by_job_id("777")
    assert found["id"] == entry["id"]
    assert daemon.find_queue_entry_by_job_id("no_such_job") is None


def test_requeue_after_yield_resets_status_and_adds_resume_env(tmp_path):
    daemon = _daemon(tmp_path)
    entries = []
    entry = job_queue.add_job(
        entries, "yielded_run", "x.sbatch", {"TURN_REWARD_MODE": "balanced_ce"},
        gpu_count=1, partition="dgx", priority=5, notes="",
    )
    entry["status"] = job_queue.STATUS_SUBMITTED
    entry["submitted_job_id"] = "888"
    job_queue.save_queue(entries, daemon.queue_path)

    daemon.requeue_after_yield(entry)
    reloaded = job_queue.find_entry(job_queue.load_queue(daemon.queue_path), entry["id"])
    assert reloaded["status"] == job_queue.STATUS_QUEUED
    assert reloaded["submitted_job_id"] is None
    assert reloaded["env"]["TURN_REWARD_MODE"] == "balanced_ce"
    assert reloaded["env"]["RUN_NAME"] == "yielded_run"
    assert reloaded["env"]["RESUME_FROM"] == "checkpoints/yielded_run"
    assert reloaded["env"]["NUM_STEPS"] == "2000"  # no metrics.jsonl -> last_step=0


def test_build_resume_env_uppercases_scalars_and_sets_resume_fields():
    snapshot = {
        "run_id": "mc_reward_balcecfg",
        "sbatch_script": "scripts/training/train_meld_ami_werewolf.sbatch",
        "hyperparameters": {
            "turn_reward_mode": "balanced_ce",
            "lora_r": 16,
            "seed": 176106,
            "run_name": "mc_reward_balcecfg",
            "some_nested": {"a": 1},
        },
        "last_known_step": 3000,
    }
    env = build_resume_env(snapshot, resume_step_increment=2000)
    assert env["TURN_REWARD_MODE"] == "balanced_ce"
    assert env["LORA_R"] == "16"
    assert env["SEED"] == "176106"
    assert "SOME_NESTED" not in env
    assert env["RUN_NAME"] == "mc_reward_balcecfg"
    assert env["RESUME_FROM"] == "checkpoints/mc_reward_balcecfg"
    assert env["NUM_STEPS"] == "5000"
