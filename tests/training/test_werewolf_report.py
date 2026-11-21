"""report.py: the status page (status.sh) and the hourly ntfy GPU report."""

import json
import os
import time

import pytest

from training.werewolf_rl import continuous as cont
from training.werewolf_rl import keeper as kp
from training.werewolf_rl import report as rp

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def no_real_texts(monkeypatch, tmp_path):
    monkeypatch.setenv("NOTIFY_CONFIG", str(tmp_path / "no-notify.env"))


class FakeSlurm:
    def __init__(self, jobs, blocker=None):
        self._jobs, self.blocker = jobs, blocker

    def jobs(self, prefix):
        return [dict(j) for j in self._jobs if j["name"].startswith(prefix)]

    def queue_blocker(self, partition):
        return self.blocker


def job(jid, role, state="RUNNING", part="dgx", gpus=8, node="dh-dgx1-2", reason="None", elapsed=3600,
        left=72000, dep=""):
    return {"id": str(jid), "name": f"wwc-run1-{role}", "state": state, "partition": part, "gpus": gpus,
            "node": node if state == "RUNNING" else "", "reason": reason, "dependency": dep,
            "elapsed": elapsed if state == "RUNNING" else None, "time_left": left if state == "RUNNING" else None}


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        if isinstance(obj, list):
            f.write("".join(json.dumps(x) + "\n" for x in obj))
        else:
            json.dump(obj, f)


@pytest.fixture
def run(tmp_path):
    run = str(tmp_path / "run1")
    kp.write_launch(run, {"run_name": "run1", "run_dir": run, "repo": REPO, "sbatch": "/x/main.sbatch",
                          "python_bin": "/venv/bin", "chains": kp.parse_chains("h100:dgxh100:4 v100:dgx:8 v100b:dgx:4"),
                          "depth": 14, "job_cap": 4, "env": {}})
    cont.ensure_run_dir(run)
    write(os.path.join(run, "game_config.json"), {"num_players": 6, "num_werewolves": 1, "werewolves_win": "all_dead"})
    now = time.time()
    hb = os.path.join(run, "heartbeats")
    # job 500 (v100, 8 GPUs): trainers on 0-3 (0 leads), players on 4-7 (7 has no heartbeat: down)
    write(os.path.join(hb, "trainer_a.json"), {"job": "500", "slot": "0", "leader": True, "accepting": True,
                                                "tasks_done": 9, "gpu_mem_gb": 28.4, "t": now})
    for slot in (1, 2, 3):
        write(os.path.join(hb, f"trainer_{slot}.json"), {"job": "500", "slot": str(slot), "leader": False,
                                                          "accepting": True, "tasks_done": 4, "t": now})
    write(os.path.join(hb, "player_v100j500s4.json"), {
        "job": "500", "slot": 4, "state": "playing", "phase": "day 2 vote", "game_t0": now - 180, "games": 12,
        "avg_minutes": 6.0, "last": {"winner": "Village", "days": 2, "minutes": 5.5}, "loaded": 2,
        "gpu_mem_gb": 18.5, "t": now})
    write(os.path.join(hb, "player_v100j500s5.json"), {"job": "500", "slot": 5, "state": "pool full, waiting",
                                                        "games": 3, "loaded": 2, "t": now})
    write(os.path.join(hb, "player_v100j500s6.json"), {"job": "500", "slot": 6, "state": "loading policy v3",
                                                        "games": 3, "loaded": 2, "t": now})
    write(os.path.join(run, "train", "leader.json"), {"owner": "a", "step": 3, "phase": "epoch1_grads"})
    # job 600 (v100b): the whole job idle
    os.makedirs(os.path.join(run, "chain", "holding"), exist_ok=True)
    with open(os.path.join(run, "chain", "holding", "v100b_600_job"), "w") as f:
        f.write("2026-11-20 15:06:08 job 600 on dh-dgx1-1, job: no python here runs torch on the GPU (PATH python x)")
    timing = os.path.join(run, "timing")
    write(os.path.join(timing, "game_v100.jsonl"),
          [{"t": now - 7200, "seconds": 200}] +   # an old-rule record (no winner / rule)
          [{"t": now - 600 * i, "seconds": 360, "winner": "Village" if i % 2 else "Werewolves", "days": 2 + i % 2,
            "rule": "all_dead"} for i in range(4)])
    write(os.path.join(timing, "learner_step.jsonl"), [{"t": now - 1800, "seconds": 1500, "step": 2}])
    write(os.path.join(run, "learner", "metrics.jsonl"),
          [{"step": 2, "village_win_rate": 0.5, "mean_days": 2.4, "mean_speak_rate": 0.12, "step_seconds": 1500,
            "epochs": [{"kl_ref": 0.0123}]}])
    with open(os.path.join(run, "weights", "LATEST"), "w") as f:
        f.write("3")
    return run, now


def slurm_for(blocker=None):
    return FakeSlurm([job(500, "v100"), job(600, "v100b", gpus=4, node="dh-dgx1-1"),
                      job(700, "h100", "PENDING", "dgxh100", 4, reason="Priority"),
                      job(701, "h100", "PENDING", "dgxh100", 4, reason="Dependency", dep="afterany:700")],
                     blocker=blocker)


def test_snapshot_rows_and_counts(run):
    run, now = run
    s = rp.snapshot(run, slurm=slurm_for(), now=now)
    assert s["held"] == {"V100": 12} and s["target"] == {"H100": 4, "V100": 12}
    v100 = next(c for c in s["chains"] if c["role"] == "v100")["running"][0]["rows"]
    assert [r["what"] for r in v100] == ["train"] * 4 + ["play"] * 4
    assert v100[0]["state"] == "leading" and v100[0]["detail"] == "step 3 (epoch1_grads); 9 tasks done; 28.4 GB"
    assert v100[1]["state"] == "training" and v100[1]["detail"].startswith("helping on step 3")
    assert v100[4]["state"] == "playing"
    assert v100[4]["detail"] == ("game 13: day 2 vote, 3 min in; 12 done (avg 6 min), last: Village in 2d; "
                                 "policy v2; 18.5 GB")
    assert v100[5]["state"] == "waiting" and v100[6]["state"] == "loading"
    assert v100[7]["state"] == "down"                        # no heartbeat in a job that ran an hour
    v100b = next(c for c in s["chains"] if c["role"] == "v100b")["running"][0]["rows"]
    assert {r["state"] for r in v100b} == {"IDLE"} and v100b[0]["detail"].startswith(
        "no python here runs torch on the GPU (since 15:06)")
    assert s["working"] == 7 and s["idle"] == 5
    g = s["games"]
    assert g["recent"] == 4 and g["other_rules"] == 1 and g["village_win_rate"] == 0.5 and g["last_hour"] == 4
    t = s["training"]
    assert t["version"] == 3 and t["steps"] == 1 and t["steps_last_hour"] == 1 and t["last"]["kl_ref"] == 0.0123


def test_status_page_and_phone_report(run):
    run, now = run
    blocker = {"job": "302079", "user": "someone", "start": "2026-11-20T20:36:45", "planned_nodes": ["dh-dgxh100-1"]}
    s = rp.snapshot(run, slurm=slurm_for(blocker), now=now)
    page = rp.format_status(s)
    assert "12/16 GPUs held (H100 0/4, V100 12/12): 7 working, 5 idle" in page
    assert "behind someone's job 302079" in page and "keeps dh-dgxh100-1 free" in page
    assert "v100b 600 on dh-dgx1-1: all 4 V100 IDLE - no python here runs torch on the GPU" in page
    assert "gpu4  play  playing   game 13: day 2 vote" in page
    assert "last 4 games: Village won 50%" in page and "(+1 under older rules)" in page
    phone = rp.format_report(s)
    assert "someone" not in phone and "behind an older job" in phone      # no other users' names in pushes
    assert " g0 train leading: step 3 (epoch1_grads)" in phone and " g4 play playing: game 13" in phone
    assert "v100b 4xV100 dh-dgx1-1 (20h00m left): all 4 IDLE" in phone
    assert len(phone) < 3900


def test_report_goes_out_once_per_clock_hour(run):
    run, now = run
    spec = kp.read_launch(run)
    assert spec["report_minutes"] == 60
    sent, state = [], {}
    send = lambda event, message, run_dir, force: sent.append((event, force)) or True  # noqa: E731
    hour = (int(now // 3600)) * 3600
    assert rp.maybe_send_report(run, spec, state, hour + 60, slurm_for(), send)
    assert not rp.maybe_send_report(run, spec, state, hour + 1800, slurm_for(), send)   # same hour
    assert rp.maybe_send_report(run, spec, state, hour + 3600 + 30, slurm_for(), send)  # next hour
    assert sent == [("report", True), ("report", True)] and state["last_report_ok"]
    assert not rp.maybe_send_report(run, dict(spec, report_minutes=0), {}, hour, slurm_for(), send)


def test_keeper_tick_sends_the_hourly_report(run, monkeypatch):
    run, now = run
    calls = []
    monkeypatch.setattr(rp, "maybe_send_report", lambda *a, **k: calls.append(a[3]) or True)
    summary = kp.tick(run, slurm=type("S", (), {"jobs": lambda self, p: [], "nodes": lambda self, ps: {},
                                               "submit": lambda self, argv, env: "1",
                                               "cancel_name": lambda self, n: None})(), now=now, wait=False)
    assert calls == [now] and "sent the hourly GPU report" in summary["actions"]
    open(os.path.join(run, "STOP"), "w").close()
    calls.clear()
    kp.tick(run, slurm=None, now=now)
    assert calls == []                                       # stopped runs send nothing
