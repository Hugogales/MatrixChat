"""Keeper (login-node gap filler / chain upkeep) and chain.sh queue top-up."""

import json
import os
import subprocess
import textwrap

import pytest

from training.werewolf_rl import continuous as cont
from training.werewolf_rl import keeper as kp

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def no_real_texts(monkeypatch, tmp_path):
    monkeypatch.setenv("NOTIFY_CONFIG", str(tmp_path / "no-notify.env"))


def make_spec(tmp_path, **over):
    run = str(tmp_path / "run1")
    os.makedirs(run, exist_ok=True)
    spec = {"run_name": "run1", "run_dir": run, "repo": REPO, "sbatch": "/x/main.sbatch", "python_bin": "/venv/bin",
            "chains": kp.parse_chains("h100:dgxh100:4 v100:dgx:8 v100b:dgx:4"), "depth": 3, "job_cap": 4,
            "env": {"TRAINER_ARGS": "--games_per_step 16", "EXTRA_PARTITIONS": ""}}
    spec.update(over)
    return kp.write_launch(run, spec)


def job(jid, role, state="PENDING", part="dgx", gpus=4, node="", reason="Resources", dep=""):
    return {"id": str(jid), "name": f"wwc-run1-{role}", "state": state, "partition": part, "gpus": gpus,
            "node": node, "reason": reason, "dependency": dep}


def free(gpus, cpus=80, mem_gb=480):
    return {"gpus": gpus, "cpus": cpus, "mem_mb": mem_gb * 1024}


def full_chains(running=False):
    """Every chain at 1 + depth jobs; heads running (holding their GPUs) or waiting."""
    jobs, jid = [], 100
    for role, part, gpus in (("h100", "dgxh100", 4), ("v100", "dgx", 8), ("v100b", "dgx", 4)):
        jobs.append(job(jid, role, "RUNNING" if running else "PENDING", part, gpus,
                        node="n" if running else "", reason="None" if running else "Resources"))
        for k in range(1, 4):
            jobs.append(job(jid + k, role, part=part, gpus=gpus, reason="Dependency", dep=f"afterany:{jid + k - 1}"))
        jid += 10
    return jobs


def test_parse_chains_and_launch_files(tmp_path):
    with pytest.raises(ValueError):
        kp.parse_chains("gap1:dgx:2")
    with pytest.raises(ValueError):
        kp.parse_chains("a:dgx:2 a:dgx:4")
    with pytest.raises(ValueError):
        kp.parse_chains("a:dgx")
    spec = make_spec(tmp_path)
    run = spec["run_dir"]
    assert kp.read_launch(run)["chains"][1] == {"role": "v100", "partition": "dgx", "gpus": 8}
    assert kp.gap_roles(spec) == ["gap1"]                       # 4 job slots - 3 chains
    env = open(os.path.join(run, "launch.env")).read()
    assert "export CHAIN_OPTS_v100='--partition=dgx --gres=gpu:8 --cpus-per-task=32 --mem=384G --time-min=00:30:00'" in env
    assert "export CHAIN_DEPTH=3" in env and "export TRAINER_ARGS='--games_per_step 16'" in env
    # the file is valid shell
    out = subprocess.run(["bash", "-c", f"source {run}/launch.env && echo $CHAIN_NAMES"], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "wwc-run1-h100,wwc-run1-v100,wwc-run1-v100b"


def test_plan_tops_up_chains_and_skips_stopped(tmp_path):
    spec = make_spec(tmp_path)
    roles = kp.by_role(spec, [job(5, "v100", gpus=8), job(6, "v100", gpus=8, reason="Dependency", dep="afterany:5")])
    queue, _, _ = kp.plan(spec, roles, {}, {}, 0.0, stopped={"v100b"})
    assert ("v100", "6", 2) in queue           # 2 queued + 2 more = 1 + depth
    assert ("h100", None, 4) in queue          # missing chain: a head + 3 follow-ups
    assert all(r != "v100b" for r, _, _ in queue)
    _, _, fill = kp.plan(spec, kp.by_role(spec, full_chains()), {"dgx": {"a": free(2)}}, {}, 0.0, stopped={"gap1"})
    assert fill == []                          # a gap slot stopped after 3 crashes stays empty
    os.makedirs(os.path.join(spec["run_dir"], "chain"), exist_ok=True)
    json.dump({"stopped": True}, open(os.path.join(spec["run_dir"], "chain", "gap1.json"), "w"))
    assert kp.stopped_chains(spec["run_dir"], spec) == {"gap1"}


def test_gap_fills_shortfall_only_while_chains_wait(tmp_path):
    spec = make_spec(tmp_path)
    nodes = {"dgx": {"a": free(1)}, "dgxh100": {"h": free(1)}}
    queue, cancel, fill = kp.plan(spec, kp.by_role(spec, full_chains()), nodes, {}, 0.0)
    assert queue == [] and cancel == [] and fill == ["gap1"]
    # chains running at their target: a gap job off the extra partitions is ended, and the slot is
    # refilled with extra GPUs on the extra partitions (dgxh100); without extra partitions, nothing.
    jobs = full_chains(running=True) + [job(900, "gap1", "RUNNING", "dgx", 1, node="a", reason="None")]
    _, cancel, fill = kp.plan(spec, kp.by_role(spec, jobs), nodes, {}, 0.0)
    assert fill == ["gap1"] and cancel and cancel[0][0] == "gap1" and "target" in cancel[0][1]
    _, cancel, fill = kp.plan(dict(spec, extra_partitions=[]), kp.by_role(spec, jobs), nodes, {}, 0.0)
    assert fill == [] and cancel and "target" in cancel[0][1]


def test_gap_gives_way_to_capped_or_blocked_chain_jobs(tmp_path):
    spec = make_spec(tmp_path)
    jobs = full_chains()
    jobs[0]["reason"] = "QOSMaxJobsPerUserLimit"           # our h100 head can't start: we're at 4 jobs
    jobs.append(job(900, "gap1", "RUNNING", "dgx", 1, node="a", reason="None"))
    _, cancel, fill = kp.plan(spec, kp.by_role(spec, jobs), {"dgx": {"a": free(0)}}, {}, 0.0)
    assert cancel[0][0] == "gap1" and "job limit" in cancel[0][1] and fill == []

    jobs = full_chains() + [job(900, "gap1", "RUNNING", "dgx", 2, node="a", reason="None")]
    # v100b head needs 4 on dgx; node a has 3 free + the gap's 2 -> the gap must go
    _, cancel, fill = kp.plan(spec, kp.by_role(spec, jobs), {"dgx": {"a": free(3)}}, {}, 0.0)
    assert cancel[0][0] == "gap1" and "needs the GPUs" in cancel[0][1] and fill == ["gap1"]
    # 4 free already: the chain job starts on its own, the gap stays
    _, cancel, _ = kp.plan(spec, kp.by_role(spec, jobs), {"dgx": {"a": free(4)}}, {}, 0.0)
    assert not any("needs the GPUs" in why for _, why in cancel)


def test_gap_patience_backoff_and_upgrade(tmp_path):
    spec = make_spec(tmp_path)
    state = {}
    jobs = full_chains() + [job(900, "gap1", "PENDING", "dgxh100", 1, reason="Priority")]
    nodes = {"dgx": {}, "dgxh100": {"h": free(1, cpus=200, mem_gb=1900)}}
    _, cancel, _ = kp.plan(spec, kp.by_role(spec, jobs), nodes, state, 1000.0)
    assert cancel == [] and state["pending_since"]["900"] == 1000.0
    _, cancel, fill = kp.plan(spec, kp.by_role(spec, jobs), nodes, state, 1000.0 + 11 * 60)
    assert cancel[0][0] == "gap1" and "waited 11 min" in cancel[0][1] and fill == ["gap1"]
    assert state["backoff"]["dgxh100:1"] > 1000.0 + 11 * 60
    assert kp.holes(spec, nodes, 16, backoff=state["backoff"], now=1000.0 + 12 * 60) == []   # backed off
    # a running 1-GPU gap job next to 3 newly free GPUs is swapped for a 4-GPU one
    # (h100 and v100b run; only the 8-GPU v100 chain waits, so the 4 GPUs are not enough for it)
    jobs = [j for j in full_chains(running=True) if j["name"] != "wwc-run1-v100"]
    jobs += [job(110, "v100", gpus=8)] + [job(901, "gap1", "RUNNING", "dgx", 1, node="a", reason="None")]
    _, cancel, fill = kp.plan(spec, kp.by_role(spec, jobs), {"dgx": {"a": free(3)}}, {}, 0.0)
    assert cancel[0][0] == "gap1" and "4-GPU hole" in cancel[0][1] and fill == ["gap1"]
    _, cancel, _ = kp.plan(spec, kp.by_role(spec, jobs), {"dgx": {"a": free(1)}}, {}, 0.0)
    assert cancel == []                         # only 1 more: not worth it


def test_holes_respect_cpu_memory_and_preference(tmp_path):
    spec = make_spec(tmp_path)
    nodes = {"dgx": {"a": free(6, cpus=8), "b": free(2)}, "dgxh100": {"h": free(3, mem_gb=100)}}
    # a: 8 CPUs -> 2 GPUs; h: 100 GB -> 2 GPUs; b: 2
    got = kp.holes(spec, nodes, 16, short={"dgx": 12})
    assert [(p, n, g) for p, n, g in got] == [("dgx", "a", 2), ("dgx", "b", 2), ("dgxh100", "h", 2)]
    assert kp.holes(spec, nodes, 1)[0][2] == 1


def test_extras_are_ended(tmp_path):
    spec = make_spec(tmp_path)
    jobs = full_chains() + [job(950, "extra", "RUNNING", "dgx", 1, node="a", reason="None")]
    _, cancel, _ = kp.plan(spec, kp.by_role(spec, jobs), {"dgx": {}}, {}, 0.0)
    assert ("extra", "extras are replaced by gap jobs") in cancel


class FakeSlurm:
    def __init__(self, jobs, nodes):
        self.queue, self._nodes, self.submitted, self.cancelled, self.next = list(jobs), nodes, [], [], 1000

    def jobs(self, prefix):
        return [dict(j) for j in self.queue if j["name"].startswith(prefix)]

    def nodes(self, partitions):
        return {p: {n: dict(r) for n, r in self._nodes.get(p, {}).items()} for p in partitions}

    def will_start(self, argv):
        raise AssertionError("test_only is off by default")

    def submit(self, argv, env):
        self.next += 1
        self.submitted.append((argv, env))
        return str(self.next)

    def cancel_name(self, name):
        self.cancelled.append(name)
        self.queue = [j for j in self.queue if j["name"] != name]


def test_tick_submits_chain_links_and_a_gap_job(tmp_path):
    spec = make_spec(tmp_path)
    run = spec["run_dir"]
    jobs = full_chains()
    jobs = [j for j in jobs if j["name"] != "wwc-run1-v100b"]            # that chain vanished
    slurm = FakeSlurm(jobs, {"dgx": {"a": free(2)}, "dgxh100": {}})
    summary = kp.tick(run, slurm=slurm, now=0.0, log=lambda *_: None, wait=False)
    links = [(a, e) for a, e in slurm.submitted if "--job-name=wwc-run1-v100b" in a]
    assert len(links) == 4
    assert not any(a.startswith("--dependency") for a in links[0][0]) and "PREV_JOB" not in links[0][1]
    assert "--dependency=afterany:1001" in links[1][0] and links[1][1]["PREV_JOB"] == "1001"
    assert links[1][1]["CHAIN_ROLE"] == "v100b" and links[1][1]["PATH"].startswith("/venv/bin:")
    gap = [(a, e) for a, e in slurm.submitted if "--job-name=wwc-run1-gap1" in a]
    assert len(gap) == 1 and "--gres=gpu:2" in gap[0][0] and gap[0][1]["CHAIN_ROLE"] == "gap1"
    assert gap[0][1]["TRAINER_ARGS"] == "--games_per_step 16" and gap[0][0][-1] == "/x/main.sbatch"
    assert any("gap1: submitted job" in a for a in summary["actions"])
    state = json.load(open(os.path.join(run, "keeper", "state.json")))
    assert state["last_tick"] == 0.0 and kp.keeper_status(run)[2]
    # dry run changes nothing; STOP stops everything
    slurm2 = FakeSlurm(full_chains(), {"dgx": {"a": free(2)}})
    kp.tick(run, slurm=slurm2, now=0.0, dry_run=True, log=lambda *_: None)
    assert slurm2.submitted == []
    open(os.path.join(run, "STOP"), "w").close()
    slurm3 = FakeSlurm([], {"dgx": {"a": free(8)}})
    kp.tick(run, slurm=slurm3, now=0.0, log=lambda *_: None)
    assert slurm3.submitted == []


def test_chain_guard_sticky_stop_and_short_runs(tmp_path):
    run = str(tmp_path / "run1")
    states = {"1": "COMPLETED", "2": "COMPLETED", "3": "COMPLETED", "4": "TIMEOUT", "9": "COMPLETED"}
    elapsed = {"1": 30, "2": 40, "3": 50, "4": 86400, "9": 5000}
    guard = lambda prev: cont.chain_guard(run, "v100", prev, state_of=states.get,  # noqa: E731
                                          elapsed_of=elapsed.get, alert=lambda *a: None)
    assert guard("9")[0]                       # a long normal run is fine
    assert guard("1")[0] and guard("2")[0]     # finishing after 30-40 s counts as a crash
    ok, msg = guard("3")
    assert not ok
    ok, msg = guard("4")                       # a queued follow-up does not revive a stopped chain
    assert not ok and "stopped earlier" in msg
    status = json.load(open(os.path.join(run, "chain", "v100.json")))
    assert status["stopped"] and "after only 50s" in status["history"][-1]["state"]


def test_chain_sh_tops_up_queue(tmp_path):
    """chain.sh keeps CHAIN_DEPTH follow-ups queued, each after the last, sized from CHAIN_OPTS_<role>."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    q, log, nxt = tmp_path / "q", tmp_path / "log", tmp_path / "next"
    q.write_text("601\n")
    nxt.write_text("700")
    (bin_dir / "squeue").write_text(f"#!/bin/bash\ncat {q}\n")
    (bin_dir / "sbatch").write_text(textwrap.dedent(f"""\
        #!/bin/bash
        n=$(( $(cat {nxt}) + 1 )); echo $n > {nxt}; echo "$*" >> {log}; echo $n >> {q}
        echo "Submitted batch job $n"
        """))
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    run = tmp_path / "run"
    (run / "chain").mkdir(parents=True)
    script = textwrap.dedent(f"""\
        export PATH={bin_dir}:$PATH RUN_DIR={run} CHAIN_SCRIPT=/x/main.sbatch CHAIN_ROLE=v100 CHAIN_DEPTH=3
        export SLURM_JOB_NAME=wwc-t-v100 SLURM_JOB_ID=500 SLURM_JOB_PARTITION=dgx CUDA_VISIBLE_DEVICES=0,1
        export CHAIN_OPTS_v100='--partition=dgx --gres=gpu:8'
        source {REPO}/scripts/training/werewolf_continuous/chain.sh
        sleep() {{ :; }}
        _chain_top_up
        CHAIN_ROLE=gap1 SLURM_JOB_NAME=wwc-t-gap1; : > {q}; _chain_top_up
        """)
    subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True)
    calls = log.read_text().splitlines()
    assert calls[0].startswith("--partition=dgx --gres=gpu:8 --job-name=wwc-t-v100 --dependency=afterany:601")
    assert "--dependency=afterany:701" in calls[1] and "PREV_JOB=701" in calls[1]
    assert len(calls) == 3                     # 601 + 2 new = depth 3, then 1 for the gap chain
    assert "--gres=gpu:2 --cpus-per-task=8 --mem=96G" in calls[2] and "afterany:500" in calls[2]


CRASH_3_THEN_STOP = 'if [ $n -le 3 ]; then exit 2; fi; touch "$RUN_DIR/STOP"; exit 0'


def _fake_cluster(tmp_path, sacct_line="", player=CRASH_3_THEN_STOP):
    """PATH with fake Slurm commands and a fake ``python`` (``-m`` runs the real one). A
    trainer/player run is ``player`` (bash; ``$n`` = how many runs so far, this one included):
    by default it crashes 3 times, then the 4th run writes STOP and exits 0."""
    import sys

    bin_dir, calls = tmp_path / "bin", tmp_path / "calls"
    bin_dir.mkdir()
    (bin_dir / "squeue").write_text(textwrap.dedent("""\
        #!/bin/bash
        case "$*" in
          *"-o %e"*) echo 2026-11-21T19:45:51 ;;
          *"-t R"*) printf 'wwc-t-v100 gres/gpu:v100:8\\nwwc-t-h100 gres/gpu:4\\nsomeone gres/gpu:2\\n' ;;
        esac
        """))
    (bin_dir / "nvidia-smi").write_text("#!/bin/bash\necho 'Tesla V100-SXM2-32GB, 32768 MiB'\n")
    (bin_dir / "sbatch").write_text(f"#!/bin/bash\necho \"sbatch $*\" >> {calls}\necho 'Submitted batch job 900'\n")
    (bin_dir / "scancel").write_text(f"#!/bin/bash\necho \"scancel $*\" >> {calls}\n")
    (bin_dir / "sacct").write_text(f"#!/bin/bash\necho '{sacct_line}'\n")
    (bin_dir / "python").write_text(textwrap.dedent(f"""\
        #!/bin/bash
        if [ "$3" = yield-check ]; then  # fake keeper verdict: keep, then give back
          k=$(( $(cat {tmp_path}/yields 2>/dev/null || echo 0) + 1 )); echo $k > {tmp_path}/yields
          [ $k -ge 2 ] && echo "another user's job 1 needs them" && exit 0
          exit 1
        fi
        [ "$1" = -m ] && exec {sys.executable} "$@"
        if [ "$1" = -c ]; then case "$2" in *__version__*) echo fake;; esac; exit 0; fi
        n=$(( $(cat {tmp_path}/runs 2>/dev/null || echo 0) + 1 )); echo $n > {tmp_path}/runs
        {player}
        """))
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    run = tmp_path / "run"
    (run / "chain").mkdir(parents=True)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", REPO=REPO, RUN_DIR=str(run), RUN_NAME="t",
               CHAIN_SCRIPT="/x/main.sbatch", CHAIN_DEPTH="1", SLURM_JOB_ID="500", SLURM_JOB_PARTITION="dgx",
               CUDA_VISIBLE_DEVICES="0", CHAIN_POLL="0.2", CHAIN_RESTART_DELAY="0",
               CHECKPOINT_DIR="c", HYPERPARAMETERS="h", MODEL_PATH="m")
    return run, calls, env


def _run_chain_job(env, role, after=""):
    """Run chain_start as a job of ``role``; ``after`` runs in the background (a watcher)."""
    script = (f"export CHAIN_ROLE={role} SLURM_JOB_NAME=wwc-t-{role}\n{after}\n"
              f"source {REPO}/scripts/training/werewolf_continuous/chain.sh\nchain_start\necho RETURNED\n")
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)


def test_chain_job_holds_a_crashed_gpu_until_retry(tmp_path):
    """A main chain's process that crashed 3 times leaves its GPU held (idle) until start.sh
    touches RUN/chain/retry; then it restarts. STOP ends the job; chain_start never returns."""
    run, calls, env = _fake_cluster(tmp_path)
    watcher = (f"( while ! ls {run}/chain/holding/v100_500_play0 >/dev/null 2>&1; do /bin/sleep 0.1; done; "
               f"/bin/sleep 1.2; touch {run}/chain/retry ) &")
    out = _run_chain_job(env, "v100", watcher)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "RETURNED" not in out.stdout
    assert "play0 idle, GPUs kept" in out.stdout and "retry requested: restarting play0" in out.stdout
    alert = next(line for line in out.stdout.splitlines() if line.startswith("[alert] gpu:"))
    assert "got 1x Tesla V100-SXM2-32GB 32768 MiB on " in alert and "(partition dgx) - v100 job 500" in alert
    assert "new for this chain, runs until Sat 19:45. working (fake)" in alert and "now holds 12 GPUs" in alert
    assert (tmp_path / "runs").read_text().strip() == "4"       # 3 crashes, then the retried run
    assert os.listdir(run / "chain" / "holding") == []
    assert "sbatch" in calls.read_text() and "scancel" not in calls.read_text()


def test_chain_job_holds_after_three_crashed_jobs_but_gap_stops(tmp_path):
    run, calls, env = _fake_cluster(tmp_path, sacct_line="FAILED|5")
    env["PREV_JOB"] = "499"
    (run / "chain" / "v100.json").write_text(json.dumps({"consecutive_crashes": 2, "history": []}))
    out = _run_chain_job(env, "v100", f"( /bin/sleep 1.5; touch {run}/STOP ) &")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "job idle, GPUs kept: its previous jobs crashed 3 times" in out.stdout
    assert "takes over from job 499" in out.stdout and "IDLE, GPUs kept: its previous jobs crashed" in out.stdout
    status = json.loads((run / "chain" / "v100.json").read_text())
    assert status["holding"] and not status["stopped"]
    assert "scancel" not in calls.read_text() and "sbatch" in calls.read_text()   # follow-ups stay queued
    assert not (tmp_path / "runs").exists()                                         # nothing ran on the GPUs

    (run / "STOP").unlink()
    (run / "chain" / "gap1.json").write_text(json.dumps({"consecutive_crashes": 2, "history": []}))
    out = _run_chain_job(env, "gap1")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "scancel -u" in calls.read_text() and json.loads((run / "chain" / "gap1.json").read_text())["stopped"]


def test_pause_keeps_the_gpu_and_resumes(tmp_path):
    """While RUN/PAUSE exists nothing runs, but the job (gap chains too) keeps its GPU;
    when start.sh removes PAUSE the process starts again."""
    first_run_sees_pause = 'if [ $n -eq 1 ]; then touch "$RUN_DIR/PAUSE"; exit 0; fi; touch "$RUN_DIR/STOP"; exit 0'
    run, calls, env = _fake_cluster(tmp_path, player=first_run_sees_pause)
    watcher = (f"( while ! ls {run}/chain/holding/gap1_500_play0 >/dev/null 2>&1; do /bin/sleep 0.1; done; "
               f"/bin/sleep 0.5; rm -f {run}/PAUSE ) &")
    out = _run_chain_job(env, "gap1", watcher)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "play0 paused, GPU kept" in out.stdout and "resumed: starting play0" in out.stdout
    assert (tmp_path / "runs").read_text().strip() == "2"
    assert "scancel" not in calls.read_text() and os.listdir(run / "chain" / "holding") == []


def test_halt_reason(tmp_path):
    run = str(tmp_path / "run")
    cont.ensure_run_dir(run)
    assert cont.halt_reason(run) is None
    open(os.path.join(run, "PAUSE"), "w").close()
    assert cont.halt_reason(run) == "PAUSE file" and not cont.stop_requested(run)
    open(os.path.join(run, "STOP"), "w").close()
    assert cont.halt_reason(run) == "STOP file"


def test_queue_summary_and_announce(tmp_path):
    spec = make_spec(tmp_path)
    jobs = [job(100, "h100", "RUNNING", "dgxh100", 4, node="dh-dgxh100-1", reason="None"),
            job(101, "h100", part="dgxh100", reason="Dependency", dep="afterany:100"),
            job(110, "v100", gpus=8, reason="Priority"), job(111, "v100", gpus=8, reason="Dependency"),
            job(200, "gap1", "RUNNING", gpus=1, node="dh-dgx1-3", reason="None")]
    assert kp.queue_summary(spec, jobs) == (
        "h100 4xH100: 100 running on dh-dgxh100-1 +1 queued; v100 8xV100: next 110 (Priority) +1 queued; "
        "v100b 4xV100: no jobs; gap1: 200 running on dh-dgx1-3 (1 GPU)")

    class Fake:
        def jobs(self, prefix):
            return jobs

    sent = []
    msg, ok = kp.announce(spec["run_dir"], "Paused.", slurm=Fake(),
                          send=lambda *a, **k: sent.append((a, k)) or True)
    assert ok and msg.startswith("Paused. squeue: h100 4xH100") and len(msg) < 480 and "IDLE" not in msg
    assert sent[0][0][0] == "status" and sent[0][1] == {"force": True}

    hold = os.path.join(spec["run_dir"], "chain", "holding")
    os.makedirs(hold)
    open(os.path.join(hold, "h100_100_job"), "w").write("job 100: no python here runs torch on the GPU")
    open(os.path.join(hold, "gap1_200_play0"), "w").write("job 200: paused (stop.sh --pause)")
    msg, _ = kp.announce(spec["run_dir"], "Back.", slurm=Fake(), send=lambda *a, **k: True)
    assert "100 running on dh-dgxh100-1 IDLE" in msg and "dh-dgx1-3 (1 GPU) paused" in msg
    assert msg.endswith("IDLE = holds its GPUs but runs nothing (see status.sh)")


def test_notify_force_overrides_the_event_filter(monkeypatch):
    from training.werewolf_rl import notify as nt

    sent = []
    monkeypatch.setattr(nt, "send_text", lambda m, cfg, event=None: (sent.append(event), (True, "fake"))[1])
    cfg = {"NOTIFY_PROVIDER": "ntfy", "NTFY_TOPIC": "test-topic", "NOTIFY_EVENTS": "crash"}
    assert nt.notify("status", "a", cfg=cfg) is False and sent == []
    assert nt.notify("status", "b", cfg=cfg, force=True) and sent == ["status"]


def test_extra_gap_job_gives_its_gpus_back_when_someone_needs_them(tmp_path):
    """A gap job on an extra partition asks the keeper every YIELD_POLL s; when told to yield it
    alerts and cancels itself and its queued follow-up (scancel --name)."""
    wait_for_cancel = f'while ! grep -q "scancel -u" {tmp_path}/calls 2>/dev/null; do /bin/sleep 0.1; done; ' \
                      'touch "$RUN_DIR/STOP"; exit 0'
    run, calls, env = _fake_cluster(tmp_path, player=wait_for_cancel)
    env.update(SLURM_JOB_PARTITION="dgxh100", EXTRA_GPU_PARTITIONS="dgxh100", YIELD_POLL="0.2")
    out = _run_chain_job(env, "gap1")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Gap slot: GPUs beyond our chain targets are given back" in out.stdout
    gave = next(line for line in out.stdout.splitlines() if "gave back" in line)
    assert "gave back 1x Tesla V100-SXM2-32GB 32768 MiB" in gave and "(partition dgxh100, gap1 job 500, extra GPUs)" in gave
    assert "another user's job 1 needs them" in gave
    assert "scancel -u" in calls.read_text() and "--name=wwc-t-gap1" in calls.read_text()


def test_chain_jobs_never_run_the_yield_watch(tmp_path):
    run, calls, env = _fake_cluster(tmp_path)
    env.update(SLURM_JOB_PARTITION="dgxh100", EXTRA_GPU_PARTITIONS="dgxh100", YIELD_POLL="0.1")
    watcher = (f"( while ! ls {run}/chain/holding/h100_500_play0 >/dev/null 2>&1; do /bin/sleep 0.1; done; "
               f"/bin/sleep 1.2; touch {run}/chain/retry ) &")
    out = _run_chain_job(env, "h100", watcher)
    assert out.returncode == 0 and not (tmp_path / "yields").exists() and "Gap slot" not in out.stdout


FREUNDS = {"NumNodes": "2-2", "NumCPUs": "32", "TresPerNode": "gres/gpu:1", "MinMemoryNode": "256G",
           "Partition": "dgxh100", "ReqNodeList": "(null)", "ExcNodeList": "(null)"}


def test_parse_request_and_fit():
    r = kp.parse_request(FREUNDS)
    assert r == {"nodes": 2, "gpus": 1, "cpus": 16, "mem_mb": 262144, "partitions": ["dgxh100"],
                 "req_nodes": set(), "exc_nodes": set()}
    one = kp.parse_request({"NumNodes": "1", "NumCPUs": "8", "TresPerJob": "gres/gpu:h100:2", "MinMemoryCPU": "4G",
                            "Partition": "dgx,dgxh100"})
    assert (one["nodes"], one["gpus"], one["cpus"], one["mem_mb"], one["partitions"]) == (1, 2, 8, 32768, ["dgx", "dgxh100"])
    nodes = {"h1": free(0, 200, 1500), "h2": free(0, 200, 1900)}
    assert not kp.request_fits(one, nodes) and kp.request_fits(one, nodes, give={"h1": (4, 16, 196608)})
    assert not kp.request_fits(r, nodes, give={"h1": (4, 16, 196608)})      # still needs a GPU on h2 too


def h100_spec(tmp_path):
    return make_spec(tmp_path)


def test_extra_gpus_kept_until_another_users_job_fits(tmp_path):
    spec = h100_spec(tmp_path)
    jobs = full_chains(running=True)
    for j in jobs:
        if j["state"] == "RUNNING" and j["partition"] == "dgxh100":
            j["node"] = "h1"
    gap = job(900, "gap1", "RUNNING", "dgxh100", 4, node="h1", reason="None")
    roles = kp.by_role(spec, jobs + [gap])
    assert kp.is_extra(spec, roles, gap)
    nodes = {"dgxh100": {"h1": free(0, 200, 1500), "h2": free(0, 200, 1900)}, "dgx": {}}
    two = dict(kp.parse_request({"NumNodes": "1", "NumCPUs": "8", "TresPerNode": "gres/gpu:2", "Partition": "dgxh100"}),
               job="555", user="x", reason="Priority")
    freunds = dict(kp.parse_request(FREUNDS), job="302079", user="x", reason="Resources")
    assert kp.yield_reason(spec, roles, gap, [freunds], nodes) is None     # can't start even if we leave
    assert "job 555" in kp.yield_reason(spec, roles, gap, [freunds, two], nodes)
    q, cancel, fill = kp.plan(spec, roles, nodes, {}, 0.0, requests=[freunds])
    assert cancel == [] and fill == []                                      # extras kept
    q, cancel, fill = kp.plan(spec, roles, nodes, {}, 0.0, requests=[two])
    assert cancel and cancel[0][0] == "gap1" and "giving back extra GPUs" in cancel[0][1] and fill == []
    # The h100 chain is not running: the gap job covers our own minimum, so it is no extra and keeps its GPUs.
    roles_short = kp.by_role(spec, [j for j in jobs if j["partition"] != "dgxh100"] + [gap])
    assert not kp.is_extra(spec, roles_short, gap) and kp.yield_reason(spec, roles_short, gap, [two], nodes) is None


def test_tick_grabs_extra_h100s_only_when_nobody_could_use_them(tmp_path):
    spec = h100_spec(tmp_path)
    run = spec["run_dir"]
    jobs = full_chains(running=True)
    nodes = {"dgxh100": {"h1": free(4, 200, 1500), "h2": free(0, 200, 1900)}, "dgx": {"d3": free(5, 40, 380)}}
    fake = FakeSlurm(jobs, nodes)
    fake.waiting_requests = lambda part: [dict(kp.parse_request(FREUNDS), job="302079", user="x", reason="Resources")]
    summary = kp.tick(run, slurm=fake, now=1000.0, wait=False)
    assert any("extra GPUs, given back" in a for a in summary["actions"]), summary["actions"]
    argv = fake.submitted[-1][0]
    assert "--partition=dgxh100" in argv and "--gres=gpu:4" in argv and "--time-min=00:30:00" in argv
    fake2 = FakeSlurm(full_chains(running=True), nodes)
    fake2.waiting_requests = lambda part: [dict(kp.parse_request({"NumNodes": "1", "NumCPUs": "8",
                                                                   "TresPerNode": "gres/gpu:2", "Partition": "dgxh100"}),
                                                job="555", user="x", reason="Priority")]
    summary = kp.tick(spec["run_dir"], slurm=fake2, now=2000.0, wait=False)
    assert fake2.submitted == [] and any("could use them" in a for a in summary["actions"])


def test_sbatch_opts_time_min(tmp_path):
    spec = make_spec(tmp_path)
    assert "--time-min=00:30:00" in kp.sbatch_opts(spec, "dgxh100", 4)
    assert not any(o.startswith("--time-min") for o in kp.sbatch_opts(dict(spec, time_min=""), "dgx", 8))
    env = open(os.path.join(spec["run_dir"], "launch.env")).read()
    assert "export CHAIN_TIME_MIN=00:30:00" in env and "export EXTRA_GPU_PARTITIONS=dgxh100" in env
    assert "--time-min=00:30:00" in env.split("CHAIN_OPTS_h100=")[1].splitlines()[0]
