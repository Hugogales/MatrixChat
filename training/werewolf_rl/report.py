"""Status of an always-on Werewolf RL run: every chain, job and GPU (status.sh, hourly ntfy report).

    python -m training.werewolf_rl.report status --run_dir RUN           # what status.sh prints
    python -m training.werewolf_rl.report hourly --run_dir RUN [--send]  # the hourly ntfy report

``snapshot`` gathers everything into one dict:

* Slurm: each chain's running job (node, elapsed, time left) and its queue, and for a
  waiting chain, the older job the scheduler is keeping nodes free for.
* Every GPU of every running job, from the heartbeats that trainer.py and player.py
  rewrite every 20 s and from chain.sh's idle marks (RUN/chain/holding).
  * A player reports its games so far, the current game's day and phase, the last
    result, its policy version and the GPU memory it uses.
  * A trainer reports whether it leads (the step and phase it is on) and how many
    tasks it has finished.
* Training: the policy version, steps (learner/metrics.jsonl, timing/learner_step.jsonl)
  and the leader lease.
* Games: the pool, games in the last hour, and recent results under the current
  rules (timing/game_*.jsonl).
* The keeper's last tick and the alerts.

``format_status`` renders it for a terminal and ``format_report`` for a phone.
The keeper's cron tick calls ``maybe_send_report`` every 5 minutes; it sends once
per ``report_minutes`` (launch.json, default 60), on the clock hour.
"""

import argparse
import glob
import json
import os
import re
import time

from . import continuous as cont
from . import keeper as kp

HEARTBEAT_DEAD = 120          # seconds without a heartbeat before a process counts as gone
STARTING_GRACE = 15 * 60      # a job this young without heartbeats is still loading its model
WORKING = {"playing", "training", "leading", "waiting", "loading", "ready"}


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        return None


def _jsonl(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except FileNotFoundError:
        pass
    return out


def dur(seconds):
    """``3h05m`` / ``42m`` / ``-``."""
    if seconds is None:
        return "-"
    seconds = int(max(seconds, 0))
    h, m = divmod(seconds // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def short(reason, limit=60):
    """The gist of an idle reason: drop the parenthesized details, cap the length."""
    text = reason.split(" (")[0] if len(reason.split(" (")[0]) >= 10 else reason
    return text if len(text) <= limit else text[:limit - 3] + "..."


def kind_of(partition):
    return kp.GPU_KIND.get(partition, partition or "GPU")


# ----------------------------------------------------------------------------
# Gathering
# ----------------------------------------------------------------------------
def read_holding(run_dir):
    """chain.sh's idle marks: ``{job: {what: {"since", "reason", "paused"}}}``; ``what`` is
    ``job`` (the whole job) or a slot like ``play3`` / ``train0``."""
    out = {}
    d = os.path.join(run_dir, "chain", "holding")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        parts = f.split("_")
        if len(parts) < 3:
            continue
        try:
            with open(os.path.join(d, f)) as fh:
                text = fh.read().strip()
        except OSError:
            continue
        m = re.match(r"(\S+ \S+) job \S+ on [^,]+, [^:]+: (.*)", text)
        since, reason = (m.group(1), m.group(2)) if m else ("", text)
        out.setdefault(parts[1], {})["_".join(parts[2:])] = {
            "since": since, "reason": reason, "paused": reason.startswith("paused")}
    return out


def _heartbeats(run_dir, prefix):
    from .distributed import alive_heartbeats

    return alive_heartbeats(run_dir, prefix, HEARTBEAT_DEAD)


def describe_player(hb, now):
    """``(state, detail)`` of a player heartbeat."""
    games, st = hb.get("games") or 0, hb.get("state") or "ready"
    bits = []
    if st == "playing":
        state = "playing"
        mins = max(0.0, (now - hb["game_t0"]) / 60) if hb.get("game_t0") else None  # node clocks differ a little
        bits.append(f"game {games + 1}: {hb.get('phase') or 'day 1'}"
                    + (f", {mins:.0f} min in" if mins is not None else ""))
    elif st.startswith("pool full"):
        state, _ = "waiting", bits.append("pool full until the next training step")
    elif st.startswith("loading"):
        state, _ = "loading", bits.append(f"{st.replace('loading ', 'new ')} (had v{hb.get('loaded', 0)})")
    elif st.startswith("holding"):
        state, _ = "holding", bits.append("another game won't fit before the time limit")
    else:
        state = "playing" if st == "between games" else st
    tail = f"{games} done" + (f" (avg {hb['avg_minutes']:g} min)" if hb.get("avg_minutes") else "")
    if hb.get("last"):
        last = hb["last"]
        tail += f", last: {last.get('winner') or 'no winner'} in {last.get('days')}d"
    bits.append(tail)
    if state != "loading":
        bits.append(f"policy v{hb.get('loaded', 0)}")
    if hb.get("gpu_mem_gb"):
        bits.append(f"{hb['gpu_mem_gb']:g} GB")
    return state, "; ".join(bits)


def describe_trainer(hb, lease):
    bits = []
    if hb.get("leader") and lease:
        state = "leading"
        bits.append(f"step {lease['step']} ({lease.get('phase') or '-'})" if lease.get("step") is not None
                    else "waiting for enough games for the next step")
    else:
        state = "training"
        bits.append(f"helping on step {lease['step']}" if lease and lease.get("step") else "waiting for a step")
    if hb.get("accepting") is False:
        state = "holding"
        bits.append("a full step no longer fits before the time limit")
    bits.append(f"{hb.get('tasks_done', 0)} tasks done")
    if hb.get("gpu_mem_gb"):
        bits.append(f"{hb['gpu_mem_gb']:g} GB")
    return state, "; ".join(bits)


def gpu_rows(spec, job, players, trainers, holding, lease, now):
    """One row per GPU of a running chain job: slot, train/play, state, detail."""
    n = job["gpus"] or 1
    ntrain = int(spec.get("env", {}).get("TRAIN_GPUS") or n // 2)
    marks = holding.get(job["id"], {})
    mine_p = [h for h in players if str(h.get("job")) == job["id"]]
    mine_t = [h for h in trainers if str(h.get("job")) == job["id"]]
    p_slot = {str(h["slot"]): h for h in mine_p if h.get("slot") is not None}
    t_slot = {str(h["slot"]): h for h in mine_t if h.get("slot") not in (None, "")}
    p_loose = [h for h in mine_p if h.get("slot") is None]
    t_loose = [h for h in mine_t if h.get("slot") in (None, "")]
    rows = []
    for i in range(n):
        what = "train" if i < ntrain else "play"
        if what == "play":
            hb = p_slot.get(str(i)) or (p_loose.pop(0) if p_loose else None)
        else:
            hb = t_slot.get(str(i)) or (t_loose.pop(0) if t_loose else None)
        mark = marks.get(f"{what}{i}") or marks.get("job")
        if hb is not None:
            state, detail = describe_player(hb, now) if what == "play" else describe_trainer(hb, lease)
        elif mark:
            state = "paused" if mark["paused"] else "IDLE"
            detail = short(mark["reason"]) + (f" (since {mark['since'][11:16]})" if mark["since"] else "")
        elif (job.get("elapsed") or 0) < STARTING_GRACE:
            state, detail = "starting", "loading the model"
        else:
            state, detail = "down", f"no heartbeat: see RUN/logs/{job['role']}_{job['id']}_{what}{i}.log"
        rows.append({"slot": i, "what": what, "state": state, "detail": detail})
    return rows


def game_stats(run_dir, rule, now):
    records = []
    for path in glob.glob(os.path.join(cont.run_paths(run_dir)["timing"], "game_*.jsonl")):
        records += _jsonl(path)
    records.sort(key=lambda r: r.get("t", 0))
    recent = [r for r in records if r.get("winner") is not None and r.get("rule") == rule][-50:]
    hour = [r for r in records if r.get("t", 0) >= now - 3600]
    out = {"total": len(records), "other_rules": sum(r.get("rule") != rule for r in records),
           "last_hour": len(hour), "recent": len(recent),
           "village_win_rate": None, "mean_days": None, "mean_minutes": None, "days": {}}
    if recent:
        out["village_win_rate"] = sum(r["winner"] == "Village" for r in recent) / len(recent)
        out["mean_days"] = sum(r.get("days", 0) for r in recent) / len(recent)
        out["mean_minutes"] = sum(r["seconds"] for r in recent) / len(recent) / 60
        for r in recent:
            out["days"][r.get("days")] = out["days"].get(r.get("days"), 0) + 1
    return out


def training_stats(run_dir, now):
    from .distributed import file_age

    metrics = _jsonl(os.path.join(cont.run_paths(run_dir)["learner"], "metrics.jsonl"))
    steps = _jsonl(os.path.join(cont.run_paths(run_dir)["timing"], "learner_step.jsonl"))
    lease_path = os.path.join(run_dir, "train", "leader.json")
    lease = _read_json(lease_path)
    age = file_age(run_dir, lease_path) if lease else None
    held = bool(lease and not lease.get("released") and age is not None and age <= 180)
    last = metrics[-1] if metrics else None
    return {
        "version": cont.latest_version(run_dir), "steps": len(metrics),
        "steps_last_hour": sum(r.get("t", 0) >= now - 3600 for r in steps),
        "last_step_ago": now - steps[-1]["t"] if steps else None,
        "mean_step_minutes": (sum(r["seconds"] for r in steps[-10:]) / len(steps[-10:]) / 60) if steps else None,
        "lease": lease if held else None,
        "last": None if last is None else {
            "step": last.get("step"), "village_win_rate": last.get("village_win_rate"),
            "mean_days": last.get("mean_days"), "speak_rate": last.get("mean_speak_rate"),
            "kl_ref": (last.get("epochs") or [{}])[-1].get("kl_ref"), "minutes": last.get("step_seconds", 0) / 60},
    }


def snapshot(run_dir, slurm=None, now=None, spec=None):
    """Everything status.sh and the hourly report show (a plain dict)."""
    now = time.time() if now is None else now
    spec = spec or kp.read_launch(run_dir)
    slurm = slurm or kp.Slurm()
    jobs = slurm.jobs(f"wwc-{spec['run_name']}-")
    roles = kp.by_role(spec, jobs)
    holding = read_holding(run_dir)
    players = _heartbeats(run_dir, "player_")
    trainers = _heartbeats(run_dir, "trainer_")
    train = training_stats(run_dir, now)
    config = _read_json(cont.run_paths(run_dir)["config"]) or {}
    rule = config.get("werewolves_win", "parity")
    chains = []
    for c in spec["chains"] + [{"role": g, "partition": None, "gpus": 0, "gap": True} for g in kp.gap_roles(spec)]:
        js = roles.get(c["role"], [])
        running = [dict(j, role=c["role"]) for j in js if j["state"] == "RUNNING"]
        waiting = [j for j in js if j["state"] != "RUNNING"]
        entry = {"role": c["role"], "partition": c["partition"], "target": c["gpus"], "gap": c.get("gap", False),
                 "running": [], "queued": len(waiting), "next": None}
        for j in running:
            entry["running"].append({"id": j["id"], "node": j["node"], "gpus": j["gpus"],
                                     "extra": c.get("gap", False) and kp.is_extra(spec, roles, j),
                                     "kind": kind_of(j["partition"]), "elapsed": j.get("elapsed"),
                                     "time_left": j.get("time_left"),
                                     "rows": gpu_rows(spec, j, players, trainers, holding, train["lease"], now)})
        if waiting and not running:
            head = waiting[0]
            entry["next"] = {"id": head["id"], "reason": head["reason"], "partition": head["partition"],
                             "blocker": None}
            if head["reason"] in ("Priority", "Resources") and hasattr(slurm, "queue_blocker"):
                entry["next"]["blocker"] = slurm.queue_blocker(head["partition"])
        chains.append(entry)
    rows = [r for c in chains for j in c["running"] for r in j["rows"]]
    held = {}
    for c in chains:
        for j in c["running"]:
            held[j["kind"]] = held.get(j["kind"], 0) + j["gpus"]
    target = {}
    for c in spec["chains"]:
        target[kind_of(c["partition"])] = target.get(kind_of(c["partition"]), 0) + c["gpus"]
    paths = cont.run_paths(run_dir)
    pool = cont.pool_games(run_dir)
    age, host, actions = kp.keeper_status(run_dir)
    kstate = _read_json(kp.paths(run_dir)["state"]) or {}
    from . import notify as nt

    ncfg = nt.load_config()
    return {
        "run": spec["run_name"], "now": now, "stop": cont.stop_requested(run_dir),
        "pause": cont.pause_requested(run_dir),
        "rules": {"players": config.get("num_players"), "wolves": config.get("num_werewolves"), "rule": rule},
        "held": held, "target": target, "working": sum(r["state"] in WORKING | {"holding"} for r in rows),
        "idle": sum(r["state"] in ("IDLE", "paused", "down") for r in rows),
        "starting": sum(r["state"] == "starting" for r in rows),
        "chains": chains, "training": train,
        "games": dict(game_stats(run_dir, rule, now), pool=len(pool),
                      fresh=sum(cont.game_version(p) >= train["version"] - 1 for p in pool),
                      archived=len(os.listdir(paths["archive"])) if os.path.isdir(paths["archive"]) else 0),
        "keeper": {"age": age, "host": host, "actions": actions[-5:],
                   "last_report": kstate.get("last_report"), "report_minutes": spec.get("report_minutes", 60)},
        "alerts": nt.provider_of(ncfg) if nt.enabled(ncfg) else None,
    }


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def _gpus_line(s):
    kinds = sorted(set(s["target"]) | set(s["held"]))
    per = ", ".join(f"{k} {s['held'].get(k, 0)}/{s['target'].get(k, 0)}" for k in kinds)
    total_t, total_h = sum(s["target"].values()), sum(s["held"].values())
    extra = f", {s['starting']} starting" if s["starting"] else ""
    return f"{total_h}/{total_t} GPUs held ({per}): {s['working']} working, {s['idle']} idle{extra}"


def _when(iso, now):
    """Slurm's ``2026-11-20T20:36:45`` as ``20:36`` (today) or ``Sat 13:29``."""
    try:
        t = time.mktime(time.strptime(iso, "%Y-%m-%dT%H:%M:%S"))
    except (TypeError, ValueError):
        return iso
    fmt = "%H:%M" if time.strftime("%F", time.localtime(t)) == time.strftime("%F", time.localtime(now)) else "%a %H:%M"
    return time.strftime(fmt, time.localtime(t))


def _waiting_text(nxt, now, with_user=True):
    text = f"waiting ({nxt['reason']})"
    b = nxt.get("blocker")
    if b:
        who = f"{b['user']}'s job {b['job']}" if with_user else "an older job"
        when = f", planned {_when(b['start'], now)}" if b.get("start") else ""
        where = f"; the scheduler keeps {', '.join(b['planned_nodes'])} free for it" if b.get("planned_nodes") else ""
        text += f" behind {who}{when}{where}"
    return text


def _rules_text(r):
    rule = "Werewolves win when every Villager is dead" if r["rule"] == "all_dead" else "Werewolves win at parity"
    return f"{r['players']} players, {r['wolves']} Werewolf; {rule}"


def format_status(s):
    """The full status page (status.sh)."""
    t, g = s["training"], s["games"]
    when = time.strftime("%a %d %b %H:%M", time.localtime(s["now"]))
    flags = ("  STOPPED (stop.sh)" if s["stop"] else "") + ("  PAUSED (stop.sh --pause; start.sh resumes)"
                                                            if s["pause"] else "")
    out = [f"Werewolf RL {s['run']}  |  {when}{flags}",
           f"  rules: {_rules_text(s['rules'])}",
           "",
           f"GPUS  {_gpus_line(s)}",
           f"  {'chain':<7} {'target':<9} {'job':<8} {'node':<13} {'ran':>6} {'left':>6}  state"]
    for c in s["chains"]:
        target = f"{c['target']}x{kind_of(c['partition'])}" if not c["gap"] else "gaps"
        for j in c["running"]:
            target = "extra" if j.get("extra") else target
            states = [r["state"] for r in j["rows"]]
            if states and all(x in ("IDLE", "paused") for x in states) and len(
                    {r["detail"] for r in j["rows"]}) == 1:
                state = f"{states[0]}: {j['rows'][0]['detail']}"
            else:
                counts = {}
                for x in states:
                    counts[x] = counts.get(x, 0) + 1
                state = ", ".join(f"{n} {x}" for x, n in counts.items())
            out.append(f"  {c['role']:<7} {target:<9} {j['id']:<8} {j['node']:<13} {dur(j['elapsed']):>6} "
                       f"{dur(j['time_left']):>6}  {state}" + (f"  (+{c['queued']} queued)" if c["queued"] else ""))
        if not c["running"]:
            nxt = c["next"]
            state = _waiting_text(nxt, s["now"]) if nxt else "no jobs"
            out.append(f"  {c['role']:<7} {target:<9} {nxt['id'] if nxt else '-':<8} {'-':<13} {'-':>6} {'-':>6}  "
                       f"{state}" + (f"  (+{c['queued'] - 1} queued)" if nxt and c["queued"] > 1 else ""))

    out += ["", "EACH GPU"]
    any_rows = False
    for c in s["chains"]:
        for j in c["running"]:
            rows = j["rows"]
            if rows and len({(r["state"], r["detail"]) for r in rows}) == 1 and rows[0]["state"] in ("IDLE", "paused"):
                out.append(f"  {c['role']} {j['id']} on {j['node']}: all {len(rows)} {j['kind']} {rows[0]['state']}"
                           f" - {rows[0]['detail']}")
                any_rows = True
                continue
            for r in rows:
                any_rows = True
                out.append(f"  {c['role']:<6} {j['id']:<7} gpu{r['slot']:<2} {r['what']:<5} {r['state']:<9} {r['detail']}")
    if not any_rows:
        out.append("  (no running jobs)")
    if any(r["state"] == "IDLE" for c in s["chains"] for j in c["running"] for r in j["rows"]):
        out.append(f"  IDLE = the job keeps those GPUs but runs nothing. Fix the cause, then start.sh {s['run']}"
                   " restarts them in place within a minute.")

    out += ["", "TRAINING"]
    line = f"  policy v{t['version']}, {t['steps']} step{'' if t['steps'] == 1 else 's'}"
    if t["steps"]:
        line += (f", {t['steps_last_hour']} in the last hour, last one {dur(t['last_step_ago'])} ago "
                 f"(avg {t['mean_step_minutes']:.0f} min/step)")
    out.append(line)
    if t["lease"]:
        step = t["lease"].get("step")
        out.append(f"  leader {t['lease'].get('owner')}: " + (f"step {step} ({t['lease'].get('phase') or '-'})"
                                                              if step is not None else "waiting for games"))
    if t["last"]:
        x = t["last"]
        out.append(f"  last step {x['step']}: village win {x['village_win_rate']:.2f}, {x['mean_days']:.2f} days, "
                   f"speak rate {x['speak_rate']:.3f}, kl_ref {x['kl_ref'] if x['kl_ref'] is None else round(x['kl_ref'], 4)}, "
                   f"{x['minutes']:.0f} min")

    out += ["", "GAMES"]
    out.append(f"  pool {g['pool']} waiting ({g['fresh']} fresh), {g['archived']} archived; "
               f"{g['last_hour']} played in the last hour, {g['total'] - g['other_rules']} in all"
               + (f" (+{g['other_rules']} under older rules)" if g["other_rules"] else ""))
    if g["recent"]:
        days = ", ".join(f"{d}d: {n}" for d, n in sorted(g["days"].items(), key=lambda kv: kv[0] or 0))
        out.append(f"  last {g['recent']} games: Village won {100 * g['village_win_rate']:.0f}%, "
                   f"{g['mean_days']:.2f} days avg ({days}), {g['mean_minutes']:.1f} min each")
        out.append("  details: python scripts/eval/analyze_werewolf_games.py")
    else:
        out.append("  no games under the current rules yet")

    k = s["keeper"]
    out += ["", "KEEPER / ALERTS"]
    out.append(f"  keeper: " + ("never ran" if k["age"] is None else f"last tick {dur(k['age'])} ago on {k['host']}")
               + ("  <-- NOT RUNNING? rerun start.sh on a login node" if (k["age"] is None or k["age"] > 900)
                  and not s["stop"] else ""))
    for a in k["actions"]:
        out.append(f"    {a}")
    alerts = f"on ({s['alerts']})" if s["alerts"] else "off (set NTFY_TOPIC in ~/.config/werewolf_rl/notify.env)"
    hourly = ("off" if not k["report_minutes"] else
              f"every {k['report_minutes']} min, last sent "
              + (time.strftime("%H:%M", time.localtime(k["last_report"])) if k.get("last_report") else "never"))
    out.append(f"  alerts: {alerts}; GPU report: {hourly}")
    return "\n".join(out)


def format_report(s):
    """The hourly phone report: one block per chain, one line per working GPU."""
    t, g = s["training"], s["games"]
    when = time.strftime("%H:%M", time.localtime(s["now"]))
    flags = (" STOPPED" if s["stop"] else "") + (" PAUSED" if s["pause"] else "")
    out = [f"Werewolf RL {s['run']} {when}{flags}", _gpus_line(s)]
    line = (f"Training: policy v{t['version']}, {t['steps']} step{'' if t['steps'] == 1 else 's'} "
            f"({t['steps_last_hour']} last hour)")
    if t["lease"] and t["lease"].get("step") is not None:
        line += f"; leader on step {t['lease'].get('step')} ({t['lease'].get('phase') or '-'})"
    out.append(line)
    line = f"Games: {g['last_hour']} last hour, pool {g['pool']}"
    if g["recent"]:
        line += (f"; last {g['recent']}: Village {100 * g['village_win_rate']:.0f}%, "
                 f"{g['mean_days']:.1f} days, {g['mean_minutes']:.0f} min")
    out.append(line)
    for c in s["chains"]:
        target = f"{c['target']}x{kind_of(c['partition'])}" if not c["gap"] else "gap"
        if not c["running"]:
            out.append(f"{c['role']} {target}: " + (_waiting_text(c["next"], s["now"], with_user=False) if c["next"]
                                                     else "no jobs"))
            continue
        for j in c["running"]:
            rows = j["rows"]
            label = f"extra {j['gpus']}x{j['kind']}, given back when needed" if j.get("extra") else target
            head = f"{c['role']} {label} {j['node']} ({dur(j['time_left'])} left)"
            if rows and len({(r["state"], r["detail"]) for r in rows}) == 1 and rows[0]["state"] in (
                    "IDLE", "paused", "starting"):
                out.append(f"{head}: all {len(rows)} {rows[0]['state']} - {rows[0]['detail']}")
                continue
            out.append(head)
            for r in rows:
                out.append(f" g{r['slot']} {r['what']} {r['state']}: {r['detail']}")
    return "\n".join(out)[:cont_max()]


def cont_max():
    from .notify import NTFY_MAX_CHARS

    return NTFY_MAX_CHARS


# ----------------------------------------------------------------------------
# Hourly sending (keeper tick)
# ----------------------------------------------------------------------------
def maybe_send_report(run_dir, spec, state, now, slurm=None, send=None):
    """Send the GPU report once per ``report_minutes`` (clock-aligned). Updates ``state``."""
    minutes = spec.get("report_minutes", 60)
    if not minutes:
        return False
    slot = int(now // (minutes * 60))
    if state.get("report_slot") == slot:
        return False
    message = format_report(snapshot(run_dir, slurm=slurm, now=now, spec=spec))
    if send is None:
        from .notify import notify as send
    ok = bool(send("report", message, run_dir, force=True))
    state.update(report_slot=slot, last_report=now, last_report_ok=ok)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("status", help="Print the full status page (status.sh).")
    st.add_argument("--run_dir", required=True)
    hr = sub.add_parser("hourly", help="Print (and with --send, send) the hourly ntfy GPU report.")
    hr.add_argument("--run_dir", required=True)
    hr.add_argument("--send", action="store_true")
    a = ap.parse_args(argv)
    run_dir = os.path.abspath(a.run_dir)
    if not os.path.isfile(kp.paths(run_dir)["json"]):
        raise SystemExit(f"{run_dir} has no launch.json: start the run with start.sh")
    s = snapshot(run_dir)
    if a.cmd == "status":
        print(format_status(s))
        return
    message = format_report(s)
    print(message)
    if a.send:
        from .notify import notify

        print(f"\n[report] {'sent' if notify('report', message, run_dir, force=True) else 'NOT sent'}")


if __name__ == "__main__":
    main()
