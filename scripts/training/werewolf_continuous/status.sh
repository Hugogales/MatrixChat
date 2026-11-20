#!/bin/bash
# Show the state of the always-on Werewolf RL loop.
#   scripts/training/werewolf_continuous/status.sh [RUN_NAME]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "$REPO"
RUN_NAME="${1:-${RUN_NAME:-run1}}"
RUN_DIR="${RUN_DIR:-$REPO/checkpoints/werewolf_rl/continuous/$RUN_NAME}"
echo "== jobs"
squeue -u "$USER" -o "%.10i %.10P %.26j %.9T %.10M %.10L %R" | awk -v p="wwc-$RUN_NAME-" 'NR==1 || index($3, p)==1'

python - "$RUN_DIR" <<'EOF'
import json, os, sys
from training.werewolf_rl import continuous as c
run = sys.argv[1]
if not os.path.isdir(run):
    sys.exit(f"no run directory {run}")
paths = c.run_paths(run)
v = c.latest_version(run)
pool = c.pool_games(run)
print(f"== policy v{v}   STOP={'yes' if c.stop_requested(run) else 'no'}")
print(f"== pool: {len(pool)} waiting, {sum(c.game_version(p) >= v - 1 for p in pool)} fresh; "
      f"{len(os.listdir(paths['archive']))} archived")
for kind in sorted(f[:-6] for f in os.listdir(paths["timing"]) if f.endswith(".jsonl")):
    avg = c.average_time(run, kind)
    n = sum(1 for _ in open(os.path.join(paths["timing"], kind + ".jsonl")))
    print(f"== avg {kind}: {avg / 60:.1f} min over {n}")
from training.werewolf_rl.distributed import alive_heartbeats

try:
    lease = json.load(open(os.path.join(run, "train", "leader.json")))
except (FileNotFoundError, ValueError):
    lease = None
trainers = alive_heartbeats(run, "trainer_", 120)
players = alive_heartbeats(run, "player_", 120)
from training.werewolf_rl.distributed import file_age
lease_age = file_age(run, os.path.join(run, "train", "leader.json"))
held = lease and not lease.get("released") and lease.get("expires", 1) != 0 and lease_age is not None and lease_age <= 180
lead = f"{lease['owner']} (step {lease.get('step')}, {lease.get('phase', '-')})" if held else "none"
print(f"== trainers alive: {len(trainers)} ({', '.join(sorted(h['kind'] + ('*' if h.get('role') == 'extra' else '') for h in trainers)) or '-'}); "
      f"players alive: {len(players)} ({', '.join(sorted(h['kind'] + ('*' if h.get('role') == 'extra' else '') for h in players)) or '-'}); "
      f"leader: {lead}   (* = borrowed)")
for role in ("h100", "v100"):
    p = os.path.join(paths["chain"], f"{role}.json")
    if os.path.isfile(p):
        s = json.load(open(p))
        last = s.get("history", [])[-1:] or [{}]
        print(f"== chain {role}: crashes in a row={s.get('consecutive_crashes', 0)}"
              f"{' STOPPED: ' + s['reason'] if s.get('stopped') else ''}; last job {last[0].get('job', '-')} "
              f"ended {last[0].get('state', '-')}")
from training.werewolf_rl import notify as nt
ncfg = nt.load_config()
if nt.enabled(ncfg):
    if nt.uses_relay(ncfg):
        age, host, queued = nt.relay_status(ncfg)
        alive = "never ran" if age is None else f"last ran {age / 60:.0f} min ago on {host}"
        warn = "  <-- NOT RUNNING? (python -m training.werewolf_rl.notify install-cron on the login node)" \
            if age is None or age > 600 else ""
        print(f"== alerts: on (gateway relay {alive}, {queued} queued){warn}")
    else:
        print(f"== alerts: on ({nt.provider_of(ncfg)}, sent straight from the jobs)")
else:
    print("== alerts: off (set NTFY_TOPIC or NOTIFY_PHONE in ~/.config/werewolf_rl/notify.env)")
m = os.path.join(paths["learner"], "metrics.jsonl")
if os.path.isfile(m):
    lines = open(m).read().splitlines()
    if lines:
        r = json.loads(lines[-1])
        e = r["epochs"][-1]
        print(f"== last step {r['step']} (trained v{r.get('policy_version', '?')} on games from "
              f"v{r.get('game_versions')}, {len(r.get('trainers', []))} trainer(s), "
              f"{r.get('reassigned_tasks', 0)} reassigned): village win {r['village_win_rate']:.2f}, "
              f"speak rate {r['mean_speak_rate']:.3f}, discussion cols {r['mean_discussion_columns_per_game']:.0f}, "
              f"kl_ref {e['kl_ref']:.4f}, critic {e['critic_loss']:.3f}, {r['step_seconds'] / 60:.1f} min")
EOF