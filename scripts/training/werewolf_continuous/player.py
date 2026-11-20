"""One game-playing process of the always-on Werewolf RL loop (one per GPU).

Plays recorded games with the newest published policy and drops each finished
game into ``RUN/pool/incoming`` for the learner. Between games it

* reloads the policy when the learner has published a newer version,
* waits while the pool already holds ``--max_pool`` fresh games (they would
  only go stale),
* checks that another game fits in the job's remaining time
  (average game time on this GPU type * (1 + ``--time_margin``)); if not, it
  holds the GPU until Slurm's time limit ends the job.

``--role extra`` (GPUs borrowed beyond the reserved set) additionally checks
the queue every ``--poll`` seconds: as soon as another user's job, or one of
our own main-chain jobs, is waiting for GPUs on this partition, the current
game is abandoned and the process exits so the GPU is freed.

A ``RUN/STOP`` file makes it exit after the current game.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
import traceback

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.werewolf_rl import continuous as cont  # noqa: E402
from training.werewolf_rl.game import add_game_args, load_game_model, play_game, single_token_names  # noqa: E402


class GameAborted(Exception):
    pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint_dir", default="checkpoints/checkpoint")
    p.add_argument("--hyperparameters", default="hyperparameters/adaptcleanv3_cand_00003.json")
    p.add_argument("--model_path", default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--slot", type=int, default=0, help="GPU slot within the job (for file names/logs).")
    p.add_argument("--role", choices=["main", "extra"], default="main")
    p.add_argument("--partition", default=os.environ.get("SLURM_JOB_PARTITION", ""))
    p.add_argument("--chain_names", default="",
                   help="Comma-separated job names of the main chain; extras also yield to these.")
    p.add_argument("--time_margin", type=float, default=0.05)
    p.add_argument("--max_pool", type=int, default=64, help="Pause while this many fresh games are waiting.")
    p.add_argument("--max_lag", type=int, default=1, help="Games this many versions behind still count as fresh.")
    p.add_argument("--poll", type=float, default=60.0, help="Seconds between queue checks (extras).")
    p.add_argument("--max_consecutive_failures", type=int, default=3)
    p.add_argument("--device", default="cuda")
    add_game_args(p)
    return p.parse_args(argv)


def gpu_kind():
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0).lower()
    return next((k for k in ("h100", "a100", "v100", "t4") if k in name), "gpu")


def main(argv=None):
    args = parse_args(argv)
    run = args.run_dir
    cont.ensure_run_dir(run)
    if not cont.apply_game_config(run, args):
        raise SystemExit(f"{cont.run_paths(run)['config']} missing; start the run with start.sh")
    job = os.environ.get("SLURM_JOB_ID", "local")
    kind = gpu_kind()
    label = f"{kind}j{job}s{args.slot}"

    def log(msg):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')} {label}] {msg}", flush=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"CUDA unavailable (torch {torch.__version__}, built for CUDA {torch.version.cuda}); "
                         "refusing to run on CPU")
    device = torch.device(args.device)
    dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
             else torch.float16 if device.type == "cuda" else torch.float32)
    model, tokenizer, _, _ = load_game_model(args.model_path, args.checkpoint_dir, args.num_players, "on",
                                             device, dtype, hyperparameters=args.hyperparameters,
                                             log=lambda m: log(m))
    model.eval()
    loaded = 0
    name_pool = single_token_names(tokenizer, args)
    own_names = tuple(n for n in args.chain_names.split(",") if n)

    # Extras: a background thread watches the queue and flags when to give the GPU back.
    yield_reason = {"why": None}
    if args.role == "extra" and args.partition:
        def watch():
            while yield_reason["why"] is None:
                waiting = cont.waiting_jobs(args.partition, own_names)
                if waiting:
                    w = waiting[0]
                    yield_reason["why"] = (f"{len(waiting)} job(s) waiting on {args.partition} "
                                           f"(e.g. {w['job']} of {w['user']}, {w['reason']})")
                    return
                time.sleep(args.poll)
        threading.Thread(target=watch, daemon=True).start()

    def abort_check():
        if yield_reason["why"]:
            raise GameAborted(yield_reason["why"])

    def release_reason():
        if cont.stop_requested(run):
            return "STOP file"
        return yield_reason["why"]

    from training.werewolf_rl.distributed import Heartbeat

    heartbeat = Heartbeat(run, f"player_{label}", {"role": args.role, "kind": kind, "job": job})

    def on_term(signum, _frame):  # Slurm time limit / scancel: stop advertising at once
        log(f"got signal {signum}; leaving")
        heartbeat.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, on_term)
    try:
        return _play_loop(args, run, model, tokenizer, name_pool, kind, label, loaded, device,
                          abort_check, release_reason, log)
    finally:
        heartbeat.stop()


def _play_loop(args, run, model, tokenizer, name_pool, kind, label, loaded, device, abort_check,
               release_reason, log):
    failures = 0
    games = 0
    log(f"ready: role={args.role} partition={args.partition or '-'}")
    while True:
        reason = release_reason()
        if reason:
            log(f"exiting: {reason}")
            return 0
        version = cont.latest_version(run)
        if version > loaded:
            try:
                cont.load_weights(run, model, version)
                loaded = version
                log(f"loaded policy v{version}")
            except (FileNotFoundError, EOFError, RuntimeError) as exc:  # being replaced right now; retry
                log(f"could not load v{version} yet ({exc}); retrying")
                time.sleep(10)
                continue
        if cont.fresh_count(run, loaded - args.max_lag) >= args.max_pool:
            time.sleep(30)
            continue
        avg = cont.average_time(run, f"game_{kind}")
        if not cont.fits(cont.remaining_seconds(), avg, args.time_margin):
            reason = cont.idle_until_killed(release_reason, poll=min(args.poll, 60), log=log)
            log(f"exiting: {reason}")
            return 0
        seed = int.from_bytes(os.urandom(6), "little")
        t0 = time.time()
        try:
            with torch.no_grad():
                result = play_game(model, tokenizer, args, seed, record=True, log=lambda *a: None,
                                   name_pool=name_pool, abort_check=abort_check)
        except GameAborted as exc:
            log(f"abandoned the current game and exiting to free the GPU: {exc}")
            return 0
        except Exception:  # noqa: BLE001 - one bad game must not end the player
            failures += 1
            log(f"game failed ({failures}/{args.max_consecutive_failures}):\n{traceback.format_exc()}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if failures >= args.max_consecutive_failures:
                return 1
            continue
        seconds = time.time() - t0
        failures = 0
        games += 1
        cont.write_game(run, result, loaded, label, seed, seconds)
        cont.record_time(run, f"game_{kind}", seconds, columns=result["matrix_columns"])
        log(f"game {games}: v{loaded} seed={seed} {seconds:.0f}s winner={result['winner']} "
            f"days={result['days_played']} columns={result['matrix_columns']}")
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
