"""One training GPU of the always-on Werewolf RL loop (see training/werewolf_rl/distributed.py).

All training GPUs -- the reserved H100 and V100 halves and any borrowed
ones -- run this. Whichever holds the lease leads (claims games, assigns
them, sums gradients, steps the optimizer, publishes weights, checkpoints);
the rest work on what it assigns. ``--role extra`` gives the GPU back as soon
as anyone else is waiting on its partition.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.werewolf_rl import continuous as cont  # noqa: E402
from training.werewolf_rl.core import add_rl_args, pick_dtype  # noqa: E402
from training.werewolf_rl.distributed import TrainerWorker  # noqa: E402
from training.werewolf_rl.game import add_game_args, load_game_model  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint_dir", default="checkpoints/checkpoint")
    p.add_argument("--hyperparameters", default="hyperparameters/adaptcleanv3_cand_00003.json")
    p.add_argument("--model_path", default="models/Qwen3-4B-Instruct-2507")
    p.add_argument("--role", choices=["main", "extra"], default="main")
    p.add_argument("--partition", default=os.environ.get("SLURM_JOB_PARTITION", ""))
    p.add_argument("--chain_names", default="", help="Comma-separated main-chain job names (extras yield to them).")
    p.add_argument("--games_per_step", type=int, default=16)
    p.add_argument("--max_game_lag", type=int, default=1)
    p.add_argument("--time_margin", type=float, default=0.05)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--keep_last", type=int, default=3)
    p.add_argument("--keep_rollouts", action="store_true")
    p.add_argument("--extra_partitions", default="")
    p.add_argument("--max_extra", type=int, default=0)
    p.add_argument("--extra_sbatch", default="scripts/training/werewolf_continuous/extra.sbatch")
    p.add_argument("--extra_name", default="")
    p.add_argument("--extras_every", type=float, default=120.0)
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--device", default="cuda")
    add_rl_args(p)
    add_game_args(p)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run = args.run_dir
    if not cont.apply_game_config(run, args):  # sampling must match the players exactly
        raise SystemExit(f"{cont.run_paths(run)['config']} missing; start the run with start.sh")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"CUDA unavailable (torch {torch.__version__}, built for CUDA {torch.version.cuda}); "
                         "refusing to run on CPU")
    device = torch.device(args.device)
    dtype = pick_dtype(args.dtype, device)

    def log(*parts):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]", *parts, flush=True)

    model, tokenizer, cfg, saved = load_game_model(args.model_path, args.checkpoint_dir, args.num_players, "on",
                                                   device, dtype, hyperparameters=args.hyperparameters, log=log,
                                                   keep_pretrained=args.kl_reference == "qwen")
    worker = TrainerWorker(run, model, tokenizer, cfg, saved, args, device, dtype, role=args.role,
                           partition=args.partition, chain_names=[x for x in args.chain_names.split(",") if x],
                           log=log)
    log(f"[trainer] {worker.wid} ready (role={args.role}, dtype={dtype})")

    def on_term(signum, _frame):  # Slurm time limit / scancel: leave at once so the leader reassigns
        log(f"[trainer] {worker.wid} got signal {signum}; leaving")
        worker.leave()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, on_term)
    reason = worker.run_forever()
    log(f"[trainer] {worker.wid} exiting: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
