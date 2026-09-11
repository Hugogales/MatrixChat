"""Submit the four fresh arms for the 24-hour race and write its registry."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "scripts" / "train_meld_ami_werewolf.sbatch"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", default=None)
    parser.add_argument("--baseline_job", default="267537")
    parser.add_argument("--baseline_run", default="run_000024")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def submit(env, dry_run=False):
    export = ",".join(["ALL"] + [f"{key}={value}" for key, value in env.items()])
    command = ["sbatch", f"--export={export}", str(SBATCH)]
    if dry_run:
        print(" ".join(command))
        return None
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse sbatch output: {result.stdout!r}")
    return match.group(1)


def main():
    args = parse_args()
    now = datetime.now(timezone.utc)
    sweep_id = args.sweep_id or now.strftime("race_%Y%m%d_%H%M%S")
    sweep_dir = ROOT / "logs" / "sweeps" / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "LEARNING_RATE": "5e-5",
        "LR_SCHEDULER": "cosine",
        "WARMUP_STEPS": "200",
        "MAX_GRAD_NORM": "1.0",
        "EVAL_EVERY": "600",
        "CHECKPOINT_EVERY": "500",
        "NUM_STEPS": "5000",
        "NUM_EPOCHS": "2",
        "SEED": "0",
        "SAMPLING_STRATEGY": "probabilistic",
        "UNFREEZE_FIRST_N_LAYERS": "2",
        "UNFREEZE_LAST_N_LAYERS": "2",
        "LAMBDA_REWARD": "0.05",
    }
    definitions = [
        ("B_stabilized", {}),
        ("C_lora_only", {
            "UNFREEZE_FIRST_N_LAYERS": "0",
            "UNFREEZE_LAST_N_LAYERS": "0",
        }),
        ("D_no_reward", {"LAMBDA_REWARD": "0"}),
        ("E_balanced_cycle", {"SAMPLING_STRATEGY": "balanced_cycle"}),
    ]
    arms = [{
        "label": "A_baseline",
        "run_name": args.baseline_run,
        "job_id": args.baseline_job,
        "params": {
            "learning_rate": "1e-4",
            "lr_scheduler": "constant",
            "unfreeze": "2+2",
            "sampling_strategy": "probabilistic",
            "lambda_reward": "0.05",
        },
        "last_probed_checkpoint_step": None,
    }]
    for label, overrides in definitions:
        run_name = f"{sweep_id}_{label}"
        env = {**common, **overrides, "RUN_NAME": run_name}
        job_id = submit(env, dry_run=args.dry_run)
        arms.append({
            "label": label,
            "run_name": run_name,
            "job_id": job_id,
            "params": env,
            "last_probed_checkpoint_step": None,
        })

    state = {
        "sweep_id": sweep_id,
        "started_at": now.isoformat(),
        "deadline": (now + timedelta(hours=24)).isoformat(),
        "manual_decisions_only": True,
        "arms": arms,
    }
    (sweep_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    (sweep_dir / "decision_log.jsonl").touch()
    print(json.dumps(state, indent=2))
    print(f"SWEEP_DIR={sweep_dir}")


if __name__ == "__main__":
    main()
