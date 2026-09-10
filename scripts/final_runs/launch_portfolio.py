#!/usr/bin/env python3
"""Materialize and launch a staged final-run portfolio plus held-out evals."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--launch", action="store_true")
    return parser.parse_args(argv)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def materialize_config(spec: dict[str, Any], stage_steps: int) -> dict[str, Any]:
    source = (_ROOT / spec["source_config"]).resolve()
    config = json.loads(source.read_text(encoding="utf-8"))
    config.update(spec.get("overrides") or {})
    config.update(
        {
            "_stop_requested": False,
            "run_name": spec["run_name"],
            "seed": int(spec["seed"]),
            "processed_data_dir": spec["processed_data_dir"],
            "resume_from": spec.get("resume_from"),
            "num_steps": int(stage_steps),
            "num_epochs": max(int(config.get("num_epochs") or 0), 20),
            "max_runtime_minutes": 0.0,
            "val_fraction": 0.05,
            "checkpoint_every": 250,
            "eval_every": 250,
            "probe_every": 250,
            "probe_max_new_tokens": 16,
            "sample_every": 0,
            "save_checkpoint": True,
            "dry_run": False,
            "device": "cuda",
        }
    )
    return config


def _sbatch(command: list[str], launch: bool) -> str:
    printable = " ".join(command)
    if not launch:
        print(f"DRY RUN: {printable}")
        return "DRY_RUN"
    result = subprocess.run(
        command, cwd=_ROOT, check=True, text=True, capture_output=True
    )
    job_id = result.stdout.strip().split(";")[0]
    print(f"submitted {job_id}: {printable}")
    return job_id


def main(argv=None) -> None:
    args = parse_args(argv)
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_steps = int(manifest["stage_steps"])
    contract = Path(manifest["development_contract"]).expanduser().resolve()
    output_root = Path(manifest["output_root"]).expanduser().resolve()
    specs = manifest["runs"]
    names = [spec["run_name"] for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("run_name values must be unique")
    if any(spec["gpu"] not in {"v100", "h100"} for spec in specs):
        raise ValueError("gpu must be v100 or h100")

    launch_record = {
        "manifest": str(manifest_path),
        "stage_steps": stage_steps,
        "development_contract": str(contract),
        "runs": [],
    }
    for spec in specs:
        run_name = spec["run_name"]
        config = materialize_config(spec, stage_steps)
        config_path = output_root / "configs" / f"{run_name}.json"
        _atomic_json(config_path, config)

        gpu = spec["gpu"]
        partition = "dgxh100" if gpu == "h100" else "dgx"
        gres = f"gpu:{gpu}:1"
        memory = "80G" if gpu == "h100" else "56G"
        train_name = f"fin_{run_name}"[:64]
        train_job = _sbatch(
            [
                "sbatch",
                "--parsable",
                f"--job-name={train_name}",
                f"--partition={partition}",
                f"--gres={gres}",
                f"--mem={memory}",
                f"--export=ALL,CONFIG_PATH={config_path}",
                "scripts/final_runs/train_final.sbatch",
            ],
            args.launch,
        )

        # Root, not "last": evaluate_final_checkpoint.sbatch resolves the
        # actual leaf dir at eval-time (prefers best_probe/ > root
        # best-val-loss > last/ terminal state) -- see
        # training/checkpoint.py::CheckpointManager.on_probe.
        checkpoint_dir = _ROOT / "checkpoints" / run_name
        eval_dir = output_root / "evaluations" / run_name / f"step_{stage_steps}"
        eval_name = f"eval_{run_name}"[:64]
        eval_command = [
            "sbatch",
            "--parsable",
            f"--job-name={eval_name}",
            f"--partition={partition}",
            f"--gres={gres}",
            f"--mem={memory}",
            f"--export=ALL,CONTRACT={contract},CHECKPOINT_DIR={checkpoint_dir},OUTPUT_DIR={eval_dir}",
            "scripts/final_runs/evaluate_final_checkpoint.sbatch",
        ]
        if train_job != "DRY_RUN":
            eval_command.insert(2, f"--dependency=afterok:{train_job}")
        eval_job = _sbatch(eval_command, args.launch)
        launch_record["runs"].append(
            {
                "run_name": run_name,
                "gpu": gpu,
                "config": str(config_path),
                "train_job_id": train_job,
                "eval_job_id": eval_job,
                "checkpoint_dir": str(checkpoint_dir),
                "evaluation_dir": str(eval_dir),
            }
        )

    _atomic_json(output_root / "launch_record.json", launch_record)
    print(json.dumps(launch_record, indent=2))


if __name__ == "__main__":
    main()
