"""Run-id and hyperparameter tracking.

Each run gets a zero-padded id like ``run_000001``. Hyperparameters are saved as
JSON under ``hyperparameters/`` and matching ``logs/{run_id}/`` and
``checkpoints/{run_id}/`` directories are created.
"""

from __future__ import annotations

import argparse
import json
import os
import re

_RUN_RE = re.compile(r"^run_(\d{6})\.json$")


def get_next_run_id(base_dir: str = "hyperparameters") -> str:
    """Return the next run id, e.g. ``run_000003``.

    Scans ``base_dir`` for files named ``run_NNNNNN.json`` and returns the next
    sequential id. Returns ``run_000001`` when none exist.
    """
    max_id = 0
    if os.path.isdir(base_dir):
        for name in os.listdir(base_dir):
            match = _RUN_RE.match(name)
            if match:
                max_id = max(max_id, int(match.group(1)))
    return f"run_{max_id + 1:06d}"


def save_hyperparams(
    args,
    run_id: str,
    base_dir: str = "hyperparameters",
    logs_dir: str = "logs",
    checkpoints_dir: str = "checkpoints",
) -> str:
    """Save args as JSON and create run-specific log/checkpoint directories.

    Returns the path to the saved hyperparameter JSON file.
    """
    os.makedirs(base_dir, exist_ok=True)

    if isinstance(args, argparse.Namespace):
        payload = vars(args)
    elif isinstance(args, dict):
        payload = args
    else:
        payload = vars(args)

    json_path = os.path.join(base_dir, f"{run_id}.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)

    os.makedirs(os.path.join(logs_dir, run_id), exist_ok=True)
    os.makedirs(os.path.join(checkpoints_dir, run_id), exist_ok=True)

    return json_path
