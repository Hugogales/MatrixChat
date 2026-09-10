#!/usr/bin/env python3
"""Run ``main.py`` from a validated, frozen JSON training configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import main as training_main
from training.config import build_parser


_IGNORED_METADATA_KEYS = {"_stop_requested"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def _option_actions() -> dict[str, argparse.Action]:
    actions: dict[str, argparse.Action] = {}
    for action in build_parser()._actions:
        if action.dest != "help":
            actions[action.dest] = action
    return actions


def _serialise_value(
    name: str, value: Any, action: argparse.Action
) -> list[str]:
    if name == "dataset_weights" and isinstance(value, dict):
        value = ",".join(f"{source}={weight}" for source, weight in value.items())
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, (list, tuple)):
        if action.nargs in ("+", "*", argparse.REMAINDER):
            return [str(item) for item in value]
        return [",".join(str(item) for item in value)]
    return [str(value)]


def config_to_argv(config: dict[str, Any]) -> list[str]:
    """Convert a saved hyperparameter JSON into ``main.py`` CLI arguments."""
    actions = _option_actions()
    unknown = sorted(
        key
        for key, value in config.items()
        if key not in actions
        and key not in _IGNORED_METADATA_KEYS
        and value is not None
    )
    if unknown:
        raise ValueError(f"Unknown training configuration keys: {unknown}")

    argv: list[str] = []
    for name, action in actions.items():
        if name not in config or config[name] is None:
            continue
        values = _serialise_value(name, config[name], action)
        if not values and action.nargs not in ("*", argparse.REMAINDER):
            raise ValueError(f"Configuration value for {name!r} cannot be empty")
        argv.append(f"--{name}")
        argv.extend(values)
    return argv


def main(argv=None) -> None:
    args = parse_args(argv)
    path = Path(args.config).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("training config must be a JSON object")
    print(f"[final-run] frozen config: {path}")
    training_main.main(config_to_argv(config))


if __name__ == "__main__":
    main()
