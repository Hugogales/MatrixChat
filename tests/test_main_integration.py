"""End-to-end smoke test of main.main() with the tiny model.

Runs in a temporary working directory so run artifacts do not pollute the repo.
"""

import json
import os

import pytest


def test_main_smoke_creates_artifacts(tmp_path, monkeypatch):
    import main

    monkeypatch.chdir(tmp_path)

    argv = [
        "--use_tiny_model", "true",
        "--num_steps", "2",
        "--device", "cpu",
        "--torch_dtype", "float32",
        "--num_agents", "2",
        "--seq_len", "6",
        "--batch_size", "2",
    ]
    main.main(argv)

    run_json = tmp_path / "hyperparameters" / "run_000001.json"
    assert run_json.exists()
    payload = json.loads(run_json.read_text())
    assert payload["num_steps"] == 2

    metrics = tmp_path / "logs" / "run_000001" / "metrics.jsonl"
    assert metrics.exists()
    lines = [json.loads(l) for l in metrics.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    assert all("loss" in row for row in lines)

    assert (tmp_path / "checkpoints" / "run_000001").is_dir()


def test_main_dry_run(tmp_path, monkeypatch):
    import main

    monkeypatch.chdir(tmp_path)
    main.main([
        "--use_tiny_model", "true",
        "--device", "cpu",
        "--torch_dtype", "float32",
        "--dry_run", "true",
    ])
    # Dry run still writes hyperparameters but skips the training loop / metrics.
    assert (tmp_path / "hyperparameters" / "run_000001.json").exists()
    assert not (tmp_path / "logs" / "run_000001" / "metrics.jsonl").exists()
