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
        "--save_checkpoint", "true",
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

    assert (tmp_path / "checkpoints" / "run_000001" / "model_state.pt").is_file()


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


class _FakeHFDataset:
    """Minimal stand-in for datasets.Dataset's shuffle/select/__len__ API."""

    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def shuffle(self, seed):
        rng_ordered = sorted(self.items, key=lambda x: (hash((x, seed)),))
        return _FakeHFDataset(rng_ordered)

    def select(self, indices):
        return _FakeHFDataset([self.items[i] for i in indices])


def test_cap_val_examples_noop_when_under_cap():
    import main

    val_ds = _FakeHFDataset(range(10))
    result = main.cap_val_examples(val_ds, max_val_examples=2000, seed=0)
    assert result is val_ds


def test_cap_val_examples_noop_when_cap_disabled():
    """Regression guard: this is the Bazinga-validation-blowup fix's escape
    hatch -- max_val_examples<=0 must fully disable the cap."""
    import main

    val_ds = _FakeHFDataset(range(50000))
    result = main.cap_val_examples(val_ds, max_val_examples=0, seed=0)
    assert result is val_ds


def test_cap_val_examples_subsamples_oversized_predefined_split():
    """Regression guard for 2026-08-26: Bazinga's undeduplicated predefined
    validation split ballooned the shared validation set from 1306 to 45919
    examples, making a single eval_every pass take ~15h instead of ~20min.
    cap_val_examples must bring any oversized validation set down to the cap."""
    import main

    val_ds = _FakeHFDataset(range(45919))
    result = main.cap_val_examples(val_ds, max_val_examples=2000, seed=176106)
    assert len(result) == 2000


def test_cap_val_examples_deterministic_for_same_seed():
    import main

    val_ds = _FakeHFDataset(range(45919))
    a = main.cap_val_examples(val_ds, max_val_examples=2000, seed=176106)
    b = main.cap_val_examples(val_ds, max_val_examples=2000, seed=176106)
    assert a.items == b.items


def test_cap_val_examples_handles_falsy_val_ds():
    import main

    assert main.cap_val_examples(None, max_val_examples=2000, seed=0) is None
