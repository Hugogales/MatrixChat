"""Tests for weight-only init_from checkpoint helpers."""

from pathlib import Path

import torch

from training.checkpoint import resolve_init_checkpoint_dir


def test_resolve_init_checkpoint_dir_falls_back_when_best_probe_missing(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    payload = {
        "inactive_embedding": {"weight": torch.zeros(1, 2)},
        "activity_head": {"weight": torch.zeros(1, 2), "bias": torch.zeros(1)},
    }
    torch.save(payload, root / "model_state.pt")
    resolved = resolve_init_checkpoint_dir(str(root), "best_probe")
    assert resolved == str(root)


def test_resolve_init_checkpoint_dir_prefers_best_probe(tmp_path: Path):
    root = tmp_path / "run"
    best = root / "best_probe"
    best.mkdir(parents=True)
    payload = {
        "inactive_embedding": {"weight": torch.zeros(1, 2)},
        "activity_head": {"weight": torch.zeros(1, 2), "bias": torch.zeros(1)},
    }
    torch.save(payload, best / "model_state.pt")
    resolved = resolve_init_checkpoint_dir(str(root), "best_probe")
    assert resolved == str(best)
