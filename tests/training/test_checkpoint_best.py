"""Tests for best-validation checkpoint selection and save/load helpers."""

from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from training.checkpoint import (
    CheckpointManager,
    apply_checkpoint_to_model,
    apply_unfrozen_base_parameters,
    collect_unfrozen_base_parameters,
    load_checkpoint_state,
    load_training_state,
    save_checkpoint,
    save_training_state,
)


class _DummyBase(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.layer = nn.Linear(hidden_size, hidden_size)

    def get_input_embeddings(self):
        return nn.Embedding(8, 4)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _DummyMatrixModel(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.base_model = _DummyBase(hidden_size)
        self.agent_embeddings = nn.Embedding(8, hidden_size)
        self.channel_embeddings = None
        self.inactive_embedding = nn.Embedding(1, hidden_size)
        self.activity_head = nn.Linear(hidden_size, 1)


def _args(**overrides):
    defaults = dict(lora_enable=False)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_model(hidden_size=4):
    return _DummyMatrixModel(hidden_size)


def test_checkpoint_manager_saves_best_on_improvement(tmp_path):
    model = _make_model()
    args = _args()
    root = tmp_path / "run"
    mgr = CheckpointManager(str(root), model, args)
    mgr.set_validation_split(True)

    assert mgr.on_validation(2.0, epoch=0, step=1, val_components={"content_loss": 1.0}) is True
    assert (root / "model_state.pt").is_file()
    assert (root / "best_checkpoint.json").is_file()
    best = json.loads((root / "best_checkpoint.json").read_text())
    assert best["best_val_loss"] == 2.0
    assert best["step"] == 1

    # Non-improvement should not overwrite metadata step.
    assert mgr.on_validation(2.5, epoch=0, step=2) is False
    best2 = json.loads((root / "best_checkpoint.json").read_text())
    assert best2["step"] == 1
    assert best2["best_val_loss"] == 2.0

    # Strict improvement updates best.
    assert mgr.on_validation(1.5, epoch=0, step=3, val_components={"content_loss": 0.5}) is True
    best3 = json.loads((root / "best_checkpoint.json").read_text())
    assert best3["step"] == 3
    assert best3["best_val_loss"] == 1.5


def test_checkpoint_manager_finalize_with_validation_saves_last(tmp_path):
    model = _make_model()
    args = _args()
    root = tmp_path / "run"
    mgr = CheckpointManager(str(root), model, args)
    mgr.set_validation_split(True)
    mgr.on_validation(1.0, epoch=0, step=5)

    # Mutate weights so last differs from best.
    model.activity_head.bias.data.fill_(99.0)
    mgr.finalize(epoch=1, step=10, last_val_loss=1.2)

    assert (root / "model_state.pt").is_file()
    assert (root / "last" / "model_state.pt").is_file()

    best_state = torch.load(root / "model_state.pt", map_location="cpu")
    last_state = torch.load(root / "last" / "model_state.pt", map_location="cpu")
    assert best_state["metadata"]["kind"] == "best"
    assert best_state["metadata"]["step"] == 5
    assert last_state["metadata"]["kind"] == "last"
    assert last_state["metadata"]["step"] == 10
    assert not torch.allclose(
        best_state["activity_head"]["bias"],
        last_state["activity_head"]["bias"],
    )


def test_checkpoint_manager_finalize_without_validation_saves_root(tmp_path):
    model = _make_model()
    args = _args()
    root = tmp_path / "run"
    mgr = CheckpointManager(str(root), model, args)
    mgr.finalize(epoch=0, step=3)

    assert (root / "model_state.pt").is_file()
    assert not (root / "last" / "model_state.pt").exists()
    state = torch.load(root / "model_state.pt", map_location="cpu")
    assert state["metadata"]["kind"] == "final"


def test_save_checkpoint_includes_metadata(tmp_path):
    model = _make_model()
    args = _args()
    metadata = {"kind": "best", "epoch": 2, "step": 100, "val_loss": 0.42}
    path = save_checkpoint(model, args, str(tmp_path / "ckpt"), metadata)
    state = torch.load(path, map_location="cpu")
    assert state["metadata"] == metadata
    assert "inactive_embedding" in state
    assert "activity_head" in state


def test_collect_and_restore_unfrozen_base_parameters():
    model = _make_model()
    for p in model.base_model.parameters():
        p.requires_grad = False
    model.base_model.layer.weight.requires_grad = True
    model.base_model.layer.weight.data.fill_(7.0)

    collected = collect_unfrozen_base_parameters(model)
    assert collected is not None
    assert "layer.weight" in collected

    model.base_model.layer.weight.data.zero_()
    missing = apply_unfrozen_base_parameters(model, collected)
    assert missing == []
    assert torch.allclose(model.base_model.layer.weight.data, torch.full_like(model.base_model.layer.weight.data, 7.0))


def test_apply_checkpoint_to_model_restores_matrix_interface(tmp_path):
    model_a = _make_model()
    model_a.inactive_embedding.weight.data.fill_(3.0)
    model_a.activity_head.bias.data.fill_(2.0)
    args = _args()
    ckpt_dir = tmp_path / "ckpt"
    save_checkpoint(model_a, args, str(ckpt_dir), {"kind": "best", "epoch": 0, "step": 1})

    model_b = _make_model()
    model_b.inactive_embedding.weight.data.zero_()
    model_b.activity_head.bias.data.zero_()
    apply_checkpoint_to_model(model_b, str(ckpt_dir))
    assert torch.allclose(model_b.inactive_embedding.weight.data, torch.full_like(model_b.inactive_embedding.weight.data, 3.0))
    assert torch.allclose(model_b.activity_head.bias.data, torch.full_like(model_b.activity_head.bias.data, 2.0))


def test_old_format_checkpoint_rejected(tmp_path):
    ckpt_dir = tmp_path / "old"
    ckpt_dir.mkdir()
    torch.save(
        {
            "matrix_config": {},
            "agent_embeddings": {"weight": torch.zeros(1, 1)},
            "silence_embedding": {"weight": torch.zeros(1, 1)},
        },
        ckpt_dir / "model_state.pt",
    )
    with pytest.raises(ValueError) as excinfo:
        load_checkpoint_state(str(ckpt_dir))
    msg = str(excinfo.value)
    assert "inactive_embedding" in msg
    assert "activity_head" in msg


def _probe(**overrides):
    defaults = dict(
        clean_handoff_rate=0.5,
        no_listener_response_rate=0.2,
        nonoverlap_listener_rate=0.3,
        overlap_rate=0.1,
        unwanted_interruption_rate=0.05,
        repeated_4gram_fraction=0.1,
    )
    defaults.update(overrides)
    return defaults


def test_checkpoint_manager_saves_best_probe_on_improvement(tmp_path):
    model = _make_model()
    args = _args()
    root = tmp_path / "run"
    mgr = CheckpointManager(str(root), model, args)

    assert mgr.on_probe(_probe(clean_handoff_rate=0.5), epoch=0, step=10) is True
    assert (root / "best_probe" / "model_state.pt").is_file()
    summary = json.loads((root / "best_probe" / "best_probe.json").read_text())
    assert summary["step"] == 10

    # A worse probe must not overwrite the saved best.
    assert mgr.on_probe(_probe(clean_handoff_rate=0.1, overlap_rate=0.6), epoch=0, step=20) is False
    summary2 = json.loads((root / "best_probe" / "best_probe.json").read_text())
    assert summary2["step"] == 10

    # A strictly better probe (higher clean handoff, lower overlap) updates it.
    assert mgr.on_probe(_probe(clean_handoff_rate=0.8, overlap_rate=0.02), epoch=0, step=30) is True
    summary3 = json.loads((root / "best_probe" / "best_probe.json").read_text())
    assert summary3["step"] == 30


def test_checkpoint_manager_best_probe_independent_of_best_val_loss(tmp_path):
    """The val-loss-best step and the probe-quality-best step can differ."""
    model = _make_model()
    args = _args()
    root = tmp_path / "run"
    mgr = CheckpointManager(str(root), model, args)
    mgr.set_validation_split(True)

    mgr.on_validation(2.0, epoch=0, step=10, val_components={"content_loss": 1.0})
    mgr.on_probe(_probe(clean_handoff_rate=0.9, overlap_rate=0.0), epoch=0, step=10)

    # Loss keeps improving (misleading), but turn-taking quality degrades.
    mgr.on_validation(1.0, epoch=0, step=20, val_components={"content_loss": 0.5})
    mgr.on_probe(_probe(clean_handoff_rate=0.1, overlap_rate=0.5), epoch=0, step=20)

    assert mgr.best_metadata["step"] == 20  # best-by-loss moved on
    assert mgr.best_probe_metadata["step"] == 10  # best-by-turn-taking stayed put
    best_probe_summary = json.loads((root / "best_probe" / "best_probe.json").read_text())
    assert best_probe_summary["step"] == 10


def test_checkpoint_manager_ignores_non_finite_val_loss(tmp_path):
    model = _make_model()
    mgr = CheckpointManager(str(tmp_path / "run"), model, _args())
    mgr.set_validation_split(True)
    assert mgr.on_validation(float("nan"), epoch=0, step=1) is False
    assert mgr.best_val_loss == math.inf
    assert not (tmp_path / "run" / "model_state.pt").exists()


def test_resumable_training_state_restores_model_optimizer_scheduler_and_position(tmp_path):
    model = _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5)
    # Create optimizer state and then snapshot recognizable weights.
    loss = model.activity_head.weight.sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    model.activity_head.bias.data.fill_(7.0)

    root = tmp_path / "resume"
    path = save_training_state(
        str(root), model, optimizer, scheduler,
        epoch=2, step=123, step_in_epoch=17, best_val_loss=0.42,
    )
    assert path.endswith("last/training_state.pt")

    restored_model = _make_model()
    state = load_training_state(str(root), restored_model)
    assert torch.allclose(
        restored_model.activity_head.bias,
        torch.full_like(restored_model.activity_head.bias, 7.0),
    )
    assert state["epoch"] == 2
    assert state["step"] == 123
    assert state["step_in_epoch"] == 17
    assert state["best_val_loss"] == 0.42
    assert state["optimizer_state"]["state"]
    assert state["scheduler_state"]["last_epoch"] == scheduler.state_dict()["last_epoch"]
