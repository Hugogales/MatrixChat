"""Old (pre-activity-head, silence-token-based) checkpoints must be rejected
with a clear error rather than silently loading into an incompatible model.
"""

import os

import pytest
import torch

from training.checkpoint import load_checkpoint_state

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _old_format_checkpoint(tmp_path):
    ckpt_dir = tmp_path / "old_run"
    ckpt_dir.mkdir()
    torch.save(
        {
            "matrix_config": {},
            "agent_embeddings": {"weight": torch.zeros(1, 1)},
            "silence_embedding": {"weight": torch.zeros(1, 1)},  # old key, no longer used
        },
        ckpt_dir / "model_state.pt",
    )
    return str(ckpt_dir)


def _new_format_checkpoint(tmp_path):
    ckpt_dir = tmp_path / "new_run"
    ckpt_dir.mkdir()
    torch.save(
        {
            "matrix_config": {"position_mode": "flat"},
            "agent_embeddings": {"weight": torch.zeros(1, 1)},
            "inactive_embedding": {"weight": torch.zeros(1, 1)},
            "activity_head": {"weight": torch.zeros(1, 1), "bias": torch.zeros(1)},
        },
        ckpt_dir / "model_state.pt",
    )
    return str(ckpt_dir)


def test_old_format_checkpoint_rejected_with_clear_error(tmp_path):
    old_dir = _old_format_checkpoint(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        load_checkpoint_state(old_dir)
    msg = str(excinfo.value)
    assert "inactive_embedding" in msg
    assert "activity_head" in msg


def test_new_format_checkpoint_loads_cleanly(tmp_path):
    new_dir = _new_format_checkpoint(tmp_path)
    state = load_checkpoint_state(new_dir)
    assert "inactive_embedding" in state
    assert "activity_head" in state
