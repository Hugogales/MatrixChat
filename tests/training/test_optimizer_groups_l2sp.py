from types import SimpleNamespace

import torch
import torch.nn as nn

from main import (
    build_optimizer_param_groups,
    l2sp_mean_squared_deviation,
    snapshot_l2sp_reference,
)
from training.checkpoint import (
    load_optimizer_state_compatible,
    optimizer_parameter_names,
    save_training_state,
)


class _GroupedBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_weight = nn.Parameter(torch.ones(3))
        self.lora_adapter = nn.Parameter(torch.ones(2))


class _GroupedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = _GroupedBase()
        self.agent_embeddings = nn.Embedding(2, 3)
        self.agent_attention = None
        self.inactive_embedding = nn.Embedding(1, 3)
        self.activity_head = nn.Linear(3, 1)
        self.channel_embeddings = None
        self.agent_same_attention_bias = None


def _args():
    return SimpleNamespace(interface_lr=0.01, lora_lr=0.02, base_lr=0.03)


def test_optimizer_groups_assign_every_trainable_parameter_once():
    model = _GroupedModel()
    groups = build_optimizer_param_groups(model, _args())

    assert [group["group_name"] for group in groups] == ["interface", "lora", "base"]
    assert [group["lr"] for group in groups] == [0.01, 0.02, 0.03]
    grouped_ids = [id(param) for group in groups for param in group["params"]]
    trainable_ids = [id(param) for param in model.parameters() if param.requires_grad]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == set(trainable_ids)


def test_l2sp_anchors_only_trainable_non_lora_base_parameters():
    model = _GroupedModel()
    model._l2sp_reference = snapshot_l2sp_reference(model)

    assert set(model._l2sp_reference) == {"base_weight"}
    model.base_model.base_weight.data.add_(2.0)
    model.base_model.lora_adapter.data.add_(9.0)
    model.agent_embeddings.weight.data.add_(9.0)

    penalty = l2sp_mean_squared_deviation(model)
    assert torch.allclose(penalty, torch.tensor(4.0))
    penalty.backward()
    assert model.base_model.base_weight.grad is not None
    assert model.base_model.lora_adapter.grad is None
    assert model.agent_embeddings.weight.grad is None


def test_legacy_single_group_optimizer_state_migrates_by_parameter_name():
    model = _GroupedModel()
    legacy_optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=0.5,
    )
    sum(param.sum() for param in model.parameters()).backward()
    legacy_optimizer.step()
    saved = legacy_optimizer.state_dict()

    grouped_optimizer = torch.optim.AdamW(build_optimizer_param_groups(model, _args()))
    load_optimizer_state_compatible(grouped_optimizer, saved, model)

    assert [group["lr"] for group in grouped_optimizer.param_groups] == [0.01, 0.02, 0.03]
    assert len(grouped_optimizer.state) == len(list(model.parameters()))


def test_training_state_saves_group_names_and_l2sp_reference(tmp_path):
    model = _GroupedModel()
    model._l2sp_reference = snapshot_l2sp_reference(model)
    optimizer = torch.optim.AdamW(build_optimizer_param_groups(model, _args()))

    path = save_training_state(
        str(tmp_path),
        model,
        optimizer,
        scheduler=None,
        epoch=0,
        step=1,
        step_in_epoch=1,
        best_val_loss=None,
    )
    state = torch.load(path, map_location="cpu")
    assert state["optimizer_parameter_names"] == optimizer_parameter_names(model, optimizer)
    assert set(state["l2sp_reference"]) == {"base_weight"}
