"""Checkpoint save/load helpers for MatrixChat training runs.

Layout (when validation is enabled during real-data training):

    checkpoints/<run_id>/model_state.pt      # best validation checkpoint (root)
    checkpoints/<run_id>/lora_adapters/      # LoRA for best checkpoint
    checkpoints/<run_id>/best_checkpoint.json
    checkpoints/<run_id>/last/model_state.pt # terminal training state
    checkpoints/<run_id>/last/lora_adapters/

When validation is disabled (toy smoke path), the terminal model is saved at the
root, preserving the previous single-checkpoint behavior.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import torch
import torch.nn as nn

_REQUIRED_CHECKPOINT_KEYS = ("inactive_embedding", "activity_head")


def _is_finite(value) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def collect_unfrozen_base_parameters(model: nn.Module) -> dict[str, torch.Tensor] | None:
    """Collect trainable, non-LoRA base-model parameters (e.g. unfrozen layers)."""
    base = getattr(model, "base_model", None)
    if base is None:
        return None
    state: dict[str, torch.Tensor] = {}
    for name, param in base.named_parameters():
        if param.requires_grad and "lora_" not in name:
            state[name] = param.detach().cpu().clone()
    return state or None


def collect_trainable_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    """Portable resume snapshot of every currently trainable tensor.

    Unlike inference checkpoints (which store PEFT adapters separately), this
    exact named-parameter map can be copied into an already-constructed model,
    avoiding any attempt to wrap a PEFT model a second time during resume.
    """
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def optimizer_parameter_names(model: nn.Module, optimizer) -> list[list[str]]:
    """Return parameter names in each optimizer group, preserving group order."""
    names_by_id = {
        id(param): name
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    groups: list[list[str]] = []
    for group in optimizer.param_groups:
        names = []
        for param in group["params"]:
            name = names_by_id.get(id(param))
            if name is None:
                raise ValueError("Optimizer contains a parameter absent from model.named_parameters()")
            names.append(name)
        groups.append(names)
    return groups


def load_optimizer_state_compatible(
    optimizer,
    saved_state: dict,
    model: nn.Module,
    saved_parameter_names: list[list[str]] | None = None,
) -> None:
    """Restore optimizer state across both grouped and legacy single-group runs.

    Legacy checkpoints did not store parameter names and used model traversal
    order in one group. New group hyperparameters are retained during that
    migration, while per-parameter moments are restored by inferred name.
    """
    current_state = optimizer.state_dict()
    saved_groups = saved_state.get("param_groups", [])
    current_groups = current_state["param_groups"]
    if (
        len(saved_groups) == len(current_groups)
        and all(
            len(saved["params"]) == len(current["params"])
            for saved, current in zip(saved_groups, current_groups)
        )
    ):
        optimizer.load_state_dict(saved_state)
        return

    if saved_parameter_names is None:
        flat_saved_ids = [
            parameter_id
            for group in saved_groups
            for parameter_id in group["params"]
        ]
        inferred_names = [
            name for name, param in model.named_parameters() if param.requires_grad
        ]
        if len(flat_saved_ids) != len(inferred_names):
            raise ValueError(
                "Cannot migrate legacy optimizer state: trainable parameter count "
                f"changed ({len(flat_saved_ids)} saved vs {len(inferred_names)} current)"
            )
        saved_parameter_names = [inferred_names]

    saved_name_by_id = {}
    for group, names in zip(saved_groups, saved_parameter_names):
        if len(group["params"]) != len(names):
            raise ValueError("Saved optimizer parameter names do not match its groups")
        saved_name_by_id.update(zip(group["params"], names))
    saved_id_by_name = {name: parameter_id for parameter_id, name in saved_name_by_id.items()}

    current_names = optimizer_parameter_names(model, optimizer)
    migrated_state = {}
    for group, names in zip(current_groups, current_names):
        for current_id, name in zip(group["params"], names):
            saved_id = saved_id_by_name.get(name)
            if saved_id in saved_state.get("state", {}):
                migrated_state[current_id] = saved_state["state"][saved_id]

    optimizer.load_state_dict({
        "state": migrated_state,
        # Keep the newly configured group names, rates, and decay settings.
        "param_groups": current_groups,
    })


def load_scheduler_state_compatible(scheduler, saved_state: dict | None) -> None:
    """Restore scheduler progress while tolerating legacy one-group schedules."""
    if saved_state is None:
        return
    state = dict(saved_state)
    group_count = len(scheduler.optimizer.param_groups)
    if len(state.get("base_lrs", [])) != group_count:
        state["base_lrs"] = [group["initial_lr"] for group in scheduler.optimizer.param_groups]
        state["_last_lr"] = [group["lr"] for group in scheduler.optimizer.param_groups]
    scheduler.load_state_dict(state)


def restore_trainable_parameters(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    named = dict(model.named_parameters())
    missing = [name for name in state if name not in named]
    if missing:
        raise ValueError(f"Resume state has unknown trainable parameters: {missing[:5]}")
    for name, tensor in state.items():
        param = named[name]
        param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))


def save_training_state(
    checkpoint_root: str,
    model: nn.Module,
    optimizer,
    scheduler,
    *,
    epoch: int,
    step: int,
    step_in_epoch: int,
    best_val_loss: float | None,
) -> str:
    """Atomically write resumable model/optimizer/scheduler/RNG state."""
    last_dir = os.path.join(checkpoint_root, "last")
    os.makedirs(last_dir, exist_ok=True)
    path = os.path.join(last_dir, "training_state.pt")
    temporary = path + ".tmp"
    payload = {
        "trainable_parameters": collect_trainable_parameters(model),
        "optimizer_state": optimizer.state_dict(),
        "optimizer_parameter_names": optimizer_parameter_names(model, optimizer),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "l2sp_reference": {
            name: tensor.detach().cpu().clone()
            for name, tensor in getattr(model, "_l2sp_reference", {}).items()
        } or None,
        "epoch": int(epoch),
        "step": int(step),
        "step_in_epoch": int(step_in_epoch),
        "best_val_loss": best_val_loss,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_training_state(checkpoint_root: str, model: nn.Module) -> dict:
    path = os.path.join(checkpoint_root, "last", "training_state.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Resumable training state not found: {path}")
    state = torch.load(path, map_location="cpu")
    restore_trainable_parameters(model, state["trainable_parameters"])
    if state.get("l2sp_reference"):
        named_base = dict(getattr(model, "base_model", model).named_parameters())
        model._l2sp_reference = {
            name: tensor.to(
                device=named_base[name].device,
                dtype=named_base[name].dtype,
            )
            for name, tensor in state["l2sp_reference"].items()
            if (
                name in named_base
                and named_base[name].requires_grad
                and "lora_" not in name
            )
        }
    return state


def build_checkpoint_payload(model: nn.Module, args, metadata: dict[str, Any] | None = None) -> dict:
    """Assemble the full ``model_state.pt`` dictionary."""
    return {
        "matrix_config": vars(args),
        "agent_embeddings": (
            model.agent_embeddings.state_dict()
            if getattr(model, "agent_embeddings", None) is not None
            else None
        ),
        "agent_attention": (
            model.agent_attention.state_dict()
            if getattr(model, "agent_attention", None) is not None
            else None
        ),
        "agent_same_attention_bias": (
            model.agent_same_attention_bias.state_dict()
            if getattr(model, "agent_same_attention_bias", None) is not None
            else None
        ),
        "agent_relation_bias": (
            model.agent_relation_bias.state_dict()
            if getattr(model, "agent_relation_bias", None) is not None
            else None
        ),
        "agent_dynamic_state": (
            model.agent_dynamic_state.state_dict()
            if getattr(model, "agent_dynamic_state", None) is not None
            else None
        ),
        "channel_embeddings": (
            model.channel_embeddings.state_dict()
            if getattr(model, "channel_embeddings", None) is not None
            else None
        ),
        "inactive_embedding": model.inactive_embedding.state_dict(),
        "activity_head": model.activity_head.state_dict(),
        "unfrozen_base_parameters": collect_unfrozen_base_parameters(model),
        "metadata": metadata,
    }


def save_checkpoint(
    model: nn.Module,
    args,
    checkpoint_dir: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Write ``model_state.pt`` (+ LoRA adapters when enabled) to ``checkpoint_dir``."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "model_state.pt")
    torch.save(build_checkpoint_payload(model, args, metadata), ckpt_path)
    if getattr(args, "lora_enable", False) and hasattr(model.base_model, "save_pretrained"):
        adapter_dir = os.path.join(checkpoint_dir, "lora_adapters")
        model.base_model.save_pretrained(adapter_dir)
    return ckpt_path


def write_best_checkpoint_summary(checkpoint_root: str, metadata: dict[str, Any]) -> str:
    """Write a small JSON summary of the current best validation checkpoint."""
    path = os.path.join(checkpoint_root, "best_checkpoint.json")
    summary = {
        "best_val_loss": metadata.get("val_loss"),
        "epoch": metadata.get("epoch"),
        "step": metadata.get("step"),
        "val_components": metadata.get("val_components"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return path


def write_best_probe_summary(checkpoint_root: str, metadata: dict[str, Any]) -> str:
    """Write a small JSON summary of the current best-by-turn-taking-probe checkpoint."""
    path = os.path.join(checkpoint_root, "best_probe", "best_probe.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    summary = {
        "best_probe_score": metadata.get("probe_score"),
        "epoch": metadata.get("epoch"),
        "step": metadata.get("step"),
        "probe": metadata.get("probe"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return path


def load_checkpoint_state(checkpoint_dir: str) -> dict:
    """Load and validate ``model_state.pt`` from a checkpoint directory.

    Raises ``ValueError`` for old (pre-activity-head) checkpoints.
    """
    path = os.path.join(checkpoint_dir, "model_state.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu")
    missing = [k for k in _REQUIRED_CHECKPOINT_KEYS if k not in state or state[k] is None]
    if missing:
        raise ValueError(
            f"Checkpoint {path!r} is missing {missing} and appears to be from the old "
            "silence-token format (pre activity-head). It is INCOMPATIBLE with the "
            "current architecture and must be retrained from scratch."
        )
    return state


def resolve_init_checkpoint_dir(checkpoint_root: str, prefer: str = "best_probe") -> str:
    """Resolve a weight-only initialization directory under a run root."""
    root = os.path.normpath(checkpoint_root)
    candidates = {
        "best_probe": os.path.join(root, "best_probe"),
        "last": os.path.join(root, "last"),
        "root": root,
    }
    if prefer in candidates:
        chosen = candidates[prefer]
        if os.path.isfile(os.path.join(chosen, "model_state.pt")):
            return chosen
        if prefer == "auto":
            pass
        else:
            # Explicit prefer missing (e.g. c106 has root weights but no
            # best_probe/) -- fall through to auto resolution below.
            prefer = "auto"
    if prefer != "auto":
        raise ValueError(f"Unknown init_checkpoint_prefer: {prefer!r}")

    for name in ("best_probe", "last", "root"):
        chosen = candidates[name]
        if os.path.isfile(os.path.join(chosen, "model_state.pt")):
            return chosen
    raise FileNotFoundError(f"No model_state.pt found under checkpoint root {root!r}")


def _is_peft_wrapped(module: nn.Module) -> bool:
    try:
        from peft import PeftModel
    except ImportError:
        return False
    return isinstance(module, PeftModel)


def _peft_base_model(module: nn.Module) -> nn.Module:
    base = module
    while _is_peft_wrapped(base):
        base = base.get_base_model()
    return base


def _load_lora_adapters(model: nn.Module, adapter_dir: str) -> None:
    """Load LoRA weights without double-wrapping an already-PEFT base model."""
    from peft import PeftModel

    base = model.base_model
    if _is_peft_wrapped(base):
        inner = base.get_base_model()
        loaded = PeftModel.from_pretrained(inner, adapter_dir)
        incompatible = base.load_state_dict(loaded.state_dict(), strict=False)
        if incompatible.missing_keys:
            preview = incompatible.missing_keys[:5]
            suffix = "..." if len(incompatible.missing_keys) > 5 else ""
            print(
                f"[warn] LoRA init missing {len(incompatible.missing_keys)} key(s): "
                f"{preview}{suffix}"
            )
        return
    model.base_model = PeftModel.from_pretrained(base, adapter_dir)


def load_best_probe_score(checkpoint_root: str) -> float | None:
    path = os.path.join(checkpoint_root, "best_probe", "best_probe.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    score = payload.get("best_probe_score")
    return float(score) if score is not None else None


def apply_checkpoint_to_model_lenient(
    model: nn.Module,
    checkpoint_dir: str,
    state: dict | None = None,
) -> dict:
    """Load compatible checkpoint weights, skipping optional mismatched modules."""
    if state is None:
        state = load_checkpoint_state(checkpoint_dir)

    adapter_dir = os.path.join(checkpoint_dir, "lora_adapters")
    if os.path.isdir(adapter_dir):
        try:
            _load_lora_adapters(model, adapter_dir)
        except Exception as exc:
            print(
                f"[warn] skipped LoRA adapter load from {adapter_dir}: "
                f"{type(exc).__name__}: {exc}"
            )

    def _load(module, saved, label: str) -> None:
        if module is None or not saved:
            return
        try:
            module.load_state_dict(saved)
        except Exception as exc:
            print(
                f"[warn] skipped {label} checkpoint weights: "
                f"{type(exc).__name__}: {exc}"
            )

    _load(
        getattr(model, "agent_embeddings", None),
        state.get("agent_embeddings"),
        "agent_embeddings",
    )
    _load(
        getattr(model, "channel_embeddings", None),
        state.get("channel_embeddings"),
        "channel_embeddings",
    )
    _load(
        getattr(model, "agent_attention", None),
        state.get("agent_attention"),
        "agent_attention",
    )
    _load(
        getattr(model, "agent_same_attention_bias", None),
        state.get("agent_same_attention_bias"),
        "agent_same_attention_bias",
    )
    _load(
        getattr(model, "agent_relation_bias", None),
        state.get("agent_relation_bias"),
        "agent_relation_bias",
    )
    _load(
        getattr(model, "agent_dynamic_state", None),
        state.get("agent_dynamic_state"),
        "agent_dynamic_state",
    )
    model.inactive_embedding.load_state_dict(state["inactive_embedding"])
    model.activity_head.load_state_dict(state["activity_head"])
    install_hooks = getattr(model, "install_agent_attention_hooks", None)
    if install_hooks is not None:
        install_hooks()

    unfrozen = state.get("unfrozen_base_parameters")
    if unfrozen:
        missing = apply_unfrozen_base_parameters(model, unfrozen)
        if missing:
            base = model.base_model
            result = base.load_state_dict(unfrozen, strict=False)
            still_missing = set(missing) - set(result.unexpected_keys)
            if still_missing:
                print(
                    f"[warn] init checkpoint missing {len(still_missing)} unfrozen base "
                    f"parameter(s): {sorted(still_missing)[:5]}"
                    f"{'...' if len(still_missing) > 5 else ''}"
                )
    return state


def init_model_from_checkpoint(
    model: nn.Module,
    checkpoint_root: str,
    *,
    prefer: str = "best_probe",
) -> dict:
    """Load model weights only from a prior run (fresh optimizer/step counter)."""
    checkpoint_dir = resolve_init_checkpoint_dir(checkpoint_root, prefer)
    state = apply_checkpoint_to_model_lenient(model, checkpoint_dir)
    return {
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_root": os.path.normpath(checkpoint_root),
        "best_probe_score": load_best_probe_score(os.path.normpath(checkpoint_root)),
    }


def apply_unfrozen_base_parameters(model: nn.Module, unfrozen: dict[str, torch.Tensor]) -> list[str]:
    """Restore explicitly unfrozen base-model weights. Returns missing key names."""
    base = getattr(model, "base_model", None)
    if base is None:
        return list(unfrozen.keys())

    named = dict(base.named_parameters())
    missing: list[str] = []
    for name, tensor in unfrozen.items():
        if name not in named:
            missing.append(name)
            continue
        param = named[name]
        param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
    return missing


def apply_checkpoint_to_model(model: nn.Module, checkpoint_dir: str, state: dict | None = None) -> dict:
    """Load LoRA adapters, matrix-interface weights, and optional unfrozen base params."""
    if state is None:
        state = load_checkpoint_state(checkpoint_dir)

    adapter_dir = os.path.join(checkpoint_dir, "lora_adapters")
    if os.path.isdir(adapter_dir):
        _load_lora_adapters(model, adapter_dir)

    if state.get("agent_embeddings") and getattr(model, "agent_embeddings", None) is not None:
        model.agent_embeddings.load_state_dict(state["agent_embeddings"])
    if state.get("channel_embeddings") and getattr(model, "channel_embeddings", None) is not None:
        model.channel_embeddings.load_state_dict(state["channel_embeddings"])
    if getattr(model, "agent_attention", None) is not None:
        if not state.get("agent_attention"):
            raise ValueError(
                "Checkpoint enables attention-level agent conditioning but "
                "contains no agent_attention state"
            )
        model.agent_attention.load_state_dict(state["agent_attention"])
    if getattr(model, "agent_same_attention_bias", None) is not None:
        saved_bias = state.get("agent_same_attention_bias")
        if saved_bias is None:
            raise ValueError(
                "Checkpoint enables same-agent attention bias but contains no saved bias"
            )
        model.agent_same_attention_bias.load_state_dict(saved_bias)
    if getattr(model, "agent_relation_bias", None) is not None:
        saved_relation_bias = state.get("agent_relation_bias")
        if saved_relation_bias is None:
            raise ValueError(
                "Checkpoint enables agent_relation_bias but contains no saved state"
            )
        model.agent_relation_bias.load_state_dict(saved_relation_bias)
    if getattr(model, "agent_dynamic_state", None) is not None:
        saved_dynamic_state = state.get("agent_dynamic_state")
        if saved_dynamic_state is None:
            raise ValueError(
                "Checkpoint enables agent_dynamic_state but contains no saved state"
            )
        model.agent_dynamic_state.load_state_dict(saved_dynamic_state)
    model.inactive_embedding.load_state_dict(state["inactive_embedding"])
    model.activity_head.load_state_dict(state["activity_head"])
    install_hooks = getattr(model, "install_agent_attention_hooks", None)
    if install_hooks is not None:
        install_hooks()

    unfrozen = state.get("unfrozen_base_parameters")
    if unfrozen:
        missing = apply_unfrozen_base_parameters(model, unfrozen)
        if missing:
            base = model.base_model
            result = base.load_state_dict(unfrozen, strict=False)
            still_missing = set(missing) - set(result.unexpected_keys)
            if still_missing:
                print(
                    f"[warn] checkpoint missing {len(still_missing)} unfrozen base parameter(s): "
                    f"{sorted(still_missing)[:5]}{'...' if len(still_missing) > 5 else ''}"
                )
    return state


class CheckpointManager:
    """Track best validation loss and maintain best/last checkpoint layout.

    Also independently tracks the best checkpoint BY IN-TRAINING TURN-TAKING
    PROBE QUALITY (``on_probe`` / ``training.race.probe_quality_score``),
    saved under ``checkpoint_root/best_probe/``. Validation loss (a mixture
    of content CE + activity BCE + reward term) and actual decoded
    conversation quality have repeatedly diverged in this project's history
    (see ``.cursor/rules/read-decoded-text-not-just-metrics.mdc`` and
    ``SUCCESS_STORIES.md``'s "more training is NOT universally better"
    finding) -- final-run evaluation should default to this checkpoint
    rather than either the val-loss-best or the terminal ``last/`` one.
    """

    def __init__(self, checkpoint_root: str, model: nn.Module, args):
        self.checkpoint_root = checkpoint_root
        self.model = model
        self.args = args
        self.best_val_loss = math.inf
        self.best_metadata: dict[str, Any] | None = None
        self.best_probe_score = -math.inf
        self.best_probe_metadata: dict[str, Any] | None = None
        self.has_validation_split = False
        self.optimizer = None
        self.scheduler = None

    def set_validation_split(self, enabled: bool) -> None:
        """Mark whether training uses a held-out validation split."""
        self.has_validation_split = bool(enabled)

    def set_training_objects(self, optimizer, scheduler) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler

    def save_resume_state(
        self, epoch: int, step: int, step_in_epoch: int, last_val_loss: float | None
    ) -> str | None:
        if self.optimizer is None:
            return None
        path = save_training_state(
            self.checkpoint_root,
            self.model,
            self.optimizer,
            self.scheduler,
            epoch=epoch,
            step=step,
            step_in_epoch=step_in_epoch,
            best_val_loss=(
                self.best_val_loss if _is_finite(self.best_val_loss) else last_val_loss
            ),
        )
        print(f"[info] saved resumable training state to {path}")
        return path

    def on_validation(
        self,
        val_loss: float,
        epoch: int,
        step: int,
        val_components: dict[str, float] | None = None,
    ) -> bool:
        """Save root best checkpoint when ``val_loss`` strictly improves."""
        if not _is_finite(val_loss):
            return False

        if val_loss >= self.best_val_loss:
            return False

        old_best = self.best_val_loss
        self.best_val_loss = float(val_loss)
        metadata = {
            "kind": "best",
            "epoch": epoch,
            "step": step,
            "val_loss": self.best_val_loss,
            "val_components": val_components,
        }
        self.best_metadata = metadata
        ckpt_path = save_checkpoint(self.model, self.args, self.checkpoint_root, metadata)
        summary_path = write_best_checkpoint_summary(self.checkpoint_root, metadata)
        old_msg = f"{old_best:.6f}" if _is_finite(old_best) else "n/a"
        print(
            f"[info] new best validation loss {self.best_val_loss:.6f} "
            f"(prev {old_msg}) at epoch {epoch} step {step}; "
            f"saved to {ckpt_path} ({summary_path})"
        )
        return True

    def on_probe(self, probe_result: dict[str, Any], epoch: int, step: int) -> bool:
        """Save a ``best_probe/`` checkpoint when the turn-taking probe score improves.

        Uses ``training.race.probe_quality_score`` on the same dict
        ``main.py``'s ``run_kitchen_sink_probe`` returns at every
        ``probe_every`` step -- no extra generation cost. Independent of
        ``on_validation``: a checkpoint can be the best-by-loss and a
        completely different step can be the best-by-turn-taking-behavior.
        """
        from training.race import probe_quality_score

        score = probe_quality_score(probe_result)
        if not math.isfinite(score) or score <= self.best_probe_score:
            return False

        old_best = self.best_probe_score
        self.best_probe_score = score
        metadata = {
            "kind": "best_probe",
            "epoch": epoch,
            "step": step,
            "probe_score": score,
            "probe": probe_result,
        }
        self.best_probe_metadata = metadata
        best_probe_dir = os.path.join(self.checkpoint_root, "best_probe")
        ckpt_path = save_checkpoint(self.model, self.args, best_probe_dir, metadata)
        summary_path = write_best_probe_summary(self.checkpoint_root, metadata)
        old_msg = f"{old_best:.4f}" if math.isfinite(old_best) else "n/a"
        print(
            f"[info] new best probe-quality score {score:.4f} "
            f"(prev {old_msg}) at epoch {epoch} step {step}; "
            f"saved to {ckpt_path} ({summary_path})"
        )
        return True

    def save_last(self, epoch: int, step: int, val_loss: float | None = None) -> str:
        """Persist the terminal training state under ``last/``."""
        last_dir = os.path.join(self.checkpoint_root, "last")
        metadata = {
            "kind": "last",
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss if _is_finite(val_loss) else None,
        }
        ckpt_path = save_checkpoint(self.model, self.args, last_dir, metadata)
        print(f"[info] saved last checkpoint to {ckpt_path}")
        return ckpt_path

    def finalize(self, epoch: int, step: int, last_val_loss: float | None = None) -> None:
        """Write terminal checkpoint(s) at end of training."""
        os.makedirs(self.checkpoint_root, exist_ok=True)
        if not self.has_validation_split:
            metadata = {
                "kind": "final",
                "epoch": epoch,
                "step": step,
                "val_loss": last_val_loss if _is_finite(last_val_loss) else None,
            }
            ckpt_path = save_checkpoint(self.model, self.args, self.checkpoint_root, metadata)
            print(f"[info] saved checkpoint to {ckpt_path} (no validation split; terminal model)")
            return

        self.save_last(epoch, step, last_val_loss)
        if self.best_metadata is not None:
            print(
                f"[info] best validation checkpoint kept at "
                f"{os.path.join(self.checkpoint_root, 'model_state.pt')} "
                f"(best val_loss={self.best_val_loss:.6f} at step {self.best_metadata['step']})"
            )
        else:
            print("[warn] validation was enabled but no finite best checkpoint was saved.")
        if self.best_probe_metadata is not None:
            print(
                f"[info] best turn-taking-probe checkpoint kept at "
                f"{os.path.join(self.checkpoint_root, 'best_probe', 'model_state.pt')} "
                f"(probe_score={self.best_probe_score:.4f} at step {self.best_probe_metadata['step']}) "
                "-- prefer this checkpoint for final-run evaluation over best-by-loss or last"
            )
