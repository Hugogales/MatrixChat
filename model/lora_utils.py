"""LoRA and parameter-freezing utilities for MatrixChat.

The matrix-interface parameters (agent embeddings, optional channel embeddings)
must remain trainable even when the base model is frozen.
"""

from __future__ import annotations

import torch.nn as nn


def _matrix_interface_modules(model: nn.Module):
    """Yield the wrapper's own (non-base) trainable modules."""
    for name in ("agent_embeddings", "channel_embeddings"):
        module = getattr(model, name, None)
        if module is not None:
            yield module


def freeze_base_model(model: nn.Module) -> None:
    """Freeze all base-model parameters; keep matrix-interface params trainable.

    ``model`` is expected to be a ``MatrixQwenForCausalLM`` with a ``base_model``
    attribute. If no ``base_model`` exists, all parameters are frozen.
    """
    base = getattr(model, "base_model", None)
    target = base if base is not None else model
    for name, param in target.named_parameters():
        # Preserve LoRA adapter params if LoRA was already injected.
        if "lora_" in name:
            continue
        param.requires_grad = False

    # Re-enable the matrix interface.
    for module in _matrix_interface_modules(model):
        for param in module.parameters():
            param.requires_grad = True


def _find_decoder_layers(model: nn.Module):
    """Locate the transformer decoder layers (``nn.ModuleList``) robustly.

    Tries common Qwen/HF paths and returns the first ``ModuleList`` found.
    """
    base = getattr(model, "base_model", model)

    candidates = [
        lambda m: m.model.layers,            # Qwen3ForCausalLM -> .model.layers
        lambda m: m.model.model.layers,      # PEFT-wrapped variants
        lambda m: m.transformer.h,           # GPT-style
        lambda m: m.layers,
    ]
    for getter in candidates:
        try:
            layers = getter(base)
        except AttributeError:
            continue
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return layers
    return None


def unfreeze_last_n_layers(model: nn.Module, n: int) -> int:
    """Unfreeze the last ``n`` transformer decoder layers.

    Returns the number of layers actually unfrozen. Raises a clear error if the
    decoder layers cannot be located when ``n > 0``.
    """
    if n <= 0:
        return 0

    layers = _find_decoder_layers(model)
    if layers is None:
        raise RuntimeError(
            "Could not locate transformer decoder layers to unfreeze. "
            "Checked .model.layers, .model.model.layers, .transformer.h, .layers."
        )

    n = min(n, len(layers))
    for layer in layers[-n:]:
        for param in layer.parameters():
            param.requires_grad = True
    return n


def apply_lora_if_enabled(model: nn.Module, args) -> nn.Module:
    """Apply PEFT LoRA to the base model if ``args.lora_enable`` is true.

    PEFT is only required when LoRA is enabled; otherwise this is a no-op and the
    original ``model`` is returned unchanged.
    """
    if not getattr(args, "lora_enable", False):
        return model

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover - exercised only without PEFT
        raise ImportError(
            "lora_enable=True requires the 'peft' package. Install it with "
            "`pip install peft` or set --lora_enable false."
        ) from exc

    target_modules = args.lora_target_modules
    if isinstance(target_modules, str):
        target_modules = [m.strip() for m in target_modules.split(",") if m.strip()]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Wrap only the base model so the matrix interface stays plain trainable params.
    model.base_model = get_peft_model(model.base_model, lora_config)
    return model


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of parameters with ``requires_grad=True``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
