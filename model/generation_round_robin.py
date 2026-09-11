"""Strict round-robin generation with a vanilla causal LM.

The scheduled agent is the only speaker in each continuation column.  Content
tokens come from a 1D causal model (unfinetuned Qwen), not MatrixChat's
activity head.  Continuation length still matches the human remainder.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch


def linearize_active_tokens(
    input_ids: torch.LongTensor,
    activity_mask: torch.BoolTensor,
) -> list[torch.LongTensor]:
    """Column-major flatten of active cells, one 1D context per batch row."""
    if input_ids.shape != activity_mask.shape:
        raise ValueError("input_ids and activity_mask must share [B, A, T]")
    batch, agents, length = input_ids.shape
    contexts: list[torch.LongTensor] = []
    for row in range(batch):
        tokens: list[torch.Tensor] = []
        for column in range(length):
            for agent in range(agents):
                if bool(activity_mask[row, agent, column]):
                    tokens.append(input_ids[row, agent, column])
        if tokens:
            contexts.append(torch.stack(tokens).long())
        else:
            contexts.append(input_ids.new_zeros((0,), dtype=torch.long))
    return contexts


def round_robin_start_agents(activity_mask: torch.BoolTensor) -> torch.LongTensor:
    """Next agent after the last unique prefix speaker; otherwise agent 0."""
    batch, agents, length = activity_mask.shape
    starts = activity_mask.new_zeros((batch,), dtype=torch.long)
    for row in range(batch):
        start = 0
        for column in range(length - 1, -1, -1):
            speakers = torch.nonzero(activity_mask[row, :, column], as_tuple=False).flatten()
            if int(speakers.numel()) == 1:
                start = int((int(speakers[0]) + 1) % agents)
                break
        starts[row] = start
    return starts


def _next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sample or argmax the last-position vocab logits ``[V]`` -> scalar id."""
    if temperature and temperature > 0:
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1).long()
    return logits.argmax(dim=-1).long()


def _call_causal_lm(model: Any, input_ids: torch.LongTensor, past: Any = None):
    """One vanilla LM step; use KV cache when the model accepts it."""
    kwargs: dict[str, Any] = {"input_ids": input_ids if past is None else input_ids[:, -1:]}
    if past is not None:
        kwargs["past_key_values"] = past
        kwargs["use_cache"] = True
    try:
        if past is None:
            output = model(input_ids=input_ids, use_cache=True)
        else:
            output = model(**kwargs)
        return output, getattr(output, "past_key_values", None)
    except TypeError:
        output = model(input_ids=input_ids)
        return output, None


@torch.no_grad()
def generate_round_robin_vanilla(
    model,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    input_activity_mask: Optional[torch.Tensor] = None,
    temperature: float = 0.0,
    activity_threshold: float = 0.5,
    placeholder_token_id: int = 0,
    input_private_mask: Optional[torch.Tensor] = None,
    agent_visibility: Optional[torch.Tensor] = None,
    return_probs: bool = False,
) -> Tuple[torch.LongTensor, torch.BoolTensor] | Tuple[torch.LongTensor, torch.BoolTensor, torch.Tensor]:
    """Generate a strict round-robin continuation with a vanilla causal LM.

    ``model`` must accept HuggingFace-style ``input_ids=[B, S]`` and return an
    object with ``logits`` of shape ``[B, S, V]``.  Matrix kwargs
    (``activity_threshold``, private masks, visibility) are accepted for call
    compatibility with :func:`model.generation.generate_matrix` and ignored.
    """
    del activity_threshold, input_private_mask, agent_visibility
    if input_ids.dim() != 3:
        raise ValueError(f"input_ids must be [B, A, T], got {tuple(input_ids.shape)}")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    was_training = bool(getattr(model, "training", False))
    eval_fn = getattr(model, "eval", None)
    if callable(eval_fn):
        eval_fn()

    batch, agents, _length = input_ids.shape
    if input_activity_mask is None:
        activity = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        activity = input_activity_mask.bool()
    if activity.shape != input_ids.shape:
        raise ValueError("input_activity_mask must match input_ids")

    contexts = linearize_active_tokens(input_ids, activity)
    starts = round_robin_start_agents(activity)
    tokens = input_ids.new_full((batch, agents, max_new_tokens), int(placeholder_token_id))
    speak = torch.zeros((batch, agents, max_new_tokens), dtype=torch.bool, device=input_ids.device)
    probs = torch.zeros((batch, agents, max_new_tokens), dtype=torch.float32, device=input_ids.device)

    for row in range(batch):
        context = contexts[row]
        if context.numel() == 0:
            context = input_ids.new_full((1,), int(placeholder_token_id))
        start = int(starts[row])
        past = None
        for step in range(max_new_tokens):
            speaker = (start + step) % agents
            model_input = context.unsqueeze(0)
            output, past = _call_causal_lm(model, model_input, past)
            logits = output.logits if hasattr(output, "logits") else output[0]
            chosen = _next_token(logits[0, -1], temperature)
            tokens[row, speaker, step] = chosen
            speak[row, speaker, step] = True
            probs[row, speaker, step] = 1.0
            context = torch.cat([context, chosen.view(1)], dim=0)

    if was_training:
        train_fn = getattr(model, "train", None)
        if callable(train_fn):
            train_fn()

    if return_probs:
        return tokens.long(), speak, probs
    return tokens.long(), speak


def load_vanilla_causal_lm(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str = "eager",
):
    """Load an unwrapped HuggingFace causal LM (no MatrixQwen / LoRA)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    kwargs = {"attn_implementation": attn_implementation}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, **kwargs)
    except TypeError:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, **kwargs
            )
        except (TypeError, ValueError):
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer
