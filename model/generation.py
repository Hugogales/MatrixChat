"""Activity-gated multi-agent generation helpers for MatrixChat.

Each step reads the model's ``activity_head`` output at the most recent column
to decide, per agent, whether to SPEAK (sample/argmax a real vocabulary token)
or YIELD (emit the placeholder input id and mark the new column inactive). No
vocabulary token is ever used to represent silence -- the decision is entirely
the activity head's, decoupled from the content vocabulary.

Multi-token autoregressive generation for real conversations is intentionally
simple here (greedy/temperature per step, no beam search).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def generation_topology_summary(
    probabilities: torch.Tensor,
    speak_mask: torch.Tensor,
    initial_previous_activity: torch.Tensor,
) -> dict:
    """Summarize soft and realized topology along an autoregressive rollout.

    ``probabilities`` and ``speak_mask`` are ``[B,A,S]`` outputs from
    :func:`generate_matrix`. ``initial_previous_activity`` is ``[B,A]`` and
    identifies who spoke in the final prompt column going into generated step
    zero. Later previous-owner states come from the model's own thresholded
    actions, not dataset teacher forcing.

    The soft p0/p_same/p_handoff/p_overlap definitions exactly match
    ``model.turn_reward``. When no unique previous owner exists, sole-speaker
    probability is assigned to p_handoff (floor acquisition), also matching
    the training reward.
    """
    if probabilities.shape != speak_mask.shape:
        raise ValueError("probabilities and speak_mask must have identical [B,A,S] shapes")
    if probabilities.dim() != 3:
        raise ValueError("probabilities and speak_mask must be [B,A,S]")
    b, a, steps = probabilities.shape
    if initial_previous_activity.shape != (b, a):
        raise ValueError(
            "initial_previous_activity must be [B,A], got "
            f"{tuple(initial_previous_activity.shape)}"
        )
    if steps == 0:
        return {
            "columns": 0,
            "soft": {"p0": 0.0, "p_same": 0.0, "p_handoff": 0.0, "p_overlap": 0.0},
            "soft_unique_owner": {"columns": 0, "same": 0.0, "handoff": 0.0},
            "realized": {"silence": 0.0, "same": 0.0, "handoff": 0.0, "overlap": 0.0},
            "realized_unique_owner_exactly_one": {
                "columns": 0, "same": 0.0, "handoff": 0.0,
            },
        }

    previous = torch.cat(
        [initial_previous_activity.bool().unsqueeze(-1), speak_mask.bool()[:, :, :-1]],
        dim=2,
    )
    previous_count = previous.sum(dim=1)
    has_owner = previous_count == 1
    previous_owner = previous.float().argmax(dim=1)

    q = probabilities.float().clamp(min=1e-6, max=1.0 - 1e-6)
    one_minus_q = 1.0 - q
    prod_all = one_minus_q.prod(dim=1, keepdim=True)
    p0 = prod_all.squeeze(1)
    p_only = q * (prod_all / one_minus_q)
    p_one = p_only.sum(dim=1)
    p_same_raw = torch.gather(
        p_only, 1, previous_owner.unsqueeze(1)
    ).squeeze(1)
    p_same = torch.where(has_owner, p_same_raw, torch.zeros_like(p_same_raw))
    p_handoff = torch.where(has_owner, p_one - p_same, p_one)
    p_overlap = (1.0 - p0 - p_one).clamp(min=0.0, max=1.0)

    current = speak_mask.bool()
    current_count = current.sum(dim=1)
    current_owner = current.float().argmax(dim=1)
    realized_silence = current_count == 0
    realized_overlap = current_count >= 2
    realized_exactly_one = current_count == 1
    realized_same = realized_exactly_one & has_owner & (current_owner == previous_owner)
    # Match reward semantics: an exactly-one acquisition after a non-unique
    # previous state is included in handoff.
    realized_handoff = realized_exactly_one & (~realized_same)

    unique_mass = (p_same + p_handoff).clamp(min=1e-8)
    unique_columns = int(has_owner.sum().detach().cpu())
    strict_columns = has_owner & realized_exactly_one
    strict_count = int(strict_columns.sum().detach().cpu())
    total = float(b * steps)

    def mean_value(value: torch.Tensor) -> float:
        return float(value.mean().detach().cpu())

    return {
        "columns": int(total),
        "soft": {
            "p0": mean_value(p0),
            "p_same": mean_value(p_same),
            "p_handoff": mean_value(p_handoff),
            "p_overlap": mean_value(p_overlap),
        },
        "soft_unique_owner": {
            "columns": unique_columns,
            "same": (
                float((p_same / unique_mass)[has_owner].mean().detach().cpu())
                if unique_columns else 0.0
            ),
            "handoff": (
                float((p_handoff / unique_mass)[has_owner].mean().detach().cpu())
                if unique_columns else 0.0
            ),
        },
        "realized": {
            "silence": float(realized_silence.sum().detach().cpu()) / total,
            "same": float(realized_same.sum().detach().cpu()) / total,
            "handoff": float(realized_handoff.sum().detach().cpu()) / total,
            "overlap": float(realized_overlap.sum().detach().cpu()) / total,
        },
        "realized_unique_owner_exactly_one": {
            "columns": strict_count,
            "same": (
                float(realized_same[strict_columns].sum().detach().cpu()) / strict_count
                if strict_count else 0.0
            ),
            "handoff": (
                float(realized_handoff[strict_columns].sum().detach().cpu()) / strict_count
                if strict_count else 0.0
            ),
        },
    }


def _decide_and_sample(
    content_logits: torch.Tensor,   # [B, A, V]
    activity_logits: torch.Tensor,  # [B, A]
    temperature: float,
    activity_threshold: float,
    placeholder_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(next_tokens[B, A], speak[B, A])`` for one step."""
    b, a, _ = content_logits.shape
    q = torch.sigmoid(activity_logits.float())
    speak = q > activity_threshold

    if temperature and temperature > 0:
        probs = torch.softmax(content_logits.float() / temperature, dim=-1)
        sampled = torch.multinomial(probs.reshape(b * a, -1), num_samples=1).reshape(b, a)
    else:
        sampled = content_logits.argmax(dim=-1)

    next_tokens = torch.where(speak, sampled, torch.full_like(sampled, placeholder_token_id))
    return next_tokens.long(), speak


@torch.no_grad()
def generate_matrix(
    model,
    input_ids: torch.LongTensor,                        # [B, A, T]
    max_new_tokens: int,
    input_activity_mask: Optional[torch.Tensor] = None,  # [B, A, T]; None = all active
    temperature: float = 0.0,
    activity_threshold: float = 0.5,
    placeholder_token_id: int = 0,
    input_private_mask: Optional[torch.Tensor] = None,   # [B, A, T]; True = private cell
    agent_visibility: Optional[torch.Tensor] = None,      # [B, A, A]; constant for the conversation
    return_probs: bool = False,
    use_kv_cache: bool = True,
) -> Tuple[torch.LongTensor, torch.BoolTensor]:
    """Autoregressively generate ``max_new_tokens`` columns per agent.

    At each step, reads the activity head at the most recent column to decide
    speak-vs-yield per agent, samples content for speaking agents, and appends
    the new column (marking its activity accordingly) before the next step.

    ``input_private_mask``/``agent_visibility`` let a prefilled private prompt
    (e.g. a synthesized Werewolf role reveal) stay hidden from agents that were
    not granted visibility, for the entire generation. Newly generated columns
    are always PUBLIC (private context is only ever part of the prompt);
    ``agent_visibility`` does not change over time and is simply forwarded.

    Returns ``(tokens, speak_mask)``, each shaped ``[B, A, max_new_tokens]``
    (the appended continuation only, not the prompt). ``speak_mask`` is True
    where the agent actually spoke that step (use it to decode/skip yields).

    If ``return_probs`` is True, ALSO returns ``probs`` -- the continuous
    P(speak) = sigmoid(activity_logit) at every step, shaped like
    ``speak_mask``. Unlike the thresholded ``speak_mask``, this is a
    calibration signal: it shows whether an agent's willingness to speak is
    trending toward/away from the decision boundary even when the discrete
    decision never flips (e.g. a listener's P(speak) climbing under silence
    pressure but never crossing ``activity_threshold``).
    """
    was_training = model.training
    model.eval()

    can_use_kv_cache = (
        use_kv_cache
        and max_new_tokens > 0
        and hasattr(model, "encode_prefix_for_generation")
        and hasattr(model, "forward_generation_column")
    )

    if can_use_kv_cache:
        if input_activity_mask is None:
            cur_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            cur_mask = input_activity_mask.bool()
        cur_private = input_private_mask.bool() if input_private_mask is not None else None

        cache, last_content, last_activity = model.encode_prefix_for_generation(
            input_ids=input_ids,
            input_activity_mask=cur_mask,
            input_private_mask=cur_private,
            agent_visibility=agent_visibility,
        )

        generated_tokens = []
        generated_speak = []
        generated_probs = [] if return_probs else None
        for _ in range(max_new_tokens):
            nxt, speak = _decide_and_sample(
                last_content, last_activity, temperature, activity_threshold, placeholder_token_id
            )
            generated_tokens.append(nxt)
            generated_speak.append(speak)
            if return_probs:
                generated_probs.append(torch.sigmoid(last_activity.float()))

            cache, last_content, last_activity = model.forward_generation_column(
                cache,
                nxt.unsqueeze(-1),
                speak.unsqueeze(-1),
                column_private=torch.zeros_like(speak, dtype=torch.bool).unsqueeze(-1),
            )

        if was_training:
            model.train()

        tokens = torch.stack(generated_tokens, dim=2).long()
        speak_mask = torch.stack(generated_speak, dim=2).bool()
        if return_probs:
            probs = torch.stack(generated_probs, dim=2).float()
            return tokens, speak_mask, probs
        return tokens, speak_mask

    cur = input_ids
    if input_activity_mask is None:
        cur_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        cur_mask = input_activity_mask.bool()

    cur_private = input_private_mask.bool() if input_private_mask is not None else None

    generated_tokens = []
    generated_speak = []
    generated_probs = [] if return_probs else None
    for _ in range(max_new_tokens):
        out = model(
            input_ids=cur, input_activity_mask=cur_mask,
            input_private_mask=cur_private, agent_visibility=agent_visibility,
        )
        last_content = out.logits[:, :, -1, :]          # [B, A, V]
        last_activity = out.activity_logits[:, :, -1]   # [B, A]

        nxt, speak = _decide_and_sample(
            last_content, last_activity, temperature, activity_threshold, placeholder_token_id
        )
        generated_tokens.append(nxt)
        generated_speak.append(speak)
        if return_probs:
            generated_probs.append(torch.sigmoid(last_activity.float()))

        cur = torch.cat([cur, nxt.unsqueeze(-1)], dim=2)
        cur_mask = torch.cat([cur_mask, speak.unsqueeze(-1)], dim=2)
        if cur_private is not None:
            cur_private = torch.cat(
                [cur_private, torch.zeros_like(speak, dtype=torch.bool).unsqueeze(-1)], dim=2
            )

    if was_training:
        model.train()

    tokens = torch.stack(generated_tokens, dim=2).long()   # [B, A, steps]
    speak_mask = torch.stack(generated_speak, dim=2).bool()  # [B, A, steps]
    if return_probs:
        probs = torch.stack(generated_probs, dim=2).float()  # [B, A, steps]
        return tokens, speak_mask, probs
    return tokens, speak_mask


@torch.no_grad()
def generate_next_tokens_for_all_agents(
    model,
    input_ids: torch.LongTensor,                        # [B, A, T]
    input_activity_mask: Optional[torch.Tensor] = None,  # [B, A, T]; None = all active
    channel_ids: Optional[torch.LongTensor] = None,
    temperature: float = 0.0,
    activity_threshold: float = 0.5,
    placeholder_token_id: int = 0,
    input_private_mask: Optional[torch.Tensor] = None,   # [B, A, T]; True = private cell
    agent_visibility: Optional[torch.Tensor] = None,      # [B, A, A]
) -> Tuple[torch.LongTensor, torch.BoolTensor]:
    """Return ``(next_tokens[B, A], speak[B, A])`` for every agent in one pass.

    Unlike the earlier query-column design, no placeholder column needs to be
    appended first: ``activity_logits[:, :, -1]`` already predicts the NEXT
    column's activity directly from the existing last column, and
    ``logits[:, :, -1, :]`` already predicts its content.
    """
    if input_ids.dim() != 3:
        raise ValueError(f"input_ids must be [B, A, T], got shape {tuple(input_ids.shape)}")

    out = model(
        input_ids=input_ids,
        input_activity_mask=input_activity_mask,
        channel_ids=channel_ids,
        input_private_mask=input_private_mask,
        agent_visibility=agent_visibility,
    )
    last_content = out.logits[:, :, -1, :]          # [B, A, V]
    last_activity = out.activity_logits[:, :, -1]   # [B, A]

    return _decide_and_sample(
        last_content, last_activity, temperature, activity_threshold, placeholder_token_id
    )
