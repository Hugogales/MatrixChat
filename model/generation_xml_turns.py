"""XML-tagged full-turn baselines for a vanilla causal LM.

Each speaker turn is wrapped as ``<agentN>...</agentN>`` (1-indexed).  The
model generates until EOS or that agent's closing tag.  Content tokens inside
the tags are written into the matched-length continuation matrix; tags and
hidden routing prompts are scaffolding only.

Protocols:
- ``xml_round_robin``: next speaker is ``(k + 1) mod A``.
- ``pass_the_baton``: after a turn, a hidden routing question asks who should
  speak next.  The routing prompt and reply are discarded, not shown.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple

import torch

from model.generation_round_robin import _call_causal_lm, _next_token, round_robin_start_agents

XML_TURN_INSTRUCTION = (
    "Continue this multi-party conversation. Each speaker turn is wrapped in "
    "XML tags named after that speaker, for example <agent1>...</agent1>. "
    "Write only that speaker's words inside their tags.\n\n"
)
MAX_XML_CONTEXT_TOKENS = 1536
ROUTING_TEMPLATE = (
    "\n<routing>Agent {agent} finished speaking. Who should speak next? "
    "Reply with only an integer from 1 to {n}.</routing>\n"
)


def agent_open_tag(agent_index: int) -> str:
    return f"<agent{int(agent_index) + 1}>"


def agent_close_tag(agent_index: int) -> str:
    return f"</agent{int(agent_index) + 1}>"


def _encode_ids(tokenizer: Any, text: str, device: torch.device) -> torch.Tensor:
    if not text:
        return torch.zeros((0,), dtype=torch.long, device=device)
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            ids = encode(text, add_special_tokens=False)
        except TypeError:
            ids = encode(text)
    else:
        encoded = tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])
    return torch.tensor([int(value) for value in ids], dtype=torch.long, device=device)


def _decode_ids(tokenizer: Any, ids: list[int] | torch.Tensor) -> str:
    if isinstance(ids, torch.Tensor):
        ids = [int(value) for value in ids.tolist()]
    if not ids:
        return ""
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return ""
    try:
        return str(decode(ids, skip_special_tokens=True))
    except TypeError:
        return str(decode(ids))


def parse_agent_choice(text: str, num_agents: int, default: int) -> int:
    """Parse a 1-indexed agent id from hidden routing text."""
    if num_agents <= 0:
        return default
    tagged = re.search(r"<agent\s*([0-9]+)>", text or "", flags=re.IGNORECASE)
    candidates = []
    if tagged:
        candidates.append(int(tagged.group(1)))
    named = re.search(r"agent\s*([0-9]+)", text or "", flags=re.IGNORECASE)
    if named:
        candidates.append(int(named.group(1)))
    candidates.extend(int(match) for match in re.findall(r"\d+", text or ""))
    for value in candidates:
        if 1 <= value <= num_agents:
            return value - 1
    if default < 0:
        return int(default)
    return int(default) % num_agents


def matrix_prefix_to_xml(
    token_ids: torch.Tensor,
    activity: torch.Tensor,
    tokenizer: Any,
) -> str:
    """Collapse a prefix matrix into XML-wrapped sole-speaker turns."""
    if token_ids.dim() != 2 or activity.shape != token_ids.shape:
        raise ValueError("token_ids and activity must be [A, T]")
    agents, length = token_ids.shape
    parts: list[str] = []
    current: int | None = None
    buffer: list[int] = []

    def flush() -> None:
        nonlocal current, buffer
        if current is None or not buffer:
            current = None
            buffer = []
            return
        text = _decode_ids(tokenizer, buffer)
        parts.append(f"{agent_open_tag(current)}{text}{agent_close_tag(current)}")
        current = None
        buffer = []

    for column in range(length):
        speakers = [agent for agent in range(agents) if bool(activity[agent, column])]
        if len(speakers) != 1:
            flush()
            for agent in speakers:
                text = _decode_ids(tokenizer, [int(token_ids[agent, column])])
                if text:
                    parts.append(f"{agent_open_tag(agent)}{text}{agent_close_tag(agent)}")
            continue
        agent = speakers[0]
        if current is not None and agent != current:
            flush()
        current = agent
        buffer.append(int(token_ids[agent, column]))
    flush()
    return "\n".join(parts)


def _initial_context(
    tokenizer: Any,
    instruction: str,
    transcript: str,
    device: torch.device,
    placeholder_token_id: int,
) -> torch.Tensor:
    instruction_ids = _encode_ids(tokenizer, instruction or "", device)
    body = _encode_ids(tokenizer, f"{transcript}\n" if transcript else "", device)
    budget = max(1, int(MAX_XML_CONTEXT_TOKENS) - int(instruction_ids.numel()))
    if body.numel() > budget:
        body = body[-budget:]
    if instruction_ids.numel() == 0 and body.numel() == 0:
        return torch.tensor([int(placeholder_token_id)], dtype=torch.long, device=device)
    if instruction_ids.numel() == 0:
        return body
    if body.numel() == 0:
        return instruction_ids
    return torch.cat([instruction_ids, body], dim=0)


def _append_tokens(
    model: Any,
    context: torch.Tensor,
    past: Any,
    new_ids: torch.Tensor,
) -> tuple[torch.Tensor, Any, Any]:
    """Append tokens to context, forwarding so logits/past stay current."""
    output = None
    if new_ids.numel() == 0:
        output, past = _call_causal_lm(model, context.unsqueeze(0), past)
        return context, past, output
    for token in new_ids:
        context = torch.cat([context, token.view(1).to(device=context.device, dtype=context.dtype)], dim=0)
        output, past = _call_causal_lm(model, context.unsqueeze(0), past)
    return context, past, output


_AGENT_MARKUP_RE = re.compile(r"</?agent\s*\d+\s*>", flags=re.IGNORECASE)


def _extract_content(generated_text: str, agent_index: int) -> tuple[str, bool, bool]:
    """Return (content, turn_done, saw_own_close).

    The turn ends at this agent's closing tag, EOS (caller), or any other
    ``<agentN>`` / ``</agentN>`` markup the model hallucinated.  Foreign tags
    are never written into the continuation matrix.
    """
    close = agent_close_tag(agent_index)
    open_tag = agent_open_tag(agent_index)
    text = (generated_text or "").replace(open_tag, "")
    saw_own_close = close in text
    content = text.split(close, 1)[0] if saw_own_close else text
    extra = _AGENT_MARKUP_RE.search(content)
    if extra:
        content = content[: extra.start()]
    if not saw_own_close and extra is None:
        for length in range(len(close), 0, -1):
            if content.endswith(close[:length]):
                content = content[: -length]
                break
    turn_done = saw_own_close or extra is not None
    return content, turn_done, saw_own_close


def _hidden_next_agent(
    model: Any,
    tokenizer: Any,
    context: torch.Tensor,
    past: Any,
    current_agent: int,
    num_agents: int,
    temperature: float,
    max_routing_tokens: int = 16,
) -> int:
    """Ask who should speak next; discard the prompt and the reply."""
    fallback = (current_agent + 1) % num_agents
    prompt = ROUTING_TEMPLATE.format(agent=current_agent + 1, n=num_agents)
    route_ids = _encode_ids(tokenizer, prompt, context.device)
    route_context, route_past, output = _append_tokens(model, context, past, route_ids)
    if output is None:
        return fallback
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pieces: list[int] = []
    logits = output.logits if hasattr(output, "logits") else output[0]
    for _ in range(max(1, int(max_routing_tokens))):
        chosen = _next_token(logits[0, -1], temperature)
        if eos_id is not None and int(chosen) == int(eos_id):
            break
        pieces.append(int(chosen))
        route_context = torch.cat([route_context, chosen.view(1)], dim=0)
        output, route_past = _call_causal_lm(model, route_context.unsqueeze(0), route_past)
        logits = output.logits if hasattr(output, "logits") else output[0]
        parsed = parse_agent_choice(_decode_ids(tokenizer, pieces), num_agents, -1)
        if parsed >= 0:
            return parsed
    return parse_agent_choice(_decode_ids(tokenizer, pieces), num_agents, fallback)


@torch.no_grad()
def generate_xml_turn_protocol(
    model,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    tokenizer: Any,
    protocol: str = "xml_round_robin",
    input_activity_mask: Optional[torch.Tensor] = None,
    temperature: float = 0.0,
    activity_threshold: float = 0.5,
    placeholder_token_id: int = 0,
    input_private_mask: Optional[torch.Tensor] = None,
    agent_visibility: Optional[torch.Tensor] = None,
    return_probs: bool = False,
    instruction: str = XML_TURN_INSTRUCTION,
) -> Tuple[torch.LongTensor, torch.BoolTensor] | Tuple[torch.LongTensor, torch.BoolTensor, torch.Tensor]:
    """Generate XML-tagged turns into a matched-length [B, A, T] matrix."""
    del activity_threshold, input_private_mask, agent_visibility
    if protocol not in {"xml_round_robin", "pass_the_baton"}:
        raise ValueError(f"unsupported xml protocol: {protocol}")
    if tokenizer is None:
        raise ValueError("tokenizer is required for XML-turn generation")
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

    device = input_ids.device
    tokens = input_ids.new_full((batch, agents, max_new_tokens), int(placeholder_token_id))
    speak = torch.zeros((batch, agents, max_new_tokens), dtype=torch.bool, device=device)
    probs = torch.zeros((batch, agents, max_new_tokens), dtype=torch.float32, device=device)
    starts = round_robin_start_agents(activity)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    max_turns = max(max_new_tokens, agents * 4)

    for row in range(batch):
        transcript = matrix_prefix_to_xml(input_ids[row], activity[row], tokenizer)
        context = _initial_context(
            tokenizer, instruction, transcript, device, int(placeholder_token_id)
        )
        past = None
        output = None
        filled = 0
        current = int(starts[row]) % agents
        empty_streak = 0
        for _turn in range(max_turns):
            if filled >= max_new_tokens:
                break
            open_ids = _encode_ids(tokenizer, agent_open_tag(current), device)
            context, past, output = _append_tokens(model, context, past, open_ids)
            if output is None:
                break
            logits = output.logits if hasattr(output, "logits") else output[0]
            generated: list[int] = []
            remaining = max_new_tokens - filled
            # Slack lets the model emit the closing tag after the content.
            for _step in range(remaining + 32):
                chosen = _next_token(logits[0, -1], temperature)
                if eos_id is not None and int(chosen) == int(eos_id):
                    break
                generated.append(int(chosen))
                context = torch.cat([context, chosen.view(1)], dim=0)
                output, past = _call_causal_lm(model, context.unsqueeze(0), past)
                logits = output.logits if hasattr(output, "logits") else output[0]
                _content_so_far, closed, _saw_own = _extract_content(
                    _decode_ids(tokenizer, generated), current
                )
                if closed:
                    break
            content, _closed, saw_own_close = _extract_content(
                _decode_ids(tokenizer, generated), current
            )
            content_ids = _encode_ids(tokenizer, content, device)
            if not saw_own_close:
                close_ids = _encode_ids(tokenizer, agent_close_tag(current), device)
                context, past, output = _append_tokens(model, context, past, close_ids)
            wrote = 0
            for token in content_ids[:remaining]:
                tokens[row, current, filled] = token
                speak[row, current, filled] = True
                probs[row, current, filled] = 1.0
                filled += 1
                wrote += 1
                if filled >= max_new_tokens:
                    break
            if wrote == 0:
                empty_streak += 1
                if empty_streak >= agents:
                    break
            else:
                empty_streak = 0
            if protocol == "xml_round_robin":
                current = (current + 1) % agents
            else:
                current = _hidden_next_agent(
                    model, tokenizer, context, past, current, agents, temperature
                )

    if was_training:
        train_fn = getattr(model, "train", None)
        if callable(train_fn):
            train_fn()

    if return_probs:
        return tokens.long(), speak, probs
    return tokens.long(), speak


def make_xml_generate_fn(tokenizer: Any, protocol: str):
    """Adapter matching generate_matrix / generate_round_robin_vanilla kwargs."""

    def generate_fn(model, input_ids, max_new_tokens, **kwargs):
        return generate_xml_turn_protocol(
            model,
            input_ids,
            max_new_tokens,
            tokenizer=tokenizer,
            protocol=protocol,
            **kwargs,
        )

    return generate_fn
