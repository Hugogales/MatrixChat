"""Tests for the silence mechanism: silent cells are invisible to other cells.

A silent cell (a slot holding ``silence_token_id``) must be masked out as a key
for every query except its own diagonal. Behaviorally, an agent that is fully
silent must have ZERO influence on another agent -- identical to that agent not
being present at all (verified in ``column`` position mode, where one agent's
position ids do not depend on how many agents share the matrix).
"""

import torch

from model.masks import build_matrix_causal_mask
from model.matrix_qwen import MatrixQwenConfig, MatrixQwenForCausalLM


def test_silence_mask_blocks_silent_keys():
    """A silent key is blocked for all queries except the diagonal."""
    num_agents, seq_len = 2, 3
    seq = num_agents * seq_len
    # Mark agent 1 at t=0 (column-major flat index 1) as silent.
    silence = torch.zeros(1, seq, dtype=torch.bool)
    silent_key = 1
    silence[0, silent_key] = True

    mask = build_matrix_causal_mask(
        batch_size=1, num_agents=num_agents, seq_len=seq_len,
        device=torch.device("cpu"), dtype=torch.float32, silence_mask=silence,
    )
    neg = torch.finfo(torch.float32).min

    # Every non-diagonal query is blocked from attending the silent key.
    for q in range(seq):
        if q == silent_key:
            assert mask[0, 0, q, silent_key].item() == 0.0  # diagonal kept
        else:
            assert mask[0, 0, q, silent_key].item() == neg

    # A non-silent key still follows the normal causal rule (key 0 visible to
    # later-column queries; here query at flat index 2 = agent0,t1).
    assert mask[0, 0, 2, 0].item() == 0.0


def _build(model_base, **cfg_kwargs):
    cfg = MatrixQwenConfig(
        base_model_name="tiny-qwen3",
        max_agents=8,
        use_agent_embeddings=False,   # isolate the attention pathway
        position_mode="column",       # agent0 positions are A-invariant in column mode
        attn_mask_mode="matrix_causal",
        allow_same_column=False,
        **cfg_kwargs,
    )
    return MatrixQwenForCausalLM(model_base, cfg).eval()


def test_learned_silence_embedding_trainable_and_used(tiny_model):
    """Silent cells use the trainable silence embedding (not the frozen token embed)."""
    from model.lora_utils import freeze_base_model

    sid = 100
    cfg = MatrixQwenConfig(
        base_model_name="tiny-qwen3", max_agents=8, use_agent_embeddings=True,
        position_mode="flat", attn_mask_mode="matrix_causal",
        silence_token_id=sid, learned_silence_embedding=True,
    )
    model = MatrixQwenForCausalLM(tiny_model, cfg)
    assert model.silence_embedding is not None

    # Survives the base freeze (trainable like agent embeddings).
    freeze_base_model(model)
    assert model.silence_embedding.weight.requires_grad is True

    # A silent cell's input = silence_embedding + agent_embedding (not the base token embed).
    b, a, t = 1, 2, 3
    ids = torch.full((b, a, t), sid)
    flat = model.flatten_matrix(ids)
    flat_agent = model.flatten_matrix(model._default_agent_ids(b, a, t, ids.device))
    emb = model.get_input_embeddings_with_agents(flat, flat_agent)
    for i in range(flat.shape[1]):
        agent = int(flat_agent[0, i])
        expected = model.silence_embedding.weight[0] + model.agent_embeddings(torch.tensor(agent))
        assert torch.allclose(emb[0, i], expected, atol=1e-5)


def test_silence_loss_weight_zero_equals_ignoring_silence(tiny_model):
    """silence_loss_weight=0 must match masking silence labels to ignore_index."""
    sid = 100
    b, a, t = 1, 2, 4
    input_ids = torch.randint(1, 90, (b, a, t))
    labels = torch.full((b, a, t), -100)
    # A mix of content targets and silence targets.
    labels[0, 0, 1] = 5
    labels[0, 0, 2] = sid
    labels[0, 1, 0] = sid
    labels[0, 1, 3] = 7

    weighted = _build(tiny_model, silence_token_id=sid, silence_loss_weight=0.0)
    plain = _build(tiny_model, silence_token_id=sid, silence_loss_weight=1.0)

    labels_masked = labels.clone()
    labels_masked[labels_masked == sid] = -100

    with torch.no_grad():
        loss_weighted = weighted(input_ids=input_ids, labels=labels).loss
        loss_ignored = plain(input_ids=input_ids, labels=labels_masked).loss

    assert torch.allclose(loss_weighted, loss_ignored, atol=1e-5)


def test_fully_silent_agent_is_invisible(tiny_model):
    """Agent 0's logits are identical whether a fully-silent agent 1 exists."""
    sid = 100  # within the tiny model's vocab (128)
    model = _build(tiny_model, silence_token_id=sid)
    t = 4
    agent0 = torch.randint(1, 90, (1, 1, t))  # avoid the silence id

    solo = torch.cat([agent0], dim=1)                       # [1, 1, t]
    silent_partner = torch.full((1, 1, t), sid)
    pair = torch.cat([agent0, silent_partner], dim=1)       # [1, 2, t], agent1 silent

    with torch.no_grad():
        logits_solo = model(input_ids=solo).logits[0, 0]    # [t, V]
        logits_pair = model(input_ids=pair).logits[0, 0]    # [t, V]

    diff = (logits_solo - logits_pair).abs().max().item()
    assert diff < 1e-4, f"silent agent leaked into agent 0 (max|Δ|={diff})"


def test_speaking_agent_is_visible_control(tiny_model):
    """Control: a non-silent partner DOES change agent 0's logits."""
    sid = 100
    model = _build(tiny_model, silence_token_id=sid)
    t = 4
    agent0 = torch.randint(1, 90, (1, 1, t))

    solo = agent0
    speaking_partner = torch.randint(1, 90, (1, 1, t))
    pair = torch.cat([agent0, speaking_partner], dim=1)

    with torch.no_grad():
        logits_solo = model(input_ids=solo).logits[0, 0]
        logits_pair = model(input_ids=pair).logits[0, 0]

    # At t=0 there is no shared past, so compare the LAST column where agent 0
    # could have attended agent 1's earlier (visible) columns.
    diff_last = (logits_solo[-1] - logits_pair[-1]).abs().max().item()
    assert diff_last > 1e-4, "a speaking partner should influence agent 0"


def test_drop_silence_matches_mask(tiny_model):
    """Dropping silent cells yields identical logits at kept cells as masking.

    Uses a ragged batch (each element has a different number of silent cells) to
    exercise the pad-to-max compaction path.
    """
    sid = 100

    def make(drop):
        cfg = MatrixQwenConfig(
            base_model_name="tiny-qwen3",
            max_agents=8,
            use_agent_embeddings=False,  # share identical params across both models
            position_mode="flat",
            attn_mask_mode="matrix_causal",
            allow_same_column=False,
            silence_token_id=sid,
            drop_silence=drop,
        )
        return MatrixQwenForCausalLM(tiny_model, cfg).eval()

    masked = make(False)
    dropped = make(True)

    b, a, t = 2, 3, 4
    ids = torch.randint(1, 90, (b, a, t))  # avoid the silence id
    # Different amounts of silence per batch element -> ragged compaction.
    ids[0, 0, 2] = sid
    ids[0, 2, 0] = sid
    ids[0, 1, 3] = sid
    ids[1, 1, 1] = sid

    with torch.no_grad():
        logits_mask = masked(input_ids=ids).logits   # [B, A, T, V]
        logits_drop = dropped(input_ids=ids).logits   # [B, A, T, V]

    keep = ids != sid  # [B, A, T]
    diff = (logits_mask[keep] - logits_drop[keep]).abs().max().item()
    assert diff < 1e-4, f"drop_silence diverged from masked reference (max|Δ|={diff})"
