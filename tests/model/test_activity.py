"""Tests for the turn-taking activity mechanism (replaces the old silence-token
design): inactive cells are invisible via an explicit mask, the inactive-cell
embedding + activity head are always present and trainable, content loss is
gated by activity, activity BCE is balanced, and the turn-taking reward matches
hand-computed values.
"""

import math

import pytest
import torch

from model.masks import build_matrix_causal_mask
from model.matrix_qwen import MatrixQwenConfig, MatrixQwenForCausalLM
from model.turn_reward import (
    FloorControlConfig,
    TurnRewardConfig,
    floor_control_reward,
    ground_truth_run_lengths,
    same_handoff_cross_entropy,
    turn_taking_reward,
)


# ---------------------------------------------------------------------------
# Attention invisibility (formerly "silence mask")
# ---------------------------------------------------------------------------


def test_inactive_mask_blocks_inactive_keys():
    """An inactive key is blocked for all queries except the diagonal."""
    num_agents, seq_len = 2, 3
    seq = num_agents * seq_len
    inactive = torch.zeros(1, seq, dtype=torch.bool)
    inactive_key = 1
    inactive[0, inactive_key] = True

    mask = build_matrix_causal_mask(
        batch_size=1, num_agents=num_agents, seq_len=seq_len,
        device=torch.device("cpu"), dtype=torch.float32, inactive_mask=inactive,
    )
    neg = torch.finfo(torch.float32).min

    for q in range(seq):
        if q == inactive_key:
            assert mask[0, 0, q, inactive_key].item() == 0.0  # diagonal kept
        else:
            assert mask[0, 0, q, inactive_key].item() == neg

    assert mask[0, 0, 2, 0].item() == 0.0  # normal causal rule unaffected


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


def test_fully_inactive_agent_is_invisible(tiny_model):
    """Agent 0's logits are identical whether a fully-inactive agent 1 exists."""
    model = _build(tiny_model)
    t = 4
    agent0 = torch.randint(1, 90, (1, 1, t))
    solo_mask = torch.ones(1, 1, t, dtype=torch.bool)

    pair = torch.cat([agent0, torch.randint(1, 90, (1, 1, t))], dim=1)  # content irrelevant when inactive
    pair_mask = torch.cat([torch.ones(1, 1, t, dtype=torch.bool), torch.zeros(1, 1, t, dtype=torch.bool)], dim=1)

    with torch.no_grad():
        logits_solo = model(input_ids=agent0, input_activity_mask=solo_mask).logits[0, 0]
        logits_pair = model(input_ids=pair, input_activity_mask=pair_mask).logits[0, 0]

    diff = (logits_solo - logits_pair).abs().max().item()
    assert diff < 1e-4, f"inactive agent leaked into agent 0 (max|Δ|={diff})"


def test_speaking_agent_is_visible_control(tiny_model):
    """Control: an ACTIVE partner DOES change agent 0's logits."""
    model = _build(tiny_model)
    t = 4
    agent0 = torch.randint(1, 90, (1, 1, t))
    speaking_partner = torch.randint(1, 90, (1, 1, t))
    pair = torch.cat([agent0, speaking_partner], dim=1)
    all_active = torch.ones(1, 2, t, dtype=torch.bool)
    solo_mask = torch.ones(1, 1, t, dtype=torch.bool)

    with torch.no_grad():
        logits_solo = model(input_ids=agent0, input_activity_mask=solo_mask).logits[0, 0]
        logits_pair = model(input_ids=pair, input_activity_mask=all_active).logits[0, 0]

    diff_last = (logits_solo[-1] - logits_pair[-1]).abs().max().item()
    assert diff_last > 1e-4, "an active partner should influence agent 0"


# ---------------------------------------------------------------------------
# Inactive-cell embedding + activity head: always present, always trainable
# ---------------------------------------------------------------------------


def test_inactive_embedding_and_activity_head_always_exist(tiny_model, matrix_config):
    model = MatrixQwenForCausalLM(tiny_model, matrix_config)
    assert model.inactive_embedding is not None
    assert model.activity_head is not None
    assert model.inactive_embedding.weight.shape == (1, model.hidden_size)
    assert model.activity_head.weight.shape == (1, model.hidden_size)


def test_inactive_embedding_trainable_and_used(tiny_model):
    """Inactive cells use the trainable inactive embedding (not the base token embed)."""
    from model.lora_utils import freeze_base_model

    cfg = MatrixQwenConfig(
        base_model_name="tiny-qwen3", max_agents=8, use_agent_embeddings=True,
        position_mode="flat", attn_mask_mode="matrix_causal",
    )
    model = MatrixQwenForCausalLM(tiny_model, cfg)

    freeze_base_model(model)
    assert model.inactive_embedding.weight.requires_grad is True
    assert model.activity_head.weight.requires_grad is True

    b, a, t = 1, 2, 3
    ids = torch.randint(1, 90, (b, a, t))
    mask = torch.zeros(b, a, t, dtype=torch.bool)  # everything inactive
    flat = model.flatten_matrix(ids)
    flat_mask = model.flatten_matrix(mask)
    flat_agent = model.flatten_matrix(model._default_agent_ids(b, a, t, ids.device))
    emb = model.get_input_embeddings_with_agents(flat, flat_agent, flat_activity_mask=flat_mask)
    for i in range(flat.shape[1]):
        agent = int(flat_agent[0, i])
        expected = model.inactive_embedding.weight[0] + model.agent_embeddings(torch.tensor(agent))
        assert torch.allclose(emb[0, i], expected, atol=1e-5)


def test_activity_head_handles_mixed_precision_base_model(tiny_model, matrix_config):
    """The base model may run in bf16 while wrapper params (agent/inactive
    embeddings, activity_head) stay fp32 for stable optimization -- the
    activity head must not crash with a dtype mismatch against the bf16
    hidden state (regression test for a real bug hit during a live training
    run: `RuntimeError: mat1 and mat2 must have the same dtype`)."""
    tiny_model_bf16 = tiny_model.to(torch.bfloat16)
    model = MatrixQwenForCausalLM(tiny_model_bf16, matrix_config).eval()
    assert model.activity_head.weight.dtype == torch.float32  # wrapper stays fp32

    b, a, t = 1, 2, 4
    ids = torch.randint(1, model.vocab_size, (b, a, t))
    with torch.no_grad():
        out = model(input_ids=ids)
    assert out.activity_logits.dtype == torch.float32
    assert torch.isfinite(out.activity_logits).all()


def test_none_activity_mask_means_all_active(tiny_model, matrix_config):
    """Default (no mask) behaves exactly like an all-active mask (back-compat)."""
    model = MatrixQwenForCausalLM(tiny_model, matrix_config).eval()
    b, a, t = 1, 2, 4
    ids = torch.randint(1, model.vocab_size, (b, a, t))
    all_active = torch.ones(b, a, t, dtype=torch.bool)

    with torch.no_grad():
        out_none = model(input_ids=ids)
        out_mask = model(input_ids=ids, input_activity_mask=all_active)

    assert torch.equal(out_none.logits, out_mask.logits)


# ---------------------------------------------------------------------------
# Loss gating: content CE only where activity_label speaks; balanced activity BCE
# ---------------------------------------------------------------------------


def test_content_loss_ignores_positions_without_labels(tiny_model, matrix_config):
    model = MatrixQwenForCausalLM(tiny_model, matrix_config).eval()
    b, a, t = 1, 2, 4
    ids = torch.randint(1, model.vocab_size, (b, a, t))
    labels = torch.full((b, a, t), -100, dtype=torch.long)
    labels[0, 0, 1] = 5  # a single content target

    out = model(input_ids=ids, labels=labels)
    assert out.content_loss is not None
    assert torch.isfinite(out.content_loss)
    assert out.activity_loss is None  # no activity_labels given
    assert torch.allclose(out.loss, out.content_loss)


def test_activity_loss_balanced_under_class_imbalance(tiny_model, matrix_config):
    """A batch with far more yield (0) than speak (1) labels shouldn't silently
    drop the speak class from the loss (the two classes are averaged equally)."""
    model = MatrixQwenForCausalLM(tiny_model, matrix_config).eval()
    b, a, t = 1, 1, 20
    ids = torch.randint(1, model.vocab_size, (b, a, t))
    activity_labels = torch.zeros(b, a, t, dtype=torch.long)
    activity_labels[0, 0, 5] = 1  # exactly one "speak" label among many "yield"
    activity_labels[0, 0, -1] = -100  # last column undefined

    out = model(input_ids=ids, activity_labels=activity_labels)
    assert out.activity_loss is not None
    assert torch.isfinite(out.activity_loss)

    # Compare against a manual balanced computation using the SAME activity logits.
    flat_labels = model.flatten_matrix(activity_labels)
    flat_logits = model.flatten_matrix(out.activity_logits)
    valid = flat_labels != -100
    pos = valid & (flat_labels == 1)
    neg = valid & (flat_labels == 0)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        flat_logits.float(), flat_labels.clamp(min=0).float(), reduction="none"
    )
    expected = 0.5 * bce[pos].mean() + 0.5 * bce[neg].mean()
    assert torch.allclose(out.activity_loss, expected, atol=1e-5)


def test_activity_pos_weight_biases_loss_toward_penalizing_missed_speak(tiny_model, matrix_config):
    """activity_pos_weight > 0.5 should weight the speak-class BCE more than
    the yield-class BCE (directly, not just via the differentiable reward)."""
    import dataclasses

    biased_config = dataclasses.replace(matrix_config, activity_pos_weight=0.8)
    model = MatrixQwenForCausalLM(tiny_model, biased_config).eval()
    b, a, t = 1, 1, 20
    ids = torch.randint(1, model.vocab_size, (b, a, t))
    activity_labels = torch.zeros(b, a, t, dtype=torch.long)
    activity_labels[0, 0, 5] = 1
    activity_labels[0, 0, -1] = -100

    out = model(input_ids=ids, activity_labels=activity_labels)
    flat_labels = model.flatten_matrix(activity_labels)
    flat_logits = model.flatten_matrix(out.activity_logits)
    valid = flat_labels != -100
    pos = valid & (flat_labels == 1)
    neg = valid & (flat_labels == 0)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        flat_logits.float(), flat_labels.clamp(min=0).float(), reduction="none"
    )
    expected = 0.8 * bce[pos].mean() + 0.2 * bce[neg].mean()
    assert torch.allclose(out.activity_loss, expected, atol=1e-5)


def test_activity_pos_weight_default_matches_prior_balanced_behavior(matrix_config):
    assert matrix_config.activity_pos_weight == 0.5


def test_lambda_content_scales_content_loss_in_total_loss(tiny_model, matrix_config):
    import dataclasses

    b, a, t = 1, 1, 6
    ids = torch.randint(1, 50, (b, a, t))
    labels = torch.randint(1, 50, (b, a, t))
    activity_labels = torch.ones(b, a, t, dtype=torch.long)

    full_weight = MatrixQwenForCausalLM(tiny_model, matrix_config).eval()
    out_full = full_weight(input_ids=ids, labels=labels, activity_labels=activity_labels)

    half_config = dataclasses.replace(matrix_config, lambda_content=0.5, lambda_reward=0.0)
    half_weight = MatrixQwenForCausalLM(tiny_model, half_config).eval()
    # Load the same weights so only lambda_content differs.
    half_weight.load_state_dict(full_weight.state_dict())
    out_half = half_weight(input_ids=ids, labels=labels, activity_labels=activity_labels)

    assert out_full.content_loss is not None and out_half.content_loss is not None
    assert torch.allclose(out_full.content_loss, out_half.content_loss, atol=1e-5)
    expected_half_total = 0.5 * out_full.content_loss + full_weight.matrix_config.lambda_activity * out_full.activity_loss
    assert torch.allclose(out_half.loss, expected_half_total, atol=1e-4)


def test_content_loss_is_zero_not_nan_when_no_valid_targets(tiny_model, matrix_config):
    """A batch with ZERO valid content labels (e.g. every target turn is a
    single token, so no internal continuation exists) must yield a defined
    zero content loss, not NaN (which would poison any running average, such
    as a validation-set mean computed by summing per-batch losses)."""
    model = MatrixQwenForCausalLM(tiny_model, matrix_config).eval()
    b, a, t = 1, 2, 4
    ids = torch.randint(1, model.vocab_size, (b, a, t))
    labels = torch.full((b, a, t), -100, dtype=torch.long)  # nothing to supervise

    out = model(input_ids=ids, labels=labels)
    assert out.content_loss is not None
    assert torch.isfinite(out.content_loss)
    assert float(out.content_loss) == 0.0
    assert torch.isfinite(out.loss)


def test_activity_labels_without_activity_mask_skips_reward(tiny_model, matrix_config):
    """The turn-taking reward requires input_activity_mask; without it, only
    the activity BCE term is computed (no reward term)."""
    model = MatrixQwenForCausalLM(tiny_model, matrix_config).eval()
    b, a, t = 1, 2, 6
    ids = torch.randint(1, model.vocab_size, (b, a, t))
    activity_labels = torch.randint(0, 2, (b, a, t))
    activity_labels[:, :, -1] = -100

    out = model(input_ids=ids, activity_labels=activity_labels)
    assert out.activity_loss is not None
    assert out.turn_reward is None


def test_loss_defined_when_only_reward_is_present(tiny_model, matrix_config):
    """If activity_labels are given but ALL are ignored (-100) while
    input_activity_mask is present, only the reward term is non-None. `loss`
    must still be defined (not None) so `.backward()` doesn't crash."""
    model = MatrixQwenForCausalLM(tiny_model, matrix_config).eval()
    b, a, t = 1, 2, 6
    ids = torch.randint(1, model.vocab_size, (b, a, t))
    mask = torch.randint(0, 2, (b, a, t)).bool()
    activity_labels = torch.full((b, a, t), -100, dtype=torch.long)  # all ignored

    out = model(input_ids=ids, input_activity_mask=mask, activity_labels=activity_labels)
    assert out.content_loss is None
    assert out.activity_loss is None
    assert out.turn_reward is not None
    assert out.loss is not None
    assert torch.isfinite(out.loss)


def test_activity_labels_with_activity_mask_computes_reward(tiny_model, matrix_config):
    model = MatrixQwenForCausalLM(tiny_model, matrix_config).eval()
    b, a, t = 1, 2, 6
    ids = torch.randint(1, model.vocab_size, (b, a, t))
    mask = torch.randint(0, 2, (b, a, t)).bool()
    activity_labels = torch.zeros(b, a, t, dtype=torch.long)
    activity_labels[:, :, :-1] = mask[:, :, 1:].long()
    activity_labels[:, :, -1] = -100

    out = model(input_ids=ids, input_activity_mask=mask, activity_labels=activity_labels)
    assert out.turn_reward is not None
    assert torch.isfinite(out.turn_reward)


# ---------------------------------------------------------------------------
# Ground-truth run lengths + turn-taking reward (hand-computed check)
# ---------------------------------------------------------------------------


def test_ground_truth_run_lengths_basic_pattern():
    # A=2, T=6. Columns: [A][A][gap][B][B][B]  (agent0=A rows, agent1=B rows)
    # col:      0    1    2    3    4    5
    # agent0:   1    1    0    0    0    0
    # agent1:   0    0    0    1    1    1
    mask = torch.tensor([[
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
    ]], dtype=torch.bool)  # [1, 2, 6]

    x, z, owner = ground_truth_run_lengths(mask)
    # x[t]/z[t]/owner[t] describe the state BEFORE column t (i.e. ending at t-1).
    # col:                 0    1    2    3    4    5
    assert x[0].tolist() == [0, 1, 2, 0, 1, 2]     # agent0 x2, reset at gap, agent1 x2 growing
    assert z[0].tolist() == [0, 0, 0, 1, 0, 0]     # exactly one all-silent column (col 2)
    assert owner[0].tolist() == [-1, 0, 0, -1, 1, 1]


def test_turn_taking_reward_matches_handcomputed_values():
    """With near-saturated logits (q ~ 0 or 1), the reward should match the
    hand-derived formula almost exactly for a simple scripted scenario."""
    # A=2, T=3: agent0 speaks col0, silence col1, agent1 speaks col2.
    mask = torch.tensor([[
        [1, 0, 0],
        [0, 0, 1],
    ]], dtype=torch.bool)  # [1, 2, 3]

    # Saturated logits: agent0 P(speak)~0 after col0 (large negative), agent1 P(speak)~1 at col2.
    BIG = 20.0
    activity_logits = torch.tensor([[
        [BIG, -BIG, -BIG],   # agent0: predicts speak@1(no,-> but col0 predicts col1)... see below
        [-BIG, -BIG, BIG],   # agent1
    ]], dtype=torch.float32)  # [1, 2, 3]; logits[:, :, c] predicts activity at c+1

    cfg = TurnRewardConfig(
        speak_grace=50.0,
        speak_tau=2500.0,
        silence_grace=0.0,
        silence_tau=20.0,
    )
    reward, stats = turn_taking_reward(activity_logits, mask, cfg=cfg)
    assert torch.isfinite(reward)

    # Manually recompute for t=1 (from logits[:, :, 0]) and t=2 (from logits[:, :, 1]).
    q = torch.sigmoid(activity_logits[:, :, :-1])  # [1, 2, 2] -> columns 1, 2
    x_gt, z_gt, owner_gt = ground_truth_run_lengths(mask)

    def expected_reward_at(t):
        qt = q[0, :, t - 1]
        p0 = float((1 - qt).prod())
        p_only = [float(qt[i] * (1 - qt[1 - i])) for i in range(2)]
        owner = int(owner_gt[0, t])
        if owner >= 0:
            p_same = p_only[owner]
            p_handoff = sum(p_only) - p_same
        else:
            p_same, p_handoff = 0.0, sum(p_only)
        s = math.exp(-max(0.0, float(x_gt[0, t]) - cfg.speak_grace) / cfg.speak_tau)
        w = 1.0 - math.exp(-float(z_gt[0, t]) / cfg.silence_tau)
        return s * p_same + p_handoff - w * p0

    expected_mean = (expected_reward_at(1) + expected_reward_at(2)) / 2.0
    assert abs(float(reward) - expected_mean) < 1e-3


def test_overlap_probability_and_sustained_penalty_are_explicit():
    # Both agents have q=0.8, so P(overlap) = 0.8 * 0.8 = 0.64.
    logit_08 = math.log(4.0)
    logits = torch.full((1, 2, 5), logit_08)
    mask = torch.ones((1, 2, 5), dtype=torch.bool)  # sustained human overlap

    flat_cfg = TurnRewardConfig(
        overlap_base_weight=0.35,
        overlap_max_weight=0.35,
        overlap_grace=0.0,
        overlap_tau=1.0,
    )
    strong_cfg = TurnRewardConfig(
        overlap_base_weight=0.35,
        overlap_max_weight=1.5,
        overlap_grace=0.0,
        overlap_tau=1.0,
    )
    flat_reward, flat_stats = turn_taking_reward(logits, mask, cfg=flat_cfg)
    strong_reward, strong_stats = turn_taking_reward(logits, mask, cfg=strong_cfg)

    assert abs(flat_stats["p_overlap_mean"] - 0.64) < 1e-5
    assert strong_stats["overlap_weight_mean"] > flat_stats["overlap_weight_mean"]
    assert float(strong_reward) < float(flat_reward)


def test_handoff_bonus_weight_scales_reward_with_p_handoff_and_defaults_to_noop():
    # Agent 0 was the sole speaker at t-1; agent 1 has higher q at t, so a
    # handoff to agent 1 is more likely than agent 0 continuing.
    logits = torch.zeros((1, 2, 3))
    logits[0, 0, 0] = 2.0   # strong "agent 0 speaks at t=1" (irrelevant to bonus)
    logits[0, 0, 1] = -1.0  # agent 0 less likely to continue at t=2
    logits[0, 1, 1] = 1.0   # agent 1 more likely to take over at t=2
    mask = torch.zeros((1, 2, 3), dtype=torch.bool)
    mask[0, 0, 0] = True  # agent 0 is the sole speaker going into column 1

    base_cfg = TurnRewardConfig(handoff_bonus_weight=0.0)
    bonus_cfg = TurnRewardConfig(handoff_bonus_weight=1.0)
    base_reward, base_stats = turn_taking_reward(logits, mask, cfg=base_cfg)
    bonus_reward, bonus_stats = turn_taking_reward(logits, mask, cfg=bonus_cfg)

    assert base_stats["p_handoff_mean"] == bonus_stats["p_handoff_mean"]
    assert base_stats["p_handoff_mean"] > 0.0
    expected_extra = bonus_cfg.handoff_bonus_weight * bonus_stats["p_handoff_mean"]
    assert abs(float(bonus_reward) - float(base_reward) - expected_extra) < 1e-4


def test_same_handoff_cross_entropy_scores_only_the_genuine_boundary():
    # agent0 speaks cols 0-1 (one run), agent1 speaks cols 2-3 (immediate
    # handoff right after agent0's run ends, then continues). Only the
    # col1->col2 transition is a genuine new arrival; the mid-run
    # continuations at col0->col1 and col2->col3 must NOT be scored, since
    # they are not turn-taking decisions at all.
    mask = torch.zeros((1, 2, 4), dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 0, 1] = True
    mask[0, 1, 2] = True
    mask[0, 1, 3] = True
    labels = torch.full((1, 2, 4), -100, dtype=torch.long)
    labels[0, :, 0] = torch.tensor([1, 0])  # next column (1): agent0 continues
    labels[0, :, 1] = torch.tensor([0, 1])  # next column (2): agent1 arrives (handoff)
    labels[0, :, 2] = torch.tensor([0, 1])  # next column (3): agent1 continues

    correct = torch.zeros((1, 2, 4), requires_grad=True)
    correct.data[0, :, 1] = torch.tensor([-4.0, 4.0])  # predicts handoff at the boundary
    wrong = torch.zeros((1, 2, 4))
    wrong[0, :, 1] = torch.tensor([4.0, -4.0])  # predicts same at the boundary

    correct_loss, stats = same_handoff_cross_entropy(correct, mask, labels)
    wrong_loss, _ = same_handoff_cross_entropy(wrong, mask, labels)
    assert float(correct_loss) < 0.05
    assert float(wrong_loss) > 4.0
    assert stats["count"] == 1  # only the col1->col2 boundary, not the two trivial continuations
    assert stats["target_handoff_rate"] == 1.0
    correct_loss.backward()
    assert correct.grad is not None


def test_same_handoff_cross_entropy_carries_owner_forward_through_silence():
    # agent0 speaks col0, then 2 silent columns, then agent0 RESUMES at col3
    # (self-resumption after a real gap, a genuine decision point) and
    # continues at col4. The carried-forward owner must survive the silence
    # gap so this resumption is correctly scored as "same", not missed
    # entirely (the pre-fix version could never score a transition spanning
    # a gap at all, since it only ever compared literally adjacent columns).
    mask = torch.zeros((1, 2, 5), dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 0, 3] = True
    mask[0, 0, 4] = True
    labels = torch.full((1, 2, 5), -100, dtype=torch.long)
    labels[0, :, 0] = torch.tensor([0, 0])  # next column (1): silence
    labels[0, :, 1] = torch.tensor([0, 0])  # next column (2): silence
    labels[0, :, 2] = torch.tensor([1, 0])  # next column (3): agent0 resumes
    labels[0, :, 3] = torch.tensor([1, 0])  # next column (4): agent0 continues

    logits = torch.zeros((1, 2, 5))
    logits[0, :, 2] = torch.tensor([4.0, -4.0])  # predicts self-resume (correct)

    loss, stats = same_handoff_cross_entropy(logits, mask, labels)
    assert stats["count"] == 1  # only the col2->col3 resumption boundary
    assert stats["target_handoff_rate"] == 0.0  # same speaker, not a handoff
    assert float(loss) < 0.05


def test_same_handoff_cross_entropy_ignores_overlap_columns():
    # col0: agent0 alone. col1: agent1 alone (a genuine handoff boundary,
    # the only transition that should be scored). col2: BOTH agents active
    # (overlap) -- the col1->col2 transition must not be scored (the
    # following state is not exactly-one-owner), and logits at column 1
    # (which only affects that unscored transition) must not change the loss.
    mask = torch.zeros((1, 2, 3), dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 1, 1] = True
    mask[0, 0, 2] = True
    mask[0, 1, 2] = True
    labels = torch.full((1, 2, 3), -100, dtype=torch.long)
    labels[0, :, 0] = torch.tensor([0, 1])  # next column (1): agent1 arrives (handoff)
    labels[0, :, 1] = torch.tensor([1, 1])  # next column (2): overlap

    logits_a = torch.zeros((1, 2, 3))
    logits_a[0, :, 0] = torch.tensor([-2.0, 2.0])
    logits_b = logits_a.clone()
    logits_b[0, :, 1] = torch.tensor([20.0, -20.0])

    loss_a, stats_a = same_handoff_cross_entropy(logits_a, mask, labels)
    loss_b, stats_b = same_handoff_cross_entropy(logits_b, mask, labels)
    assert torch.allclose(loss_a, loss_b)
    assert stats_a["count"] == stats_b["count"] == 1
    assert stats_a["target_handoff_rate"] == 1.0


# ---------------------------------------------------------------------------
# floor_control_reward (2026-08-24 rearchitecture: balanced 4-way proper
# scoring, no hand-tuned shape parameters -- see model/turn_reward.py).
# ---------------------------------------------------------------------------


def test_floor_control_reward_matches_handcomputed_values():
    """A=2, T=3: agent0 speaks col0-1 (a SAME transition into col1), agent1
    speaks col2 (a HANDOFF transition into col2). Non-saturated q values so
    the exact analytic reward can be hand-verified."""
    mask = torch.tensor([[
        [1, 1, 0],
        [0, 0, 1],
    ]], dtype=torch.bool)  # [1, 2, 3]

    def logit(p):
        return math.log(p / (1.0 - p))

    activity_logits = torch.zeros((1, 2, 3))
    # Column 0 -> predicts column 1 activity (ground truth: SAME, agent0).
    activity_logits[0, 0, 0] = logit(0.6)
    activity_logits[0, 1, 0] = logit(0.1)
    # Column 1 -> predicts column 2 activity (ground truth: HANDOFF to agent1).
    activity_logits[0, 0, 1] = logit(0.2)
    activity_logits[0, 1, 1] = logit(0.7)

    cfg = FloorControlConfig()
    reward, stats = floor_control_reward(activity_logits, mask, cfg=cfg)

    p_same = 0.6 * (1 - 0.1)      # 0.54
    p_only0 = 0.2 * (1 - 0.7)     # 0.06
    p_only1 = 0.7 * (1 - 0.2)     # 0.56
    p_handoff = (p_only0 + p_only1) - p_only0   # p_same-at-t2 is p_only0 (prior owner=0)

    same_reward = math.log(p_same) - math.log(cfg.ref_same)
    handoff_reward = math.log(p_handoff) - math.log(cfg.ref_handoff)
    expected = (same_reward + handoff_reward) / 2.0

    assert abs(float(reward) - expected) < 1e-4
    assert stats["same_count"] == 1
    assert stats["handoff_count"] == 1
    assert stats["silence_count"] == 0
    assert stats["overlap_count"] == 0


def test_floor_control_reward_same_after_gap_is_same_not_handoff():
    """agent0 speaks col0, 2 silent columns, agent0 RESUMES at col3. The
    resumption must be scored as SAME (carry-forward owner), not HANDOFF --
    this is the exact imprecision the 2026-08-24 rearchitecture fixes
    relative to the old ground_truth_run_lengths-based owner tracking."""
    mask = torch.zeros((1, 2, 4), dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 0, 3] = True

    logits = torch.zeros((1, 2, 4))
    reward, stats = floor_control_reward(logits, mask)

    assert stats["silence_count"] == 2   # columns 1, 2
    assert stats["same_count"] == 1      # column 3: agent0 resumes -> SAME
    assert stats["handoff_count"] == 0
    assert torch.isfinite(reward)


def test_floor_control_reward_no_prior_owner_counts_as_handoff():
    """The very first speaker of a conversation (no carried-forward owner
    yet) is a genuine floor-take, scored as HANDOFF, not SAME."""
    mask = torch.zeros((1, 2, 2), dtype=torch.bool)
    mask[0, 0, 1] = True  # agent0 is the very first-ever speaker, at col1
    logits = torch.zeros((1, 2, 2))
    _, stats = floor_control_reward(logits, mask)
    assert stats["handoff_count"] == 1
    assert stats["same_count"] == 0


def test_floor_control_reward_punishes_multi_claim_hedging():
    """Defensive/exploit test: when ground truth is SAME (agent0 continues),
    an agent0-confident prediction that ALSO hedges by raising other
    agents' P(speak) must score WORSE, not better -- ruling out the
    joint-activation exploit that made the old conditionally-renormalized
    same_handoff_cross_entropy gameable (see floor_control_reward's
    docstring). Because p_same is an exact leave-one-out product, raising
    ANY other agent's probability strictly multiplies p_same down."""
    mask = torch.zeros((1, 3, 2), dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 0, 1] = True  # agent0 continues -> ground truth SAME at t=1

    def logit(p):
        return math.log(p / (1.0 - p))

    honest = torch.zeros((1, 3, 2))
    honest[0, 0, 0] = logit(0.9)
    honest[0, 1, 0] = logit(0.05)
    honest[0, 2, 0] = logit(0.05)

    hedging = honest.clone()
    hedging[0, 1, 0] = logit(0.5)
    hedging[0, 2, 0] = logit(0.5)

    honest_reward, honest_stats = floor_control_reward(honest, mask)
    hedging_reward, hedging_stats = floor_control_reward(hedging, mask)

    assert honest_stats["same_count"] == 1
    assert float(honest_reward) > float(hedging_reward)
    # Concretely: p_same collapses from 0.9*0.95*0.95=0.81225 to 0.9*0.5*0.5=0.225.
    assert honest_stats["p_same_mean"] > hedging_stats["p_same_mean"]


def test_floor_control_reward_balances_rare_category_against_common_one():
    """4 trivial SAME columns (model nails them) + 1 rare HANDOFF column
    (model gets it badly wrong) must weigh the HANDOFF mistake as much as
    the SAME columns collectively, not dilute it 4:1 the way a naive
    unweighted per-column average would."""
    mask = torch.zeros((1, 2, 6), dtype=torch.bool)
    mask[0, 0, 0:5] = True   # agent0 sole speaker, columns 0-4
    mask[0, 1, 5] = True     # agent1 takes over at column 5 (handoff)

    def logit(p):
        return math.log(p / (1.0 - p))

    logits = torch.zeros((1, 2, 6))
    # Columns 0-3 (predicting columns 1-4, all SAME): confident + correct.
    logits[0, 0, 0:4] = logit(0.95)
    logits[0, 1, 0:4] = logit(0.02)
    # Column 4 (predicting column 5, HANDOFF): confidently WRONG.
    logits[0, 0, 4] = logit(0.95)
    logits[0, 1, 4] = logit(0.02)

    reward, stats = floor_control_reward(logits, mask)
    assert stats["same_count"] == 4
    assert stats["handoff_count"] == 1

    naive_avg = (4 * stats["same_reward_mean"] + 1 * stats["handoff_reward_mean"]) / 5.0
    balanced_avg = (stats["same_reward_mean"] + stats["handoff_reward_mean"]) / 2.0

    assert abs(float(reward) - balanced_avg) < 1e-5
    # The rare, badly-wrong HANDOFF column pulls the balanced mean well
    # below what a naive (count-weighted) average would show.
    assert balanced_avg < naive_avg
    assert stats["handoff_reward_mean"] < -1.0  # confidently wrong -> strongly negative


def test_floor_control_reward_category_weights_default_to_unweighted_mean():
    """Explicit weight_*=1.0 must reproduce the plain (default-cfg) mean
    exactly -- confirms the reweighting feature is backward compatible."""
    mask = torch.zeros((1, 2, 6), dtype=torch.bool)
    mask[0, 0, 0:5] = True
    mask[0, 1, 5] = True
    logits = torch.randn((1, 2, 6))

    reward_default, _ = floor_control_reward(logits, mask)
    reward_explicit, _ = floor_control_reward(
        logits, mask,
        cfg=FloorControlConfig(
            weight_silence=1.0, weight_same=1.0, weight_handoff=1.0, weight_overlap=1.0,
        ),
    )
    assert torch.allclose(reward_default, reward_explicit)


def test_floor_control_reward_overlap_weight_amplifies_overlap_contribution():
    """Raising weight_overlap must make a fixed overlap mistake pull the
    combined reward down further, without changing any category's own
    (per-category) reward_mean -- only the combination weighting changes."""
    mask = torch.zeros((1, 3, 4), dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 1, 1] = True
    mask[0, 0, 2] = True
    mask[0, 1, 2] = True  # column 2 (predicted from column 1) is OVERLAP
    mask[0, 2, 3] = True

    logits = torch.randn((1, 3, 4))
    reward_unweighted, stats_unweighted = floor_control_reward(logits, mask)
    reward_weighted, stats_weighted = floor_control_reward(
        logits, mask, cfg=FloorControlConfig(weight_overlap=5.0)
    )

    # Per-category means are unaffected by the combination weighting.
    assert stats_unweighted["overlap_reward_mean"] == pytest.approx(
        stats_weighted["overlap_reward_mean"], abs=1e-6
    )
    assert stats_unweighted["same_reward_mean"] == pytest.approx(
        stats_weighted["same_reward_mean"], abs=1e-6
    )
    # But the combined mean_reward shifts toward the (typically negative,
    # rarely-seen) overlap category's own mean once it is weighted up.
    assert not torch.allclose(reward_unweighted, reward_weighted)
    overlap_mean = stats_unweighted["overlap_reward_mean"]
    other_means = [
        stats_unweighted[f"{name}_reward_mean"]
        for name in ("silence", "same", "handoff")
        if stats_unweighted[f"{name}_count"] > 0
    ]
    if other_means and overlap_mean < min(other_means):
        assert reward_weighted < reward_unweighted
    elif other_means and overlap_mean > max(other_means):
        assert reward_weighted > reward_unweighted


def test_floor_control_reward_zero_weight_excludes_category():
    """weight_overlap=0.0 must fully exclude overlap columns from the
    combined mean, matching a hand-computed 3-way average."""
    mask = torch.zeros((1, 3, 4), dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 1, 1] = True
    mask[0, 0, 2] = True
    mask[0, 1, 2] = True  # overlap
    mask[0, 2, 3] = True

    logits = torch.randn((1, 3, 4))
    reward, stats = floor_control_reward(logits, mask, cfg=FloorControlConfig(weight_overlap=0.0))
    kept = [
        stats[f"{name}_reward_mean"]
        for name in ("silence", "same", "handoff")
        if stats[f"{name}_count"] > 0
    ]
    expected = sum(kept) / len(kept)
    assert float(reward) == pytest.approx(expected, abs=1e-5)


def test_floor_control_reward_adaptive_alpha_tracks_human_batch_frequency():
    """alpha=1 with no prior must combine category means in proportion to
    their observed human counts, exactly recovering the per-column mean."""
    mask = torch.zeros((1, 2, 6), dtype=torch.bool)
    mask[0, 0, 0:5] = True   # four predicted SAME columns
    mask[0, 1, 5] = True     # one predicted HANDOFF column
    logits = torch.randn((1, 2, 6))

    reward, stats = floor_control_reward(
        logits,
        mask,
        cfg=FloorControlConfig(
            adaptive_weight_alpha=1.0,
            adaptive_prior_strength=0.0,
        ),
    )
    expected = (
        4.0 * stats["same_reward_mean"] + stats["handoff_reward_mean"]
    ) / 5.0
    assert float(reward) == pytest.approx(expected, abs=1e-5)
    assert stats["same_effective_weight"] == pytest.approx(0.8)
    assert stats["handoff_effective_weight"] == pytest.approx(0.2)


def test_floor_control_reward_adaptive_alpha_zero_preserves_equal_weights():
    mask = torch.zeros((1, 2, 6), dtype=torch.bool)
    mask[0, 0, 0:5] = True
    mask[0, 1, 5] = True
    logits = torch.randn((1, 2, 6))

    reward, stats = floor_control_reward(
        logits,
        mask,
        cfg=FloorControlConfig(adaptive_weight_alpha=0.0),
    )
    expected = (stats["same_reward_mean"] + stats["handoff_reward_mean"]) / 2.0
    assert float(reward) == pytest.approx(expected, abs=1e-5)
    assert stats["same_effective_weight"] == pytest.approx(1.0)
    assert stats["handoff_effective_weight"] == pytest.approx(1.0)


def test_floor_control_reward_has_gradient():
    mask = torch.zeros((1, 2, 4), dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 0, 1] = True
    mask[0, 1, 2] = True
    mask[0, 1, 3] = True

    logits = torch.zeros((1, 2, 4), requires_grad=True)
    reward, _ = floor_control_reward(logits, mask)
    reward.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_floor_control_reward_short_sequence_is_zero():
    mask = torch.zeros((1, 2, 1), dtype=torch.bool)
    logits = torch.zeros((1, 2, 1))
    reward, stats = floor_control_reward(logits, mask)
    assert float(reward) == 0.0
    assert stats["same_count"] == 0
    assert stats["handoff_count"] == 0
    assert stats["silence_count"] == 0
    assert stats["overlap_count"] == 0
