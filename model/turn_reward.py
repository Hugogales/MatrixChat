"""Differentiable Stage-A turn-taking reward.

Rewards the model for having (probabilistically) exactly one agent speaking at
a time:

- A direct HANDOFF to a different agent is fully rewarded (no penalty).
- The SAME agent continuing is rewarded, decaying gently only after a long
  monopoly (``speak_grace`` columns of free continuation, decaying with time
  constant ``speak_tau``).
- Prolonged SILENCE (nobody speaking) is increasingly penalized, growing with
  time constant ``silence_tau``; short pauses are tolerated.
- OVERLAP (more than one agent active) is explicitly penalized from its exact
  probability. The onset penalty is moderate, then ramps when overlap persists.

Stage A (this module): the "who was speaking when" state -- the run-length
counters ``x`` (same-speaker duration) and ``z`` (all-silent duration) used to
shape the reward -- comes from GROUND-TRUTH ``input_activity_mask`` (teacher
forcing), NOT from the model's own sampled actions.

IMPORTANT -- mandatory Stage-A -> Stage-B migration
----------------------------------------------------
Before this reward is used in online RL / self-play (Stage B), ``x`` and ``z``
MUST be recomputed from SAMPLED/REALIZED model actions along the rollout, not
from dataset ground truth, and the counter updates must NOT be backpropagated
through. See the RL Turn-Taking Design plan for the full migration checklist.
Using ground-truth counters online would reward the model based on the
dataset's trajectory rather than the consequences of its own decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@torch.no_grad()
def ground_truth_run_lengths(
    activity_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-column ground-truth turn-taking state from ``activity_mask``.

    Parameters
    ----------
    activity_mask:
        Bool/long tensor ``[B, A, T]``, True/1 where the agent is active
        (speaking) at that column.

    Returns
    -------
    x_prev:
        Long ``[B, T]`` -- consecutive columns (ending at ``t-1``) where the
        SAME single agent was the sole speaker. 0 if column ``t-1`` had 0 or
        >=2 speakers, or ``t == 0`` (no predecessor).
    z_prev:
        Long ``[B, T]`` -- consecutive columns (ending at ``t-1``) where ALL
        agents were silent. 0 if column ``t-1`` had >=1 speaker, or ``t == 0``.
    owner_prev:
        Long ``[B, T]`` -- the sole speaker at column ``t-1``, or ``-1`` if
        there was not exactly one speaker at ``t-1`` (or ``t == 0``).

    This is a pure ground-truth computation (no gradient): it only scans the
    dataset's known activity pattern, sequentially over time. ``T`` is bounded
    by the (already-capped) matrix length, so a Python loop here is cheap
    relative to the transformer forward pass.
    """
    mask = activity_mask.bool()
    b, a, t = mask.shape
    device = mask.device

    n_speakers = mask.sum(dim=1)              # [B, T]
    exactly_one = n_speakers == 1              # [B, T]
    all_silent = n_speakers == 0               # [B, T]
    # argmax gives the (only) True index when exactly one speaker; meaningless
    # otherwise, so it is masked to -1 below.
    speaker_idx = mask.float().argmax(dim=1)   # [B, T]
    speaker_idx = torch.where(
        exactly_one, speaker_idx, torch.full_like(speaker_idx, -1)
    )

    x = torch.zeros(b, t, dtype=torch.long, device=device)
    z = torch.zeros(b, t, dtype=torch.long, device=device)
    owner = torch.full((b, t), -1, dtype=torch.long, device=device)

    prev_owner = torch.full((b,), -1, dtype=torch.long, device=device)
    prev_x = torch.zeros(b, dtype=torch.long, device=device)
    prev_z = torch.zeros(b, dtype=torch.long, device=device)
    for col in range(t):
        owner[:, col] = prev_owner
        x[:, col] = prev_x
        z[:, col] = prev_z

        cur_owner = speaker_idx[:, col]
        cur_silent = all_silent[:, col]
        has_owner = cur_owner != -1
        same_owner = has_owner & (cur_owner == prev_owner)
        overlap = (~cur_silent) & (~has_owner)  # >=2 speakers this column

        new_x = torch.where(
            same_owner, prev_x + 1, torch.where(has_owner, torch.ones_like(prev_x), torch.zeros_like(prev_x))
        )
        new_x = torch.where(overlap, torch.zeros_like(new_x), new_x)
        new_z = torch.where(cur_silent, prev_z + 1, torch.zeros_like(prev_z))
        new_owner = torch.where(has_owner, cur_owner, torch.full_like(prev_owner, -1))

        prev_owner, prev_x, prev_z = new_owner, new_x, new_z

    return x, z, owner


@dataclass
class TurnRewardConfig:
    """Shape parameters for the turn-taking reward (see module docstring)."""

    speak_grace: float = 50.0
    speak_tau: float = 2500.0
    silence_grace: float = 1.0
    silence_tau: float = 20.0
    overlap_base_weight: float = 0.35
    overlap_max_weight: float = 1.5
    overlap_grace: float = 1.0
    overlap_tau: float = 3.0
    # Extra, independently-tunable credit for a clean handoff (a DIFFERENT
    # agent taking over from the previous owner), on top of the baseline
    # credit `p_handoff` already receives as part of "exactly one speaker".
    # 0.0 preserves old behavior exactly.
    handoff_bonus_weight: float = 0.0


@torch.no_grad()
def ground_truth_overlap_run_lengths(activity_mask: torch.Tensor) -> torch.Tensor:
    """Consecutive overlap columns ending immediately before each column."""
    overlap = activity_mask.bool().sum(dim=1) >= 2  # [B, T]
    b, t = overlap.shape
    out = torch.zeros(b, t, dtype=torch.long, device=activity_mask.device)
    previous = torch.zeros(b, dtype=torch.long, device=activity_mask.device)
    for col in range(t):
        out[:, col] = previous
        previous = torch.where(overlap[:, col], previous + 1, torch.zeros_like(previous))
    return out


@torch.no_grad()
def carry_forward_owner_and_arrivals(
    activity_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward-fill the last known sole owner through silence/overlap gaps.

    A naive column-to-column owner comparison conflates two very different
    events: (a) the same speaker's utterance simply continuing into the next
    token column (the overwhelming majority of "same-owner" column pairs,
    and NOT a turn-taking decision at all), and (b) a genuine new run
    starting -- either an immediate handoff or the SAME speaker retaking the
    floor after a real gap (silence/overlap). Only (b) is a real decision
    point; this helper identifies exactly those columns.

    Returns
    -------
    carry_owner:
        Long ``[B, T]`` -- the most recent sole owner as of (and including)
        column ``t``, persisting through any silence/overlap columns. ``-1``
        before any sole owner has ever appeared in the sequence.
    new_arrival:
        Bool ``[B, T]`` -- True at column ``t`` iff exactly one agent is
        active at ``t`` AND that agent was NOT already the sole active
        speaker at column ``t-1`` (i.e. this column starts a fresh run --
        whether via an immediate handoff, a resumption after a gap, or the
        very first run in the sequence). Always False at ``t == 0``.
    """
    mask = activity_mask.bool()
    b, _a, t = mask.shape
    device = mask.device

    n_speakers = mask.sum(dim=1)
    exactly_one = n_speakers == 1
    speaker_idx = mask.float().argmax(dim=1)
    speaker_idx = torch.where(
        exactly_one, speaker_idx, torch.full_like(speaker_idx, -1)
    )

    carry_owner = torch.full((b, t), -1, dtype=torch.long, device=device)
    new_arrival = torch.zeros((b, t), dtype=torch.bool, device=device)

    prev_carry = torch.full((b,), -1, dtype=torch.long, device=device)
    prev_active_owner = torch.full((b,), -1, dtype=torch.long, device=device)
    for col in range(t):
        cur_owner = speaker_idx[:, col]
        has_owner = cur_owner != -1
        if col > 0:
            new_arrival[:, col] = has_owner & (cur_owner != prev_active_owner)
        carry_owner[:, col] = torch.where(has_owner, cur_owner, prev_carry)
        prev_carry = carry_owner[:, col]
        prev_active_owner = cur_owner

    return carry_owner, new_arrival


def same_handoff_cross_entropy(
    activity_logits: torch.Tensor,
    activity_mask_gt: torch.Tensor,
    activity_labels_gt: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, dict]:
    """Proper-scoring loss for WHO becomes the next speaker at a genuine
    run boundary -- NOT every trivial mid-utterance column.

    IMPORTANT (fixed 08-12, see SUCCESS_STORIES.md): an earlier version of
    this loss compared every consecutive unique-owner column pair, which is
    dominated by trivial "obviously still mid-utterance" transitions -- a
    single 30-token utterance contributes ~29 such non-decision "same"
    transitions and only 1 genuine decision. That inflated apparent
    same-speaker rate (~90%) is the OPPOSITE of the true human rate at real
    decision points. Measuring instead at genuine run boundaries (collapsing
    each utterance into one event, resolving forward through any
    silence/overlap to find who becomes the next sole speaker) gives 83.6%
    handoff / 16.4% self-resume in this project's validation mixture --
    inverted from the naive per-column figure.

    This version only scores columns where the ground truth marks a genuine
    new-arrival (see ``carry_forward_owner_and_arrivals``), comparing the
    arriving owner against the most recently carried-forward real owner
    (persisting through any gap) rather than the literal previous column's
    owner. Silence/overlap gaps themselves, and the many trivial mid-run
    columns, are excluded from scoring entirely -- they are not decisions.
    """
    b, a, t = activity_logits.shape
    if t < 2:
        zero = activity_logits.sum() * 0.0
        return zero, {
            "count": 0,
            "target_handoff_rate": 0.0,
            "predicted_handoff_rate": 0.0,
        }

    labels = activity_labels_gt[:, :, :-1]
    valid = (labels != -100).all(dim=1)
    following = labels == 1
    following_count = following.sum(dim=1)
    following_owner = following.float().argmax(dim=1)

    carry_owner, new_arrival = carry_forward_owner_and_arrivals(activity_mask_gt)
    prior_owner = carry_owner[:, :-1]           # carried-forward state as of column c
    arrival_next = new_arrival[:, 1:]           # True iff column c+1 starts a fresh run

    scored = valid & (following_count == 1) & arrival_next & (prior_owner != -1)
    if not bool(scored.any()):
        zero = activity_logits.sum() * 0.0
        return zero, {
            "count": 0,
            "target_handoff_rate": 0.0,
            "predicted_handoff_rate": 0.0,
        }

    target_handoff = following_owner != prior_owner

    # Logit column c predicts activity_labels[:, :, c], i.e. activity at c+1.
    q = torch.sigmoid(activity_logits[:, :, :-1].float())
    one_minus_q = (1.0 - q).clamp(min=1e-6, max=1.0)
    prod_all = one_minus_q.prod(dim=1, keepdim=True)
    p_only = q * (prod_all / one_minus_q)
    safe_prior = prior_owner.clamp(min=0)
    p_prior = torch.gather(p_only, 1, safe_prior.unsqueeze(1)).squeeze(1)
    p_handoff = (p_only.sum(dim=1) - p_prior).clamp(min=0.0)
    exactly_one_mass = (p_prior + p_handoff).clamp(min=eps)
    predicted_handoff = (p_handoff / exactly_one_mass).clamp(
        min=eps, max=1.0 - eps
    )
    selected_probability = torch.where(
        target_handoff, predicted_handoff, 1.0 - predicted_handoff
    )
    loss = -selected_probability[scored].log().mean()
    return loss, {
        "count": int(scored.sum().detach().cpu()),
        "target_handoff_rate": float(target_handoff[scored].float().mean().detach().cpu()),
        "predicted_handoff_rate": float(predicted_handoff[scored].mean().detach().cpu()),
    }


@dataclass
class FloorControlConfig:
    """Reference (baseline) per-column category rates for
    :func:`floor_control_reward`'s log-ratio framing.

    These four numbers are FIXED constants, not learned or Optuna-searched.
    They only set where the reported ``mean_reward``'s zero point sits:
    predicting the true per-column outcome MORE confidently than this
    baseline nets a positive mean reward; predicting it LESS confidently
    nets negative. They are subtracted as a constant per example, so they
    cannot affect gradients or the loss's argmin -- see the module
    docstring's "why this can't be gamed" section. Getting them slightly
    wrong only shifts the logged number's zero point, not what is trained.

    Defaults are this project's own measured rates on the candidate106
    training mixture (MELD 0.15 / AMI 0.50 / Werewolf 0.35; see
    SUCCESS_STORIES.md's calibration section), corrected from the naive
    "owner resets on any gap" numbers reported there to the carry-forward
    owner definition used here (a same-speaker resumption after a brief
    silence/overlap gap counts as SAME, not HANDOFF): of the reported 8.08%
    raw "new arrival" rate, the run-boundary-conditioned split (83.6%
    handoff / 16.4% self-resume, also in SUCCESS_STORIES.md) implies
    ~6.75% true cross-agent handoff and ~1.32% self-resume-after-gap
    (folded into "same" below).
    """

    ref_silence: float = 0.0784
    ref_same: float = 0.7805
    ref_handoff: float = 0.0675
    ref_overlap: float = 0.0735
    eps: float = 1e-6

    # Weights on the 4 per-category means when combining them into
    # mean_reward (see floor_control_reward's docstring). All 1.0 (default)
    # reproduces the original unweighted mean exactly. Raising a category's
    # weight makes the model's calibration AT THAT category's columns matter
    # more to the training signal, without changing what "well-calibrated"
    # means for any individual category (each category's log-score stays a
    # strictly proper scoring rule on its own columns -- reweighting the
    # OUTER combination is not exploitable the way an additive linear reward
    # would be, because probability mass across categories is still
    # conserved; see the module's "why this is hard to game" analysis).
    #
    # Motivation (2026-08-28, see TRAINING_RUN_LOG.md): the true final-run
    # evaluation contract (real unrestricted continuation of held-out
    # conversations, not curated broad-sweep prompts) found ~35-40% of
    # examples land in a "catastrophic" failure bucket dominated by runaway
    # OVERLAP (2+ agents both generating substantial text simultaneously for
    # many consecutive columns) -- both for the plain baseline and the
    # balanced_ce+DynamicAgentState recipe, somewhat worse for the latter.
    # Equal weighting gives overlap's ~7% of human columns a full 1/4 of the
    # reward gradient, roughly 10x the per-column influence of SAME. That may
    # itself help cause the elevated free-running overlap rate. Adaptive
    # human-frequency weighting below lets us interpolate away from that
    # artificial 25/25/25/25 target while retaining some rare-event emphasis.
    # NOTE this only fixes teacher-forced calibration, not the deeper
    # exposure-bias issue (the model never trains on recovering from ITS OWN
    # overlap mistakes since x/z/owner state is always ground-truth during
    # Stage A); see the module docstring's Stage A -> B migration note.
    weight_silence: float = 1.0
    weight_same: float = 1.0
    weight_handoff: float = 1.0
    weight_overlap: float = 1.0
    # Optionally adapt the outer category weights to the human category
    # prevalence in the current selected training batch:
    #
    #   effective_weight[c] = weight[c] * smoothed_prevalence[c] ** alpha
    #
    # alpha=0 preserves equal-category balancing exactly; alpha=1 recovers
    # ordinary per-column human-frequency weighting (apart from smoothing);
    # values in (0, 1) retain some rare-event emphasis without forcing every
    # category to contribute 25% regardless of what humans did in this batch.
    adaptive_weight_alpha: float = 0.0
    adaptive_prior_strength: float = 32.0


def floor_control_reward(
    activity_logits: torch.Tensor,       # [B, A, T] logits predicting activity at t+1
    activity_mask_gt: torch.Tensor,      # [B, A, T] ground-truth activity (bool/long)
    cfg: Optional[FloorControlConfig] = None,
    valid_time_mask: Optional[torch.Tensor] = None,  # [B, T] True where column t is real
) -> Tuple[torch.Tensor, dict]:
    """Balanced 4-way proper-scoring floor-control loss (replaces the linear
    ``turn_taking_reward`` shape-parameter apparatus above).

    Why this exists (2026-08-24 rearchitecture)
    --------------------------------------------
    ``turn_taking_reward`` has two structural problems, both traced back to
    treating a per-column MEAN LOSS as if it were a temporally-discounted
    RL return:

    1. **It discourages handoffs from the initial drop in expected reward.**
       ``p_overlap`` is penalized starting from the very first unit of
       overlap probability (``overlap_base_weight`` applies with no true
       zero-penalty grace, unlike silence's genuine grace period). But a
       clean handoff, expressed as a CONTINUOUS relaxation of per-agent
       probabilities, mechanically requires the outgoing and incoming
       speaker to both carry some probability mass during the shift -- the
       formula calls that state "overlap" and taxes it immediately, so the
       gradient path from "confidently same" to "confidently handoff"
       passes through a penalized region before it reaches the reward on
       the other side. Early in training, when the model cannot yet
       execute a crisp one-column transition, any attempted handoff nets a
       worse expected reward than never attempting one at all -- a local
       optimum that reinforces monopolization.
    2. **The ``speak_grace``/``speak_tau`` decay imports an RL-style
       "time constant" into something that isn't a discounted return.**
       This loss is a per-column mean over the batch with no bootstrapping
       and no episode-level credit assignment -- "how long has this run
       lasted" is a fine FEATURE for the model to condition on via its own
       causal context, but hand-coding its effect via an exponential decay
       tuned against columns/tokens (not real time) requires guessing a
       time constant relative to episode length; ``speak_tau=2500`` turned
       out to be so much larger than typical episode lengths that even a
       fully-monopolizing agent barely sees its reward decay within a
       realistic training window.

    Design: instead of a hand-tuned additive combination of separately-
    shaped terms, treat "what kind of column is this" as a proper 4-way
    classification (SILENCE / SAME / HANDOFF / OVERLAP) and score the
    model's predicted distribution over those 4 categories against the
    REAL ground-truth category at that exact column via cross-entropy (a
    strictly proper scoring rule). No decay, no grace period, no "time
    constant" -- every column is scored identically and independently
    against what actually happened there.

    Why this is hard to game (defensive analysis)
    -----------------------------------------------
    - The four predicted probabilities (p_silence=p0, p_same, p_handoff,
      p_overlap) are an EXACT partition of the [0, 1] simplex by
      construction: ``p0 + p_only_sum + p_overlap == 1`` always (from
      ``p_overlap = 1 - p0 - p_only_sum``), and ``p_same + p_handoff ==
      p_only_sum`` exactly (a partition of "exactly one speaker" mass by
      owner identity, not two independently-tunable quantities). There is
      no way to raise the score attributed to one category without an
      equal-and-opposite fall somewhere else in the partition -- unlike an
      additive linear reward, where ``p_same`` and ``p_handoff`` can both
      be pushed up nearly independently of ``p_overlap``.
    - Cross-entropy (the log score) is a STRICTLY PROPER scoring rule: its
      unique minimum is calibration to the true conditional distribution.
      There is mathematically no better strategy than genuinely predicting
      when a handoff/silence/overlap/continuation is coming -- concretely
      ruling out the exploit seen with the (already-disabled-by-default)
      ``same_handoff_cross_entropy`` term, where the conditional
      renormalization by "exactly-one mass" made it BLIND to how much
      probability leaked into overlap, letting several agents jointly
      raise their activity to farm handoff credit while paying only a
      lagging, capped overlap penalty. Here, raising overlap probability
      when the truth is NOT overlap directly lowers ``p_true`` (whatever
      category truth actually is), because mass is conserved.
    - The 4 per-category means are combined with EQUAL weight regardless of
      how common each category is (mirrors the existing balanced activity
      BCE). This is what prevents "always predict SAME" -- overwhelmingly
      the model's cheapest, safest per-column bet since it makes up ~78%
      of columns -- from dominating the training signal; the HANDOFF
      category's mean is scored purely on how well-calibrated the model is
      AT the (rarer) columns where a handoff genuinely occurs, with no
      dilution from the vastly more common trivial mid-utterance columns.

    Owner tracking uses :func:`carry_forward_owner_and_arrivals` (the
    corrected, gap-persistent definition), not the older
    :func:`ground_truth_run_lengths`, so a same-speaker resumption after a
    brief pause is scored as SAME, not conflated with a genuine handoff.

    Returns ``(mean_reward, stats)`` with the SAME calling convention as
    ``turn_taking_reward`` (``loss -= lambda_reward * mean_reward``):
    ``mean_reward`` is a log-likelihood-RATIO against a fixed reference
    distribution (:class:`FloorControlConfig`), so it is genuinely signed --
    positive when the model beats naive-baseline guessing on the true
    per-column outcome (on average across the batch), negative when it does
    worse. It carries gradient w.r.t. ``activity_logits``; higher is better.
    """
    cfg = cfg or FloorControlConfig()
    b, a, t = activity_logits.shape
    if t < 2:
        zero = activity_logits.sum() * 0.0
        stats = {"mean_reward": 0.0}
        for name in ("silence", "same", "handoff", "overlap"):
            stats[f"{name}_count"] = 0
            stats[f"{name}_reward_mean"] = 0.0
        return zero, stats

    mask = activity_mask_gt.bool()
    n_speakers = mask.sum(dim=1)               # [B, T]
    exactly_one = n_speakers == 1
    speaker_idx = mask.float().argmax(dim=1)
    speaker_idx = torch.where(
        exactly_one, speaker_idx, torch.full_like(speaker_idx, -1)
    )
    carry_owner, _new_arrival = carry_forward_owner_and_arrivals(activity_mask_gt)

    # Ground-truth category at column t (t = 1..T-1), using the carried-
    # forward owner AS OF t-1 (i.e. before this column is decided) --
    # exactly matching what activity_logits[:, :, t-1] is predicting.
    n_speakers_t = n_speakers[:, 1:]           # [B, T-1]
    speaker_t = speaker_idx[:, 1:]              # [B, T-1]
    prior_owner = carry_owner[:, :-1]           # [B, T-1]

    is_silence = n_speakers_t == 0
    is_overlap = n_speakers_t >= 2
    is_single = n_speakers_t == 1
    is_same = is_single & (speaker_t == prior_owner) & (prior_owner >= 0)
    is_handoff = is_single & ~is_same

    # Predicted probabilities: identical construction to turn_taking_reward.
    q = torch.sigmoid(activity_logits[:, :, :-1].float())     # [B, A, T-1]
    one_minus_q = (1.0 - q).clamp(min=1e-6, max=1.0)
    prod_all = one_minus_q.prod(dim=1, keepdim=True)          # [B, 1, T-1]
    p0 = prod_all.squeeze(1)                                  # [B, T-1]
    leave_one_out = prod_all / one_minus_q                    # [B, A, T-1]
    p_only = q * leave_one_out                                # [B, A, T-1]
    p_only_sum = p_only.sum(dim=1)                            # [B, T-1]
    p_overlap = (1.0 - p0 - p_only_sum).clamp(min=0.0, max=1.0)

    has_prior = prior_owner >= 0
    safe_prior = prior_owner.clamp(min=0)
    p_prior = torch.gather(p_only, 1, safe_prior.unsqueeze(1)).squeeze(1)
    p_same = torch.where(has_prior, p_prior, torch.zeros_like(p_prior))
    p_handoff = (p_only_sum - p_same).clamp(min=0.0)

    eps = cfg.eps
    p_true = torch.where(is_silence, p0, torch.zeros_like(p0))
    p_true = torch.where(is_overlap, p_overlap, p_true)
    p_true = torch.where(is_same, p_same, p_true)
    p_true = torch.where(is_handoff, p_handoff, p_true)
    log_p_true = p_true.clamp(min=eps).log()

    ref = torch.where(
        is_silence, torch.full_like(p0, cfg.ref_silence), torch.zeros_like(p0)
    )
    ref = torch.where(is_overlap, torch.full_like(p0, cfg.ref_overlap), ref)
    ref = torch.where(is_same, torch.full_like(p0, cfg.ref_same), ref)
    ref = torch.where(is_handoff, torch.full_like(p0, cfg.ref_handoff), ref)
    log_ref = ref.clamp(min=eps).log()   # constant w.r.t. activity_logits

    per_col_reward = log_p_true - log_ref   # [B, T-1]

    if valid_time_mask is not None:
        vmask = valid_time_mask[:, 1:].bool()
    else:
        vmask = torch.ones_like(is_silence)

    base_category_weights = {
        "silence": cfg.weight_silence,
        "same": cfg.weight_same,
        "handoff": cfg.weight_handoff,
        "overlap": cfg.weight_overlap,
    }
    category_selections = (
        ("silence", is_silence, cfg.ref_silence),
        ("same", is_same, cfg.ref_same),
        ("handoff", is_handoff, cfg.ref_handoff),
        ("overlap", is_overlap, cfg.ref_overlap),
    )
    valid_count = float(vmask.sum().detach().cpu())
    prior_strength = float(cfg.adaptive_prior_strength)
    alpha = float(cfg.adaptive_weight_alpha)
    prevalence_denom = valid_count + prior_strength
    category_weights = {}
    for name, sel, reference_rate in category_selections:
        count_value = float((sel & vmask).sum().detach().cpu())
        prevalence = (
            (count_value + prior_strength * float(reference_rate)) / prevalence_denom
            if prevalence_denom > 0
            else float(reference_rate)
        )
        category_weights[name] = float(base_category_weights[name]) * prevalence ** alpha

    weighted_means = []
    total_weight = 0.0
    stats: dict = {}
    for name, sel, _reference_rate in category_selections:
        m = (sel & vmask).float()
        count = m.sum()
        stats[f"{name}_count"] = int(count.detach().cpu())
        stats[f"{name}_effective_weight"] = category_weights[name]
        if count > 0:
            cat_mean = (per_col_reward * m).sum() / count
            stats[f"{name}_reward_mean"] = float(cat_mean.detach().cpu())
            weight = float(category_weights[name])
            weighted_means.append(cat_mean * weight)
            total_weight += weight
        else:
            stats[f"{name}_reward_mean"] = 0.0

    if weighted_means and total_weight > 0:
        mean_reward = torch.stack(weighted_means).sum() / total_weight
    else:
        mean_reward = activity_logits.sum() * 0.0

    denom = vmask.float().sum().clamp(min=1.0)
    stats["p0_mean"] = float((p0 * vmask).sum() / denom)
    stats["p_same_mean"] = float((p_same * vmask).sum() / denom)
    stats["p_handoff_mean"] = float((p_handoff * vmask).sum() / denom)
    stats["p_overlap_mean"] = float((p_overlap * vmask).sum() / denom)
    stats["mean_reward"] = float(mean_reward.detach().cpu())
    return mean_reward, stats


def turn_taking_reward(
    activity_logits: torch.Tensor,       # [B, A, T] logits predicting activity at t+1
    activity_mask_gt: torch.Tensor,      # [B, A, T] ground-truth activity (bool/long)
    cfg: Optional[TurnRewardConfig] = None,
    valid_time_mask: Optional[torch.Tensor] = None,  # [B, T] True where column t is real
) -> Tuple[torch.Tensor, dict]:
    """Differentiable two-term turn-taking reward (Stage A, ground-truth durations).

    ``activity_logits[:, :, c]`` is the model's prediction (from context up to
    and including column ``c``) of whether each agent speaks at column ``c+1``.
    So ``q = sigmoid(activity_logits[:, :, t-1])`` is the predicted probability
    distribution of "who speaks at column t", which is exactly what the reward
    shapes.

    Returns ``(mean_reward, stats)``. ``mean_reward`` is a scalar tensor with
    gradient w.r.t. ``activity_logits``; higher is better. The caller should
    subtract ``lambda_reward * mean_reward`` from the total loss (i.e. add
    ``-lambda_reward * mean_reward``).
    """
    cfg = cfg or TurnRewardConfig()
    b, a, t = activity_logits.shape
    if t < 2:
        zero = activity_logits.sum() * 0.0
        return zero, {
            "p0_mean": 0.0,
            "p_same_mean": 0.0,
            "p_handoff_mean": 0.0,
            "p_overlap_mean": 0.0,
            "overlap_weight_mean": 0.0,
        }

    x_prev, z_prev, owner_prev = ground_truth_run_lengths(activity_mask_gt)  # [B, T] each
    overlap_prev = ground_truth_overlap_run_lengths(activity_mask_gt)

    # q[:, :, t] = P(agent speaks AT column t), predicted from activity_logits[:, :, t-1].
    # Defined for t = 1..T-1 (there is no predecessor for column 0).
    q = torch.sigmoid(activity_logits[:, :, :-1].float())  # [B, A, T-1] -> columns 1..T-1

    one_minus_q = (1.0 - q).clamp(min=1e-6, max=1.0)
    prod_all = one_minus_q.prod(dim=1, keepdim=True)         # [B, 1, T-1]
    p0 = prod_all.squeeze(1)                                 # [B, T-1] all-silent probability
    leave_one_out = prod_all / one_minus_q                   # [B, A, T-1] ~= prod_{b!=a}(1-q_b)
    p_only = q * leave_one_out                                # [B, A, T-1] P(exactly agent a speaks)

    owner_t = owner_prev[:, 1:]                # owner going into column t (from column t-1)
    has_owner = owner_t >= 0                   # [B, T-1]
    owner_idx = owner_t.clamp(min=0)           # [B, T-1]

    p_same = torch.gather(p_only, 1, owner_idx.unsqueeze(1)).squeeze(1)  # [B, T-1]
    p_same = torch.where(has_owner, p_same, torch.zeros_like(p_same))

    p_only_sum = p_only.sum(dim=1)             # [B, T-1] P(exactly one agent speaks, any agent)
    p_handoff = torch.where(has_owner, p_only_sum - p_same, p_only_sum)
    p_overlap = (1.0 - p0 - p_only_sum).clamp(min=0.0, max=1.0)

    x_t = x_prev[:, 1:].float()
    z_t = z_prev[:, 1:].float()
    overlap_t = overlap_prev[:, 1:].float()

    s_x = torch.exp(-torch.clamp(x_t - cfg.speak_grace, min=0.0) / cfg.speak_tau)
    w_sil = 1.0 - torch.exp(
        -torch.clamp(z_t - cfg.silence_grace, min=0.0) / cfg.silence_tau
    )
    overlap_ramp = 1.0 - torch.exp(
        -torch.clamp(overlap_t - cfg.overlap_grace, min=0.0) / cfg.overlap_tau
    )
    w_overlap = cfg.overlap_base_weight + (
        cfg.overlap_max_weight - cfg.overlap_base_weight
    ) * overlap_ramp

    reward = (
        s_x * p_same
        + p_handoff
        + cfg.handoff_bonus_weight * p_handoff
        - w_sil * p0
        - w_overlap * p_overlap
    )  # [B, T-1]

    if valid_time_mask is not None:
        vmask = valid_time_mask[:, 1:].float()
    else:
        vmask = torch.ones_like(reward)

    denom = vmask.sum().clamp(min=1.0)
    mean_reward = (reward * vmask).sum() / denom

    stats = {
        "p0_mean": float((p0 * vmask).sum() / denom),
        "p_same_mean": float((p_same * vmask).sum() / denom),
        "p_handoff_mean": float((p_handoff * vmask).sum() / denom),
        "p_overlap_mean": float((p_overlap * vmask).sum() / denom),
        "overlap_weight_mean": float((w_overlap * vmask).sum() / denom),
    }
    return mean_reward, stats
