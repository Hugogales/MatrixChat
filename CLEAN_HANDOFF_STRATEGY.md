# Clean Handoff Strategy (Sept 17, 2026)

## Key Finding: MELD is Perfectly Clean, AMI/Werewolf Are Messy

From the processed dataset manifest analysis:

| Dataset | Examples | Avg Length | Overlap % | Assessment |
|---------|----------|------------|-----------|------------|
| **MELD** | 921 | 132.7 cols | **0.0%** | ✓ Perfectly clean (scripted TV) |
| **AMI** | 4,505 | 366.4 cols | **8.9%** | ⚠ Significant overlaps (meetings) |
| **Werewolf** | 2,372 | 285.0 cols | **6.6%** | ⚠ Significant overlaps (games) |

**User's qualitative observation:** The c106 model performed best on MELD-like prompts (clean, scripted handoffs).

**Current c106 training mix:** 15% MELD, 50% AMI, 35% Werewolf → **heavily weighted toward messy data**

## Hypothesis

Teaching the model on clean handoffs (MELD) will produce better turn-taking behavior than training on natural-but-messy data (AMI/Werewolf with 6-9% overlap). The model is learning to imitate interruptions and overlaps when we actually want crisp, deliberate handoffs.

## L2SP Clarification

**L2SP = L2 Regularization to Starting Point**

It penalizes the squared distance between current weights and their pretrained (Qwen) values:

```
L_l2sp = lambda_l2sp * sum((w_current - w_pretrained)^2)
```

This keeps the model anchored to Qwen's knowledge while adapting to turn-taking. Current c106 uses `lambda_l2sp=0.02`.

**User guidance:** L2SP is sufficient; full KL divergence is not needed.

## Reward Mechanism Context

c106 already uses **much stricter** overlap penalties than defaults:
- `overlap_base_weight`: 1.674 (default 0.35) - ~5x stricter
- `overlap_max_weight`: 5.567 (default 1.5) - ~3.7x stricter
- `overlap_tau`: 2.0 (default 3.0) - faster ramp to max penalty
- `handoff_bonus_weight`: 1.1 (default 0.0) - explicit bonus for clean handoffs

Despite these strong penalties, c106 still shows non-zero overlap rates in evaluation. This suggests:
1. The **data itself teaches overlap** (AMI/Werewolf examples have 6-9% overlap in their ground truth)
2. Even with penalties, the content loss dominates (λ_content=1.0, λ_reward=0.05 → reward is only 5% of total loss)

## Four Experimental Arms

All use the idle V100 capacity (dgx partition is empty, H100s are saturated). All start from c106 warm-start.

### Arm 1: Pure MELD (100% clean baseline)
**Rationale:** Test the hypothesis directly - train ONLY on perfectly clean data.

```bash
RUN_NAME=clean_arm1_pure_meld
INIT_FROM=checkpoints/final40_h100_c106_s176106
INIT_CHECKPOINT_PREFER=root
DATASET_WEIGHTS="meld=1.0,ami=0.0,werewolf=0.0"
NUM_STEPS=1500
NUM_EPOCHS=20
SEED=777001
```

**Expected outcome:** Near-zero overlap in evaluation, but potentially less diverse conversational patterns (MELD is scripted, shorter, simpler).

**Risk:** MELD has only 921 examples (vs. 4505 AMI + 2372 Werewolf). May overfit.

### Arm 2: MELD-Dominant Mix (70% MELD, 15% AMI, 15% Werewolf)
**Rationale:** Heavily favor clean data while keeping some natural diversity.

```bash
RUN_NAME=clean_arm2_meld_dominant
INIT_FROM=checkpoints/final40_h100_c106_s176106
INIT_CHECKPOINT_PREFER=root
DATASET_WEIGHTS="meld=0.7,ami=0.15,werewolf=0.15"
NUM_STEPS=1500
NUM_EPOCHS=20
SEED=777002
```

**Expected outcome:** Much cleaner handoffs than c106 (which used 15% MELD) while preserving natural timing/diversity from AMI/Werewolf minority.

**Comparison to c106:** c106 was 15% MELD, 50% AMI, 35% Werewolf (85% messy). This arm is 70% clean, 30% messy (inverted).

### Arm 3: Even Stricter Overlap Penalties
**Rationale:** Keep c106's data mix but penalize overlaps even harder in the reward.

```bash
RUN_NAME=clean_arm3_stricter_penalty
INIT_FROM=checkpoints/final40_h100_c106_s176106
INIT_CHECKPOINT_PREFER=root
DATASET_WEIGHTS="meld=0.15,ami=0.5,werewolf=0.35"  # Same as c106
OVERLAP_BASE_WEIGHT=3.5  # c106: 1.674 → ~2x increase
OVERLAP_MAX_WEIGHT=10.0  # c106: 5.567 → ~1.8x increase
OVERLAP_TAU=1.5  # c106: 2.0 → faster ramp
HANDOFF_BONUS_WEIGHT=2.0  # c106: 1.1 → nearly double the handoff bonus
NUM_STEPS=1500
NUM_EPOCHS=20
SEED=777003
```

**Expected outcome:** Tests whether stronger penalties alone (without changing data mix) can overcome the messy-data influence.

**Risk:** If λ_reward stays at 0.05, these penalties might still be too weak vs. content loss (1.0).

### Arm 4: Combined Approach (MELD-dominant + stricter penalties)
**Rationale:** Best of both worlds - cleaner data AND stronger reward shaping.

```bash
RUN_NAME=clean_arm4_combined
INIT_FROM=checkpoints/final40_h100_c106_s176106
INIT_CHECKPOINT_PREFER=root
DATASET_WEIGHTS="meld=0.7,ami=0.15,werewolf=0.15"  # Like Arm 2
OVERLAP_BASE_WEIGHT=3.5
OVERLAP_MAX_WEIGHT=10.0
OVERLAP_TAU=1.5
HANDOFF_BONUS_WEIGHT=2.0
NUM_STEPS=1500
NUM_EPOCHS=20
SEED=777004
```

**Expected outcome:** Strongest push toward clean handoffs. May produce the crispest turn-taking but risk over-optimizing for the reward at the expense of content quality.

## Resource Strategy

**Current Rosie status (Sept 17, 06:40):**
- V100s (dgx): 3 nodes idle, 0 jobs running ✓ **USE THIS**
- H100s (dgxh100): Fully saturated with another user's array job

**Submission plan:**
1. Use the existing `scripts/ops/job_queue.py` infrastructure (respects `--max_concurrent`)
2. Submit all 4 arms to the queue with `--max_concurrent 3` (one per V100 node)
3. Each arm runs ~1500 steps × 4 batch_size × 8 grad_accum = 48k effective examples
4. At ~90 sec/step (c106's pace on V100), each arm = ~37.5 hours
5. Probe every 250 steps → 6 checkpoints per arm + continuous probes
6. If another user queues jobs, the GPU yield daemon will automatically release capacity

**DO NOT** preempt other users. The yield daemon is already configured in c106's configuration.

## Evaluation Criteria

For each arm, assess:
1. **Overlap rate** (checkpoint probes): speaker_change, overlap_given_change
2. **Silence rate**: judge_score (clean handoffs should have appropriate pauses, not dead silence)
3. **Qualitative**: Read decoded outputs from probes - check for echo/self-repetition
4. **Comparison to c106 baseline**: c106's publication eval showed 45.9% clean handoff rate, 9.8% overlap rate

**Success = significantly lower overlap rate (<5%) without regressing on judge_score or introducing new degeneracies.**

## Next Decision Point

After these 4 arms complete (~2-3 days if all run sequentially on 3 V100s):
1. If Arm 1/2 succeed → consider reprocessing AMI/Werewolf to filter out overlap-heavy segments
2. If Arm 3/4 succeed → consider increasing λ_reward (currently 0.05) to make penalties matter more
3. If none work → the problem may be architectural (content loss always dominates) or measurement drift

## Reprocessing Strategy (future, if Arms 1/2 succeed)

If MELD-heavy training works well, we could:
1. Filter AMI/Werewolf to keep only examples with <2% overlap (would retain ~91% of AMI, ~93% of Werewolf by volume)
2. Rebuild the processed dataset with filtered versions
3. Train a "filtered natural" model: 30% MELD, 35% filtered_AMI, 35% filtered_Werewolf

This preserves natural timing/vocabulary/context while removing the worst overlap examples that teach bad habits.
