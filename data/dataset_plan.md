# Dataset Plan (Stage 2) -- IMPLEMENTED

Stage 2 teaches the matrix interface (who-speaks-when, turn-taking) while
preserving Qwen's general behavior. The pipeline turns multi-party
conversations into pre-tokenized, matrix-formatted (`[A, T]`) Arrow shards on
`/data`, so training only memory-maps ready-made integer tensors.

**Architecture note (breaking change from the earlier silence-token design):**
there is no vocabulary "silence token" anymore. Every matrix cell has an
explicit `input_activity_mask` (is this cell active/speaking right now?) and
`activity_labels` (will this agent speak at the NEXT column?), and the model
has a dedicated binary **activity head** (speak/yield) alongside the normal
vocabulary head -- see `model/matrix_qwen.py` and `model/turn_reward.py`.
Processed shards from before this change are missing these fields and MUST be
regenerated with `scripts/data_prep/preprocess_meld_ami_werewolf.sbatch`.

## Tiered dataset blend

- Structure tier (3+ agents, turn-taking; NO content loss):
  - `molweni` -- Ubuntu multi-party chat (avg ~3.5 speakers, 2-9). GitHub source.
- Breadth tier (register; content-bearing):
  - `meld` -- Friends, casual multi-party. HF `neoluigi/MELD-MPCA`.
  - `ami` -- CC BY 4.0 multi-party meetings with word timestamps, natural
    overlaps, backchannels, and pauses. Content-bearing.
  - `werewolf` -- Werewolf-Among-Us (191 usable One Night Ultimate Werewolf
    games: 151 YouTube + 40 Ego4D) with per-utterance timestamps and
    game-level roles. Content-bearing; see "Werewolf private context" below.
  - `conversation_chronicles` -- large casual dyadic multi-session. HF.
- Retention tier (anti-forgetting; content-bearing):
  - `qwen_distill` -- self-distilled Qwen3-4B responses (assistant = target seat).

Knowledge retention comes from (1) LoRA + freezing and (2) the clean
self-distilled replay tier -- NOT merely from adding more human chat. The
remaining noisy structure-only set (Molweni) is `content_bearing=False`:
context + turn-taking structure only, so the model never learns its jargon.

## Conversion (turn-major matrix)

Each turn occupies a contiguous block of time columns owned by its speaker's
row; other rows are INACTIVE in those columns (`input_activity_mask = False`,
input = an arbitrary `placeholder_token_id` always overridden by the model's
learned inactive-cell embedding). No EOS delimiter; turns end by the speaker
going inactive, and a random `[min_turn_gap, max_turn_gap]` inactive gap follows
each turn. Per example:
- Agent rows are randomly permuted (no fixed row<->role binding).
- `activity_labels[a, t] = input_activity_mask[a, t+1]` for EVERY agent and
  EVERY source (known ground truth "who speaks when"). One rule covers
  floor-taking (inactive->active), continuing (active->active), and stopping
  (active->inactive). The last column has no successor and is `ignore_index`.
- Target seat: for content-bearing sources, one speaker (preferring an
  `assistant` role, else rotated) carries shifted next-token CONTENT labels
  (`label[a,t] = input[a,t+1]`), but ONLY while continuing to speak into the
  next column -- never on a turn's last token (stopping is an ACTIVITY
  decision, not a content one). All other speakers and all structure-only
  sources carry NO content labels.
- Each example is tagged with an integer `source_id` (for weighted sampling +
  per-dataset loss breakdown).

Note: the matrix forward uses an UNSHIFTED custom CE for content, so the
converter emits already-shifted content labels.

### AMI hybrid conversational timeline

AMI does not use the synthetic-gap turn-major path. The adapter resolves
manual NXT dialogue acts to word-level forced-alignment times. Conversion then:

- packs sole-speaker text densely (speech rate never creates fake yielding);
- splits where the active speaker set changes and packs overlapping speakers
  concurrently in matrix rows;
- inserts all-inactive columns only when nobody speaks for more than one
  second, at one column per additional second (maximum eight);
- chunks at pause/dialogue boundaries so `num_agents*T <= 1536`;
- supervises the first content token after a pause from the final inactive cell.

AMI's official meeting-set splits are retained so `a/b/c/d` sessions from one
scenario cannot leak between training and validation.

**Context lookback across chunk boundaries.** Chunking a long conversation
means every chunk after the first is structurally a cold start -- no
preceding tokens at all -- even though it's genuinely mid-conversation.
`ConvertConfig.context_lookback_columns` (CLI: `--context_lookback_columns`,
default 0) prefixes each non-first chunk with up to N REAL columns copied
from immediately before its start (real `input_ids`/`input_activity_mask`,
not synthetic filler), reserved out of the same `max_flat_len` budget. Those
prefix columns are context only: `labels`/`activity_labels` stay
`ignore_index` there (they were already the label-bearing "new" portion of
the previous chunk, so re-supervising them would be redundant, not new
signal). Applies to both AMI and Werewolf (both use this hybrid path); MELD's
turn-major conversion never chunks, so it's unaffected.

## Werewolf private context (per-agent visibility)

Werewolf reuses the same hybrid timed converter as AMI (real per-utterance
`Dialogue` timestamps), but adds a genuinely new mechanism: **per-agent
attention visibility**, since Werewolf is the first source where some context
must be hidden from some agents.

**Known data limitation:** the raw transcripts cover only the *public*
day-phase discussion. Night-phase actions (who the Seer inspected, who the
Robber/Troublemaker swapped) are never recorded, so nothing here fabricates
their results. The only private knowledge synthesized is grounded in what a
player would actually know at the start of the night in One Night Ultimate
Werewolf: their own dealt role (`startRoles`); mutual teammate knowledge for
Werewolves and Masons; and the Minion's one-directional knowledge of who the
Werewolves are (never reciprocated).

**Sequential prelude** (before the real `Dialogue` begins), one pair of turns
per player, in roster order -- deliberately sequential, never simultaneous,
so it needs no special-casing in the turn-taking reward/silence logic (it is
just an ordinary run of solo turns):
1. A PUBLIC self-introduction turn (`"Hi, I'm {name}."`) -- gives every agent
   a name identifier to correlate against later dialogue mentions.
2. A PRIVATE role-reveal turn (`Turn.visible_to`) naming the player's own role
   and, when applicable, teammates by name AND role.

**Randomized names:** the real corpus has only ~59 recurring usernames across
191 games. Every game draws a fresh, unique set of replacement names from a
200+ name pool (seeded per-game, so reproducible but independent across
games), stacking with the existing row permutation so neither row position
nor name string is ever a stable proxy for role/team/identity. Real names are
used only internally to align `Dialogue.speaker` with `playerNames`/
`startRoles`; in-dialogue mentions of a real name are substituted (whole-word,
single atomic pass) to the same game's randomized name.

**Attention mechanism:** `data/schema.py`'s `Turn.visible_to` (extra speaker
indices, beyond the speaker itself, permitted to view a turn) is converted
into `is_private_mask [A,T]` and `agent_visibility [A,A]` (see
`data/convert.py`), and `model/masks.py`'s `build_matrix_causal_mask` blocks
any query agent not in `agent_visibility` from attending a private key cell as
a KEY, on top of the existing causal/inactive rules. This blocks DIRECT
attention to another agent's private cells; it does not (and structurally
cannot) prevent an unauthorized agent from later observing the owner's own
PUBLIC speech, even though that speech was computed from context including
the owner's own private cells -- that asymmetry is intentional and mirrors
real social-deduction play (your role card is unreadable, your subsequent
public behavior is fair game). Verified end-to-end in `tests/test_masks.py`.

Sources without private turns (MELD, AMI, ...) are completely unaffected:
`is_private_mask`/`agent_visibility` are only added to a shard when at least
one turn uses `visible_to`, and `collate_matrix_batch` defaults them to fully
public when absent from a given example, so mixed-source batches stay valid
without touching already-processed shards for other sources.

Werewolf's official `train`/`val`/`test` split files are used directly
(`val` normalized to `validation`); `group_id` disambiguates by
`YT_ID`/`EG_ID` **and** `video_name` (the former alone is not a unique video
key in this corpus and collides across unrelated recordings -- see
`_game_id`'s docstring) so no game leaks across splits.

## Code layout

- `data/schema.py` -- `Conversation` / `Turn` (incl. `visible_to`) / `TimedWord`.
- `data/adapters/{ami,molweni,meld,werewolf,conversation_chronicles,qwen_distill}.py`
  + `data/adapters/spec.py` registry (stable `source_id`, default weights,
  optional `hf_allow_patterns` to skip large non-text assets in a snapshot).
- `data/convert.py` -- `convert_conversation(...)` (turn-major) and
  `convert_timed_conversation(...)` (hybrid timeline; also emits
  `is_private_mask`/`agent_visibility` when a source uses private turns).
- `data/manifest.py` -- per-source manifest (shards, counts, default weight).
- `data/build_dataset.py` -- CLI: `--mode download|convert`.
- `data/distill.py` -- self-distillation generation.
- `training/multi_source.py` -- weighted interleave sampler, batch collation
  (pads `input_activity_mask`/`activity_labels`/`is_private_mask`/
  `agent_visibility`, defaulting the latter two to fully-public when a source
  doesn't use them), per-source loss breakdown.
- `training/config.py` -- `--processed_data_dir`, `--dataset_weights`,
  `--placeholder_token_id`, `--lambda_activity`, `--lambda_reward`,
  `--speak_grace`, `--speak_tau`, `--silence_grace`, `--silence_tau`,
  `--overlap_base_weight`, `--overlap_max_weight`, `--overlap_grace`,
  `--overlap_tau`, `--activity_threshold`.
- `model/turn_reward.py` -- ground-truth run-length scanner + the differentiable
  Stage-A turn-taking reward (see "Turn-taking loss" below).
- `model/masks.py` -- `build_matrix_causal_mask`, incl. the private-cell
  attention ACL (`private_mask`/`agent_visibility`).

## Jobs (Rosie)

Current pipeline (all three sources, equally weighted):

1. `scripts/data_prep/download_data.sh` -- run on an internet node; populates
   `$RAW_DIR=/data/$USER/matrixchat/raw` (HF cache + GitHub clones).
2. `scripts/data_prep/distill_qwen.sbatch` -- GPU; generates the `qwen_distill` source.
3. `scripts/data_prep/preprocess_meld_ami_werewolf.sbatch` -- CPU, offline
   (`HF_HUB_OFFLINE=1`); converts MELD + timed AMI + Werewolf raw data ->
   Arrow shards in `$PROCESSED_DIR` + `manifest.json`.
4. `scripts/training/train_meld_ami_werewolf.sbatch` -- trains all three
   sources EQUALLY weighted (~1/3 each) on a V100; most hyperparameters are
   environment-overridable (see the script header).

(Earlier single- and two-source variants of the preprocess/train scripts have
been superseded and removed; see `TRAINING_RUN_LOG.md` for their history.)

Artifacts on `/data`: `raw/` (downloads), `processed/<source>/` (Arrow shards),
`processed/manifest.json`.

## Turn-taking loss (Stage A)

The model has two heads per cell: the normal vocabulary head (content) and a
binary `activity_head` (speak/yield), fed by the SAME final hidden state --
see `model/matrix_qwen.py`. Total loss:

```
loss = content_CE (gated: only cells with a content label)
     + lambda_activity * activity_BCE (balanced across the speak/yield classes)
     - lambda_reward   * mean(turn_taking_reward)
```

`turn_taking_reward` (in `model/turn_reward.py`) rewards exactly-one-speaker
columns (including a full-credit direct handoff to a different agent), gently
decays reward for one agent monopolizing the floor past `speak_grace` columns
(time constant `speak_tau`), and increasingly penalizes prolonged all-silent
columns past a grace period (time constant `silence_tau`). Predicted overlap is
explicitly penalized using `P(overlap) = 1 - P(none) - P(exactly one)`: weight
0.35 at onset, ramping toward 1.5 after a one-column grace (`overlap_tau=3`).
**Stage A uses GROUND-TRUTH `input_activity_mask` to compute
same-speaker/all-silent/overlap run lengths (teacher forcing)** -- this is
intentional for supervised training but must NOT be reused verbatim for online
RL/self-play: Stage B must derive those run lengths from the model's own
SAMPLED actions along a rollout and must not backprop through the counters.
See the RL Turn-Taking Design plan for the full migration checklist.

## Hyperparameters (new)

- `--dataset_weights "molweni=0.25,meld=0.15,..."` -- per-source sampling
  probabilities (empty -> manifest defaults).
- `--placeholder_token_id` -- dummy input id for inactive cells (always
  overridden by the learned inactive-cell embedding).
- `--lambda_activity`, `--lambda_reward`, `--speak_grace`, `--speak_tau`,
  `--silence_grace`, `--silence_tau`, `--overlap_base_weight`,
  `--overlap_max_weight`, `--overlap_grace`, `--overlap_tau` -- the
  turn-taking loss weights/shape above.
- `--activity_threshold` -- generation-time `P(speak)` cutoff.

## IO / GPU utilization

All heavy CPU work (tokenize + matrix convert + label construction) is offline.
Training memory-maps Arrow, uses `DataLoader(num_workers>0, pin_memory,
prefetch, persistent_workers)`, a `DistributedSampler` per DDP rank, and length
bucketing. (The earlier token-id-based `drop_silence` compaction was removed
with the silence-token design; an activity-mask-based compaction is a possible
future optimization, not yet implemented.)

## Training-stage status

The `main.py` training loop (DataLoader, per-source loss logging, qualitative
sampling, checkpointing) is wired -- see `scripts/training/train_meld_ami_werewolf.sbatch`
for a worked example. Multi-GPU (LoRA + freeze top-layers + DDP via Accelerate) is
still future work. KV cache stays deferred to the Werewolf/Avalon RL phase.

## Candidate datasets considered

LMSYS-Chat-1M and UltraChat were considered but deprioritized (2-party / heavy IO);
`ishiki-labs/multi-party-dialogue` is decision-point extracts (truncated context),
usable only as an auxiliary SPEAK/SILENT signal.
