# MatrixChat Project Memory

> Persistent project memory for coding agents. Keep this file and
> `.claude/MATRIX_CHAT_PROJECT.md` identical and up to date.

## 1. Research idea

MatrixChat is a PhD research prototype that modifies how Transformer-based LLMs
represent multi-party communication.

Standard dialogue is a single flat token stream. MatrixChat instead represents a
conversation as a **matrix**:

- **rows** = agents / speakers / players
- **columns** = shared time steps / shared interaction positions
- **cells** = token slots for each agent at each shared time step

Long-term tensor shapes:

```
input_ids:      [batch, agents, time]
agent_ids:      [batch, agents, time]
position/time:  [batch, agents, time]
labels:         [batch, agents, time]
logits:         [batch, agents, time, vocab]
```

Each agent owns one row. Agents may attend to relevant public context.
Per-agent PRIVATE context + a genuine attention-visibility ACL are now
implemented (Werewolf source; see `model/masks.py`'s `private_mask`/
`agent_visibility` and `data.schema.Turn.visible_to`) -- see section 2 below.

This milestone is intentionally simple: a **testable wrapper** around Qwen3, not
the final architecture.

## 2. What has been implemented (Milestone 1)

- `model/matrix_qwen.py`: `MatrixQwenConfig`, `MatrixCausalLMOutput`,
  `MatrixQwenForCausalLM` wrapper.
  - Forward accepts `input_ids [B, A, T]`, returns `logits [B, A, T, V]`.
  - Column-major flatten `[B, A, T] -> [B, T*A]` and unflatten back.
  - Learned agent embeddings added to token embeddings before Qwen.
  - Optional channel embeddings.
  - Two position modes: `flat` and `column`.
  - Custom cross-entropy loss over matrix-aligned labels (NOT Qwen's shifted loss),
    GATED: only computed where a content label exists (else a defined 0.0, never
    NaN -- an all-`-100` batch previously produced NaN via `F.cross_entropy` and
    silently poisoned running averages like validation loss; fixed and regression-
    tested in `test_activity.py::test_content_loss_is_zero_not_nan_when_no_valid_targets`).
  - **No vocabulary "silence token".** Every cell has an explicit
    `input_activity_mask` (is it active/speaking right now?) and a dedicated
    binary **`activity_head`** (`nn.Linear(hidden, 1)`, always present) predicting
    `P(this agent speaks at the NEXT column)`, fed by the SAME final hidden state
    as the vocabulary head (`output_hidden_states=True`). Inactive cells use a
    trainable `inactive_embedding` (always present) instead of the base model's
    embedding of an arbitrary `placeholder_token_id`, so "being silent" has a
    real learned input representation.
  - `activity_labels[a,t] = input_activity_mask[a,t+1]` for EVERY agent (known
    ground truth), supervised with a BALANCED BCE (speak-class and yield-class
    averaged equally, so the abundant yield class can't collapse the policy to
    always-silent). Content labels are gated to only apply while the target
    continues speaking into the next column (stopping is an activity decision).
  - `model/turn_reward.py`: differentiable Stage-A turn-taking reward --
    rewards exactly-one-speaker columns (a direct handoff is fully rewarded),
    decays the same-speaker-continuation reward past `speak_grace` columns
    (time constant `speak_tau`), and ramps an all-silence penalty past a grace
    period (time constant `silence_tau`). Overlap has an explicit differentiable
    `P(overlap)` penalty: onset weight 0.35, ramping to 1.5 after a one-column
    grace (`overlap_tau=3`). Uses GROUND-TRUTH `input_activity_mask` to compute
    same-speaker/all-silent/overlap run lengths (teacher forcing) -- Stage A only.
    **MANDATORY Stage-A -> Stage-B migration**: before online RL/self-play, the
    run-length counters MUST come from the model's own SAMPLED actions along a
    rollout, not dataset ground truth, and must not be backpropagated through.
    Total loss: `content_CE + lambda_activity*activity_BCE - lambda_reward*mean(reward)`.
  - dtype: agent/channel/inactive embeddings stay fp32 and are cast to the base
    model's dtype before adding, so a bf16/fp16 base model + fp32 wrapper
    embeddings work.
  - The earlier `drop_silence` compaction (token-id-based) was REMOVED with the
    silence-token design; an activity-mask-based compaction is possible future
    work, not yet implemented.
- `model/masks.py`: `build_matrix_causal_mask` -> additive `[B, 1, S, S]` mask.
  - `allow_same_column=False`: a cell sees all previous columns plus **itself**
    (diagonal); other agents in the same current column are blocked.
  - `allow_same_column=True`: the whole current column (`tk <= tq`) is visible.
  - `inactive_mask` (optional `[B, S]` bool, renamed from `silence_mask`): cells
    that are inactive are made **invisible** -- no query attends them as a key
    except the diagonal (so their own softmax row stays well-defined; output
    discarded). This also hides an inactive cell from the same agent's later
    columns. (The coord-based `build_matrix_mask_from_coords` compaction helper
    was removed along with `drop_silence`.)
- `model/generation.py`: activity-gated generation.
  - `generate_next_tokens_for_all_agents` / `generate_matrix` -> `(tokens, speak_mask)`.
    No placeholder query-column trick needed: `activity_logits[:, :, -1]` already
    predicts the NEXT column's activity from the existing last column.
- `model/lora_utils.py`: freeze/unfreeze/LoRA/param-count helpers; the matrix
  interface (agent/channel embeddings, `inactive_embedding`, `activity_head`) is
  always kept trainable through `freeze_base_model`.
- `training/multi_source.py::activity_metrics`: diagnostics for the activity
  head -- thresholded accuracy vs ground truth, the empirical `speak_rate`
  (fraction of cells truly "speak"), and the model's own `predicted_speak_rate`
  (to catch a collapse to always-yield/always-speak). Logged every
  step/eval in `main.py` (`train_activity_accuracy`, `train_speak_rate`,
  `train_predicted_speak_rate`, and the `val_components` equivalents) and
  shown live in the tqdm postfix (`acc=`).
- `scripts/plot_metrics.py`: exports a run's `metrics.jsonl` as PNGs (total
  loss, content loss, activity loss, turn reward, activity accuracy/speak-rate
  panel) -- `python scripts/plot_metrics.py --run_dir logs/run_000018`.
- **Class-imbalance design note**: the activity BCE does NOT need a manually
  tuned "silence weight" -- it already averages the speak-class mean loss and
  the yield-class mean loss 50/50 every batch (`model/matrix_qwen.py`), which
  is full inverse-frequency reweighting recomputed continuously, strictly
  stronger than a fixed global weight. `activity_metrics`' `speak_rate` /
  `predicted_speak_rate` above make the actual imbalance and the model's
  response to it visible without needing a separate hyperparameter.
- `training/config.py`: argparse config with validation.
- `training/run_ids.py`: run-id + hyperparameter/log/checkpoint tracking.
- `training/checkpoint.py`: reusable save/load helpers + `CheckpointManager`
  (best validation checkpoint at root, terminal state under `last/`, metadata in
  `model_state.pt`, `best_checkpoint.json` summary, unfrozen base-layer restore).
- `training/toy_batch.py`: tiny Qwen3 factory + toy matrix batch.
- `main.py`: smoke training loop with run-id tracking (plus `--dry_run`,
  `--print_trainable_params`, `--save_checkpoint`, JSONL metrics).
- `scripts/train_matrix_qwen.sbatch`: SLURM entrypoint tailored for Rosie.
- `scripts/multi_agent_tokens.py`: loads the real Qwen, generates one decision
  (speak or yield, via the activity head) per agent per matrix forward pass,
  prints columns (one per agent). Args: `--prompt`/`--prompts`, `--num_agents`,
  `--max_new_tokens`, `--temperature`, `--activity_threshold`, `--chat`,
  `--forced_silent_agents` (force activity False every step), `--stop_on_eos`,
  `--checkpoint_dir` (loads via `training.checkpoint.apply_checkpoint_to_model`:
  LoRA + agent embeddings + inactive embedding + activity head + optional
  unfrozen base layers; rejects pre-activity-head checkpoints with a clear error
  via `load_checkpoint_state`). Per-agent EOS (`--stop_on_eos`, default on): when an
  agent emits a content EOS id (Qwen3: `<|im_end|>`=151645 or
  `<|endoftext|>`=151643), it's marked finished (shown with a `⏹` prefix) and
  forced to yield thereafter; generation halts once ALL agents are finished,
  with `--max_new_tokens` as the backstop.
- `scripts/turn_taking_probe.py`: raw-prefill (no chat template) turn-taking
  scenarios (`listeners`/`handoff`/`midspeech`) against a trained checkpoint;
  `--debug_topk` prints each agent's `P(speak)` and top content tokens per step.
- `scripts/verify_cross_agent.py`: logit-level probe proving agents share PAST
  context (changing agent 1's past moves agent 0's logits) with no future / no
  same-column leakage. Verified on real Qwen3-4B: check1 max|Δ|~9.2, checks 2&3 = 0.0.
- Real Qwen3-4B weights live (gitignored) under `models/Qwen3-4B-Instruct-2507/`.
- **Stage-2 data pipeline** (offline download + tokenize/convert): tiered blend
  (structure: molweni [no content loss]; breadth: meld, AMI timed meetings,
  werewolf [private-context timed], conversation_chronicles; retention:
  qwen_distill self-distill). AMI/Werewolf are content-bearing: solo speech
  is dense, true overlap is parallel, and all-speaker pauses over one second
  become bounded inactive columns. Turn-major matrix conversion producing
  `input_activity_mask`/`activity_labels` for every agent + rotated
  target-seat shifted content labels (gated to continued speech), agent-row
  permutation, `source_id` tagging. Code: `data/schema.py` (incl.
  `Turn.visible_to`), `data/adapters/*`, `data/convert.py` (turn-major +
  hybrid timed, incl. `is_private_mask`/`agent_visibility`), `data/manifest.py`,
  `data/build_dataset.py` (CLI download|convert|peek), `data/distill.py`; jobs
  `scripts/{download_data.sh,distill_qwen.sbatch,preprocess_data.sbatch,
  preprocess_meld_ami.sbatch,train_meld_ami.sbatch,
  preprocess_meld_ami_werewolf.sbatch,train_meld_ami_werewolf.sbatch}`;
  training hooks `training/multi_source.py` (weighted interleave, collate incl.
  activity + private-visibility fields, per-source loss). Config:
  `--processed_data_dir/--dataset_weights/--placeholder_token_id/
  --lambda_activity/--lambda_reward/--speak_grace/--speak_tau/
  --silence_grace/--silence_tau/--overlap_base_weight/--overlap_max_weight/
  --overlap_grace/--overlap_tau`.
  See `data/dataset_plan.md`. **Verified**: MELD reprocessed to the new schema
  (921 examples) and a real end-to-end CPU smoke (real Qwen3-4B, real MELD data,
  2 training steps + eval + checkpoint save/load + generation) confirmed
  finite `content_loss`/`activity_loss`/`turn_reward` and a working
  checkpoint round-trip.
  **Timed AMI verified (2026-07-18)**: official manual archive downloaded and
  fully converted to 2,370 bounded examples (1,662 train / 344 validation /
  364 test), with 83,678 real overlap columns and 26,891 preserved pause
  columns; no meeting-group split leakage. MELD now has a deterministic
  conversation-level 5% validation split (877/44). A mixed real-shard tiny
  forward/backward produced finite content/activity/reward losses and explicit
  `P(overlap)` diagnostics. Production data lives in
  `$HOME/matrixchat/processed_meld_ami`.
  **Werewolf private context verified (2026-07-18)**: real corpus downloaded
  (JSON/txt only, ~7.7MB, skipping ~9.9GB of unused video/audio via
  `AdapterSpec.hf_allow_patterns`) and fully converted to 923 examples from
  191 games (631 train / 112 validation / 180 test via the corpus's own split
  files; `group_id` disambiguates by `YT_ID`/`EG_ID` **and** `video_name` --
  the former alone collides across unrelated recordings, confirmed and fixed
  against the real data). A mixed MELD+AMI+Werewolf real-shard tiny
  forward/backward produced finite losses with 52 real private cells in the
  batch. The private-cell attention ACL was verified to give an EXACT
  (`torch.equal`), bit-for-bit zero logit difference for an unauthorized
  viewer agent when the owner has no legitimate indirect (public-speech)
  channel -- see `model/masks.py`'s docstring for why a nonzero difference is
  *expected and correct* once the owner speaks publicly again.
  **Fixed bug (2026-07-18, found on Rosie via a live 3-source training job
  crash):** `datasets.interleave_datasets` unions column schemas across
  sources -- once Werewolf's shard (which has `is_private_mask`/
  `agent_visibility`) is interleaved with MELD/AMI (which don't), every row's
  dict gets those keys, but MELD/AMI-origin rows get the value `None`, not a
  missing key. `training/multi_source.py::collate_matrix_batch` previously
  checked `"key" in ex` (always true post-interleave); fixed to
  `ex.get("key") is not None`. Regression test added; re-verified against the
  real crashing dataset AND a live 10-step smoke job on Rosie (completed
  0:0, `val_by_source` covering all three sources).
- `tests/`: pytest suite (**123 tests**, 0 skipped when PEFT is installed) across:
  - `test_multi_source.py` -- `activity_metrics`: perfect predictions, ignore-index
    handling, class-imbalance reflected in `speak_rate`, all-ignored -> NaN,
    `threshold` argument.
  - `test_matrix_shapes.py` -- shapes, variable agents, flatten/unflatten roundtrip.
  - `test_forward_backward.py` -- finite loss + agent-embedding grad norm > 0;
    the toy batch now also exercises activity BCE + turn-taking reward
    (activity-head grad checked too).
  - `test_position_ids.py` -- exact flat/column position-id values.
  - `test_masks.py` -- mask shape, same-column semantics, first-column self.
  - `test_causality.py` -- finite logits, no future leakage, no same-column
    cross-agent leakage.
  - `test_wrapper_features.py` -- channel embeddings, position-mode effect, loss
    ignore_index, num_agents > max_agents raises.
  - `test_generation.py` -- activity-gated generation: yield emits the
    placeholder (never a sampled token), `--activity_threshold` gates
    speak/yield, `generate_matrix` shapes.
  - `test_lora_freezing.py` -- freeze/unfreeze/LoRA; matrix-interface trainable
    param count now includes `inactive_embedding` + `activity_head` too.
  - `test_config.py` -- str2bool, target-module parsing, arg validation.
  - `test_main_integration.py` -- end-to-end `main.main()` smoke + dry-run.
  - `test_activity.py` -- inactive-cell attention invisibility (`inactive_mask`);
    `inactive_embedding`/`activity_head` always present + trainable; default
    (no mask) == all-active back-compat; content loss gated + defined-zero (not
    NaN) when no valid targets; balanced activity BCE under class imbalance;
    `ground_truth_run_lengths` hand-traced; `turn_taking_reward` matches a
    hand-derived formula.
  - `test_convert.py` -- Stage-2 converter: turn-major shapes, `activity_labels`
    covering every agent/transition, content labels gated to continued speech
    (never on a turn's last token), structure-only sources carry no content
    labels but full activity labels, permutation-invariant content labeling,
    source tagging.
  - `test_ami_adapter.py` / `test_timed_convert.py` -- NXT range/timing parsing,
    stable AMI source ID, dense solo speech, true overlap, same-speaker pauses,
    zero-duration event placement, bounded chunks, post-pause content labels,
    and private-turn `is_private_mask`/`agent_visibility` correctness in both
    the turn-major and timed converters.
  - `test_werewolf_adapter.py` -- real split-file schema parsing, pinned
    `source_id`, public self-intro/private role-reveal content and visibility,
    mutual Werewolf/Mason teammate reveals, one-directional Minion knowledge,
    sequential (non-overlapping) prelude, per-game randomized names (unique,
    reproducible, differing across games), consistent in-dialogue name
    substitution, Avalon-only/malformed-record skipping, and the
    `YT_ID`+`video_name`+`Game_ID` group-id collision fix.
  - `test_masks.py` -- private-cell ACL: non-permitted viewer blocked, self and
    permitted viewers allowed, diagonal always visible, public cells
    unaffected, and an end-to-end model-level check that an unauthorized
    agent's logits are EXACTLY unchanged (`torch.equal`) when the owner has no
    other legitimate channel.
  - `test_checkpoint_migration.py` -- old (pre-activity-head) checkpoints are
    rejected with a clear error via `training.checkpoint.load_checkpoint_state`;
    new-format checkpoints load cleanly.
  - `test_checkpoint_best.py` -- best/last layout, metadata, no-validation
    fallback, unfrozen-base restore, non-finite val_loss ignored.

## 3. What is intentionally NOT implemented yet

- Private/public visibility masks ARE now implemented (see section 2:
  `Turn.visible_to`, `is_private_mask`/`agent_visibility`,
  `model/masks.py`'s ACL) for the Werewolf source. NOT yet done: exposing this
  through the qualitative sampling probe in `main.py`/demo scripts (private
  context can be prefilled via `model/generation.py`'s
  `input_private_mask`/`agent_visibility` params, but no CLI script does so
  yet), and per-message (rather than per-agent-pair) visibility groups (out of
  scope for Werewolf, where each agent has only one uniform private audience).
- Relation-aware attention bias.
- 2D / multi-axis RoPE (only standard 1D Qwen RoPE; `column` mode just shares
  position ids per time column).
- A full Werewolf/Avalon RL environment (only the SUPERVISED data source +
  private-context/visibility infrastructure exists so far; online rollout,
  reward shaping from game outcome, and PPO are still Stage B work).
- Online RL / self-play (Stage B): the turn-taking reward currently uses
  ground-truth run lengths (Stage A / teacher forcing); Stage B requires
  recomputing them from sampled model actions -- see the mandatory migration
  note above and the RL Turn-Taking Design plan.
- Night-phase Werewolf actions (Seer inspections, Robber/Troublemaker swaps):
  not recorded in the raw transcripts, so not fabricated or supervised --
  documented limitation in `data/adapters/werewolf.py` and `dataset_plan.md`.
- Activity-mask-based compaction (an analogue of the removed `drop_silence`).
- Multi-token autoregressive generation loop for real conversations beyond the
  simple greedy/temperature-per-step helpers in `model/generation.py`.

## 4. How to run tests

```bash
pip install -r requirements.txt
pytest -q
```

Tests run on CPU, offline, using a tiny randomly initialized Qwen3 model. They do
NOT download `Qwen/Qwen3-4B-Instruct-2507`.

## 5. How to run the smoke training / sbatch

Local smoke run:

```bash
python main.py --use_tiny_model true --num_steps 3
```

On Rosie (SLURM):

```bash
sbatch scripts/train_matrix_qwen.sbatch
```

All hyperparameters are set as bash variables inside the sbatch file and passed
to `main.py` as CLI args (no key settings hidden in Python defaults).

## 6. Current known limitations / caveats

- `column` position mode is a distribution shift for a pretrained Qwen and is
  experimental, not final.
- Custom 4D attention masks require eager attention; the tiny model is built with
  `attn_implementation="eager"`. With `transformers 5.x` the tiny Qwen3 accepts
  the additive `[B, 1, S, S]` mask directly under eager attention -- no fallback
  was needed. If a future backend rejects 4D masks, fall back to a 2D mask for
  smoke tests while keeping the 4D builder.
- Loss is custom CE over matrix-aligned labels with `ignore_index=-100`. Do not
  place labels on slots whose input already contains the target token unless
  intended (tests use query slots).
- LoRA is optional; PEFT only required when `--lora_enable true`.
- **LoRA + freeze ordering caveat**: `freeze_base_model` must run BEFORE
  `apply_lora_if_enabled` (PEFT marks its adapters trainable, so they survive the
  freeze). `freeze_base_model` also defensively skips params whose name contains
  `lora_`. If you freeze after injecting LoRA without this guard, the adapters get
  frozen and only the agent embeddings train.
- **Mask self-attention caveat (IMPORTANT design decision)**: a literal "block
  the entire current column" rule (the original spec wording) leaves the first
  time column `t=0` with ZERO allowed keys. An all-masked softmax row does not
  error -- it degenerates into *uniform attention over the whole sequence,
  including future tokens* -- which silently leaks the future into `t=0` and then
  forward into every later column. Empirically measured leakage was ~0.05 on
  column-0 logits and ~0.03 cross-agent before the fix. Fix: always allow the
  **diagonal** (a cell attends to itself) when `allow_same_column=False`. This
  preserves "no cross-agent same-column leakage", keeps every row well-defined,
  and makes the wrapper fully causal (verified leakage == 0.0 in
  `tests/test_causality.py`). This intentionally deviates from the spec's literal
  "q at t=1 cannot attend t=1" -- now q can attend its own t=1 slot but still not
  the other agent's t=1 slot.
- transformers >=5 renamed `from_pretrained(torch_dtype=...)` to `dtype=...`;
  `main.py` tries `dtype` first and falls back to `torch_dtype`.
- **Content-loss NaN caveat (found via real end-to-end smoke testing)**:
  `F.cross_entropy` over an all-`-100` batch (e.g. an example whose target
  turns are all single tokens, so no internal continuation exists to
  supervise) returns NaN. Previously this silently poisoned any running sum
  (e.g. the validation-loss average across a val set observed `val_loss=NaN`
  in a real MELD run). Fixed: `content_loss` is a defined `0.0` (no gradient
  contribution) when a batch has zero valid content labels; regression-tested
  in `test_activity.py::test_content_loss_is_zero_not_nan_when_no_valid_targets`.
- **Checkpoint incompatibility**: checkpoints saved before the activity-head
  architecture (silence-token design) are missing `inactive_embedding` /
  `activity_head` and CANNOT be loaded by the current
  `multi_agent_tokens.py` / `turn_taking_probe.py` -- both raise a clear
  `ValueError` (`load_checkpoint_state`) rather than silently mismatching.
  Retrain from scratch with the current code.
- **`loss=None` edge case (found by parallel review agent, fixed)**: `loss`
  was only initialized when `content_loss is not None or activity_loss is not
  None`, so a call with `activity_labels` given but ALL entries `-100` (plus
  `input_activity_mask` given, plus no `labels`) computed a finite
  `turn_reward` yet left `out.loss=None` -- `.backward()` would crash. Fixed by
  widening the condition to include `turn_reward_mean is not None`
  (`model/matrix_qwen.py`); regression-tested in
  `test_activity.py::test_loss_defined_when_only_reward_is_present`.
- **Toy smoke path now exercises activity/reward too**: `make_toy_matrix_batch`
  previously returned only `(input_ids, labels)`, so the fastest smoke path
  (`main.py` with no `--processed_data_dir`) never touched the activity BCE or
  turn-taking reward code -- a real training bug there could hide behind a
  passing toy smoke run. Fixed: it now also returns a degenerate but
  self-consistent `input_activity_mask` (all active) and `activity_labels`
  (all speak-next except the last column), and `main.py::train_on_toy_batch`
  passes them through.

## Verification status (Milestone 1 -- PASSING)

Verified on Windows + Python 3.12 + torch 2.7.0+cpu + transformers 5.12.1 + peft 0.19.1:

- `pytest -q` -> **41 passed** (CPU, offline, tiny Qwen3). Coverage includes:
  shapes, variable agents, flatten/unflatten roundtrip, forward/backward grad,
  position-id values, mask shape/semantics/first-column, **causality + no-leakage**,
  channel embeddings, position-mode effect, loss ignore_index, LoRA/freezing,
  config parsing/validation, and an end-to-end `main.main()` smoke + dry-run.
- `python main.py --use_tiny_model true --num_steps 3` -> runs, loss decreases,
  run-id tracking + JSON + log/checkpoint dirs created.
- LoRA run (`--lora_enable true --freeze_base_model true`) -> 7,680 trainable
  params (agent embeddings + LoRA adapters); base otherwise frozen.
- CUDA requested without a GPU falls back to CPU; bf16/fp16 on CPU falls back to
  fp32 -- both with warnings.
- Causality probe: changing a future-column token leaves earlier-column logits
  bit-identical (diff 0.0); changing one agent's same-column token leaves the
  other agent's same-column logits bit-identical (diff 0.0).
- `transformers>=4.51.0` satisfied by installed `transformers 5.12.1`.

## Verification status (Activity-head / turn-taking architecture -- PASSING)

Verified on Rosie (Linux, Python 3.12, torch 2.6.0+cu124, transformers 4.56.2,
peft 0.19.1; also re-run in this dev environment):

- Three independent parallel review agents were run over (1) model
  activity-head/reward math, (2) data conversion/collation, (3)
  scripts/training/checkpoint consistency. (2) found no bugs; (3) found no
  stale references or CLI mismatches but flagged the toy-path gap above; (1)
  found the `loss=None` edge case above. Both were fixed and regression-tested;
  full suite re-verified green after the fixes.
- `pytest -q` -> **81 passed** (CPU, offline, tiny Qwen3), including
  `test_activity.py` (13 tests: inactive-cell attention invisibility, always-
  trainable inactive embedding + activity head, back-compat when no mask is
  given, content-loss NaN-vs-defined-zero, balanced activity BCE, hand-traced
  `ground_truth_run_lengths`, hand-derived `turn_taking_reward`),
  `test_convert.py` (rewritten for the activity-mask schema), and
  `test_checkpoint_migration.py` (old-checkpoint rejection).
- MELD reprocessed end-to-end with the new schema: 921 examples, verified via
  `data.inspect` (activity labels correctly mark floor-taking/continuing/
  stopping for every agent, e.g. a gap column immediately before a turn is
  labeled "speak-next").
- Real end-to-end CPU smoke with `--use_tiny_model false` (actual
  `Qwen/Qwen3-4B-Instruct-2507`, actual MELD data): model load -> DataLoader ->
  collate (incl. `input_activity_mask`/`activity_labels`) -> 2 forward/backward
  steps -> validation `evaluate()` -> checkpoint save, ALL producing finite
  `train_loss`/`content_loss`/`activity_loss`/`turn_reward`. This run is what
  surfaced and confirmed the content-loss NaN bug and its fix (see caveats).
- Checkpoint round-trip verified: `scripts/multi_agent_tokens.py
  --checkpoint_dir <run>` loads LoRA + agent embeddings + inactive embedding +
  activity head and generates with real activity-gated decisions (both agents
  spoke every step in a 4-step smoke, as expected pre-training).
- Old-format (silence-token) checkpoints confirmed to raise a clear
  `ValueError` instead of silently loading into an incompatible model.

## 7. Next steps

1. Compare `flat` vs `column` position modes on the real Qwen3-4B with the
   activity-head architecture.
2. Evaluate turn-taking quality on a real training run (speak/yield precision-
   recall, exactly-one-speaker rate, floor-taking/stopping recall).
3. Stage B: online rollout self-play with SAMPLED (not ground-truth) run
   lengths, PPO, and a KL-to-reference anchor -- see the mandatory migration
   note and the RL Turn-Taking Design plan.
4. Prototype private/public visibility masks.
5. Explore 2D / multi-axis RoPE.
6. Build a minimal multi-agent environment (Werewolf/Avalon) for RL later.

### Performance / scaling roadmap (planned, not yet implemented)

- **KV cache + incremental matrix mask: DEFERRED to the online Werewolf/Avalon RL
  stage.** Rationale: training is teacher-forced (the whole `A*T` matrix is one
  forward/backward pass), so a KV cache gives ZERO training speedup -- it only
  helps autoregressive token-by-token generation, which dominates cost during RL
  rollouts. Build it then (inactive/EOS-finished agents naturally drop out of
  the cache). Current generation uses `use_cache=False` and recomputes the full
  sequence each step (`O(N^2)`), which is fine for now.
- **Training-phase perf levers (the ones that matter now):** SDPA attention
  (our additive 4D mask is SDPA-compatible; faster than eager), an
  activity-mask-based compaction (successor to the removed `drop_silence`, not
  yet implemented), LoRA + selective freezing, gradient checkpointing, and
  multi-GPU DDP.
- **Multi-GPU training:** start with **LoRA + freeze most layers + DDP via HF
  Accelerate** (gradients all-reduced/shared across GPUs; tiny trainable surface
  fits a T4). Freeze the BOTTOM layers with NO adapters so backprop can stop at
  the first trainable (top) layer -- to realize this, add `--lora_layers`
  (PEFT `layers_to_transform`) so LoRA is confined to the top N layers. Use
  FSDP/ZeRO only if full fine-tuning (V100/H100 nodes). Re-baseline on a GPU
  before optimizing (CPU timings are misleading).

## 8. Rosie supercomputer (MSOE)

This repo is intended to run on **Rosie**, MSOE's academic HPC cluster
(https://msoe.dev/#/about), NOT on a local machine. Tests are the only thing
designed to run locally/offline.

### Hardware (27 compute nodes)

| Type | Count | CPU | RAM | GPU |
| --- | --- | --- | --- | --- |
| Management / login | 4 | Xeon Gold 6240 (72) | 187 GB | none |
| Compute (`teaching`) | 18 | Xeon Gold 6240 (72) | 376 GB | 4x Tesla T4 |
| HighMem Compute | 2 | Xeon Gold 6240 (72) | 752 GB | 4x Tesla T4 |
| DGX-1 | 3 | Xeon E5-2698 v4 (80) | 503 GB | 8x Tesla V100-SXM2 |
| DGX H100 (2024) | 2 | 224 cores total | -- | 16x H100 total (8/system, NVLink), ~1.2 TB GPU mem, 800 GB InfiniBand |

Aggregate: ~1,000 CPU cores, 10 TB RAM, 100 TB NVMe + 400 TB SSD, 100+ GPUs.

### Storage

- Two 100 TB high-speed pools: your **home folder** and the shared **`/data`**
  share (datasets/code). Mounted at the same path on every node.

### Access & software

- **Open OnDemand** web portal (Jupyter / batch) and **SSH**.
- Software via **Singularity containers** + the **NGC** catalog.
- Scheduler: **SLURM**.

### Researcher account

This user has a **researcher account**: any number of concurrent jobs, max
walltime **1 week**. Use in sbatch:

```bash
#SBATCH --account=undergrad_research
```

### GPU tiers / how to target them

Confirm exact partition + GRES names on the cluster first:

```bash
sinfo -o "%P %N %G"
```

- `teaching` / HighMem -> 4x **Tesla T4** per node (smoke + small jobs).
- DGX-1 -> 8x **Tesla V100-SXM2** per node (e.g. `--gres=gpu:v100:N`).
- DGX H100 -> 8x **H100** per node, NVLink (e.g. `--gres=gpu:h100:N`) -> best for
  the real `Qwen/Qwen3-4B-Instruct-2507` run.

### SLURM cheatsheet

```bash
sbatch scripts/train_matrix_qwen.sbatch       # submit
squeue -u $USER                               # my jobs
sinfo -o "%P %N %G"                           # partitions / nodes / GRES
scancel <jobid>                               # cancel
srun --account=undergrad_research --partition=teaching --gres=gpu:1 --pty bash  # interactive
```

### Interactive shell inside the MSOE Singularity container

This is the working format for an interactive bash terminal on Rosie. `--nv`
exposes the GPUs; `-B /data:/data` binds the shared share. Use `--gres=gpu:0`
for a CPU-only shell, or `gpu:N` to request GPUs:

```bash
srun --pty --partition=teaching --gres=gpu:0 --cpus-per-task=4 --time=1-00:00:00 \
  singularity shell --nv -B /data:/data /data/containers/msoe-tf2x.sif
```

## 9. Repo structure

```
.
├── .cursor/MATRIX_CHAT_PROJECT.md
├── .claude/MATRIX_CHAT_PROJECT.md
├── model/
│   ├── __init__.py
│   ├── matrix_qwen.py         # wrapper: agent/inactive embeddings, activity head, losses
│   ├── masks.py                # build_matrix_causal_mask (inactive_mask param)
│   ├── turn_reward.py          # ground_truth_run_lengths + turn_taking_reward
│   ├── generation.py           # activity-gated generate_matrix / generate_next_tokens_for_all_agents
│   └── lora_utils.py
├── training/
│   ├── __init__.py
│   ├── config.py
│   ├── run_ids.py
│   ├── toy_batch.py
│   └── multi_source.py         # weighted interleave, collate (incl. activity fields), per-source loss
├── data/
│   ├── schema.py                # Conversation / Turn / TimedWord intermediate
│   ├── convert.py               # turn-major + hybrid timed matrix conversion
│   ├── manifest.py
│   ├── build_dataset.py         # CLI: download | convert | peek
│   ├── inspect.py               # render processed shards (activity labels legend)
│   ├── distill.py
│   ├── adapters/{ami,molweni,meld,werewolf,conversation_chronicles,qwen_distill,spec}.py
│   ├── README.md
│   └── dataset_plan.md
├── tests/
│   ├── conftest.py            # project-root import + tiny_model / matrix_config fixtures
│   ├── test_matrix_shapes.py
│   ├── test_forward_backward.py
│   ├── test_position_ids.py
│   ├── test_masks.py
│   ├── test_causality.py
│   ├── test_wrapper_features.py
│   ├── test_generation.py
│   ├── test_lora_freezing.py
│   ├── test_config.py
│   ├── test_main_integration.py
│   ├── test_activity.py         # activity head / inactive embedding / turn-taking reward
│   ├── test_convert.py
│   ├── test_meld_adapter.py
│   ├── test_ami_adapter.py
│   ├── test_timed_convert.py
│   ├── test_werewolf_adapter.py  # private context, teammates, name randomization
│   ├── test_checkpoint_migration.py
│   ├── test_checkpoint_best.py    # best-validation checkpoint selection
│   └── test_multi_source.py     # activity_metrics diagnostics
├── scripts/
│   ├── train_matrix_qwen.sbatch   # generic SLURM entrypoint (toy or real data path)
│   ├── train_meld.sbatch          # dedicated, fully-tuned real MELD training job
│   ├── preprocess_data.sbatch     # CPU: raw -> Arrow shards + manifest
│   ├── preprocess_meld_ami.sbatch / train_meld_ami.sbatch
│   ├── preprocess_meld_ami_werewolf.sbatch / train_meld_ami_werewolf.sbatch
│   ├── download_data.sh
│   ├── distill_qwen.sbatch
│   ├── multi_agent_tokens.py      # activity-gated multi-agent generation demo
│   ├── turn_taking_probe.py       # listeners/handoff/midspeech turn-taking scenarios
│   ├── plot_metrics.py            # export a run's metrics.jsonl as PNG curves
│   └── verify_cross_agent.py
├── logs/.gitkeep
├── hyperparameters/.gitkeep
├── checkpoints/.gitkeep
├── main.py                     # training entrypoint (toy smoke path + real Stage-2 data path)
├── requirements.txt
├── README.md
├── pytest.ini
└── .gitignore
```

## 10. Key shapes reference

- `input_ids`, `input_activity_mask`, `activity_labels`: `[B, A, T]`
- flatten (column-major): `x.permute(0, 2, 1).reshape(B, T*A)` -> `[B, T*A]`
- flat index `i` -> `time = i // A`, `agent = i % A`
- logits: `[B, A, T, V]`; activity_logits: `[B, A, T]`; flat_logits: `[B, T*A, V]`
- position ids (A=3, T=4):
  - flat:   `[0,1,2,3,4,5,6,7,8,9,10,11]`
  - column: `[0,0,0,1,1,1,2,2,2,3,3,3]`
- generation: no query-column trick needed -- `activity_logits[:, :, -1]` already
  predicts the next column's activity from the existing last column; returns
  `(tokens, speak_mask)` each `[B, A]` (or `[B, A, steps]` for `generate_matrix`).
