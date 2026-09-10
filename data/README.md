# data/

Dataset conversion utilities for the Stage-2 pipeline: multi-party
conversations -> pre-tokenized, matrix-formatted (`[A, T]`) Arrow shards with
an explicit `input_activity_mask` / `activity_labels` turn-taking schema (see
[`dataset_plan.md`](dataset_plan.md) for the full design).

- `schema.py` -- `Conversation` / `Turn` (incl. `visible_to` for private
  context) plus optional `TimedWord` metadata.
- `adapters/ami.py` -- official AMI NXT dialogue acts resolved to timed words
  (CC BY 4.0). `adapters/werewolf.py` -- Werewolf-Among-Us with synthesized
  per-agent private role context (see below). Other adapters cover molweni,
  meld, conversation_chronicles, and qwen_distill.
- `convert.py` -- ordinary turn-major conversion plus the hybrid timed
  timeline (AMI, Werewolf): dense solo speech, parallel overlap blocks,
  meaningful all-speaker pauses, and per-agent `is_private_mask`/
  `agent_visibility` when a source uses `Turn.visible_to`.
- `manifest.py` -- per-source manifest (shard paths, counts, default weights).
- `build_dataset.py` -- CLI: `--mode download|convert|peek`.
- `inspect.py` -- render processed shards as a readable matrix for sanity-checking.
- `distill.py` -- self-distillation generation (the `qwen_distill` source).

The toy random matrix batch (`training/toy_batch.py`, no real data) still
exists for fast offline smoke tests independent of this pipeline.

MELD + AMI + Werewolf workflow (equally weighted, ~1/3 each -- current pipeline):

```bash
SOURCES="meld ami werewolf" bash scripts/data_prep/download_data.sh
sbatch scripts/data_prep/preprocess_meld_ami_werewolf.sbatch
sbatch scripts/training/train_meld_ami_werewolf.sbatch
```

AMI pauses up to one second are compressed; each additional second becomes
one inactive column (capped at eight). No vocabulary silence token is used.
MELD dialogue also gets per-scene random name substitution (see
`adapters/meld.py`) so the model doesn't overfit to the show's recurring
character names.

AMI/Werewolf conversations are long enough that they get split into multiple
chunks (`convert_timed_conversation`'s `max_flat_len` budget); every chunk
after the first otherwise starts structurally "cold" with zero preceding
tokens even though it's really mid-conversation. `--context_lookback_columns`
(default 0, opt-in) prefixes each non-first chunk with up to N REAL columns
copied from immediately before it -- genuine prior dialogue, not synthetic
filler -- as unsupervised context (their `labels`/`activity_labels` stay
`ignore_index`, since they were already the label-bearing portion of the
previous chunk). MELD is untouched by this flag since its turn-major
conversion doesn't chunk at all. Enable it via:

```bash
CONTEXT_LOOKBACK_COLUMNS=64 PROCESSED_DIR=$HOME/matrixchat/processed_meld_ami_lookback \
  sbatch scripts/data_prep/preprocess_meld_ami_werewolf.sbatch
```

(Earlier single- and two-source workflows have been superseded and removed;
see `TRAINING_RUN_LOG.md` for their history.)

Werewolf gives each agent a private, per-game-randomized-name role reveal
(own role, plus teammates for Werewolf/Mason, or one-directional Werewolf
knowledge for the Minion) that other agents cannot directly attend to -- see
`dataset_plan.md`'s "Werewolf private context" section for the full design
and its documented data limitation (no recorded night-phase actions).

On Rosie, the shared `/data` share is also available and mounted at the same path
on every compute node; large datasets should live there rather than in this repo.
