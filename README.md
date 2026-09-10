# MatrixChat

A PhD research prototype that represents multi-party LLM conversation as a
**matrix** (agents x time) instead of a single flat token stream, wrapped around
a Qwen3 causal language model.

## Current milestone

**Milestone 1: a testable matrix wrapper.** This is intentionally NOT the final
architecture and NOT a useful trained model. The goal is correctness and
testability:

- Add an agent dimension to the LLM input/output: `[B, A, T]` -> `[B, A, T, V]`.
- Flatten the matrix column-major into a Qwen-compatible sequence.
- Add learned agent embeddings (and optional channel embeddings).
- Support two position modes: `flat` and `column`.
- Support a matrix-causal attention mask.
- One forward pass produces logits for all agents at once.
- A generation helper predicts one next token for every agent in one pass.
- Pass forward/backward tests with a tiny randomly initialized Qwen3 (offline).

## Installation

```bash
pip install -r requirements.txt
```

`torch`, `transformers>=4.51.0` (for `Qwen3Config`/`Qwen3ForCausalLM`), `numpy`,
and `pytest` are required for tests. `peft` is only required when LoRA is enabled.

## Run tests

```bash
pytest -q
```

Tests run on CPU, offline, with a tiny random Qwen3 model (~91k params). They
never download `Qwen/Qwen3-4B-Instruct-2507`. The suite (**41 tests**) covers
shapes, forward/backward gradients, position ids, mask semantics, causality and
no-leakage, channel embeddings, loss masking, LoRA/freezing, config validation,
and an end-to-end `main` smoke run.

## Local smoke training

```bash
python main.py --use_tiny_model true --num_steps 3
```

Prints the run id, hyperparameter JSON path, log directory, and checkpoint
directory, then runs a few forward/backward/optimizer steps on a toy batch.

## Submit on Rosie (SLURM)

```bash
sbatch scripts/training/train_meld_ami_werewolf.sbatch
```

All hyperparameters are set as bash variables (env-overridable) in the sbatch
file and passed to `main.py` as CLI args. See the Rosie section below.

## Repo structure

```
model/          matrix wrapper, masks, activity/turn-taking reward, generation, LoRA utils
training/       argparse config, run-id tracking, tiny-model + toy batch, multi-source loader
tests/          pytest suite (CPU, offline)
scripts/        SLURM entrypoints + tooling, grouped by purpose -- see scripts/README.md
logs/           per-run logs  (logs/{run_id}/, incl. metrics.jsonl, samples.jsonl)
hyperparameters/ saved run_NNNNNN.json hyperparameter files
checkpoints/    per-run checkpoints (checkpoints/{run_id}/)
data/           dataset pipeline: schema, adapters, converter, manifest, dataset_plan.md
main.py         training entrypoint (toy smoke path + real Stage-2 data path)
```

## Checkpoints

When `--save_checkpoint true` and a validation split is used (`--val_fraction > 0`):

| Path | Meaning |
| --- | --- |
| `checkpoints/{run_id}/model_state.pt` | **Best** validation-loss checkpoint (load this for inference) |
| `checkpoints/{run_id}/lora_adapters/` | LoRA weights for the best checkpoint |
| `checkpoints/{run_id}/best_checkpoint.json` | Summary (best val loss, step, components) |
| `checkpoints/{run_id}/last/model_state.pt` | Terminal training state (for comparison/debugging) |

Without a validation split (toy smoke path), the terminal model is saved at the
root (`checkpoints/{run_id}/model_state.pt`).

Load the best checkpoint for multi-agent generation:

```bash
python scripts/eval/multi_agent_tokens.py \
  --checkpoint_dir checkpoints/run_000010 \
  --prompts "Hi!" "How are you?" --max_new_tokens 40
```

**Note:** A training job already running before this feature was added must be
restarted to use best-checkpoint selection.

## Continuous conversation-quality signal (`--probe_every`)

Loss can look great while every generated "conversation" is silent or
degenerate -- confirmed directly: the best-loss checkpoint of one race had a
*worse* clean-handoff rate than an earlier, worse-loss checkpoint (see
`SUCCESS_STORIES.md`'s 07-27 correction). `--probe_every N` runs the fixed
handoff/listener probe suite (`scripts/eval/evaluate_checkpoint_suite.py`)
against the LIVE training model every `N` optimizer steps -- no checkpoint
reload, generation-based, in-process -- and logs the headline rates
(`clean_handoff_rate`, `no_listener_response_rate`, `overlap_rate`,
`repeated_4gram_fraction`, ...) into `metrics.jsonl` under a `"probe"` key,
right alongside the loss curve. `python scripts/analysis/plot_metrics.py`
renders it as `probe.png` automatically whenever a run used it.

Disabled by default (`--probe_every 0`) since it's a real generation-based
check that measurably slows down the steps it fires on; a reasonable
default when enabled is to match `--checkpoint_every`. See
`--probe_num_agents`/`--probe_scaling_agent_counts`/`--probe_max_new_tokens`/
`--probe_include_secret_scenarios` for scope, and the corresponding
`PROBE_EVERY`/`PROBE_NUM_AGENTS`/`PROBE_MAX_NEW_TOKENS` env vars in
`scripts/training/train_meld_ami_werewolf.sbatch`.

## Shapes

| Tensor | Shape |
| --- | --- |
| `input_ids`, `labels`, `agent_ids` | `[B, A, T]` |
| `input_activity_mask`, `activity_labels` | `[B, A, T]` |
| flattened (column-major) | `[B, T*A]` |
| `logits` | `[B, A, T, V]` |
| `activity_logits` | `[B, A, T]` |
| `flat_logits` | `[B, T*A, V]` |
| attention mask | `[B, 1, S, S]`, `S = A*T` |
| generation output | `(tokens, speak_mask)` each `[B, A]` (or `[B, A, steps]`) |

## Turn-taking: activity head (no vocabulary "silence" token)

Every cell's final hidden state feeds TWO heads: the normal vocabulary head
(content) and a binary `activity_head` predicting `P(this agent speaks at the
NEXT column)`. There is no reserved vocabulary token for silence. Cells that
are not currently active (`input_activity_mask == False`) use a trainable
`inactive_embedding` instead of looking up a (frozen, meaningless) placeholder
token id, so "being silent" has a real, learned input representation.

Training combines three terms (see `model/matrix_qwen.py`, `model/turn_reward.py`):

```
loss = content_CE (gated: only cells with a content label)
     + lambda_activity * activity_BCE (balanced across speak/yield classes)
     - lambda_reward   * mean(turn_taking_reward)
```

`turn_taking_reward` rewards exactly-one-speaker columns (a direct handoff to a
different agent is fully rewarded), gently decays the reward for one agent
monopolizing the floor, and increasingly penalizes prolonged all-silence;
overlap gets neither term. It uses GROUND-TRUTH activity to compute run
lengths (teacher forcing) -- this is a Stage-A-only design choice that must
change before online RL/self-play (see `data/dataset_plan.md` and the RL
Turn-Taking Design plan for the mandatory migration).

At generation time (`model/generation.py`), each agent's `P(speak)` is read
from the activity head; above `--activity_threshold` it samples/argmaxes a
real content token, otherwise it emits the placeholder and the new column is
marked inactive.

### Class imbalance: the activity BCE is already balanced

The "yield" (silent) class is far more common in real conversation than the
"speak" class. The activity BCE handles this automatically: it is computed as
the mean loss over speak-labeled cells and the mean loss over yield-labeled
cells **separately, then averaged 50/50** (`model/matrix_qwen.py`). This is
equivalent to full inverse-frequency class weighting recomputed every batch --
strictly stronger than any fixed global weight, and it self-adjusts if the
imbalance ratio drifts across sources/batches, so there's no separate
"silence weight" hyperparameter to tune.

To see the actual numbers (how imbalanced the data really is, and whether the
model is tracking it or collapsing to always-yield/always-speak),
`training/multi_source.py::activity_metrics` reports, and `main.py` logs every
step/eval:

- `train_activity_accuracy` / `val_components.activity_accuracy` -- thresholded
  (`P(speak) > --activity_threshold`) accuracy against the ground-truth label.
- `train_speak_rate` / `val_components.speak_rate` -- the ground-truth fraction
  of cells that are actually "speak" (the empirical imbalance).
- `train_predicted_speak_rate` / `val_components.predicted_speak_rate` -- the
  fraction the MODEL predicts as "speak"; compare against `speak_rate` to spot
  a collapse to a trivial always-yield (or always-speak) policy.

Export a run's curves (content loss, turn reward, activity loss, total loss,
and the accuracy/speak-rate panel above) as PNGs:

```bash
python scripts/analysis/plot_metrics.py --run_dir logs/run_000018
```

Column-major flatten: `flat = input_ids.permute(0, 2, 1).reshape(B, T*A)`.
A flat index `i` decodes as `time = i // A`, `agent = i % A`.

## Position modes (`flat` vs `column`)

For `A=3, T=4`:

- **flat**: `[0,1,2,3,4,5,6,7,8,9,10,11]` -- ordinary Qwen positions after
  flattening. Safest; the model sees a normal sequence.
- **column**: `[0,0,0,1,1,1,2,2,2,3,3,3]` -- all agents in the same time column
  share a position id. A first attempt to stop Qwen treating same-time agent
  slots as ordinary later tokens. This is a distribution shift, experimental, not
  final.

## Attention mask

`build_matrix_causal_mask` returns an additive `[B, 1, S, S]` mask:

- `allow_same_column=False` (default): a cell attends to all **previous** time
  columns plus **itself**; other agents in the *same* current column are blocked
  (avoids same-step teacher-forcing leakage across agents).
- `allow_same_column=True`: the whole current column is visible.

Note: the self-diagonal is always allowed. A literal "block the entire current
column" rule would leave the first column (`t=0`) with no valid keys, and an
all-masked softmax row degenerates into uniform attention over the whole
sequence -- silently leaking the future. Allowing the diagonal keeps the wrapper
fully causal (verified in `tests/test_causality.py`).

### Private-cell visibility (per-agent ACL)

`build_matrix_causal_mask` also accepts optional `private_mask [B,S]` and
`agent_visibility [B,A,A]`: a private cell (see `data.schema.Turn.visible_to`,
used by the Werewolf source) is blocked as a *key* for any query agent not
listed in `agent_visibility` (self is always permitted). This blocks DIRECT
attention to another agent's private cells; it does not (and structurally
cannot) prevent an agent from later observing the owner's own PUBLIC speech,
even though that speech's hidden states were computed from context including
the owner's private cells -- an owner always attends to its own past. That
asymmetry is intentional: your private context is unreadable to others, but
your subsequent public behavior (which it legitimately informs) is fair game
to observe, exactly like real social-deduction play. Verified end-to-end in
`tests/test_masks.py` (an isolated case with no legitimate indirect channel
gives an exact, bit-for-bit zero logit difference for the unauthorized agent).

## Loss

Content loss is a **custom cross-entropy** over matrix-aligned labels with
`ignore_index=-100`, NOT the base model's internal shifted causal-LM loss. This
lets query slots act as prediction slots without the flat next-token shift
secretly defining the objective. It is combined with the activity BCE and
turn-taking reward described above into the total loss.

## MELD + AMI + Werewolf training (private per-agent context)

The AMI adapter reads the official manual NXT annotations (CC BY 4.0),
including forced-aligned word start/end times. Its hybrid converter keeps solo
speech dense, aligns genuine interruptions in parallel agent rows, and retains
meaningful all-speaker pauses without introducing a vocabulary silence token.

The Werewolf adapter (191 usable One Night Ultimate Werewolf games: 151
YouTube + 40 Ego4D, real per-utterance timestamps) synthesizes, for each
player and in a fixed roster order: a PUBLIC self-introduction turn (own
name) followed by a PRIVATE role-reveal turn (own role, plus teammates by
name+role for Werewolf/Mason, or one-directional Werewolf knowledge for the
Minion). Names are freshly randomized per game from a 400+ pool so no small
recurring name set becomes a learnable proxy (same pool/mechanism used by the
MELD adapter's per-scene character-name substitution -- see
`data/adapters/_names.py`). Private turns are enforced by a genuine
attention-visibility mechanism (see "Private-cell visibility" above), not
merely a training-signal hint.

```bash
# Internet/login node
SOURCES="meld ami werewolf" bash scripts/data_prep/download_data.sh

# CPU preprocessing, then V100 training
sbatch scripts/data_prep/preprocess_meld_ami_werewolf.sbatch
sbatch scripts/training/train_meld_ami_werewolf.sbatch
```

This job uses `--dataset_weights "meld=0.3334,ami=0.3333,werewolf=0.3333"` so
all three sources are sampled equally (~1/3 each) regardless of their differing
manifest sizes. Known limitation: the raw transcripts only cover *public*
day-phase dialogue, so no night-phase action results (Seer inspections,
Robber/Troublemaker swaps) are fabricated or supervised.

## Rosie supercomputer (MSOE)

This repo is designed to run on **Rosie**, MSOE's academic HPC cluster
(https://msoe.dev/#/about). Only the tests are designed to run locally/offline.

### Hardware (27 compute nodes)

| Type | Count | RAM | GPU |
| --- | --- | --- | --- |
| Management / login | 4 | 187 GB | none |
| Compute (`teaching`) | 18 | 376 GB | 4x Tesla T4 |
| HighMem Compute | 2 | 752 GB | 4x Tesla T4 |
| DGX-1 | 3 | 503 GB | 8x Tesla V100-SXM2 |
| DGX H100 (2024) | 2 | -- | 16x H100 total (8/system, NVLink), ~1.2 TB GPU mem |

Aggregate: ~1,000 CPU cores, 10 TB RAM, 100 TB NVMe + 400 TB SSD, 100+ GPUs.
Shared storage: your home folder and the `/data` share, mounted identically on
every node. Access via Open OnDemand (web) or SSH; software via Singularity +
NGC containers.

### Researcher account

This project uses a **researcher account** (`#SBATCH --account=undergrad_research`):
unlimited concurrent jobs, max walltime **1 week**.

### Targeting GPU tiers

Confirm exact partition / GRES names first:

```bash
sinfo -o "%P %N %G"
```

- `teaching` / HighMem -> 4x Tesla T4 (smoke + small jobs).
- DGX-1 -> 8x Tesla V100-SXM2 (`--gres=gpu:v100:N`).
- DGX H100 -> 8x H100, NVLink (`--gres=gpu:h100:N`) -> best for the real
  `Qwen/Qwen3-4B-Instruct-2507` run (set `USE_TINY_MODEL=false`).

### Interactive shell on Rosie (Singularity container)

Use this to get an interactive bash terminal inside the MSOE TensorFlow/PyTorch
container (the `--nv` flag exposes the GPUs, `-B /data:/data` binds the shared
share). Adjust `--gres=gpu:N` to request GPUs (use `gpu:0` for a CPU-only shell):

```bash
srun --pty --partition=teaching --gres=gpu:0 --cpus-per-task=4 --time=1-00:00:00 \
  singularity shell --nv -B /data:/data /data/containers/msoe-tf2x.sif
```

### SLURM cheatsheet

```bash
sbatch scripts/training/train_meld_ami_werewolf.sbatch
squeue -u $USER
sinfo -o "%P %N %G"
scancel <jobid>
srun --account=undergrad_research --partition=teaching --gres=gpu:1 --pty bash
# Interactive shell inside the MSOE container (CPU-only example: gpu:0):
srun --pty --partition=teaching --gres=gpu:0 --cpus-per-task=4 --time=1-00:00:00 \
  singularity shell --nv -B /data:/data /data/containers/msoe-tf2x.sif
```

## Current limitations

- `column` mode is an experimental distribution shift for a pretrained model.
- Custom 4D masks require eager attention (tiny model uses `attn_implementation="eager"`).
- The turn-taking reward uses GROUND-TRUTH activity for run lengths (Stage A,
  teacher forcing) -- online RL/self-play (Stage B) must recompute run lengths
  from the model's own sampled actions instead (see `data/dataset_plan.md`).
- No 2D RoPE, no online multi-agent RL environment yet (private/public visibility
  masks ARE implemented -- see "Private-cell visibility" above).
- KV cache is deferred to the RL phase; generation recomputes the full sequence
  each step.

## Next steps

1. Compare `flat` vs `column` position modes on the real Qwen3-4B.
2. Evaluate turn-taking quality (speak/yield precision-recall, exactly-one-speaker
   rate, floor-taking/stopping recall) on a trained checkpoint.
3. Stage B: online rollout self-play with sampled (not ground-truth) run lengths,
   PPO, and a KL-to-reference anchor (see the RL Turn-Taking Design plan).
4. Prototype private/public visibility masks.
5. Explore 2D / multi-axis RoPE.
6. Build a minimal multi-agent environment (Werewolf/Avalon) for RL.
