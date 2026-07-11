# Dataset Plan (Stage 2) -- IMPLEMENTED

Stage 2 teaches the matrix interface (who-speaks-when, turn-taking, silence)
while preserving Qwen's general behavior. The pipeline turns multi-party
conversations into pre-tokenized, matrix-formatted (`[A, T]`) Arrow shards on
`/data`, so training only memory-maps ready-made integer tensors.

## Tiered dataset blend

- Structure tier (3+ agents, turn-taking, silence; NO content loss):
  - `molweni` -- Ubuntu multi-party chat (avg ~3.5 speakers, 2-9). GitHub source.
  - `werewolf` -- Werewolf-Among-Us social-deduction transcripts. Bridge to RL.
- Breadth tier (register; content-bearing):
  - `meld` -- Friends, casual multi-party. HF `neoluigi/MELD-MPCA`.
  - `conversation_chronicles` -- large casual dyadic multi-session. HF.
- Retention tier (anti-forgetting; content-bearing):
  - `qwen_distill` -- self-distilled Qwen3-4B responses (assistant = target seat).

Knowledge retention comes from (1) LoRA + freezing and (2) the clean
self-distilled replay tier -- NOT merely from adding more human chat. Noisy
multi-party sets (Molweni/Werewolf) are `content_bearing=False`: context +
turn-taking structure only, so the model never learns their typos/jargon.

## Conversion (turn-major matrix)

Each turn occupies a contiguous block of time columns owned by its speaker's row;
other rows hold the silence token (id 151669). Per example:
- Agent rows are randomly permuted (no fixed row<->role binding).
- Target seat: for content-bearing sources, one speaker (preferring an
  `assistant` role, else rotated) carries shifted next-token CONTENT labels
  (`label[a,t] = input[a,t+1]`), plus EOS at its turn ends. All other speakers and
  all structure-only sources carry NO content labels.
- Silence labels: candidate = non-speaking agents at turn boundaries (decision
  points). Amount is tunable to avoid collapse to "always silent":
  - content-bearing: keep `silence_loss_ratio x #content_labels`;
  - structure-only: keep all decision-point candidates (silence is the signal).
- Each example is tagged with an integer `source_id` (for weighted sampling +
  per-dataset loss breakdown).

Note: the matrix forward uses an UNSHIFTED custom CE, so the converter emits
already-shifted labels.

## Code layout

- `data/schema.py` -- `Conversation` / `Turn` intermediate.
- `data/adapters/{molweni,meld,werewolf,conversation_chronicles,qwen_distill}.py`
  + `data/adapters/spec.py` registry (stable `source_id`, default weights).
- `data/convert.py` -- `convert_conversation(...)` (core logic) + `ConvertConfig`.
- `data/manifest.py` -- per-source manifest (shards, counts, default weight).
- `data/build_dataset.py` -- CLI: `--mode download|convert`.
- `data/distill.py` -- self-distillation generation.
- `training/multi_source.py` -- weighted interleave sampler, batch collation,
  per-source loss breakdown (training-stage hooks).
- `training/config.py` -- `--processed_data_dir`, `--dataset_weights`,
  `--silence_token_id`, `--drop_silence`, `--silence_loss_ratio`,
  `--silence_decision_points_only`.

## Jobs (Rosie)

1. `scripts/download_data.sh` -- run on an internet node; populates
   `$RAW_DIR=/data/$USER/matrixchat/raw` (HF cache + GitHub clones).
2. `scripts/distill_qwen.sbatch` -- GPU; generates the `qwen_distill` source.
3. `scripts/preprocess_data.sbatch` -- CPU, offline (`HF_HUB_OFFLINE=1`); converts
   raw -> Arrow shards in `$PROCESSED_DIR` + `manifest.json`.

Artifacts on `/data`: `raw/` (downloads), `processed/<source>/` (Arrow shards),
`processed/manifest.json`.

## Hyperparameters (new)

- `--dataset_weights "molweni=0.25,meld=0.15,..."` -- per-source sampling
  probabilities (empty -> manifest defaults).
- `--silence_loss_ratio`, `--silence_decision_points_only` -- how much silence is
  supervised (the overfitting guard).

## IO / GPU utilization

All heavy CPU work (tokenize + matrix convert + label masking) is offline. Training
memory-maps Arrow, uses `DataLoader(num_workers>0, pin_memory, prefetch,
persistent_workers)`, a `DistributedSampler` per DDP rank, and length bucketing;
`drop_silence` shrinks the sparse turn-major matrices.

## Not yet wired (training stage)

The `main.py` training loop integration (DataLoader + per-source loss logging) and
multi-GPU (LoRA + freeze top-layers + DDP via Accelerate) come in the training
stage. KV cache stays deferred to the Werewolf/Avalon RL phase.

## Candidate datasets considered

LMSYS-Chat-1M and UltraChat were considered but deprioritized (2-party / heavy IO);
`ishiki-labs/multi-party-dialogue` is decision-point extracts (truncated context),
usable only as an auxiliary SPEAK/SILENT signal.
