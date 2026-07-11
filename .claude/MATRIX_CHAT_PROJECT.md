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

Each agent owns one row. Agents may attend to relevant public context. Later
versions add private context and visibility masks.

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
  - Custom cross-entropy loss over matrix-aligned labels (NOT Qwen's shifted loss).
  - `silence_token_id` (optional): cells with this id are masked out (invisible)
    via `build_matrix_causal_mask`'s `silence_mask`. Plumbing only -- the model
    is NOT yet trained to *choose* silence; a reserved unused id (e.g. 151669 for
    Qwen3-4B, vocab_size 151936 > tokenizer len 151669) avoids any embedding resize.
  - `drop_silence` (optional): when set with `silence_token_id`, silent cells are
    physically DROPPED before the base model (compaction) instead of masked, so
    per-token compute scales with active cells, not `A*T`. Survivors keep their
    ORIGINAL `position_ids` (RoPE unchanged) and `(time, agent)` coords (matrix
    rule preserved); ragged batches are front-packed and padded to the batch max
    (padding blocked by the mask via `build_matrix_mask_from_coords`); logits are
    scattered back into `[B, A, T, V]` (silent slots zero). Proven to match the
    masked path bit-for-bit at kept cells (`test_silence.py::test_drop_silence_matches_mask`).
  - dtype: agent/channel embeddings stay fp32 and are cast to the base model's
    dtype before adding, so a bf16/fp16 base model + fp32 wrapper embeddings work.
- `model/masks.py`: `build_matrix_causal_mask` -> additive `[B, 1, S, S]` mask.
  - `allow_same_column=False`: a cell sees all previous columns plus **itself**
    (diagonal); other agents in the same current column are blocked.
  - `allow_same_column=True`: the whole current column (`tk <= tq`) is visible.
  - `silence_mask` (optional `[B, S]` bool): cells holding `silence_token_id`
    are made **invisible** -- no query attends them as a key except the diagonal
    (so their own softmax row stays well-defined; output discarded). This also
    hides a silent cell from the same agent's later columns.
  - `build_matrix_mask_from_coords(time, agent, valid, ...)`: coord-based mask for
    the silence-DROP/compaction path (non-rectangular packed sequences); blocks
    padding keys, keeps the diagonal, applies the same matrix-causal rule.
- `model/generation.py`: `generate_next_tokens_for_all_agents` -> `[B, A]` in one pass.
- `model/lora_utils.py`: freeze/unfreeze/LoRA/param-count helpers.
- `training/config.py`: argparse config with validation.
- `training/run_ids.py`: run-id + hyperparameter/log/checkpoint tracking.
- `training/toy_batch.py`: tiny Qwen3 factory + toy matrix batch.
- `main.py`: smoke training loop with run-id tracking (plus `--dry_run`,
  `--print_trainable_params`, `--save_checkpoint`, JSONL metrics).
- `scripts/train_matrix_qwen.sbatch`: SLURM entrypoint tailored for Rosie.
- `scripts/multi_agent_tokens.py`: loads the real Qwen, generates one token per
  agent per matrix forward pass, prints columns (one per agent). Args: `--prompt`/
  `--prompts`, `--num_agents`, `--max_new_tokens`, `--temperature`, `--chat`,
  `--silence_token_id`/`--silent_agents`, `--drop_silence`, `--stop_on_eos`, etc.
  Per-agent EOS (`--stop_on_eos`, default on): when an agent emits an EOS id
  (Qwen3: `<|im_end|>`=151645 or `<|endoftext|>`=151643), it's marked finished
  (shown with a `⏹` prefix), its remaining row is filled with the silence token
  (so it goes invisible / droppable), and generation halts once ALL agents are
  finished, with `--max_new_tokens` as the backstop.
- `scripts/verify_cross_agent.py`: logit-level probe proving agents share PAST
  context (changing agent 1's past moves agent 0's logits) with no future / no
  same-column leakage. Verified on real Qwen3-4B: check1 max|Δ|~9.2, checks 2&3 = 0.0.
- Real Qwen3-4B weights live (gitignored) under `models/Qwen3-4B-Instruct-2507/`.
- **Stage-2 data pipeline** (offline download + tokenize/convert): tiered blend
  (structure: molweni, werewolf [no content loss]; breadth: meld,
  conversation_chronicles; retention: qwen_distill self-distill). Turn-major
  matrix conversion with rotated target-seat shifted content labels + EOS,
  silence subsampling (`silence_loss_ratio`, decision-points-only), agent-row
  permutation, `source_id` tagging. Code: `data/schema.py`, `data/adapters/*`,
  `data/convert.py`, `data/manifest.py`, `data/build_dataset.py` (CLI
  download|convert), `data/distill.py`; jobs `scripts/{download_data.sh,
  distill_qwen.sbatch,preprocess_data.sbatch}`; training hooks
  `training/multi_source.py` (weighted interleave, collate, per-source loss).
  Config: `--processed_data_dir/--dataset_weights/--silence_token_id/
  --drop_silence/--silence_loss_ratio/--silence_decision_points_only`. See
  `data/dataset_plan.md`.
- `tests/`: pytest suite (**50 tests**, 1 skipped without PEFT) across:
  - `test_matrix_shapes.py` -- shapes, variable agents, flatten/unflatten roundtrip.
  - `test_forward_backward.py` -- finite loss + agent-embedding grad norm > 0.
  - `test_position_ids.py` -- exact flat/column position-id values.
  - `test_masks.py` -- mask shape, same-column semantics, first-column self.
  - `test_causality.py` -- finite logits, no future leakage, no same-column
    cross-agent leakage.
  - `test_wrapper_features.py` -- channel embeddings, position-mode effect, loss
    ignore_index, num_agents > max_agents raises.
  - `test_generation.py` -- next-token shape `[B, A]`, greedy + sampling.
  - `test_lora_freezing.py` -- freeze/unfreeze/LoRA (skips cleanly if no PEFT).
  - `test_config.py` -- str2bool, target-module parsing, arg validation.
  - `test_main_integration.py` -- end-to-end `main.main()` smoke + dry-run.
  - `test_silence.py` -- silence mask blocks silent keys (keeps diagonal); a
    fully-silent agent is invisible (== not present, column mode); speaking-agent
    control confirms visibility; `drop_silence` compaction matches the masked
    path at kept cells on a ragged batch.
  - `test_convert.py` -- Stage-2 converter: turn-major shapes, shifted
    target-seat content labels + EOS, structure-only sources carry no content
    labels, silence subsampling honors `silence_loss_ratio`, permutation-invariant
    content labeling, source tagging.

## 3. What is intentionally NOT implemented yet

- True private/public visibility masks.
- Relation-aware attention bias.
- 2D / multi-axis RoPE (only standard 1D Qwen RoPE; `column` mode just shares
  position ids per time column).
- Werewolf/Avalon-style environments.
- Simultaneous speaking / interruption / backchannel actions.
- Real dataset pipelines (only a toy random batch). See `data/dataset_plan.md`.
- Multi-token autoregressive generation (only one next token per agent).

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

## 7. Next steps

1. Wire a real (small) dataset converter into matrix format (see dataset plan).
2. Experiment with `flat` vs `column` position modes on the real Qwen3-4B.
3. Prototype private/public visibility masks.
4. Explore 2D / multi-axis RoPE.
5. Build a minimal multi-agent environment (Werewolf/Avalon) for RL later.

### Performance / scaling roadmap (planned, not yet implemented)

- **KV cache + incremental matrix mask: DEFERRED to the online Werewolf/Avalon RL
  stage.** Rationale: training is teacher-forced (the whole `A*T` matrix is one
  forward/backward pass), so a KV cache gives ZERO training speedup -- it only
  helps autoregressive token-by-token generation, which dominates cost during RL
  rollouts. Build it then (silence/EOS-finished agents naturally drop out of the
  cache). Current generation uses `use_cache=False` and recomputes the full
  sequence each step (`O(N^2)`), which is fine for now.
- **Training-phase perf levers (the ones that matter now):** SDPA attention
  (our additive 4D mask is SDPA-compatible; faster than eager), silence
  compaction (`drop_silence`), LoRA + selective freezing, gradient checkpointing,
  and multi-GPU DDP.
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
│   ├── matrix_qwen.py
│   ├── masks.py
│   ├── generation.py
│   └── lora_utils.py
├── training/
│   ├── __init__.py
│   ├── config.py
│   ├── run_ids.py
│   └── toy_batch.py
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
│   └── test_main_integration.py
├── scripts/train_matrix_qwen.sbatch
├── logs/.gitkeep
├── hyperparameters/.gitkeep
├── checkpoints/.gitkeep
├── data/{README.md, dataset_plan.md}
├── main.py
├── requirements.txt
├── README.md
├── pytest.ini
└── .gitignore
```

## 10. Key shapes reference

- `input_ids`: `[B, A, T]`
- flatten (column-major): `x.permute(0, 2, 1).reshape(B, T*A)` -> `[B, T*A]`
- flat index `i` -> `time = i // A`, `agent = i % A`
- logits: `[B, A, T, V]`; flat_logits: `[B, T*A, V]`
- position ids (A=3, T=4):
  - flat:   `[0,1,2,3,4,5,6,7,8,9,10,11]`
  - column: `[0,0,0,1,1,1,2,2,2,3,3,3]`
- generation: append a query column -> next tokens `[B, A]`.
