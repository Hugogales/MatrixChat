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
sbatch scripts/train_matrix_qwen.sbatch
```

All hyperparameters are set as bash variables in the sbatch file and passed to
`main.py` as CLI args. See the Rosie section below.

## Repo structure

```
model/         matrix wrapper, masks, generation, LoRA utils
training/       argparse config, run-id tracking, tiny-model + toy batch
tests/          pytest suite (CPU, offline)
scripts/        train_matrix_qwen.sbatch  (Rosie SLURM entrypoint)
logs/           per-run logs  (logs/{run_id}/)
hyperparameters/ saved run_NNNNNN.json hyperparameter files
checkpoints/    per-run checkpoints (checkpoints/{run_id}/)
data/           dataset plan (no data pipelines yet)
main.py         smoke training entrypoint
```

## Shapes

| Tensor | Shape |
| --- | --- |
| `input_ids`, `labels`, `agent_ids` | `[B, A, T]` |
| flattened (column-major) | `[B, T*A]` |
| `logits` | `[B, A, T, V]` |
| `flat_logits` | `[B, T*A, V]` |
| attention mask | `[B, 1, S, S]`, `S = A*T` |
| generation output | `[B, A]` |

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

## Loss

Loss is a **custom cross-entropy** over matrix-aligned labels with
`ignore_index=-100`, NOT the base model's internal shifted causal-LM loss. This
lets query slots act as prediction slots without the flat next-token shift
secretly defining the objective.

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
sbatch scripts/train_matrix_qwen.sbatch
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
- Only one next token per agent in generation (no autoregressive loop yet).
- No real datasets, no private/public visibility, no 2D RoPE, no environments.

## Next steps

1. Wire a small dataset converter to matrix format (see `data/dataset_plan.md`).
2. Compare `flat` vs `column` position modes on the real Qwen3-4B.
3. Prototype private/public visibility masks.
4. Explore 2D / multi-axis RoPE.
5. Build a minimal multi-agent environment for later RL.
