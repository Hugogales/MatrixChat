# MatrixChat best-model handoff

This bundle contains the best verified MatrixChat checkpoint at packaging time:
`final40_h100_c106_s176106` (usually shortened to **c106**).

## What is in the ZIP

```text
matrixchat_best_model_c106_20260911/
├── AGENT_READ_THIS.md
├── TRAINING_RUN_LOG.md
├── SHA256SUMS
├── hyperparameters/
│   └── final40_h100_c106_s176106.json
└── checkpoints/
    └── final40_h100_c106_s176106/
        ├── model_state.pt
        ├── best_checkpoint.json
        └── lora_adapters/
            ├── adapter_config.json
            ├── adapter_model.safetensors
            └── README.md
```

`model_state.pt` and `lora_adapters/` are the best validation checkpoint.
The much larger `last/training_state.pt` is intentionally omitted: it contains
optimizer, scheduler, and RNG state needed only for an exact same-run resume.
This portable bundle supports inference, evaluation, and fresh warm-start
training.

The base Qwen model is not bundled. You must separately provide
`Qwen/Qwen3-4B-Instruct-2507` at `models/Qwen3-4B-Instruct-2507`, or set
`MODEL_PATH`/`--model_path` to another local copy.

## Install the weights

From the MatrixChat repository root:

```bash
unzip /path/to/matrixchat_best_model_c106_20260911.zip
cp -a matrixchat_best_model_c106_20260911/checkpoints/. checkpoints/
cp -a matrixchat_best_model_c106_20260911/hyperparameters/. hyperparameters/
```

The expected path is then:

```text
checkpoints/final40_h100_c106_s176106/model_state.pt
```

Verify the bundle before using it:

```bash
cd matrixchat_best_model_c106_20260911
sha256sum -c SHA256SUMS
```

## Run a conversation

On Rosie, from the repository root:

```bash
CHECKPOINT_DIR=checkpoints/final40_h100_c106_s176106 \
SEED_TEXT="What should our group discuss today?" \
NUM_AGENTS=3 \
sbatch scripts/eval/run_best_model_conversations.sbatch
```

The readable transcript is written to the Slurm output and structured JSON to
`logs/best_model_conversations/`.

For an interactive allocation or another CUDA machine:

```bash
python scripts/eval/demo_freeform_conversation.py \
  --model_path models/Qwen3-4B-Instruct-2507 \
  --checkpoint_dir checkpoints/final40_h100_c106_s176106 \
  --seed_text "What should our group discuss today?" \
  --num_agents 3 \
  --max_new_tokens 120 \
  --temperature 0.8 \
  --device cuda
```

## Run a quick checkpoint probe

```bash
python scripts/eval/turn_taking_probe.py \
  --model_path models/Qwen3-4B-Instruct-2507 \
  --checkpoint_dir checkpoints/final40_h100_c106_s176106 \
  --scenario handoff \
  --num_agents 3 \
  --max_new_tokens 40 \
  --device cuda
```

## Run the held-out evaluation

The frozen contract and processed datasets are not in this model bundle.
Once they are present, submit:

```bash
CONTRACT=logs/final_runs_20260827_recontract/publication_full.json \
CHECKPOINT_DIR=checkpoints/final40_h100_c106_s176106 \
CHECKPOINT_PREFER=root \
OUTPUT_DIR=paper_results_c106 \
TRUST_FROZEN_CONTRACT=1 \
sbatch scripts/final_runs/evaluate_final_checkpoint.sbatch
```

`CHECKPOINT_PREFER=root` is important for c106 because its packaged best weights
are at the checkpoint root, not under `best_probe/`.

## Start a new training run from these weights

Use a **warm start** to initialize a new experiment from c106 while starting a
fresh optimizer and step counter:

```bash
RUN_NAME=my_c106_warm_start \
INIT_FROM=checkpoints/final40_h100_c106_s176106 \
INIT_CHECKPOINT_PREFER=root \
NUM_STEPS=1500 \
NUM_EPOCHS=20 \
PROCESSED_DIR="$HOME/matrixchat/processed_final_20260814_lookback192_allspk" \
sbatch scripts/training/train_meld_ami_werewolf.sbatch
```

To adapt another run file, add environment defaults:

```bash
INIT_FROM="${INIT_FROM:-checkpoints/final40_h100_c106_s176106}"
INIT_CHECKPOINT_PREFER="${INIT_CHECKPOINT_PREFER:-root}"
```

and pass these arguments to `main.py`:

```bash
--init_from "$INIT_FROM" \
--init_checkpoint_prefer "$INIT_CHECKPOINT_PREFER"
```

Do not combine `--init_from` with `--resume_from`. `--init_from` loads model
weights into a fresh run; `--resume_from` requires `last/training_state.pt` and
continues the prior optimizer/scheduler/global step. If performing an exact
resume from a full checkpoint, set `NUM_STEPS` and `NUM_EPOCHS` high enough that
the planned step cap is greater than the restored global step.

## Change job resources or hyperparameters

All commands must be run from the repository root. Most sbatch files expose
their settings as environment variables:

```bash
RUN_NAME=experiment_01 SEED=42 LEARNING_RATE=5e-5 \
DATASET_WEIGHTS="ami=0.5,meld=0.15,werewolf=0.35" \
sbatch scripts/training/train_meld_ami_werewolf.sbatch
```

Resource directives are changed with sbatch flags without editing the file:

```bash
sbatch --partition=dgxh100 --gres=gpu:h100:1 --mem=96G \
  scripts/training/train_meld_ami_werewolf.sbatch
```

Before launching, confirm `MODEL_PATH`, `PROCESSED_DIR`, checkpoint paths, the
partition/GRES names, and output directories. Monitor with `squeue -u "$USER"`
and inspect `logs/slurm/<job-name>_<job-id>.out` and `.err`.

## Important files

- `README.md`: project overview and common workflows.
- `scripts/README.md`: script-by-script map.
- `TRAINING_RUN_LOG.md`: historical Slurm run ledger and experimental outcomes.
- `SUCCESS_STORIES.md`: verified findings and methodological warnings.
- `hyperparameters/final40_h100_c106_s176106.json`: exact c106 training config.
- `training/checkpoint.py`: checkpoint loading, warm-start, and resume behavior.
