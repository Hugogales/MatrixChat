#!/bin/bash
# Always-on Werewolf RL: one EXTRA (borrowed) GPU. Submitted by the leader only
# when the partition has idle GPUs and nobody is waiting there. On start it
# picks its role: a GPU borrowed before does the opposite of last time (train
# <-> play); a new one takes whichever role has fewer live extras. It holds
# the GPU until its time limit like the main jobs, but exits (abandoning its
# current game / training share, which is reassigned) as soon as another
# user's job -- or one of our main-chain jobs -- is waiting on this partition.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -uo pipefail
cd "$REPO" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f "$RUN_DIR/STOP" ] && { echo "[extra] STOP present; not starting"; exit 0; }
PART="${EXTRA_PARTITION:-$SLURM_JOB_PARTITION}"
ROLE="$(python -m training.werewolf_rl.continuous extra-role --run_dir "$RUN_DIR")"
echo "[extra] $(date '+%F %T') job $SLURM_JOB_ID on $(hostname), partition $PART: $ROLE"
LOG="$RUN_DIR/logs/extra_${SLURM_JOB_ID}_${ROLE}.log"

if [ "$ROLE" = train ]; then
  # shellcheck disable=SC2086
  python scripts/training/werewolf_continuous/trainer.py \
    --run_dir "$RUN_DIR" --role extra --partition "$PART" --chain_names "$CHAIN_NAMES" \
    --checkpoint_dir "$CHECKPOINT_DIR" --hyperparameters "$HYPERPARAMETERS" --model_path "$MODEL_PATH" \
    --extra_partitions "$EXTRA_PARTITIONS" --max_extra "$MAX_EXTRA" --extra_name "$EXTRA_NAME" \
    --extra_sbatch "$REPO/scripts/training/werewolf_continuous/extra.sbatch" \
    $TRAINER_ARGS >> "$LOG" 2>&1
else
  # shellcheck disable=SC2086
  python scripts/training/werewolf_continuous/player.py \
    --run_dir "$RUN_DIR" --role extra --partition "$PART" --chain_names "$CHAIN_NAMES" \
    --checkpoint_dir "$CHECKPOINT_DIR" --hyperparameters "$HYPERPARAMETERS" --model_path "$MODEL_PATH" \
    $PLAYER_ARGS >> "$LOG" 2>&1
fi
