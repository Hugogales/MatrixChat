#!/usr/bin/env bash
# Publication-contract screening for search finalists vs paper baseline c106.
set -euo pipefail

ROOT="/home/ad.msoe.edu/garrido-lestacheh/UR/MatrixChat"
cd "$ROOT"
mkdir -p logs/slurm logs/final_runs_20260902_publication_screening

CONTRACT="logs/final_runs_20260827_recontract/publication_full.json"
BASELINE_DIR="paper_results_176106_20260819"

launch_one() {
  local name="$1"
  local ckpt="$2"
  local prefer="$3"
  local outdir="paper_results_${name}_20260902"
  local eval_job post_job
  eval_job=$(sbatch --parsable --partition=dgx --gres=gpu:v100:1 --exclude=dh-dgx1-1 \
    --job-name="pub_eval_${name}" \
    --export=ALL,CONTRACT="$CONTRACT",CHECKPOINT_DIR="$ckpt",OUTPUT_DIR="$outdir",TEMPERATURE=1.0,GENERATION_SEEDS=0,ATTN_IMPLEMENTATION=sdpa,EVAL_BATCH_SIZE=4,CHECKPOINT_PREFER="$prefer",TRUST_FROZEN_CONTRACT=1 \
    scripts/final_runs/evaluate_final_checkpoint.sbatch)
  post_job=$(sbatch --parsable --dependency=afterok:"$eval_job" \
    --job-name="pub_post_${name}" \
    --export=ALL,INPUT_DIR="$outdir",BASELINE_DIR="$BASELINE_DIR",OUTPUT_DIR="$outdir" \
    scripts/final_runs/publication_bundle_postprocess.sbatch)
  echo "$name eval=$eval_job post=$post_job -> $outdir"
}

echo "Queueing publication screening evals (exclude flaky dh-dgx1-1)..."
launch_one "h100expl_00001" "checkpoints/h100expl_cand_00001" "best_probe"
launch_one "h100expl_00038_r1s1500" "checkpoints/h100expl_cand_00038/rung1_step1500" "root"
launch_one "c106_baseline" "checkpoints/final40_h100_c106_s176106" "root"
