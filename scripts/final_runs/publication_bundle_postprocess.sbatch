#!/bin/bash
# Post-eval publication bundle: judge, AMI+Werewolf analysis, comparison tables,
# violin figures, best conversations, and RESULTS.md.
#SBATCH --account=undergrad_research
#SBATCH --job-name=pub_bundle_post
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/ad.msoe.edu/garrido-lestacheh/UR/MatrixChat
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

CONTAINER="${CONTAINER:-/data/containers/msoe-tf2x.sif}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the publication bundle directory}"
INPUT="${INPUT:-$OUTPUT_DIR/paired_continuations.jsonl}"
JUDGED="$OUTPUT_DIR/paired_continuations_judged.jsonl"
ANALYSIS_AMI_WW="$OUTPUT_DIR/analysis_ami_werewolf"
FIGURES="$OUTPUT_DIR/figures_violins"
BEST="$OUTPUT_DIR/best_conversations"

ROUND_ROBIN_JUDGED="${ROUND_ROBIN_JUDGED:-logs/final_runs_20260820_baselines/qwen_xml_round_robin_t1_judged/paired_continuations_judged.jsonl}"
PASS_BATON_JUDGED="${PASS_BATON_JUDGED:-logs/final_runs_20260820_baselines/qwen_pass_the_baton_t1_judged/paired_continuations_judged.jsonl}"
ROUND_ROBIN_SUMMARY="${ROUND_ROBIN_SUMMARY:-logs/final_runs_20260820_baselines/qwen_xml_round_robin_t1_judged/analysis_judged/summary.json}"
PASS_BATON_SUMMARY="${PASS_BATON_SUMMARY:-logs/final_runs_20260820_baselines/qwen_pass_the_baton_t1_judged/analysis_judged/summary.json}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-paper_results_176106_20260819/analysis_ami_werewolf/summary.json}"
BASELINE_STATS="${BASELINE_STATS:-paper_results_176106_20260819/analysis_ami_werewolf/stats.json}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-adaptive_hfreq_a075_s176106 (best_probe, 2000 steps)}"

if [ ! -f "$INPUT" ]; then
  echo "[error] missing eval output: $INPUT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$ANALYSIS_AMI_WW" "$FIGURES" "$BEST"

singularity exec -B /data:/data "$CONTAINER" \
  python scripts/eval/judge_held_out_jsonl.py \
    --input "$INPUT" \
    --output "$JUDGED"

singularity exec -B /data:/data "$CONTAINER" \
  python scripts/analysis/analyze_held_out_continuations.py "$JUDGED" \
    --output-dir "$ANALYSIS_AMI_WW" \
    --exclude-sources meld \
    --bootstrap-resamples 10000 \
    --seed 0

singularity exec -B /data:/data "$CONTAINER" \
  python scripts/analysis/compare_human_matrixchat_roundrobin.py \
    --matrixchat-summary "$ANALYSIS_AMI_WW/summary.json" \
    --matrixchat-stats "$ANALYSIS_AMI_WW/stats.json" \
    --round-robin-summary "$ROUND_ROBIN_SUMMARY" \
    --round-robin-stats "${ROUND_ROBIN_SUMMARY%/summary.json}/stats.json" \
    --pass-the-baton-summary "$PASS_BATON_SUMMARY" \
    --pass-the-baton-stats "${PASS_BATON_SUMMARY%/summary.json}/stats.json" \
    --output-dir "$OUTPUT_DIR"

singularity exec -B /data:/data "$CONTAINER" \
  python scripts/analysis/plot_paper_violins.py \
    --input "$JUDGED" \
    --round-robin-input "$ROUND_ROBIN_JUDGED" \
    --pass-the-baton-input "$PASS_BATON_JUDGED" \
    --output-dir "$FIGURES"

singularity exec -B /data:/data "$CONTAINER" \
  python scripts/analysis/export_best_conversations.py \
    --input "$JUDGED" \
    --output-dir "$BEST"

singularity exec -B /data:/data "$CONTAINER" \
  python scripts/analysis/render_publication_results_md.py \
    --summary "$ANALYSIS_AMI_WW/summary.json" \
    --stats "$ANALYSIS_AMI_WW/stats.json" \
    --output "$OUTPUT_DIR/RESULTS.md" \
    --figures-dir figures_violins \
    --checkpoint-label "$CHECKPOINT_LABEL" \
    --baseline-summary "$BASELINE_SUMMARY" \
    --baseline-stats "$BASELINE_STATS" \
    --baseline-label "final40_h100_c106_s176106 (paper baseline, 1500 steps)"

echo "[info] publication bundle complete under $OUTPUT_DIR"
