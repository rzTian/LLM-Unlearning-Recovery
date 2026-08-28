#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00-00:30
#SBATCH --output=./results/tofu/logs/05-plots-%j-%N.out
#SBATCH --job-name=tofu-05-plots

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery}

KEY_RECOVERY_ROOT=${KEY_RECOVERY_ROOT:-results/tofu/key_recovery}
RECOVERY_ROOT=${RECOVERY_ROOT:-results/tofu/recovery}
OUTPUT_DIR=${OUTPUT_DIR:-results/tofu/05_plots}

# Reviewer-response plots for Q1 generalization evidence.
# This script intentionally calls only tofu/05_reviewer_plots.py; it does not run 05_collect_v1_1.py.
REVIEWER_OUTPUT_DIR=${REVIEWER_OUTPUT_DIR:-${OUTPUT_DIR}/reviewer_q1}
REVIEWER_SPLIT=${REVIEWER_SPLIT:-forget10}
REVIEWER_CONSTRAINT_SCOPE=${REVIEWER_CONSTRAINT_SCOPE:-same_fact_group_content_vocab}
REVIEWER_METHODS=${REVIEWER_METHODS:-grad_ascent,KL,grad_diff,npo}

# Select which checkpoint to read for each method. These are selection-only knobs;
# epoch/lr are not shown in the generated figures.
# Example:
#   REVIEWER_METHOD_EPOCHS="grad_ascent:20,KL:20,grad_diff:10,npo:20"
#   REVIEWER_METHOD_LRS="grad_ascent:1e-05,KL:1e-05,grad_diff:1e-05,npo:0.0005"
REVIEWER_METHOD_EPOCHS=${REVIEWER_METHOD_EPOCHS:-grad_ascent:20,KL:20,grad_diff:10,npo:10}
REVIEWER_METHOD_LRS=${REVIEWER_METHOD_LRS:-grad_ascent:1e-05,KL:1e-05,grad_diff:5e-05,npo:0.0005}
REVIEWER_SELECTED_EPOCH=${REVIEWER_SELECTED_EPOCH:-}
REVIEWER_FIG_FORMAT=${REVIEWER_FIG_FORMAT:-png,pdf}
REVIEWER_BROKEN_AXIS=${REVIEWER_BROKEN_AXIS:-1}

if command -v module >/dev/null 2>&1; then
  module load gcc arrow/18.1.0 cuda
  module load python/3.10
  module load scipy-stack
fi
if [ -f "$HOME/ENV-3.10/bin/activate" ]; then
  source "$HOME/ENV-3.10/bin/activate"
fi

cd "$PROJECT_DIR"
mkdir -p results/tofu/logs "$OUTPUT_DIR"

ARGS=(
  --key_recovery_root "$KEY_RECOVERY_ROOT"
  --recovery_root "$RECOVERY_ROOT"
  --output_dir "$REVIEWER_OUTPUT_DIR"
  --split "$REVIEWER_SPLIT"
  --constraint_scope "$REVIEWER_CONSTRAINT_SCOPE"
  --methods "$REVIEWER_METHODS"
  --method_epochs "$REVIEWER_METHOD_EPOCHS"
  --fig_format "$REVIEWER_FIG_FORMAT"
  --make_plots
)

if [ -n "$REVIEWER_METHOD_LRS" ]; then
  ARGS+=(--method_lrs "$REVIEWER_METHOD_LRS")
fi
if [ -n "$REVIEWER_SELECTED_EPOCH" ]; then
  ARGS+=(--selected_epoch "$REVIEWER_SELECTED_EPOCH")
fi
if [ "$REVIEWER_BROKEN_AXIS" = "0" ]; then
  ARGS+=(--no-broken_axis)
fi

python tofu/05_reviewer_plots_v2.py "${ARGS[@]}"
