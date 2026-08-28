#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00-00:30
#SBATCH --output=./results/tofu/logs/05-collect-%j-%N.out
#SBATCH --job-name=tofu-05-collect

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery}

MEMORY_ROOT=${MEMORY_ROOT:-results/tofu/key_memory}
KEY_RECOVERY_ROOT=${KEY_RECOVERY_ROOT:-results/tofu/key_recovery}
RECOVERY_ROOT=${RECOVERY_ROOT:-results/tofu/recovery}

OUTPUT_DIR=${OUTPUT_DIR:-results/tofu/05_reports_v1}
MAKE_PLOTS=${MAKE_PLOTS:-1}
SELECTED_EPOCH=${SELECTED_EPOCH:-20}
SELECTED_SPLIT=${SELECTED_SPLIT:-}
MIN_FACT_GROUP_COUNT=${MIN_FACT_GROUP_COUNT:-10}
MAIN_CONSTRAINT_SCOPE=${MAIN_CONSTRAINT_SCOPE:-full_vocab}
# same_genre_fact_group_content_vocab same_fact_group_content_vocab same_genre_content_vocab full_vocab
SEQUENCE_SPLIT=${SEQUENCE_SPLIT:-}

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
  --memory_root "$MEMORY_ROOT"
  --key_recovery_root "$KEY_RECOVERY_ROOT"
  --recovery_root "$RECOVERY_ROOT"
  --output_dir "$OUTPUT_DIR"
  --min_fact_group_count "$MIN_FACT_GROUP_COUNT"
  --main_constraint_scope "$MAIN_CONSTRAINT_SCOPE"
)

[ "$MAKE_PLOTS" = "1" ] && ARGS+=(--make_plots)
[ -n "$SEQUENCE_SPLIT" ] && ARGS+=(--sequence_split "$SEQUENCE_SPLIT")

if [ -n "$SELECTED_EPOCH" ]; then
  ARGS+=(--selected_epoch "$SELECTED_EPOCH")
fi
if [ -n "$SELECTED_SPLIT" ]; then
  ARGS+=(--selected_split "$SELECTED_SPLIT")
fi

python tofu/05_collect_v1_1.py "${ARGS[@]}"
