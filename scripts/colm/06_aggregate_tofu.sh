#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00-00:20
#SBATCH --output=./results/tofu/logs/aggregate-%j-%N.out
#SBATCH --job-name=tofu-aggregate

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery}
RESULTS_DIR=${RESULTS_DIR:-results/tofu}
OUTPUT_CSV=${OUTPUT_CSV:-results/tofu/summary.csv}

echo "[aggregate] PROJECT_DIR=$PROJECT_DIR"
echo "[aggregate] RESULTS_DIR=$RESULTS_DIR OUTPUT_CSV=$OUTPUT_CSV"

if command -v module >/dev/null 2>&1; then
  module load gcc arrow/18.1.0 cuda
  module load python/3.10
  module load scipy-stack
fi
if [ -f "$HOME/ENV-3.10/bin/activate" ]; then
  source "$HOME/ENV-3.10/bin/activate"
fi

cd "$PROJECT_DIR"
mkdir -p results/tofu/logs
python tofu/aggregate_tofu_results.py --results_dir "$RESULTS_DIR" --output_csv "$OUTPUT_CSV"
