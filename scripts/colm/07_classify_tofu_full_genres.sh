#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00-00:10
#SBATCH --output=./results/tofu/logs/classify-%j-%N.out
#SBATCH --job-name=tofu-classify

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery}
FULL_JSON=${FULL_JSON:-TOFU/full.json}
GENRE_DIR=${GENRE_DIR:-TOFU/genres}

echo "[classify] PROJECT_DIR=$PROJECT_DIR"
echo "[classify] FULL_JSON=$FULL_JSON GENRE_DIR=$GENRE_DIR"

if command -v module >/dev/null 2>&1; then
  module load gcc arrow/18.1.0 cuda
  module load python/3.10
  module load scipy-stack
fi
if [ -f "$HOME/ENV-3.10/bin/activate" ]; then
  source "$HOME/ENV-3.10/bin/activate"
fi

cd "$PROJECT_DIR"
mkdir -p "$GENRE_DIR" results/tofu/logs
python tofu/classify_full_to_genres.py --input "$FULL_JSON" --output_dir "$GENRE_DIR" --single_label --write_reports --print_oversized
