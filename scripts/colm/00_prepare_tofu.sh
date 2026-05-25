#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00-00:20
#SBATCH --output=./results/tofu/logs/prepare-%j-%N.out
#SBATCH --job-name=tofu-prepare

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery}
RAW_DIR=${RAW_DIR:-TOFU}
PROCESSED_DIR=${PROCESSED_DIR:-tofu/processed}
GENRE_DIR=${GENRE_DIR:-TOFU/genres}

echo "[prepare] PROJECT_DIR=$PROJECT_DIR"
echo "[prepare] RAW_DIR=$RAW_DIR"
echo "[prepare] PROCESSED_DIR=$PROCESSED_DIR"
echo "[prepare] GENRE_DIR=$GENRE_DIR"

if command -v module >/dev/null 2>&1; then
  module load gcc arrow/18.1.0 cuda
  module load python/3.10
  module load scipy-stack
fi
if [ -f "$HOME/ENV-3.10/bin/activate" ]; then
  source "$HOME/ENV-3.10/bin/activate"
fi

cd "$PROJECT_DIR"
mkdir -p "$PROCESSED_DIR" "$GENRE_DIR" results/tofu/logs

python tofu/prepare_tofu_data.py --raw_dir "$RAW_DIR" --output_dir "$PROCESSED_DIR"
python tofu/classify_full_to_genres.py --full_json "$RAW_DIR/full.json" --output_dir "$GENRE_DIR"

echo "[prepare] Wrote processed files under $PROCESSED_DIR"
echo "[prepare] Wrote genre files under $GENRE_DIR"
