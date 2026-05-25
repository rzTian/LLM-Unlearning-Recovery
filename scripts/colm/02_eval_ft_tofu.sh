#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=00-01:00
#SBATCH --output=./results/tofu/logs/eval-ft-%j-%a-%N.out
#SBATCH --job-name=tofu-eval-ft
#SBATCH --array=0-13

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery}
MODEL_NAME=${MODEL_NAME:-deepseek-ai/deepseek-llm-7b-chat}
FT_ROOT=${FT_ROOT:-results/tofu/adapters/ft}
RESULTS_ROOT=${RESULTS_ROOT:-results/tofu/eval}
LR=${LR:-0.0002}
EPOCHS=${EPOCHS:-10}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
LORA_RANK=${LORA_RANK:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-40}
LIMIT_ARGS=()
if [ -n "${LIMIT:-}" ]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

adapter_suffix="lr${LR}_WD${WEIGHT_DECAY}_loraRank${LORA_RANK}_loraDrop${LORA_DROPOUT}_GradStsp${GRAD_ACC_STEPS}/epoch-${EPOCHS}"
target_adapter="$FT_ROOT/target_full/$adapter_suffix"
retrain90_adapter="$FT_ROOT/oracle_retrain90/$adapter_suffix"
retrain95_adapter="$FT_ROOT/oracle_retrain95/$adapter_suffix"
retrain99_adapter="$FT_ROOT/oracle_retrain99/$adapter_suffix"

jobs=(
  "target_full|$target_adapter|forget10"
  "target_full|$target_adapter|forget05"
  "target_full|$target_adapter|forget01"
  "target_full|$target_adapter|retain90"
  "target_full|$target_adapter|retain95"
  "target_full|$target_adapter|retain99"
  "target_full|$target_adapter|real_authors"
  "target_full|$target_adapter|world_facts"
  "oracle_retrain90|$retrain90_adapter|forget10"
  "oracle_retrain90|$retrain90_adapter|retain90"
  "oracle_retrain95|$retrain95_adapter|forget05"
  "oracle_retrain95|$retrain95_adapter|retain95"
  "oracle_retrain99|$retrain99_adapter|forget01"
  "oracle_retrain99|$retrain99_adapter|retain99"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
IFS='|' read -r MODEL_TAG ADAPTER_DIR SPLIT <<< "${jobs[$IDX]}"
EVAL_DATA="tofu/processed/${SPLIT}.json"
OUT_DIR="$RESULTS_ROOT/$SPLIT/$MODEL_TAG"

echo "[eval-ft] IDX=$IDX MODEL_TAG=$MODEL_TAG SPLIT=$SPLIT"
echo "[eval-ft] ADAPTER_DIR=$ADAPTER_DIR"
echo "[eval-ft] EVAL_DATA=$EVAL_DATA OUT_DIR=$OUT_DIR LIMIT=${LIMIT:-none}"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source "$HOME/ENV-3.10/bin/activate"
cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR" results/tofu/logs

python tofu/evaluate_tofu.py \
  --base_model_name "$MODEL_NAME" \
  --adapter_dir "$ADAPTER_DIR" \
  --eval_data "$EVAL_DATA" \
  --split "$SPLIT" \
  --model_tag "$MODEL_TAG" \
  --output_dir "$OUT_DIR" \
  "${LIMIT_ARGS[@]}"
