#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=00-12:00
#SBATCH --output=./results/tofu/logs/audit-ft-rank-%j-%a-%N.out
#SBATCH --job-name=tofu-audit-ft-rank
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --array=0-1

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery}
MODEL_NAME=${MODEL_NAME:-deepseek-ai/deepseek-llm-7b-chat}
FT_ROOT=${FT_ROOT:-results/tofu/adapters/ft}
RESULTS_ROOT=${RESULTS_ROOT:-results/tofu/audit_rank}

POOL_DATA=${POOL_DATA:-TOFU/full.json}
CANDIDATE_POOL=${CANDIDATE_POOL:-category}

MODEL_TAG_LIST=${MODEL_TAG_LIST:-"target_full oracle_retrain90"}
SPLIT_LIST=${SPLIT_LIST:-"forget10"}
EPOCH_LIST=${EPOCH_LIST:-"25"}

read -r -a model_tags <<< "$MODEL_TAG_LIST"
read -r -a split_list <<< "$SPLIT_LIST"
read -r -a epoch_list <<< "$EPOCH_LIST"

LR_FT=${LR_FT:-0.0002}
WD_FT=${WD_FT:-0.01}
LORA_RANK_FT=${LORA_RANK_FT:-128}
LORA_DROPOUT_FT=${LORA_DROPOUT_FT:-0.0}
GRAD_ACC_STEPS_FT=${GRAD_ACC_STEPS_FT:-40}

LOG_EVERY=${LOG_EVERY:-10}
MAX_LENGTH=${MAX_LENGTH:-512}
CACHE_DIR=${CACHE_DIR:-}

LIMIT_ARGS=()
if [ -n "${LIMIT:-}" ]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

LOCAL_ARGS=(--local_files_only)
if [ -n "$CACHE_DIR" ]; then
  LOCAL_ARGS+=(--cache_dir "$CACHE_DIR")
fi

IDX=${SLURM_ARRAY_TASK_ID:-0}

NUM_MODELS=${#model_tags[@]}
NUM_SPLITS=${#split_list[@]}
NUM_EPOCHS=${#epoch_list[@]}
TOTAL=$((NUM_MODELS * NUM_SPLITS * NUM_EPOCHS))

if [ "$IDX" -ge "$TOTAL" ]; then
  echo "[audit-ft] IDX=$IDX >= TOTAL=$TOTAL, skip."
  exit 0
fi

epoch_idx=$((IDX % NUM_EPOCHS))
split_idx=$(((IDX / NUM_EPOCHS) % NUM_SPLITS))
model_idx=$((IDX / (NUM_EPOCHS * NUM_SPLITS)))

MODEL_TAG=${model_tags[$model_idx]}
SPLIT=${split_list[$split_idx]}
EPOCH=${epoch_list[$epoch_idx]}

PARAM_TAG="lr${LR_FT}_WD${WD_FT}_loraRank${LORA_RANK_FT}_loraDrop${LORA_DROPOUT_FT}_GradStsp${GRAD_ACC_STEPS_FT}"
ADAPTER_DIR="$FT_ROOT/$MODEL_TAG/$PARAM_TAG/epoch-${EPOCH}"

EVAL_DATA="tofu/processed/${SPLIT}.json"
OUT_DIR="$RESULTS_ROOT/$MODEL_TAG/$PARAM_TAG"

SUMMARY_FILENAME="epoch-${EPOCH}-${SPLIT}-summary.json"
DETAILS_FILENAME="epoch-${EPOCH}-${SPLIT}-details.jsonl"
EPOCH_CSV="$OUT_DIR/${SPLIT}-epoch_curve.csv"

echo "[audit-ft] IDX=$IDX model_idx=$model_idx split_idx=$split_idx epoch_idx=$epoch_idx"
echo "[audit-ft] MODEL_TAG=$MODEL_TAG SPLIT=$SPLIT EPOCH=$EPOCH"
echo "[audit-ft] ADAPTER_DIR=$ADAPTER_DIR"
echo "[audit-ft] EVAL_DATA=$EVAL_DATA"
echo "[audit-ft] OUT_DIR=$OUT_DIR"
echo "[audit-ft] SUMMARY_FILENAME=$SUMMARY_FILENAME"
echo "[audit-ft] DETAILS_FILENAME=$DETAILS_FILENAME"
echo "[audit-ft] EPOCH_CSV=$EPOCH_CSV"
echo "[audit-ft] POOL_DATA=$POOL_DATA CANDIDATE_POOL=$CANDIDATE_POOL LIMIT=${LIMIT:-none}"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source "$HOME/ENV-3.10/bin/activate"

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR" results/tofu/logs

if [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
  echo "[audit-ft][ERROR] adapter_config.json not found: $ADAPTER_DIR"
  exit 1
fi

export PYTHONUNBUFFERED=1

python -u tofu/audit_tofu_rank.py \
  --base_model_name "$MODEL_NAME" \
  --adapter_dir "$ADAPTER_DIR" \
  --eval_data "$EVAL_DATA" \
  --pool_data "$POOL_DATA" \
  --output_dir "$OUT_DIR" \
  --model_tag "$MODEL_TAG" \
  --split "$SPLIT" \
  --candidate_pool "$CANDIDATE_POOL" \
  --epoch "$EPOCH" \
  --lr "$LR_FT" \
  --weight_decay "$WD_FT" \
  --lora_rank "$LORA_RANK_FT" \
  --lora_dropout "$LORA_DROPOUT_FT" \
  --grad_acc_steps "$GRAD_ACC_STEPS_FT" \
  --summary_filename "$SUMMARY_FILENAME" \
  --details_filename "$DETAILS_FILENAME" \
  --epoch_csv "$EPOCH_CSV" \
  --max_length "$MAX_LENGTH" \
  --log_every "$LOG_EVERY" \
  "${LOCAL_ARGS[@]}" \
  "${LIMIT_ARGS[@]}"