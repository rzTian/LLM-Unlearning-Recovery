#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=00-06:00
#SBATCH --output=./results/tofu/logs/audit-unl-rank-%j-%a-%N.out
#SBATCH --job-name=tofu-audit-unl-rank
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --array=0-29

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery}
MODEL_NAME=${MODEL_NAME:-deepseek-ai/deepseek-llm-7b-chat}

FT_ROOT=${FT_ROOT:-results/tofu/adapters/ft/target_full}
UNL_ROOT=${UNL_ROOT:-results/tofu/adapters/unlearned}
RESULTS_ROOT=${RESULTS_ROOT:-results/tofu/audit_rank}

UNLEARN_SET=${UNLEARN_SET:-tofu_fgt10_ret90}
POOL_DATA=${POOL_DATA:-TOFU/full.json}
CANDIDATE_POOL=${CANDIDATE_POOL:-category}

LR_LIST=(0.0001 0.0005 5e-05) # 1e-05 
METHOD_LIST=${METHOD_LIST:-"npo"} # grad_ascent grad_diff KL 
SPLIT_LIST=${SPLIT_LIST:-"forget10"}
EPOCH_LIST=${EPOCH_LIST:-"2 4 6 8 10 12 14 16 18 20"}

read -r -a lr_list <<< "$LR_LIST"
read -r -a method_list <<< "$METHOD_LIST"
read -r -a split_list <<< "$SPLIT_LIST"
read -r -a epoch_list <<< "$EPOCH_LIST"

LR_FT=${LR_FT:-0.0002}
EPS_FT=${EPS_FT:-25}
WD_FT=${WD_FT:-0.01}
LORA_RANK_FT=${LORA_RANK_FT:-128}
LORA_DROPOUT_FT=${LORA_DROPOUT_FT:-0.0}
GRAD_ACC_STEPS_FT=${GRAD_ACC_STEPS_FT:-40}

WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
LORA_RANK=${LORA_RANK:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-40}
REG=${REG:-1.0}
BETA=${BETA:-0.1}

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

NUM_LR=${#lr_list[@]}
NUM_METHODS=${#method_list[@]}
NUM_SPLITS=${#split_list[@]}
NUM_EPOCHS=${#epoch_list[@]}
TOTAL=$((NUM_METHODS * NUM_SPLITS * NUM_EPOCHS))

if [ "$IDX" -ge "$TOTAL" ]; then
  echo "[audit-unl] IDX=$IDX >= TOTAL=$TOTAL, skip."
  exit 0
fi

epoch_idx=$((IDX % NUM_EPOCHS))
split_idx=$(((IDX / NUM_EPOCHS) % NUM_SPLITS))
method_idx=$(((IDX / (NUM_EPOCHS * NUM_SPLITS)) % NUM_METHODS))
lr_idx=$((IDX / (NUM_EPOCHS * NUM_SPLITS * NUM_METHODS)))

LR=${lr_list[$lr_idx]}
METHOD=${method_list[$method_idx]}
SPLIT=${split_list[$split_idx]}
EPOCH=${epoch_list[$epoch_idx]}

FT_PARAM_TAG="lr${LR_FT}_WD${WD_FT}_loraRank${LORA_RANK_FT}_loraDrop${LORA_DROPOUT_FT}_GradStsp${GRAD_ACC_STEPS_FT}"
TARGET_ADAPTER="$FT_ROOT/$FT_PARAM_TAG/epoch-${EPS_FT}"

UNL_PARAM_TAG="${UNLEARN_SET}-lr${LR}_WD${WEIGHT_DECAY}_loraRank${LORA_RANK}_loraDrop${LORA_DROPOUT}_GradStep${GRAD_ACC_STEPS}_reg${REG}"
if [ "$BETA" != "0.1" ]; then
  UNL_PARAM_TAG="${UNL_PARAM_TAG}_beta${BETA}"
fi

UNLEARN_ADAPTER="$UNL_ROOT/$UNL_PARAM_TAG/$METHOD/epoch-${EPOCH}"

EVAL_DATA="tofu/processed/${SPLIT}.json"
MODEL_TAG="unlearned_${METHOD}"
OUT_DIR="$RESULTS_ROOT/unlearned/$UNL_PARAM_TAG/$METHOD"

SUMMARY_FILENAME="epoch-${EPOCH}-${SPLIT}-summary.json"
DETAILS_FILENAME="epoch-${EPOCH}-${SPLIT}-details.jsonl"
EPOCH_CSV="$OUT_DIR/${SPLIT}-epoch_curve.csv"

echo "[audit-unl] IDX=$IDX method_idx=$method_idx split_idx=$split_idx epoch_idx=$epoch_idx"
echo "[audit-unl] METHOD=$METHOD SPLIT=$SPLIT EPOCH=$EPOCH"
echo "[audit-unl] EPS_FT=$EPS_FT TARGET_ADAPTER=$TARGET_ADAPTER"
echo "[audit-unl] UNLEARN_ADAPTER=$UNLEARN_ADAPTER"
echo "[audit-unl] OUT_DIR=$OUT_DIR"
echo "[audit-unl] SUMMARY_FILENAME=$SUMMARY_FILENAME"
echo "[audit-unl] DETAILS_FILENAME=$DETAILS_FILENAME"
echo "[audit-unl] EPOCH_CSV=$EPOCH_CSV"
echo "[audit-unl] POOL_DATA=$POOL_DATA CANDIDATE_POOL=$CANDIDATE_POOL LIMIT=${LIMIT:-none}"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source "$HOME/ENV-3.10/bin/activate"

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR" results/tofu/logs

if [ ! -f "$TARGET_ADAPTER/adapter_config.json" ]; then
  echo "[audit-unl][ERROR] target adapter_config.json not found: $TARGET_ADAPTER"
  exit 1
fi

if [ ! -f "$UNLEARN_ADAPTER/adapter_config.json" ]; then
  echo "[audit-unl][ERROR] unlearn adapter_config.json not found: $UNLEARN_ADAPTER"
  exit 1
fi

export PYTHONUNBUFFERED=1

python -u tofu/audit_tofu_rank.py \
  --base_model_name "$MODEL_NAME" \
  --target_adapter_dir "$TARGET_ADAPTER" \
  --adapter_dir "$UNLEARN_ADAPTER" \
  --eval_data "$EVAL_DATA" \
  --pool_data "$POOL_DATA" \
  --output_dir "$OUT_DIR" \
  --model_tag "$MODEL_TAG" \
  --split "$SPLIT" \
  --candidate_pool "$CANDIDATE_POOL" \
  --epoch "$EPOCH" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --lora_rank "$LORA_RANK" \
  --lora_dropout "$LORA_DROPOUT" \
  --grad_acc_steps "$GRAD_ACC_STEPS" \
  --reg "$REG" \
  --beta "$BETA" \
  --summary_filename "$SUMMARY_FILENAME" \
  --details_filename "$DETAILS_FILENAME" \
  --epoch_csv "$EPOCH_CSV" \
  --max_length "$MAX_LENGTH" \
  --log_every "$LOG_EVERY" \
  "${LOCAL_ARGS[@]}" \
  "${LIMIT_ARGS[@]}"