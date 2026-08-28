#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=00-12:00
#SBATCH --output=./results/tofu/logs/audit-grp-rank-%j-%a-%N.out
#SBATCH --job-name=tofu-audit-grp-rank
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --array=0-29

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery}
MODEL_NAME=${MODEL_NAME:-deepseek-ai/deepseek-llm-7b-chat}
RESULTS_ROOT=${RESULTS_ROOT:-results/tofu/audit_grp_rank}
GENRES_DIR=${GENRES_DIR:-TOFU/genres}
AUDIT_MODE=${AUDIT_MODE:-unl}
SPLIT_LIST=${SPLIT_LIST:-"forget10"}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}
LOG_EVERY=${LOG_EVERY:-10}
MAX_LENGTH=${MAX_LENGTH:-512}
CACHE_DIR=${CACHE_DIR:-}

FT_ROOT=${FT_ROOT:-results/tofu/adapters/ft}
UNL_ROOT=${UNL_ROOT:-results/tofu/adapters/unlearned}

LR_FT=${LR_FT:-0.0002}
WD_FT=${WD_FT:-0.01}
LORA_RANK_FT=${LORA_RANK_FT:-128}
LORA_DROPOUT_FT=${LORA_DROPOUT_FT:-0.0}
GRAD_ACC_STEPS_FT=${GRAD_ACC_STEPS_FT:-40}

EPS_FT=${EPS_FT:-25}
LR=${LR:-1e-05}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
LORA_RANK=${LORA_RANK:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-40}
REG=${REG:-1.0}
BETA=${BETA:-0.1}
UNLEARN_SET=${UNLEARN_SET:-tofu_fgt10_ret90}

read -r -a split_list <<< "$SPLIT_LIST"

LIMIT_ARGS=()
if [ -n "${LIMIT:-}" ]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

LOCAL_ARGS=()
if [ "$LOCAL_FILES_ONLY" = "1" ]; then
  LOCAL_ARGS=(--local_files_only)
fi
if [ -n "$CACHE_DIR" ]; then
  LOCAL_ARGS+=(--cache_dir "$CACHE_DIR")
fi

IDX=${SLURM_ARRAY_TASK_ID:-0}

if [ "$AUDIT_MODE" = "ft" ]; then
  MODEL_TAG_LIST=${MODEL_TAG_LIST:-"target_full oracle_retrain90"}
  EPOCH_LIST=${EPOCH_LIST:-"25"}
  read -r -a model_tags <<< "$MODEL_TAG_LIST"
  read -r -a epoch_list <<< "$EPOCH_LIST"

  NUM_MODELS=${#model_tags[@]}
  NUM_METHODS=0
  NUM_SPLITS=${#split_list[@]}
  NUM_EPOCHS=${#epoch_list[@]}
  TOTAL=$((NUM_MODELS * NUM_SPLITS * NUM_EPOCHS))

  if [ "$IDX" -ge "$TOTAL" ]; then
    echo "[audit-grp] AUDIT_MODE=$AUDIT_MODE IDX=$IDX >= TOTAL=$TOTAL, skip."
    exit 0
  fi

  epoch_idx=$((IDX % NUM_EPOCHS))
  split_idx=$(((IDX / NUM_EPOCHS) % NUM_SPLITS))
  model_idx=$((IDX / (NUM_EPOCHS * NUM_SPLITS)))
  method_idx=-1

  MODEL_TAG=${model_tags[$model_idx]}
  METHOD=""
  SPLIT=${split_list[$split_idx]}
  EPOCH=${epoch_list[$epoch_idx]}

  PARAM_TAG="lr${LR_FT}_WD${WD_FT}_loraRank${LORA_RANK_FT}_loraDrop${LORA_DROPOUT_FT}_GradStsp${GRAD_ACC_STEPS_FT}"
  ADAPTER_DIR="$FT_ROOT/$MODEL_TAG/$PARAM_TAG/epoch-${EPOCH}"
  TARGET_ADAPTER_DIR=""
  OUT_DIR="$RESULTS_ROOT/$MODEL_TAG/$PARAM_TAG"

  PY_LR="$LR_FT"
  PY_WD="$WD_FT"
  PY_RANK="$LORA_RANK_FT"
  PY_DROPOUT="$LORA_DROPOUT_FT"
  PY_GRAD="$GRAD_ACC_STEPS_FT"
  PY_REG=""
  PY_BETA=""

elif [ "$AUDIT_MODE" = "unl" ]; then
  METHOD_LIST=${METHOD_LIST:-"grad_ascent grad_diff KL"} # npo
  EPOCH_LIST=${EPOCH_LIST:-"2 4 6 8 10 12 14 16 18 20"}
  read -r -a method_list <<< "$METHOD_LIST"
  read -r -a epoch_list <<< "$EPOCH_LIST"

  NUM_MODELS=0
  NUM_METHODS=${#method_list[@]}
  NUM_SPLITS=${#split_list[@]}
  NUM_EPOCHS=${#epoch_list[@]}
  TOTAL=$((NUM_METHODS * NUM_SPLITS * NUM_EPOCHS))

  if [ "$IDX" -ge "$TOTAL" ]; then
    echo "[audit-grp] AUDIT_MODE=$AUDIT_MODE IDX=$IDX >= TOTAL=$TOTAL, skip."
    exit 0
  fi

  epoch_idx=$((IDX % NUM_EPOCHS))
  split_idx=$(((IDX / NUM_EPOCHS) % NUM_SPLITS))
  method_idx=$((IDX / (NUM_EPOCHS * NUM_SPLITS)))
  model_idx=-1

  METHOD=${method_list[$method_idx]}
  MODEL_TAG="unlearned_${METHOD}"
  SPLIT=${split_list[$split_idx]}
  EPOCH=${epoch_list[$epoch_idx]}

  FT_PARAM_TAG="lr${LR_FT}_WD${WD_FT}_loraRank${LORA_RANK_FT}_loraDrop${LORA_DROPOUT_FT}_GradStsp${GRAD_ACC_STEPS_FT}"
  TARGET_ADAPTER_DIR="$FT_ROOT/target_full/$FT_PARAM_TAG/epoch-${EPS_FT}"

  UNL_PARAM_TAG="${UNLEARN_SET}-lr${LR}_WD${WEIGHT_DECAY}_loraRank${LORA_RANK}_loraDrop${LORA_DROPOUT}_GradStep${GRAD_ACC_STEPS}_reg${REG}"
  if [ "$BETA" != "0.1" ]; then
    UNL_PARAM_TAG="${UNL_PARAM_TAG}_beta${BETA}"
  fi

  ADAPTER_DIR="$UNL_ROOT/$UNL_PARAM_TAG/$METHOD/epoch-${EPOCH}"
  OUT_DIR="$RESULTS_ROOT/unlearned/$UNL_PARAM_TAG/$METHOD"

  PY_LR="$LR"
  PY_WD="$WEIGHT_DECAY"
  PY_RANK="$LORA_RANK"
  PY_DROPOUT="$LORA_DROPOUT"
  PY_GRAD="$GRAD_ACC_STEPS"
  PY_REG="$REG"
  PY_BETA="$BETA"
else
  echo "[audit-grp][ERROR] AUDIT_MODE must be ft or unl, got: $AUDIT_MODE"
  exit 1
fi

EVAL_DATA="tofu/processed/${SPLIT}.json"
SUMMARY_FILENAME="epoch-${EPOCH}-${SPLIT}-summary.json"
DETAILS_FILENAME="epoch-${EPOCH}-${SPLIT}-details.jsonl"
EPOCH_CSV="$OUT_DIR/${SPLIT}-epoch_curve.csv"

echo "[audit-grp] AUDIT_MODE=$AUDIT_MODE"
echo "[audit-grp] IDX=$IDX"
echo "[audit-grp] model_idx=$model_idx method_idx=$method_idx split_idx=$split_idx epoch_idx=$epoch_idx"
echo "[audit-grp] NUM_MODELS=$NUM_MODELS NUM_METHODS=$NUM_METHODS NUM_SPLITS=$NUM_SPLITS NUM_EPOCHS=$NUM_EPOCHS TOTAL=$TOTAL"
echo "[audit-grp] MODEL_TAG=$MODEL_TAG METHOD=${METHOD:-none} SPLIT=$SPLIT EPOCH=$EPOCH"
echo "[audit-grp] adapter_dir=$ADAPTER_DIR"
echo "[audit-grp] target_adapter_dir=${TARGET_ADAPTER_DIR:-none}"
echo "[audit-grp] eval_data=$EVAL_DATA genres_dir=$GENRES_DIR"
echo "[audit-grp] output_dir=$OUT_DIR"
echo "[audit-grp] summary_filename=$SUMMARY_FILENAME"
echo "[audit-grp] details_filename=$DETAILS_FILENAME"
echo "[audit-grp] epoch_csv=$EPOCH_CSV"
echo "[audit-grp] LIMIT=${LIMIT:-none} LOCAL_FILES_ONLY=$LOCAL_FILES_ONLY"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source "$HOME/ENV-3.10/bin/activate"

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR" results/tofu/logs

if [ "$AUDIT_MODE" = "unl" ] && [ ! -f "$TARGET_ADAPTER_DIR/adapter_config.json" ]; then
  echo "[audit-grp][ERROR] target adapter_config.json not found: $TARGET_ADAPTER_DIR"
  exit 1
fi

if [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
  echo "[audit-grp][ERROR] adapter_config.json not found: $ADAPTER_DIR"
  exit 1
fi

export PYTHONUNBUFFERED=1

PY_ARGS=(
  --base_model_name "$MODEL_NAME"
  --adapter_dir "$ADAPTER_DIR"
  --eval_data "$EVAL_DATA"
  --genres_dir "$GENRES_DIR"
  --output_dir "$OUT_DIR"
  --model_tag "$MODEL_TAG"
  --split "$SPLIT"
  --epoch "$EPOCH"
  --lr "$PY_LR"
  --weight_decay "$PY_WD"
  --lora_rank "$PY_RANK"
  --lora_dropout "$PY_DROPOUT"
  --grad_acc_steps "$PY_GRAD"
  --summary_filename "$SUMMARY_FILENAME"
  --details_filename "$DETAILS_FILENAME"
  --epoch_csv "$EPOCH_CSV"
  --max_length "$MAX_LENGTH"
  --log_every "$LOG_EVERY"
)

if [ -n "$TARGET_ADAPTER_DIR" ]; then
  PY_ARGS+=(--target_adapter_dir "$TARGET_ADAPTER_DIR")
fi
if [ -n "$PY_REG" ]; then
  PY_ARGS+=(--reg "$PY_REG")
fi
if [ -n "$PY_BETA" ]; then
  PY_ARGS+=(--beta "$PY_BETA")
fi

python -u tofu/audit_tofu_grp_rank.py \
  "${PY_ARGS[@]}" \
  "${LOCAL_ARGS[@]}" \
  "${LIMIT_ARGS[@]}"
