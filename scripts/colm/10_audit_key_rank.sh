#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=00-3:00
#SBATCH --output=./results/tofu/logs/audit-key-rank-v3-%j-%a-%N.out
#SBATCH --job-name=tofu-audit-key-rank-v3
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --array=0-7

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery}
MODEL_NAME=${MODEL_NAME:-deepseek-ai/deepseek-llm-7b-chat}
RESULTS_ROOT=${RESULTS_ROOT:-results/tofu/audit_key_rank}
AUDIT_MODE=${AUDIT_MODE:-unl}   # unl / ft
AUDIT_CONFIG_LIST=${AUDIT_CONFIG_LIST:-"factgroup_content factgroup_type genre_allkey genre_content"}
METHOD_LIST=${METHOD_LIST:-"grad_ascent KL"} # grad_diff npo
MODEL_TAG_LIST=${MODEL_TAG_LIST:-"target_full oracle_retrain90"}
SPLIT_LIST=${SPLIT_LIST:-"forget10"}
EPOCH_LIST=${EPOCH_LIST:-"10"}
TOKENIZED_KEY_FILE=${TOKENIZED_KEY_FILE:-"TOFU/keys/tokenized/deepseek-ai__deepseek-llm-7b-chat/full_key_tokens_tokenized.jsonl"}

LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}
LOG_EVERY=${LOG_EVERY:-10}
MAX_LENGTH=${MAX_LENGTH:-512}
CACHE_DIR=${CACHE_DIR:-}

DO_SPAN_RANK=${DO_SPAN_RANK:-1}
SPAN_CANDIDATE_SCOPE=${SPAN_CANDIDATE_SCOPE:-auto}
SPAN_RANK_MAX_CANDIDATES=${SPAN_RANK_MAX_CANDIDATES:-300}
SPAN_RANK_MODE=${SPAN_RANK_MODE:-normal_and_proxy_rig}
SPAN_RANK_TOPK_TO_SAVE=${SPAN_RANK_TOPK_TO_SAVE:-10}
SPAN_RANK_BATCH_SIZE=${SPAN_RANK_BATCH_SIZE:-16}

SAVE_CANDIDATE_SCORES=${SAVE_CANDIDATE_SCORES:-0}
CANDIDATE_SCORES_MODE=${CANDIDATE_SCORES_MODE:-top_bottom_gold}
CANDIDATE_SCORES_TOPK=${CANDIDATE_SCORES_TOPK:-10}

FT_ROOT=${FT_ROOT:-results/tofu/adapters/ft}
UNL_ROOT=${UNL_ROOT:-results/tofu/adapters/unlearned}
LR_FT=${LR_FT:-0.0002}
WD_FT=${WD_FT:-0.01}
LORA_RANK_FT=${LORA_RANK_FT:-128}
LORA_DROPOUT_FT=${LORA_DROPOUT_FT:-0.0}
GRAD_ACC_STEPS_FT=${GRAD_ACC_STEPS_FT:-40}
EPS_FT=${EPS_FT:-25}
FT_RUN=${FT_RUN:-"lr${LR_FT}_WD${WD_FT}_loraRank${LORA_RANK_FT}_loraDrop${LORA_DROPOUT_FT}_GradStsp${GRAD_ACC_STEPS_FT}"}

WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
LORA_RANK=${LORA_RANK:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-40}
REG=${REG:-1.0}
BETA=${BETA:-0.1}
UNLEARN_SET=${UNLEARN_SET:-tofu_fgt10_ret90}

get_default_unlearn_lr() {
  local method="$1"
  case "$method" in
    grad_ascent|grad_diff|KL)
      echo "1e-05"
      ;;
    npo)
      echo "0.0005"
      ;;
    *)
      echo "[audit-key-v3] Unknown method: $method" >&2
      exit 1
      ;;
  esac
}

fail_missing_adapter() {
  local label="$1"
  local path="$2"
  echo "[audit-key-v3][ERROR] $label missing: $path" >&2
  echo "[audit-key-v3][ERROR] METHOD=${METHOD:-}" >&2
  echo "[audit-key-v3][ERROR] METHOD_DIR=${METHOD_DIR:-}" >&2
  echo "[audit-key-v3][ERROR] UNLEARN_LR=${UNLEARN_LR:-}" >&2
  echo "[audit-key-v3][ERROR] UNLEARN_RUN=${UNLEARN_RUN:-}" >&2
  echo "[audit-key-v3][ERROR] EPOCH=${EPOCH:-}" >&2
  echo "[audit-key-v3][ERROR] SPLIT=${SPLIT:-}" >&2
  echo "[audit-key-v3][ERROR] AUDIT_CONFIG=${AUDIT_CONFIG:-}" >&2
  exit 1
}

read -r -a audit_config_list <<< "$AUDIT_CONFIG_LIST"
read -r -a method_list <<< "$METHOD_LIST"
read -r -a model_tag_list <<< "$MODEL_TAG_LIST"
read -r -a split_list <<< "$SPLIT_LIST"
read -r -a epoch_list <<< "$EPOCH_LIST"

NUM_CONFIGS=${#audit_config_list[@]}
NUM_METHODS=${#method_list[@]}
NUM_MODELS=${#model_tag_list[@]}
NUM_SPLITS=${#split_list[@]}
NUM_EPOCHS=${#epoch_list[@]}
IDX=${SLURM_ARRAY_TASK_ID:-0}

if [ "$AUDIT_MODE" = "ft" ]; then
  TOTAL=$((NUM_MODELS * NUM_SPLITS * NUM_EPOCHS * NUM_CONFIGS))
  if [ "$IDX" -ge "$TOTAL" ]; then
    echo "[audit-key-v3] IDX=$IDX >= TOTAL=$TOTAL, skip."
    exit 0
  fi
  config_idx=$((IDX % NUM_CONFIGS))
  epoch_idx=$(((IDX / NUM_CONFIGS) % NUM_EPOCHS))
  split_idx=$(((IDX / (NUM_CONFIGS * NUM_EPOCHS)) % NUM_SPLITS))
  model_idx=$((IDX / (NUM_CONFIGS * NUM_EPOCHS * NUM_SPLITS)))
  AUDIT_CONFIG=${audit_config_list[$config_idx]}
  MODEL_TAG=${model_tag_list[$model_idx]}
  SPLIT=${split_list[$split_idx]}
  EPOCH=${epoch_list[$epoch_idx]}
  METHOD=""
elif [ "$AUDIT_MODE" = "unl" ]; then
  TOTAL=$((NUM_METHODS * NUM_SPLITS * NUM_EPOCHS * NUM_CONFIGS))
  if [ "$IDX" -ge "$TOTAL" ]; then
    echo "[audit-key-v3] IDX=$IDX >= TOTAL=$TOTAL, skip."
    exit 0
  fi
  config_idx=$((IDX % NUM_CONFIGS))
  epoch_idx=$(((IDX / NUM_CONFIGS) % NUM_EPOCHS))
  split_idx=$(((IDX / (NUM_CONFIGS * NUM_EPOCHS)) % NUM_SPLITS))
  method_idx=$((IDX / (NUM_CONFIGS * NUM_EPOCHS * NUM_SPLITS)))
  AUDIT_CONFIG=${audit_config_list[$config_idx]}
  METHOD=${method_list[$method_idx]}
  SPLIT=${split_list[$split_idx]}
  EPOCH=${epoch_list[$epoch_idx]}
  MODEL_TAG="unlearned_${METHOD}"
else
  echo "[audit-key-v3][ERROR] AUDIT_MODE must be unl or ft."
  exit 1
fi

EVAL_DATA="tofu/processed/${SPLIT}.json"

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

if [ "$AUDIT_MODE" = "unl" ]; then
  MODEL_FAMILY="unlearned"
  METHOD_DIR=${METHOD_DIR_OVERRIDE:-$METHOD}
  UNLEARN_LR=$(get_default_unlearn_lr "$METHOD")
  if [ -n "${UNLEARN_LR_OVERRIDE:-}" ]; then
    UNLEARN_LR="$UNLEARN_LR_OVERRIDE"
  fi
  UNLEARN_RUN=${UNLEARN_RUN:-"${UNLEARN_SET}-lr${UNLEARN_LR}_WD${WEIGHT_DECAY}_loraRank${LORA_RANK}_loraDrop${LORA_DROPOUT}_GradStep${GRAD_ACC_STEPS}_reg${REG}"}
  if [ "$BETA" != "0.1" ] && [[ "$UNLEARN_RUN" != *_beta* ]]; then
    UNLEARN_RUN="${UNLEARN_RUN}_beta${BETA}"
  fi
  TARGET_FT_RUN="lr${LR_FT}_WD${WD_FT}_loraRank${LORA_RANK_FT}_loraDrop${LORA_DROPOUT_FT}_GradStsp${GRAD_ACC_STEPS_FT}"
  TARGET_ADAPTER_DIR="$FT_ROOT/target_full/$TARGET_FT_RUN/epoch-${EPS_FT}"
  ADAPTER_DIR="$UNL_ROOT/$UNLEARN_RUN/$METHOD_DIR/epoch-${EPOCH}"
  RUN_KEY="$UNLEARN_RUN"
  OUT_DIR="$RESULTS_ROOT/$AUDIT_CONFIG/unlearned/$UNLEARN_RUN/$METHOD"
  PY_LR="$UNLEARN_LR"
  PY_WD="$WEIGHT_DECAY"
  PY_RANK="$LORA_RANK"
  PY_DROPOUT="$LORA_DROPOUT"
  PY_GRAD="$GRAD_ACC_STEPS"
  PY_REG="$REG"
  PY_BETA="$BETA"
elif [ "$AUDIT_MODE" = "ft" ]; then
  MODEL_FAMILY="ft"
  METHOD_DIR=""
  UNLEARN_LR=""
  UNLEARN_RUN=""
  RUN_KEY="$FT_RUN"
  ADAPTER_DIR="$FT_ROOT/$MODEL_TAG/$FT_RUN/epoch-${EPOCH}"
  TARGET_ADAPTER_DIR=""
  OUT_DIR="$RESULTS_ROOT/$AUDIT_CONFIG/ft/$MODEL_TAG/$RUN_KEY"
  PY_LR="$LR_FT"
  PY_WD="$WD_FT"
  PY_RANK="$LORA_RANK_FT"
  PY_DROPOUT="$LORA_DROPOUT_FT"
  PY_GRAD="$GRAD_ACC_STEPS_FT"
  PY_REG=""
  PY_BETA=""
fi

SUMMARY_FILENAME="epoch-${EPOCH}-${SPLIT}-summary.json"
DETAILS_FILENAME="epoch-${EPOCH}-${SPLIT}-details.jsonl"
EPOCH_CSV="$OUT_DIR/epoch_curve-${SPLIT}.csv"

echo "[audit-key-v3] mode=$AUDIT_MODE idx=$IDX/$TOTAL"
echo "[audit-key-v3] config=$AUDIT_CONFIG method=${METHOD:-} method_dir=${METHOD_DIR:-} split=$SPLIT epoch=$EPOCH"
echo "[audit-key-v3] model_tag=$MODEL_TAG unlearn_lr=${UNLEARN_LR:-} run=${UNLEARN_RUN:-$RUN_KEY}"
echo "[audit-key-v3] tokenized_key_file=$TOKENIZED_KEY_FILE"
echo "[audit-key-v3] out_dir=$OUT_DIR"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source "$HOME/ENV-3.10/bin/activate"

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR" results/tofu/logs

if [ ! -f "$TOKENIZED_KEY_FILE" ]; then
  echo "[audit-key-v3][ERROR] tokenized key file not found: $TOKENIZED_KEY_FILE"
  exit 1
fi
if [ ! -f "$EVAL_DATA" ]; then
  echo "[audit-key-v3][ERROR] eval data not found: $EVAL_DATA"
  exit 1
fi
if [ "$AUDIT_MODE" = "unl" ] && [ ! -f "$TARGET_ADAPTER_DIR/adapter_config.json" ]; then
  fail_missing_adapter "target adapter" "$TARGET_ADAPTER_DIR"
fi
if [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
  fail_missing_adapter "adapter" "$ADAPTER_DIR"
fi

export PYTHONUNBUFFERED=1

PY_ARGS=(
  --base_model_name "$MODEL_NAME"
  --adapter_dir "$ADAPTER_DIR"
  --eval_data "$EVAL_DATA"
  --tokenized_key_file "$TOKENIZED_KEY_FILE"
  --output_dir "$OUT_DIR"
  --audit_config "$AUDIT_CONFIG"
  --model_family "$MODEL_FAMILY"
  --method "$METHOD"
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
if [ "$AUDIT_MODE" = "unl" ]; then
  PY_ARGS+=(--unlearn_run "$RUN_KEY")
fi
if [ "$AUDIT_MODE" = "ft" ]; then
  PY_ARGS+=(--ft_run "$RUN_KEY")
fi
if [ -n "$PY_REG" ]; then
  PY_ARGS+=(--reg "$PY_REG")
fi
if [ -n "$PY_BETA" ]; then
  PY_ARGS+=(--beta "$PY_BETA")
fi
if [ "$DO_SPAN_RANK" = "1" ]; then
  PY_ARGS+=(
    --do_span_rank
    --span_candidate_scope "$SPAN_CANDIDATE_SCOPE"
    --span_rank_max_candidates "$SPAN_RANK_MAX_CANDIDATES"
    --span_rank_mode "$SPAN_RANK_MODE"
    --span_rank_topk_to_save "$SPAN_RANK_TOPK_TO_SAVE"
    --span_rank_batch_size "$SPAN_RANK_BATCH_SIZE"
  )
fi
if [ "$SAVE_CANDIDATE_SCORES" = "1" ]; then
  PY_ARGS+=(
    --save_candidate_scores
    --candidate_scores_mode "$CANDIDATE_SCORES_MODE"
    --candidate_scores_topk "$CANDIDATE_SCORES_TOPK"
  )
fi

python -u tofu/audit_key_rank.py \
  "${PY_ARGS[@]}" \
  "${LOCAL_ARGS[@]}" \
  "${LIMIT_ARGS[@]}"
