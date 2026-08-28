#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=00-03:00
#SBATCH --output=./results/tofu/logs/05-key-recovery-%j-%a-%N.out
#SBATCH --job-name=tofu-05-key-recovery
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --array=0-39

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery}
MODEL_NAME=${MODEL_NAME:-deepseek-ai/deepseek-llm-7b-chat}
RESULTS_ROOT=${RESULTS_ROOT:-results/tofu/key_recovery}
AUDIT_MODE=${AUDIT_MODE:-ft}

MODEL_TAG_LIST=${MODEL_TAG_LIST:-"target_full oracle_retrain90"}
METHOD_LIST=${METHOD_LIST:-"grad_ascent KL grad_diff npo"} # 
DATASET_LIST=${DATASET_LIST:-"forget10"}
EPOCH_LIST=${EPOCH_LIST:-"2 4 6 8 10 12 14 16 18 20"}
CONSTRAINT_SCOPE_LIST=${CONSTRAINT_SCOPE_LIST:-"full_vocab"}
# same_genre_fact_group_content_vocab same_fact_group_content_vocab same_genre_content_vocab 
TOKENIZED_KEY_FILE=${TOKENIZED_KEY_FILE:-"TOFU/keys/tokenized/deepseek-ai__deepseek-llm-7b-chat/full_key_tokens_tokenized.jsonl"}

FT_RUN=${FT_RUN:-"lr0.0002_WD0.01_loraRank128_loraDrop0.0_GradStsp40"}
METHOD_LR_MAP=${METHOD_LR_MAP:-"grad_ascent:1e-05,KL:1e-05,grad_diff:1e-05,npo:1e-05"}
UNLEARN_LR_OVERRIDE=${UNLEARN_LR_OVERRIDE:-}
UNLEARN_RUN=${UNLEARN_RUN:-}

MAX_KEY_TOKENS=${MAX_KEY_TOKENS:-16}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-1}
NLL_BATCH_SIZE=${NLL_BATCH_SIZE:-1}
MIN_CANDIDATE_TOKEN_COUNT=${MIN_CANDIDATE_TOKEN_COUNT:-50}
CONTENT_RECALL_HIT_THRESHOLD=${CONTENT_RECALL_HIT_THRESHOLD:-0.5}
SKIP_CONSTRAINED_GENERATION=${SKIP_CONSTRAINED_GENERATION:-0}
SKIP_MASKED_NLL=${SKIP_MASKED_NLL:-0}
SAVE_CANDIDATE_DEBUG=${SAVE_CANDIDATE_DEBUG:-0}
LIMIT=${LIMIT:-}

FT_ROOT=${FT_ROOT:-results/tofu/adapters/ft}
UNL_ROOT=${UNL_ROOT:-results/tofu/adapters/unlearned}
TARGET_FT_RUN=${TARGET_FT_RUN:-"lr0.0002_WD0.01_loraRank128_loraDrop0.0_GradStsp40"}
TARGET_FT_EPOCH=${TARGET_FT_EPOCH:-40}

read -r -a model_tags <<< "$MODEL_TAG_LIST"
read -r -a methods <<< "$METHOD_LIST"
read -r -a datasets <<< "$DATASET_LIST"
read -r -a epochs <<< "$EPOCH_LIST"
read -r -a scopes <<< "$CONSTRAINT_SCOPE_LIST"

get_lr_for_method() {
  local m="$1"
  IFS=',' read -r -a kvs <<< "$METHOD_LR_MAP"
  for kv in "${kvs[@]}"; do
    local k="${kv%%:*}"; local v="${kv#*:}"
    if [ "$k" = "$m" ]; then
      echo "$v"
      return
    fi
  done
  echo ""
}

fail_missing_adapter() {
  local path="$1"
  echo "AUDIT_MODE=$AUDIT_MODE" >&2
  echo "MODEL_TAG=${MODEL_TAG:-}" >&2
  echo "METHOD=${METHOD:-}" >&2
  echo "UNLEARN_LR=${UNLEARN_LR:-}" >&2
  echo "UNLEARN_RUN=${RUN_KEY:-}" >&2
  echo "EPOCH=${EPOCH:-}" >&2
  echo "SPLIT=${SPLIT:-}" >&2
  echo "expected adapter path=$path" >&2
  exit 1
}

IDX=${SLURM_ARRAY_TASK_ID:-0}
NUM_DATASETS=${#datasets[@]}
NUM_EPOCHS=${#epochs[@]}
NUM_SCOPES=${#scopes[@]}

if [ "$AUDIT_MODE" = "ft" ]; then
  NUM_MODELS=${#model_tags[@]}
  TOTAL=$((NUM_MODELS * NUM_DATASETS * NUM_EPOCHS * NUM_SCOPES))
  [ "$IDX" -ge "$TOTAL" ] && echo "[05-recovery] IDX=$IDX >= TOTAL=$TOTAL, skip" && exit 0
  scope_idx=$((IDX % NUM_SCOPES))
  epoch_idx=$(((IDX / NUM_SCOPES) % NUM_EPOCHS))
  dataset_idx=$(((IDX / (NUM_SCOPES * NUM_EPOCHS)) % NUM_DATASETS))
  model_idx=$((IDX / (NUM_SCOPES * NUM_EPOCHS * NUM_DATASETS)))
  SCOPE=${scopes[$scope_idx]}
  EPOCH=${epochs[$epoch_idx]}
  SPLIT=${datasets[$dataset_idx]}
  MODEL_TAG=${model_tags[$model_idx]}
  METHOD=""
  MODEL_FAMILY="ft"
  ADAPTER_DIR="$FT_ROOT/$MODEL_TAG/$FT_RUN/epoch-$EPOCH"
  TARGET_ADAPTER_DIR=""
  RUN_KEY="$FT_RUN"
  OUT_DIR="$RESULTS_ROOT/$SCOPE/ft/$MODEL_TAG/$FT_RUN"
elif [ "$AUDIT_MODE" = "unl" ] || [ "$AUDIT_MODE" = "unlearned" ]; then
  NUM_METHODS=${#methods[@]}
  TOTAL=$((NUM_METHODS * NUM_DATASETS * NUM_EPOCHS * NUM_SCOPES))
  [ "$IDX" -ge "$TOTAL" ] && echo "[05-recovery] IDX=$IDX >= TOTAL=$TOTAL, skip" && exit 0
  scope_idx=$((IDX % NUM_SCOPES))
  epoch_idx=$(((IDX / NUM_SCOPES) % NUM_EPOCHS))
  dataset_idx=$(((IDX / (NUM_SCOPES * NUM_EPOCHS)) % NUM_DATASETS))
  method_idx=$((IDX / (NUM_SCOPES * NUM_EPOCHS * NUM_DATASETS)))
  SCOPE=${scopes[$scope_idx]}
  EPOCH=${epochs[$epoch_idx]}
  SPLIT=${datasets[$dataset_idx]}
  METHOD=${methods[$method_idx]}
  MODEL_TAG="unlearned_${METHOD}"
  MODEL_FAMILY="unlearned"
  UNLEARN_LR=$(get_lr_for_method "$METHOD")
  [ -n "$UNLEARN_LR_OVERRIDE" ] && UNLEARN_LR="$UNLEARN_LR_OVERRIDE"
  [ -z "$UNLEARN_LR" ] && echo "[05-recovery][ERROR] no lr for method=$METHOD" && exit 1
  RUN_KEY=${UNLEARN_RUN:-"tofu_fgt10_ret90-lr${UNLEARN_LR}_WD0.01_loraRank128_loraDrop0.0_GradStep40_reg1.0"}
  TARGET_ADAPTER_DIR="$FT_ROOT/target_full/$TARGET_FT_RUN/epoch-$TARGET_FT_EPOCH"
  ADAPTER_DIR="$UNL_ROOT/$RUN_KEY/$METHOD/epoch-$EPOCH"
  OUT_DIR="$RESULTS_ROOT/$SCOPE/unlearned/$RUN_KEY/$METHOD"
else
  echo "[05-recovery][ERROR] AUDIT_MODE must be ft or unl" >&2
  exit 1
fi

EVAL_DATA="tofu/processed/${SPLIT}.json"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source "$HOME/ENV-3.10/bin/activate"

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR" results/tofu/logs

[ ! -f "$TOKENIZED_KEY_FILE" ] && echo "[05-recovery][ERROR] tokenized key file missing: $TOKENIZED_KEY_FILE" && exit 1
[ ! -f "$EVAL_DATA" ] && echo "[05-recovery][ERROR] eval data missing: $EVAL_DATA" && exit 1
if [ "$AUDIT_MODE" = "unl" ] || [ "$AUDIT_MODE" = "unlearned" ]; then
  [ ! -f "$TARGET_ADAPTER_DIR/adapter_config.json" ] && fail_missing_adapter "$TARGET_ADAPTER_DIR"
fi
[ ! -f "$ADAPTER_DIR/adapter_config.json" ] && fail_missing_adapter "$ADAPTER_DIR"

ARGS=(
  --eval_data "$EVAL_DATA"
  --split "$SPLIT"
  --tokenized_key_file "$TOKENIZED_KEY_FILE"
  --base_model_name "$MODEL_NAME"
  --adapter_dir "$ADAPTER_DIR"
  --model_family "$MODEL_FAMILY"
  --model_tag "$MODEL_TAG"
  --method "$METHOD"
  --unlearn_run "$RUN_KEY"
  --epoch "$EPOCH"
  --constraint_scope "$SCOPE"
  --output_dir "$OUT_DIR"
  --max_key_tokens "$MAX_KEY_TOKENS"
  --generation_batch_size "$GENERATION_BATCH_SIZE"
  --nll_batch_size "$NLL_BATCH_SIZE"
  --min_candidate_token_count "$MIN_CANDIDATE_TOKEN_COUNT"
  --content_recall_hit_threshold "$CONTENT_RECALL_HIT_THRESHOLD"
  --local_files_only
)
[ -n "$TARGET_ADAPTER_DIR" ] && ARGS+=(--target_adapter_dir "$TARGET_ADAPTER_DIR")
[ -n "$LIMIT" ] && ARGS+=(--limit "$LIMIT")
[ "$SKIP_CONSTRAINED_GENERATION" = "1" ] && ARGS+=(--skip_constrained_generation)
[ "$SKIP_MASKED_NLL" = "1" ] && ARGS+=(--skip_masked_nll)
[ "$SAVE_CANDIDATE_DEBUG" = "1" ] && ARGS+=(--save_candidate_debug)

python -u tofu/05_eval_key_recovery.py "${ARGS[@]}"
