#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=00-01:00
#SBATCH --output=./results/tofu/logs/audit-rank-%j-%a-%N.out
#SBATCH --job-name=tofu-audit-rank
#SBATCH --array=0-5

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery}
MODEL_NAME=${MODEL_NAME:-deepseek-ai/deepseek-llm-7b-chat}
FT_ROOT=${FT_ROOT:-results/tofu/adapters/ft}
UNL_ROOT=${UNL_ROOT:-results/tofu/adapters/unlearned}
RESULTS_ROOT=${RESULTS_ROOT:-results/tofu/audit_rank}
SPLIT=${SPLIT:-forget10}
POOL_DATA=${POOL_DATA:-TOFU/full.json}
CANDIDATE_POOL=${CANDIDATE_POOL:-category}
UNLEARN_SET=${UNLEARN_SET:-tofu_fgt10_ret90}

LR_FT=${LR_FT:-0.0002}
EPS_FT=${EPS_FT:-10}
WD_FT=${WD_FT:-0.01}
LORA_RANK_FT=${LORA_RANK_FT:-128}
LORA_DROPOUT_FT=${LORA_DROPOUT_FT:-0.0}
GRAD_ACC_STEPS_FT=${GRAD_ACC_STEPS_FT:-40}

LR=${LR:-0.00001}
EPOCHS=${EPOCHS:-1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
LORA_RANK=${LORA_RANK:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-40}
REG=${REG:-1.0}
BETA=${BETA:-0.1}
LIMIT_ARGS=()
if [ -n "${LIMIT:-}" ]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

ft_suffix="lr${LR_FT}_WD${WD_FT}_loraRank${LORA_RANK_FT}_loraDrop${LORA_DROPOUT_FT}_GradStsp${GRAD_ACC_STEPS_FT}/epoch-${EPS_FT}"
target_adapter="$FT_ROOT/target_full/$ft_suffix"
oracle_adapter="$FT_ROOT/oracle_retrain90/$ft_suffix"
unl_child="${UNLEARN_SET}-lr${LR}_WD${WEIGHT_DECAY}_loraRank${LORA_RANK}_loraDrop${LORA_DROPOUT}_GradStep${GRAD_ACC_STEPS}_reg${REG}"
if [ "$BETA" != "0.1" ]; then
  unl_child="${unl_child}_beta${BETA}"
fi

model_tags=(target_full oracle_retrain90 unlearned_grad_ascent unlearned_grad_diff unlearned_KL unlearned_npo)
methods=("" "" grad_ascent grad_diff KL npo)
IDX=${MODEL_IDX:-${SLURM_ARRAY_TASK_ID:-0}}
MODEL_TAG=${model_tags[$IDX]}
METHOD=${methods[$IDX]}
OUT_DIR="$RESULTS_ROOT/$SPLIT/$MODEL_TAG"

echo "[audit] IDX=$IDX MODEL_TAG=$MODEL_TAG METHOD=${METHOD:-none} SPLIT=$SPLIT"
echo "[audit] POOL_DATA=$POOL_DATA CANDIDATE_POOL=$CANDIDATE_POOL LIMIT=${LIMIT:-none}"
echo "[audit] target_adapter=$target_adapter oracle_adapter=$oracle_adapter"

if command -v module >/dev/null 2>&1; then
  module load gcc arrow/18.1.0 cuda
  module load python/3.10
  module load scipy-stack
fi
if [ -f "$HOME/ENV-3.10/bin/activate" ]; then
  source "$HOME/ENV-3.10/bin/activate"
fi

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR" results/tofu/logs

if [ "$MODEL_TAG" = "target_full" ]; then
  python tofu/audit_tofu_rank.py \
    --base_model_name "$MODEL_NAME" \
    --adapter_dir "${TARGET_ADAPTER_DIR:-$target_adapter}" \
    --eval_data "tofu/processed/${SPLIT}.json" \
    --pool_data "$POOL_DATA" \
    --output_dir "$OUT_DIR" \
    --model_tag "$MODEL_TAG" \
    --split "$SPLIT" \
    --candidate_pool "$CANDIDATE_POOL" \
    "${LIMIT_ARGS[@]}"
elif [ "$MODEL_TAG" = "oracle_retrain90" ]; then
  python tofu/audit_tofu_rank.py \
    --base_model_name "$MODEL_NAME" \
    --adapter_dir "${ORACLE_ADAPTER_DIR:-$oracle_adapter}" \
    --eval_data "tofu/processed/${SPLIT}.json" \
    --pool_data "$POOL_DATA" \
    --output_dir "$OUT_DIR" \
    --model_tag "$MODEL_TAG" \
    --split "$SPLIT" \
    --candidate_pool "$CANDIDATE_POOL" \
    "${LIMIT_ARGS[@]}"
else
  unlearn_adapter="$UNL_ROOT/$unl_child/$METHOD/epoch-${EPOCHS}"
  echo "[audit] unlearn_adapter=$unlearn_adapter"
  python tofu/audit_tofu_rank.py \
    --base_model_name "$MODEL_NAME" \
    --target_adapter_dir "${TARGET_ADAPTER_DIR:-$target_adapter}" \
    --adapter_dir "${UNLEARN_ADAPTER_DIR:-$unlearn_adapter}" \
    --eval_data "tofu/processed/${SPLIT}.json" \
    --pool_data "$POOL_DATA" \
    --output_dir "$OUT_DIR" \
    --model_tag "$MODEL_TAG" \
    --split "$SPLIT" \
    --candidate_pool "$CANDIDATE_POOL" \
    "${LIMIT_ARGS[@]}"
fi
