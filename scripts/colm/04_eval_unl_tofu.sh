#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=00-01:00
#SBATCH --output=./results/tofu/logs/eval-unl-%j-%a-%N.out
#SBATCH --job-name=tofu-eval-unl
#SBATCH --array=0-15

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery}
MODEL_NAME=${MODEL_NAME:-deepseek-ai/deepseek-llm-7b-chat}
FT_ROOT=${FT_ROOT:-results/tofu/adapters/ft/target_full}
UNL_ROOT=${UNL_ROOT:-results/tofu/adapters/unlearned}
RESULTS_ROOT=${RESULTS_ROOT:-results/tofu/eval}
UNLEARN_SET=${UNLEARN_SET:-tofu_fgt10_ret90}
FORGET_SPLIT=${FORGET_SPLIT:-forget10}
RETAIN_SPLIT=${RETAIN_SPLIT:-retain90}

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

method_list=(grad_diff KL grad_ascent npo)
split_list=("$FORGET_SPLIT" "$RETAIN_SPLIT" real_authors world_facts)
IDX=${SLURM_ARRAY_TASK_ID:-0}
method_idx=$((IDX / 4))
split_idx=$((IDX % 4))
METHOD=${method_list[$method_idx]}
SPLIT=${split_list[$split_idx]}

target_adapter="$FT_ROOT/lr${LR_FT}_WD${WD_FT}_loraRank${LORA_RANK_FT}_loraDrop${LORA_DROPOUT_FT}_GradStsp${GRAD_ACC_STEPS_FT}/epoch-${EPS_FT}"
unl_child="${UNLEARN_SET}-lr${LR}_WD${WEIGHT_DECAY}_loraRank${LORA_RANK}_loraDrop${LORA_DROPOUT}_GradStep${GRAD_ACC_STEPS}_reg${REG}"
if [ "$BETA" != "0.1" ]; then
  unl_child="${unl_child}_beta${BETA}"
fi
unlearn_adapter="$UNL_ROOT/$unl_child/$METHOD/epoch-${EPOCHS}"
MODEL_TAG="unlearned_${METHOD}_lr${LR}_ep${EPOCHS}"
OUT_DIR="$RESULTS_ROOT/$SPLIT/$MODEL_TAG"

echo "[eval-unl] IDX=$IDX METHOD=$METHOD SPLIT=$SPLIT"
echo "[eval-unl] target_adapter=$target_adapter"
echo "[eval-unl] unlearn_adapter=$unlearn_adapter"
echo "[eval-unl] OUT_DIR=$OUT_DIR LIMIT=${LIMIT:-none}"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source "$HOME/ENV-3.10/bin/activate"
cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR" results/tofu/logs

python tofu/evaluate_tofu.py \
  --base_model_name "$MODEL_NAME" \
  --target_adapter_dir "$target_adapter" \
  --adapter_dir "$unlearn_adapter" \
  --eval_data "tofu/processed/${SPLIT}.json" \
  --split "$SPLIT" \
  --model_tag "$MODEL_TAG" \
  --output_dir "$OUT_DIR" \
  "${LIMIT_ARGS[@]}"
