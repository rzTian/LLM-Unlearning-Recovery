#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=00-06:00
#SBATCH --output=./results/tofu/logs/eval-ft-%j-%a-%N.out
#SBATCH --job-name=tofu-eval-ft
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --array=0-19

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery}
MODEL_NAME=${MODEL_NAME:-deepseek-ai/deepseek-llm-7b-chat}
FT_ROOT=${FT_ROOT:-results/tofu/adapters/ft}
RESULTS_ROOT=${RESULTS_ROOT:-results/tofu/eval}
LOG_EVERY=${LOG_EVERY:-10}

LR=${LR:-0.0002}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
LORA_RANK=${LORA_RANK:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-40}

# 默认评估训练中每 5 epoch 保存的 checkpoint。
# 如果只想评估 5 和 10，可以提交时：
# EPOCH_LIST="5 10" sbatch --array=0-27 scripts/colm/02_eval_ft_tofu.sh
EPOCH_LIST=${EPOCH_LIST:-"5 10 15 20 25 30 35 40 45 50"}
read -r -a epoch_list <<< "$EPOCH_LIST"

LIMIT_ARGS=()
if [ -n "${LIMIT:-}" ]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

# 这里第二列不再直接写 adapter path，而是写 adapter tag。
# 真正 adapter path 等 epoch 确定后再拼。
jobs=(
  # "target_full|target_full|full"
  # "target_full|target_full|forget10"
  # "target_full|target_full|forget05"
  # "target_full|target_full|forget01"
  # "target_full|target_full|retain90"
  # "target_full|target_full|retain95"
  # "target_full|target_full|retain99"
  "target_full|target_full|real_authors"
  "target_full|target_full|world_facts"
  # "oracle_retrain90|oracle_retrain90|forget10"
  # "oracle_retrain90|oracle_retrain90|retain90"
  # "oracle_retrain95|oracle_retrain95|forget05"
  # "oracle_retrain95|oracle_retrain95|retain95"
  # "oracle_retrain99|oracle_retrain99|forget01"
  # "oracle_retrain99|oracle_retrain99|retain99"
)

NUM_JOBS=${#jobs[@]}
NUM_EPOCHS=${#epoch_list[@]}
TOTAL=$((NUM_JOBS * NUM_EPOCHS))

IDX=${SLURM_ARRAY_TASK_ID:-0}

# 因为 #SBATCH --array 是静态的，如果 EPOCH_LIST 被缩短，超出的 array task 直接退出。
if [ "$IDX" -ge "$TOTAL" ]; then
  echo "[eval-ft] IDX=$IDX >= TOTAL=$TOTAL, skip."
  exit 0
fi

job_idx=$((IDX % NUM_JOBS))
epoch_idx=$((IDX / NUM_JOBS))
EPOCHS=${epoch_list[$epoch_idx]}

IFS='|' read -r MODEL_TAG ADAPTER_TAG SPLIT <<< "${jobs[$job_idx]}"

PARAM_TAG="lr${LR}_WD${WEIGHT_DECAY}_loraRank${LORA_RANK}_loraDrop${LORA_DROPOUT}_GradStsp${GRAD_ACC_STEPS}"
adapter_suffix="${PARAM_TAG}/epoch-${EPOCHS}"
ADAPTER_DIR="$FT_ROOT/$ADAPTER_TAG/$adapter_suffix"

EVAL_DATA="tofu/processed/${SPLIT}.json"

# 结果按模型名 + 参数组集中存放。
OUT_DIR="$RESULTS_ROOT/$MODEL_TAG/$PARAM_TAG"
SUMMARY_FILENAME="epoch-${EPOCHS}-${SPLIT}-summary.json"
DETAILS_FILENAME="epoch-${EPOCHS}-${SPLIT}-details.jsonl"
EPOCH_CSV="$OUT_DIR/${SPLIT}-epoch_curve.csv"

echo "[eval-ft] IDX=$IDX job_idx=$job_idx epoch_idx=$epoch_idx"
echo "[eval-ft] MODEL_TAG=$MODEL_TAG ADAPTER_TAG=$ADAPTER_TAG SPLIT=$SPLIT EPOCH=$EPOCHS"
echo "[eval-ft] ADAPTER_DIR=$ADAPTER_DIR"
echo "[eval-ft] EVAL_DATA=$EVAL_DATA"
echo "[eval-ft] OUT_DIR=$OUT_DIR"
echo "[eval-ft] SUMMARY_FILENAME=$SUMMARY_FILENAME"
echo "[eval-ft] DETAILS_FILENAME=$DETAILS_FILENAME"
echo "[eval-ft] EPOCH_CSV=$EPOCH_CSV"
echo "[eval-ft] LIMIT=${LIMIT:-none}"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source "$HOME/ENV-3.10/bin/activate"
cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR" results/tofu/logs

if [ ! -d "$ADAPTER_DIR" ]; then
  echo "[eval-ft][ERROR] Adapter dir not found: $ADAPTER_DIR"
  exit 1
fi

if [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
  echo "[eval-ft][ERROR] adapter_config.json not found in: $ADAPTER_DIR"
  exit 1
fi

python tofu/evaluate_tofu.py \
  --base_model_name "$MODEL_NAME" \
  --adapter_dir "$ADAPTER_DIR" \
  --eval_data "$EVAL_DATA" \
  --split "$SPLIT" \
  --model_tag "$MODEL_TAG" \
  --epoch "$EPOCHS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --lora_rank "$LORA_RANK" \
  --lora_dropout "$LORA_DROPOUT" \
  --grad_acc_steps "$GRAD_ACC_STEPS" \
  --output_dir "$OUT_DIR" \
  --summary_filename "$SUMMARY_FILENAME" \
  --details_filename "$DETAILS_FILENAME" \
  --epoch_csv "$EPOCH_CSV" \
  --log_every "$LOG_EVERY" \
  --local_files_only \
  "${LIMIT_ARGS[@]}"