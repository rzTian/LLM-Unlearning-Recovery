#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=498G
#SBATCH --time=00-03:00
#SBATCH --output=./results/tofu/logs/train-%j-%a-%N.out
#SBATCH --job-name=tofu-train
#SBATCH --array=0-3

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery}
MODEL_NAME=${MODEL_NAME:-deepseek-ai/deepseek-llm-7b-chat}
MODEL_DIR=${MODEL_DIR:-results/tofu/adapters/ft}
LOG_DIR=${LOG_DIR:-results/tofu/logs/ft}
LR=${LR:-0.0002}
EPOCHS=${EPOCHS:-10}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
LORA_RANK=${LORA_RANK:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-40}

split_list=(full retain90 retain95 retain99)
tag_list=(target_full oracle_retrain90 oracle_retrain95 oracle_retrain99)

IDX=${SLURM_ARRAY_TASK_ID:-0}
SPLIT=${split_list[$IDX]}
TAG=${tag_list[$IDX]}
DATA="tofu/processed/${SPLIT}.json"

echo "[train] IDX=$IDX TAG=$TAG SPLIT=$SPLIT DATA=$DATA"
echo "[train] MODEL_NAME=$MODEL_NAME MODEL_DIR=$MODEL_DIR LOG_DIR=$LOG_DIR"
echo "[train] lr=$LR epochs=$EPOCHS wd=$WEIGHT_DECAY rank=$LORA_RANK dropout=$LORA_DROPOUT grad_acc=$GRAD_ACC_STEPS"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source "$HOME/ENV-3.10/bin/activate"
cd "$PROJECT_DIR"
mkdir -p "$MODEL_DIR/$TAG" "$LOG_DIR/$TAG" results/tofu/logs

accelerate launch --multi_gpu Finetune.py \
  --datasetName TOFU \
  --dataDIR "$DATA" \
  --model_name "$MODEL_NAME" \
  --logDIR "$LOG_DIR/$TAG" \
  --modelDIR "$MODEL_DIR/$TAG" \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --weight_decay "$WEIGHT_DECAY" \
  --LoRA_rank "$LORA_RANK" \
  --lora_dropout "$LORA_DROPOUT" \
  --grad_acc_steps "$GRAD_ACC_STEPS"
