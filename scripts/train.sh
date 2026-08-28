#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=498G
#SBATCH --time=00-02:59
#SBATCH --output=./results_extra/train-%j-%N.out
#SBATCH --job-name=train

set -euo pipefail

MODEL_NAME="deepseek-ai/deepseek-llm-7b-chat"
DATA_DIR="training_dataset.json"
LOG_DIR="fine_tuned_deepseek_7b_log"
MODEL_DIR="fine_tuned_deepseek_7b"
LR="0.0005"
EPOCHS="30"
WEIGHT_DECAY="0.01"
LORA_RANK="256"
LORA_DROPOUT="0.0"
GRAD_ACC_STEPS="40"

echo "Train: model=${MODEL_NAME}, data=${DATA_DIR}"
echo "Config: lr=${LR}, epochs=${EPOCHS}, wd=${WEIGHT_DECAY}, rank=${LORA_RANK}, grad_acc=${GRAD_ACC_STEPS}"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source "$HOME/ENV-3.10/bin/activate"
cd "$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery"
mkdir -p results_extra

accelerate launch --multi_gpu Finetune.py \
  --model_name "$MODEL_NAME" \
  --dataDIR "$DATA_DIR" \
  --logDIR "$LOG_DIR" \
  --modelDIR "$MODEL_DIR" \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --weight_decay "$WEIGHT_DECAY" \
  --LoRA_rank "$LORA_RANK" \
  --lora_dropout "$LORA_DROPOUT" \
  --grad_acc_steps "$GRAD_ACC_STEPS"
