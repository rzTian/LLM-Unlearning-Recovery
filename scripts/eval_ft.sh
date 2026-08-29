#!/bin/bash
#SBATCH --account=<your-slurm-account>
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=00-01:00
#SBATCH --output=./results_extra/eval-ft-%j-%N.out
#SBATCH --job-name=eval-ft

set -euo pipefail

MODEL_NAME="deepseek-ai/deepseek-llm-7b-chat"
UNLEARN_SET="unlearn-N20-A1-yrb"
LOG_DIR="fine_tuned_deepseek_7b_log"
MODEL_DIR="fine_tuned_deepseek_7b"
LR="0.0005"
EPOCHS="30"
WEIGHT_DECAY="0.01"
LORA_RANK="256"
LORA_DROPOUT="0.0"
GRAD_ACC_STEPS="40"

echo "Evaluate finetuned model: set=${UNLEARN_SET}"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source "$HOME/ENV-3.10/bin/activate"
cd "<path-to-repo>"  # e.g. "$HOME/LLM-Unlearning-Recovery"
mkdir -p results_extra

python evaluate.py \
  --model_name "$MODEL_NAME" \
  --logDIR "$LOG_DIR" \
  --modelDIR "$MODEL_DIR" \
  --unlearnSet "$UNLEARN_SET" \
  --datasetType "forget" \
  --modelType "learned" \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --weight_decay "$WEIGHT_DECAY" \
  --LoRA_rank "$LORA_RANK" \
  --lora_dropout "$LORA_DROPOUT" \
  --grad_acc_steps "$GRAD_ACC_STEPS"

python evaluate.py \
  --model_name "$MODEL_NAME" \
  --logDIR "$LOG_DIR" \
  --modelDIR "$MODEL_DIR" \
  --unlearnSet "$UNLEARN_SET" \
  --datasetType "retain" \
  --modelType "learned" \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --weight_decay "$WEIGHT_DECAY" \
  --LoRA_rank "$LORA_RANK" \
  --lora_dropout "$LORA_DROPOUT" \
  --grad_acc_steps "$GRAD_ACC_STEPS"

python evaluate.py \
  --model_name "$MODEL_NAME" \
  --logDIR "$LOG_DIR" \
  --modelDIR "$MODEL_DIR" \
  --unlearnSet "$UNLEARN_SET" \
  --datasetType "remain" \
  --modelType "learned" \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --weight_decay "$WEIGHT_DECAY" \
  --LoRA_rank "$LORA_RANK" \
  --lora_dropout "$LORA_DROPOUT" \
  --grad_acc_steps "$GRAD_ACC_STEPS"
