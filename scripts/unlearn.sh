#!/bin/bash
#SBATCH --account=<your-slurm-account>
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=498G
#SBATCH --time=00-01:20
#SBATCH --output=./results_extra/unlearn-%j-%N.out
#SBATCH --job-name=unlearn

set -euo pipefail

MODEL_NAME="deepseek-ai/deepseek-llm-7b-chat"
UNLEARN_SET="unlearn-N20-A1-yrb"
FORGET_SET="forget.json"
RETAIN_SET="retain-same_fn_attr.json"
FT_MODEL_DIR="fine_tuned_deepseek_7b"
LOG_DIR="unlearn_deepseek_7b_log"
UNLEARN_MODEL_DIR="unlearn_deepseek_7b"
METHOD="grad_diff"
LR="0.0001"
EPOCHS="50"
WEIGHT_DECAY="0.01"
REG_WEIGHT="5.0"
BETA="0.1"
LORA_RANK="256"
LORA_DROPOUT="0.0"
GRAD_ACC_STEPS="80"

FT_LR="0.0005"
FT_EPOCHS="30"
FT_WEIGHT_DECAY="0.01"
FT_LORA_RANK="256"
FT_LORA_DROPOUT="0.0"
FT_GRAD_ACC_STEPS="40"

echo "Unlearn: method=${METHOD}, set=${UNLEARN_SET}"
echo "Config: lr=${LR}, epochs=${EPOCHS}, wd=${WEIGHT_DECAY}, reg=${REG_WEIGHT}, rank=${LORA_RANK}"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source "$HOME/ENV-3.10/bin/activate"
cd "<path-to-repo>"  # e.g. "$HOME/LLM-Unlearning-Recovery"
mkdir -p results_extra

accelerate launch --multi_gpu unlearn.py \
  --model_name "$MODEL_NAME" \
  --finetune_model_DIR "$FT_MODEL_DIR" \
  --logDIR "$LOG_DIR" \
  --unlearn_model_DIR "$UNLEARN_MODEL_DIR" \
  --unlearnSet "$UNLEARN_SET" \
  --forgetSetDir "$FORGET_SET" \
  --retainSetDir "$RETAIN_SET" \
  --lr_ft "$FT_LR" \
  --eps_ft "$FT_EPOCHS" \
  --wd_ft "$FT_WEIGHT_DECAY" \
  --LoRA_rank_ft "$FT_LORA_RANK" \
  --lora_dropout_ft "$FT_LORA_DROPOUT" \
  --grad_acc_steps_ft "$FT_GRAD_ACC_STEPS" \
  --unlearn_method "$METHOD" \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --weight_decay "$WEIGHT_DECAY" \
  --LoRA_rank "$LORA_RANK" \
  --lora_dropout "$LORA_DROPOUT" \
  --reg_weights "$REG_WEIGHT" \
  --grad_acc_steps "$GRAD_ACC_STEPS" \
  --beta "$BETA"
