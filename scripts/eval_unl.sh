#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=00-01:30
#SBATCH --output=./results_extra/eval-unl-%j-%N.out
#SBATCH --job-name=eval-unl

set -euo pipefail

MODEL_NAME="deepseek-ai/deepseek-llm-7b-chat"
UNLEARN_SET="unlearn-N20-A1-yrb"
LOG_DIR_FT="fine_tuned_deepseek_7b_log"
MODEL_DIR_FT="fine_tuned_deepseek_7b"
LOG_DIR_UNL="unlearn_deepseek_7b_log"
MODEL_DIR_UNL="unlearn_deepseek_7b"
METHOD="grad_diff"
QUANT="none"

FT_LR="0.0005"
FT_EPOCHS="30"
FT_WEIGHT_DECAY="0.01"
FT_LORA_RANK="256"
FT_LORA_DROPOUT="0.0"
FT_GRAD_ACC_STEPS="40"

UNL_LR="0.0001"
UNL_EPOCHS="50"
UNL_WEIGHT_DECAY="0.01"
UNL_REG_WEIGHT="5.0"
UNL_BETA="0.1"
UNL_LORA_RANK="256"
UNL_GRAD_ACC_STEPS="80"

echo "Evaluate unlearned model: method=${METHOD}, set=${UNLEARN_SET}, quant=${QUANT}"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source "$HOME/ENV-3.10/bin/activate"
cd "$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery"
mkdir -p results_extra

python evaluate.py \
  --model_name "$MODEL_NAME" \
  --logDIR "$LOG_DIR_FT" \
  --modelDIR "$MODEL_DIR_FT" \
  --logDIR_fgt "$LOG_DIR_UNL" \
  --modelDIR_fgt "$MODEL_DIR_UNL" \
  --unlearnSet "$UNLEARN_SET" \
  --modelType "unlearned" \
  --lr "$FT_LR" \
  --epochs "$FT_EPOCHS" \
  --weight_decay "$FT_WEIGHT_DECAY" \
  --LoRA_rank "$FT_LORA_RANK" \
  --lora_dropout "$FT_LORA_DROPOUT" \
  --grad_acc_steps "$FT_GRAD_ACC_STEPS" \
  --unlearn_method "$METHOD" \
  --lr_fgt "$UNL_LR" \
  --eps_fgt "$UNL_EPOCHS" \
  --reg_weights_fgt "$UNL_REG_WEIGHT" \
  --wd_fgt "$UNL_WEIGHT_DECAY" \
  --LoRA_rank_fgt "$UNL_LORA_RANK" \
  --grad_acc_steps_fgt "$UNL_GRAD_ACC_STEPS" \
  --beta_fgt "$UNL_BETA" \
  --quant "$QUANT" \
  --datasetType "forget"

python evaluate.py \
  --model_name "$MODEL_NAME" \
  --logDIR "$LOG_DIR_FT" \
  --modelDIR "$MODEL_DIR_FT" \
  --logDIR_fgt "$LOG_DIR_UNL" \
  --modelDIR_fgt "$MODEL_DIR_UNL" \
  --unlearnSet "$UNLEARN_SET" \
  --modelType "unlearned" \
  --lr "$FT_LR" \
  --epochs "$FT_EPOCHS" \
  --weight_decay "$FT_WEIGHT_DECAY" \
  --LoRA_rank "$FT_LORA_RANK" \
  --lora_dropout "$FT_LORA_DROPOUT" \
  --grad_acc_steps "$FT_GRAD_ACC_STEPS" \
  --unlearn_method "$METHOD" \
  --lr_fgt "$UNL_LR" \
  --eps_fgt "$UNL_EPOCHS" \
  --reg_weights_fgt "$UNL_REG_WEIGHT" \
  --wd_fgt "$UNL_WEIGHT_DECAY" \
  --LoRA_rank_fgt "$UNL_LORA_RANK" \
  --grad_acc_steps_fgt "$UNL_GRAD_ACC_STEPS" \
  --beta_fgt "$UNL_BETA" \
  --quant "$QUANT" \
  --datasetType "retain_sfa"

python evaluate.py \
  --model_name "$MODEL_NAME" \
  --logDIR "$LOG_DIR_FT" \
  --modelDIR "$MODEL_DIR_FT" \
  --logDIR_fgt "$LOG_DIR_UNL" \
  --modelDIR_fgt "$MODEL_DIR_UNL" \
  --unlearnSet "$UNLEARN_SET" \
  --modelType "unlearned" \
  --lr "$FT_LR" \
  --epochs "$FT_EPOCHS" \
  --weight_decay "$FT_WEIGHT_DECAY" \
  --LoRA_rank "$FT_LORA_RANK" \
  --lora_dropout "$FT_LORA_DROPOUT" \
  --grad_acc_steps "$FT_GRAD_ACC_STEPS" \
  --unlearn_method "$METHOD" \
  --lr_fgt "$UNL_LR" \
  --eps_fgt "$UNL_EPOCHS" \
  --reg_weights_fgt "$UNL_REG_WEIGHT" \
  --wd_fgt "$UNL_WEIGHT_DECAY" \
  --LoRA_rank_fgt "$UNL_LORA_RANK" \
  --grad_acc_steps_fgt "$UNL_GRAD_ACC_STEPS" \
  --beta_fgt "$UNL_BETA" \
  --quant "$QUANT" \
  --datasetType "remain_sfa"
