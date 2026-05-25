#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=498G
#SBATCH --time=00-03:00
#SBATCH --output=./results/tofu/logs/unlearn-%j-%a-%N.out
#SBATCH --job-name=tofu-unlearn
#SBATCH --array=0-31

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery}
MODEL_NAME=${MODEL_NAME:-deepseek-ai/deepseek-llm-7b-chat}
FT_ROOT=${FT_ROOT:-results/tofu/adapters/ft/target_full}
UNL_ROOT=${UNL_ROOT:-results/tofu/adapters/unlearned}
LOG_DIR=${LOG_DIR:-results/tofu/logs/unlearned}

UNLEARN_SET=${UNLEARN_SET:-tofu_fgt10_ret90}
PAIR_DIR=${PAIR_DIR:-tofu/processed/$UNLEARN_SET}
FORGET_SET=${FORGET_SET:-$PAIR_DIR/forget.json}
RETAIN_SET=${RETAIN_SET:-$PAIR_DIR/retain.json}

LR_FT=${LR_FT:-0.0002}
EPS_FT=${EPS_FT:-10}
WD_FT=${WD_FT:-0.01}
LORA_RANK_FT=${LORA_RANK_FT:-128}
LORA_DROPOUT_FT=${LORA_DROPOUT_FT:-0.0}
GRAD_ACC_STEPS_FT=${GRAD_ACC_STEPS_FT:-40}

method_list=(grad_diff KL grad_ascent npo)
lr_list=(0.00001 0.00005)
epoch_list=(1 3 5 10)
REG=${REG:-1.0}
BETA=${BETA:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
LORA_RANK=${LORA_RANK:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-40}

IDX=${SLURM_ARRAY_TASK_ID:-0}
method_idx=$((IDX / 8))
lr_idx=$(((IDX / 4) % 2))
epoch_idx=$((IDX % 4))
METHOD=${method_list[$method_idx]}
LR=${lr_list[$lr_idx]}
EPOCHS=${epoch_list[$epoch_idx]}

echo "[unlearn] IDX=$IDX method_idx=$method_idx lr_idx=$lr_idx epoch_idx=$epoch_idx"
echo "[unlearn] method=$METHOD lr=$LR epochs=$EPOCHS reg=$REG beta=$BETA wd=$WEIGHT_DECAY rank=$LORA_RANK grad_acc=$GRAD_ACC_STEPS"
echo "[unlearn] target_ft_root=$FT_ROOT ft_epoch=$EPS_FT unlearn_set=$UNLEARN_SET"
echo "[unlearn] forget=$FORGET_SET retain=$RETAIN_SET output_root=$UNL_ROOT"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source "$HOME/ENV-3.10/bin/activate"
cd "$PROJECT_DIR"
mkdir -p "$UNL_ROOT" "$LOG_DIR" results/tofu/logs

accelerate launch --multi_gpu unlearn.py \
  --datasetName TOFU \
  --model_name "$MODEL_NAME" \
  --finetune_model_DIR "$FT_ROOT" \
  --logDIR "$LOG_DIR" \
  --unlearn_model_DIR "$UNL_ROOT" \
  --unlearnSet "$UNLEARN_SET" \
  --forgetSetDir "$FORGET_SET" \
  --retainSetDir "$RETAIN_SET" \
  --lr_ft "$LR_FT" \
  --eps_ft "$EPS_FT" \
  --wd_ft "$WD_FT" \
  --LoRA_rank_ft "$LORA_RANK_FT" \
  --lora_dropout_ft "$LORA_DROPOUT_FT" \
  --grad_acc_steps_ft "$GRAD_ACC_STEPS_FT" \
  --unlearn_method "$METHOD" \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --weight_decay "$WEIGHT_DECAY" \
  --LoRA_rank "$LORA_RANK" \
  --lora_dropout "$LORA_DROPOUT" \
  --reg_weights "$REG" \
  --grad_acc_steps "$GRAD_ACC_STEPS" \
  --beta "$BETA"
