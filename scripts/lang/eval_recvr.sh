#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=00-01:00
#SBATCH --output=./results_extra/lang-eval-recvr-%j-%a-%N.out
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --job-name=lang-eval-recvr
#SBATCH --array=0-47

set -euo pipefail

beta_list=(0.1)
r_list=(256)
rg_list=(5.0)
wd_list=(0.01)
lr_list=(0.0002)
recvr_list=('beam')
set_list=("unlearn-N20-A1-yrb" "unlearn-N20-A1-bld" "unlearn-N5-A1-pcd10" "unlearn-N5-A1-sin10")
method_list=("langevin" "dp_random_label" "noisy_grad_diff")
epoch_list=(4 8 12 16 20 24)

IDX=${SLURM_ARRAY_TASK_ID:-0}
base_idx=$((IDX / 6))
epoch_idx=$((IDX % 6))

method_idx=$((base_idx % 2))
set_idx=$(((base_idx / 2) % 4))
lr_idx=$(((base_idx / 12) % 1))

BETA=${beta_list[0]}
R=${r_list[0]}
RG=${rg_list[0]}
WD=${wd_list[0]}
LR=${lr_list[$lr_idx]}
EPOCHS=${epoch_list[$epoch_idx]}
METHOD=${method_list[$method_idx]}

UNLEARN_SET=${set_list[$set_idx]}
FORGET_TYPE="forget"
RECOVER_TYPE=${recvr_list[0]}
BEAM_K=10
BEAM_C=30
BEAM_N=1

MDL_NAME="deepseek-ai/deepseek-llm-7b-chat"
LOG_DIR_FT="fine_tuned_deepseek_7b_log"
MDL_DIR_FT="fine_tuned_deepseek_7b"
LOG_DIR_UNL="unlearn_deepseek_7b_lang_log"
MDL_DIR_UNL="unlearn_deepseek_7b_lang"
LOG_DIR_RCV="recovery_lang_results"

echo "method=$METHOD | set=$UNLEARN_SET | lr=$LR | epoch=$EPOCHS | recover=$RECOVER_TYPE"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

echo "Run flip=0"
python recovery.py --model_name $MDL_NAME --logDIR $LOG_DIR_FT --modelDIR $MDL_DIR_FT --logDIR_fgt $LOG_DIR_UNL --modelDIR_fgt $MDL_DIR_UNL --logDIR_recvr $LOG_DIR_RCV \
  --unlearnSet $UNLEARN_SET --datasetType $FORGET_TYPE --modelType unlearned \
  --recover_type $RECOVER_TYPE --flip 0 --K $BEAM_K --C $BEAM_C --N $BEAM_N \
  --lr 0.0005 --epochs 30 --weight_decay 0.01 --LoRA_rank 256 --lora_dropout 0.0 --grad_acc_steps 40 \
  --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --wd_fgt $WD --LoRA_rank_fgt $R --grad_acc_steps_fgt 10 --beta_fgt $BETA

echo "Run flip=1"
python recovery.py --model_name $MDL_NAME --logDIR $LOG_DIR_FT --modelDIR $MDL_DIR_FT --logDIR_fgt $LOG_DIR_UNL --modelDIR_fgt $MDL_DIR_UNL --logDIR_recvr $LOG_DIR_RCV \
  --unlearnSet $UNLEARN_SET --datasetType $FORGET_TYPE --modelType unlearned \
  --recover_type $RECOVER_TYPE --flip 1 --K $BEAM_K --C $BEAM_C --N $BEAM_N \
  --lr 0.0005 --epochs 30 --weight_decay 0.01 --LoRA_rank 256 --lora_dropout 0.0 --grad_acc_steps 40 \
  --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --wd_fgt $WD --LoRA_rank_fgt $R --grad_acc_steps_fgt 10 --beta_fgt $BETA
