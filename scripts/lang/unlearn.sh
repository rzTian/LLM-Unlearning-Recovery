#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=00-03:00
#SBATCH --output=./results_extra/lang-unlearn-%j-%a-%N.out
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --job-name=lang-unlearn
#SBATCH --array=0

set -euo pipefail

beta_list=(0.1)
r_list=(256)
rg_list=(5.0)
wd_list=(0.01)
lr_list=(0.0002)
set_list=("unlearn-N20-A1-yrb" "unlearn-N20-A1-bld" "unlearn-N5-A1-pcd10" "unlearn-N5-A1-sin10")
method_list=("langevin" "dp_random_label" "noisy_grad_diff")

IDX=${SLURM_ARRAY_TASK_ID:-0}
method_idx=$((IDX % 2))
set_idx=$(((IDX / 2) % 4))
lr_idx=$(((IDX / 12) % 1))

BETA=${beta_list[0]}
R=${r_list[0]}
RG=${rg_list[0]}
WD=${wd_list[0]}
LR=${lr_list[$lr_idx]}
METHOD=${method_list[$method_idx]}
EPOCHS=24
UNLEARN_SET=${set_list[$set_idx]}
FORGET_SET="forget.json"
RETAIN_SET="retain-same_fn_attr.json"
UNLEARN_GRAD_ACC=10

echo "method=$METHOD"
echo "unlearn_set=$UNLEARN_SET"
echo "lr=$LR"
echo "epochs=$EPOCHS"
echo "grad_acc_steps=$UNLEARN_GRAD_ACC"

echo "output_dir=unlearn_deepseek_7b_lang"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

python unlearn.py --model_name deepseek-ai/deepseek-llm-7b-chat \
  --finetune_model_DIR fine_tuned_deepseek_7b --logDIR unlearn_deepseek_7b_lang_log --unlearn_model_DIR unlearn_deepseek_7b_lang \
  --unlearnSet $UNLEARN_SET --forgetSetDir $FORGET_SET --retainSetDir $RETAIN_SET \
  --lr_ft 0.0005 --eps_ft 30 --wd_ft 0.01 --LoRA_rank_ft 256 --lora_dropout_ft 0.0 --grad_acc_steps_ft 40 \
  --unlearn_method $METHOD --lr $LR --epochs $EPOCHS --weight_decay $WD \
  --LoRA_rank $R --lora_dropout 0.0 --reg_weights $RG --grad_acc_steps $UNLEARN_GRAD_ACC --beta $BETA \
  --dp_random_label_use_retain