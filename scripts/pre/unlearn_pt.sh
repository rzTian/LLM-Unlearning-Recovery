#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-01:00
#SBATCH --output=./results_extra/unlearn-pt-%j-%a-%N.out
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=unlearn-pt
#SBATCH --array=8-15

# ===== unlearning hyperparams =====
r_list=(0)
rg_list=(1.0)
wd_list=(0.01 0.0)
lr_list=(1e-4)
set_list=("unlearn-N5-A1-yrb10" "unlearn-N5-A1-bld10" "unlearn-N5-A1-pcd10" "unlearn-N5-A1-sin10")
method_list=("grad_ascent" "grad_diff" "KL" "npo")

IDX=$SLURM_ARRAY_TASK_ID
r_idx=$(((IDX / 16) % 1))
rg_idx=$(((IDX / 16) % 1))
wd_idx=$(((IDX / 8) % 2))
lr_idx=$(((IDX / 8) % 1))
set_idx=$(((IDX / 4) % 2))
method_idx=$(((IDX % 4)))

R=${r_list[$r_idx]}
RG=${rg_list[$rg_idx]}
WD=${wd_list[$wd_idx]}
LR=${lr_list[$lr_idx]}
METHOD=${method_list[$method_idx]}
EPOCHS=50

UNLEARN_SET=${set_list[$set_idx]}
FORGET_SET="forget.json"
RETAIN_SET="retain-same_fn_attr.json"

# echo "🔧 PT source: epoch=$SRC_EPOCH | lr=$SRC_LR | wd=$SRC_WD | rank=$SRC_R | GS=$SRC_GS"
echo "🔧 Unlearn  : method=$METHOD | epoch=$EPOCHS | lr=$LR | wd=$WD | rank=$R | reg=$RG"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

python unlearn.py --source_model_type pt --model_name gpt2 \
  --finetune_model_DIR pretrain_gpt2 --logDIR unlearn_gpt2_pt_log --unlearn_model_DIR unlearn_gpt2_pt \
  --unlearnSet $UNLEARN_SET --forgetSetDir $FORGET_SET --retainSetDir $RETAIN_SET \
  --lr_ft 0.0001  --eps_ft 45 --wd_ft 0.0  --LoRA_rank_ft 0  --lora_dropout_ft 0.0 --grad_acc_steps_ft 40 \
  --unlearn_method $METHOD --lr $LR --epochs $EPOCHS --weight_decay $WD \
  --LoRA_rank $R --lora_dropout 0.0 --reg_weights $RG --grad_acc_steps 80