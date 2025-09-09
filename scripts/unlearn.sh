#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --nodes=1                # Request 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=498G        # memory per node
#SBATCH --time=00-01:00  # time (DD-HH:MM)
#SBATCH --output=./results_extra/unlearn-%j-%a-%N.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0-47

rg_list=(1.0)
r_list=(32)
gs_list=(10)
lr_list=(0.001)
epoch_list=(25)
set_list=("unlearn-N1-A1-bld" "unlearn-N1-A1-pcd" "unlearn-N1-A1-sin" "unlearn-N1-A1-yrb")
retain_list=("retain-same_attr.json" "retain-same_fn_attr.json")
method_list=('grad_ascent' 'grad_diff' 'KL' 'po' 'dpo' 'npo')

IDX=$SLURM_ARRAY_TASK_ID
rg_idx=$(((IDX / 48) % 1))
r_idx=$(((IDX / 48) % 1))
gs_idx=$(((IDX / 48) % 1))
lr_idx=$(((IDX / 48) % 1))
epoch_idx=$(((IDX / 48) % 1))
retain_idx=$(((IDX / 12) % 4))
set_idx=$(((IDX / 6) % 2))
method_idx=$((IDX % 6))

RG=${rg_list[$rg_idx]}
R=${r_list[$r_idx]}
GS=${gs_list[$gs_idx]}
LR=${lr_list[$lr_idx]}
EPOCHS=${epoch_list[$epoch_idx]}
METHOD=${method_list[$method_idx]}

UNLEARN_SET=${set_list[$set_idx]}
FORGET_SET="forget.json"
RETAIN_SET=${retain_list[$retain_idx]}

echo "🔧 当前配置: LoRA rank=$R | reg=$RG | gradstep=$GS | lr=$LR | epochs=$EPOCHS | method=$METHOD"
echo "🔧 当前配置: Unlearn Set=$UNLEARN_SET | Forget Set=$FORGET_SET | Retain Set=$RETAIN_SET"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

accelerate launch --multi_gpu unlearn.py \
    --unlearnSet $UNLEARN_SET --forgetSetDir $FORGET_SET --retainSetDir $RETAIN_SET \
    --lr_ft 0.001  --eps_ft 15 --wd_ft 0.0  --LoRA_rank_ft 128  --lora_dropout_ft 0.0 \
    --unlearn_method $METHOD  --lr $LR  --epochs $EPOCHS  --weight_decay 0.0 \
    --LoRA_rank $R  --lora_dropout 0.0  --reg_weights $RG --grad_acc_steps $GS