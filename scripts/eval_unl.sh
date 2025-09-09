#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-00:50  # time (DD-HH:MM)
#SBATCH --output=./results_extra/eval-unl-%j-%a-%N.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0-143

rg_list=(1.0)
r_list=(32 64 128 156)
gs_list=(4 8 16 32 64 128)
lr_list=(0.001 0.0005 0.0001 0.00005 0.00001)
epoch_list=(2 4 6 8 10 12 14 16)
set_list=("unlearn-N1-A1-sin" "unlearn-N1-A1-pcd" "unlearn-N1-A1-bld" "unlearn-N1-A1-yrb")
method_list=('grad_diff' 'KL' 'po' 'dpo' 'npo' 'grad_ascent')

IDX=$SLURM_ARRAY_TASK_ID
rg_idx=$((IDX / 960))
r_idx=$(((IDX / 240) % 4))
ga_idx=$(((IDX / 40) % 6))
lr_idx=$(((IDX / 8) % 5))
epoch_idx=$(((IDX / 1) % 8))
set_idx=$(((IDX / 1) % 1))
method_idx=$((IDX % 1))

RG=${rg_list[$rg_idx]}
R=${r_list[$r_idx]}
GS=${gs_list[$gs_idx]}
LR=${lr_list[$lr_idx]}
EPOCHS=${epoch_list[$epoch_idx]}
METHOD=${method_list[$method_idx]}

UNLEARN_SET=${set_list[$set_idx]}
FORGET_TYPE="forget"
RETAIN_TYPE="retain_sa"
REMAIN_TYPE="remain_sa"

echo "🔧 当前配置: LoRA rank=$R | reg=$RG | gradstep=$GS | lr=$LR | epochs=$EPOCHS | method=$METHOD"
echo "🔧 当前配置: Unlearn Set=$UNLEARN_SET | Forget Type=$FORGET_TYPE | Retain Type=$RETAIN_TYPE | Remain Type=$REMAIN_TYPE"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

python evaluate.py  --datasetType "common"  --unlearnSet $UNLEARN_SET  --modelType 'unlearned' \
    --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --LoRA_rank_fgt $R --grad_acc_steps_fgt $GS
python evaluate.py  --datasetType $FORGET_TYPE  --unlearnSet $UNLEARN_SET  --modelType 'unlearned' \
    --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --LoRA_rank_fgt $R --grad_acc_steps_fgt $GS
python evaluate.py  --datasetType $RETAIN_TYPE  --unlearnSet $UNLEARN_SET  --modelType 'unlearned' \
    --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --LoRA_rank_fgt $R --grad_acc_steps_fgt $GS
python evaluate.py  --datasetType $REMAIN_TYPE  --unlearnSet $UNLEARN_SET  --modelType 'unlearned' \
    --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --LoRA_rank_fgt $R --grad_acc_steps_fgt $GS