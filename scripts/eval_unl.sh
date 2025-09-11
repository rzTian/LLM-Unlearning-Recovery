#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-00:10  # time (DD-HH:MM)
#SBATCH --output=./results_extra/eval-unl-%j-%a-%N.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0-103

rg_list=(5.0 10.0)
r_list=(256)
wd_list=(0.0 0.01)
lr_list=(0.0001 0.00007)
set_list=("unlearn-N20-A1-yrb" "unlearn-N20-A1-bld" "unlearn-N20-A1-pcd" "unlearn-N20-A1-sin")
method_list=('grad_diff' 'KL' 'po' 'dpo' 'npo' 'grad_ascent')
epoch_list=(2 6 10 14 18 22 26 30 34 38 42 46 50)

IDX=$SLURM_ARRAY_TASK_ID
rg_idx=$((IDX / 52))
r_idx=$(((IDX / 52) % 1))
wd_idx=$(((IDX / 26) % 2))
lr_idx=$(((IDX / 13) % 2))
set_idx=$(((IDX / 13) % 1))
method_idx=$(((IDX / 13) % 1))
epoch_idx=$((IDX % 13))

RG=${rg_list[$rg_idx]}
R=${r_list[$r_idx]}
WD=${wd_list[$wd_idx]}
LR=${lr_list[$lr_idx]}
METHOD=${method_list[$method_idx]}
EPOCHS=${epoch_list[$epoch_idx]}

UNLEARN_SET=${set_list[$set_idx]}
FORGET_TYPE="forget"
RETAIN_TYPE="retain_sfa"
REMAIN_TYPE="remain_sfa"

echo "🔧 当前配置: method=$METHOD"
echo "🔧 当前配置: lr=$LR | reg=$RG | WD=$WD | LoRA rank=$R | epochs=$EPOCHS"
echo "🔧 当前配置: Unlearn Set=$UNLEARN_SET | Forget Type=$FORGET_TYPE | Retain Type=$RETAIN_TYPE | Remain Type=$REMAIN_TYPE"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

python evaluate.py  --datasetType $FORGET_TYPE  --unlearnSet $UNLEARN_SET  --modelType 'unlearned' \
    --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --wd_fgt $WD --LoRA_rank_fgt $R --grad_acc_steps_fgt 10
python evaluate.py  --datasetType $RETAIN_TYPE  --unlearnSet $UNLEARN_SET  --modelType 'unlearned' \
    --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --wd_fgt $WD --LoRA_rank_fgt $R --grad_acc_steps_fgt 10
python evaluate.py  --datasetType $REMAIN_TYPE  --unlearnSet $UNLEARN_SET  --modelType 'unlearned' \
    --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --wd_fgt $WD --LoRA_rank_fgt $R --grad_acc_steps_fgt 10
# python evaluate.py  --datasetType "common"  --unlearnSet $UNLEARN_SET  --modelType 'unlearned' \
#     --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --LoRA_rank_fgt $R --grad_acc_steps_fgt $GS