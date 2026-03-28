#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-01:30  # time (DD-HH:MM)
#SBATCH --output=./results_extra/eval-unl-%j-%a-%N.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=11

set_list=("unlearn-N5-A1-sin10")
quant_list=("none" "int8" "int4")
gs_list=(10 10 10 80 10)
r_list=(64 64 64 256 64)
reg_list=(5.0 5.0 5.0 2.0 5.0)
wd_list=(0.01 0.01 0.01 0.0 0.01)
lr_list=(0.0002 0.0002 0.0002 0.0005 0.0005)
method_list=('grad_ascent' 'grad_diff' 'KL' 'po' 'npo')
epoch_list=(120 100 100 40 20)

IDX=$SLURM_ARRAY_TASK_ID
set_idx=$(((IDX / 15) % 1))
quant_idx=$(((IDX / 5) % 3))
method_idx=$(((IDX / 1) % 5))

QUANT=${quant_list[$quant_idx]}
BETA=0.1
GS=${gs_list[$method_idx]}
R=${r_list[$method_idx]}
RG=${reg_list[$method_idx]}
WD=${wd_list[$method_idx]}
LR=${lr_list[$method_idx]}
METHOD=${method_list[$method_idx]}
EPOCHS=${epoch_list[$method_idx]}

UNLEARN_SET=${set_list[$set_idx]}
FORGET_TYPE="forget"
RETAIN_TYPE="retain_sfa"
REMAIN_TYPE="remain_sfa"

echo "🔧 当前配置: method=$METHOD"
echo "🔧 当前配置: lr=$LR | reg=$RG | WD=$WD | LoRA rank=$R | epochs=$EPOCHS | gs=$GS | beta=$BETA | quant=$QUANT"
echo "🔧 当前配置: Unlearn Set=$UNLEARN_SET | Forget Type=$FORGET_TYPE | Retain Type=$RETAIN_TYPE | Remain Type=$REMAIN_TYPE"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

python evaluate.py  --datasetType $FORGET_TYPE  --unlearnSet $UNLEARN_SET  --modelType 'unlearned' \
    --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --wd_fgt $WD --LoRA_rank_fgt $R --grad_acc_steps_fgt $GS --beta_fgt $BETA --quant $QUANT
python evaluate.py  --datasetType $RETAIN_TYPE  --unlearnSet $UNLEARN_SET  --modelType 'unlearned' \
    --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --wd_fgt $WD --LoRA_rank_fgt $R --grad_acc_steps_fgt $GS --beta_fgt $BETA --quant $QUANT
python evaluate.py  --datasetType $REMAIN_TYPE  --unlearnSet $UNLEARN_SET  --modelType 'unlearned' \
    --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --wd_fgt $WD --LoRA_rank_fgt $R --grad_acc_steps_fgt $GS --beta_fgt $BETA --quant $QUANT
# python evaluate.py  --datasetType "common"  --unlearnSet $UNLEARN_SET  --modelType 'unlearned' \
#     --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --wd_fgt $WD --LoRA_rank_fgt $R --grad_acc_steps_fgt $GS --beta_fgt $BETA --quant $QUANT