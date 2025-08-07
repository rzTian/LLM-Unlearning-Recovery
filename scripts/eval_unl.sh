#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-00:50  # time (DD-HH:MM)
#SBATCH --output=./results_extra/eval-unl-%a-%N-%j.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0-179

r_list=(32)
epoch_list=(2 4 6 8 10 12 14 16 18 20 22 24)
lr_list=(0.001)
rg_list=(1.0)
method_list=('grad_ascent' 'grad_diff' 'KL' 'dpo' 'npo')
set_list=("unlearn-N1-A1-bt" "unlearn-N1-A1-sin" "unlearn-N1-A1-pc")

IDX=$SLURM_ARRAY_TASK_ID
r_idx=$(((IDX / 60) % 1))
epoch_idx=$(((IDX / 15) % 12))
lr_idx=$(((IDX / 5) % 1))
rg_idx=$(((IDX / 5) % 1))
method_idx=$((IDX % 5))
set_idx=$(((IDX / 5) % 3))

R=${r_list[$r_idx]}
LR=${lr_list[$lr_idx]}
RG=${rg_list[$rg_idx]}
EPOCHS=${epoch_list[$epoch_idx]}
METHOD=${method_list[$method_idx]}

UNLEARN_SET=${set_list[$set_idx]}
FORGET_TYPE="forget"
RETAIN_TYPE="retain_sfa"
REMAIN_TYPE="remain_sfa"

echo "🔧 当前配置: LoRA rank=$R | epochs=$EPOCHS | lr=$LR | reg=$RG | method=$METHOD"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

python evaluate.py  --datasetType $FORGET_TYPE  --unlearnSet $UNLEARN_SET  --modelType 'unlearned' \
    --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --LoRA_rank_fgt $R
python evaluate.py  --datasetType $RETAIN_TYPE  --unlearnSet $UNLEARN_SET  --modelType 'unlearned' \
    --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --LoRA_rank_fgt $R
python evaluate.py  --datasetType $REMAIN_TYPE  --unlearnSet $UNLEARN_SET  --modelType 'unlearned' \
    --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --LoRA_rank_fgt $R