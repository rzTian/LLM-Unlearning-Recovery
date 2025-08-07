#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-00:40  # time (DD-HH:MM)
#SBATCH --output=./results_extra/recovery-%a-%N-%j.out  # %N for node name, %j for jobID, %a for array ID
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

# python recovery.py  --unlearnSet "unlearn-N1" --datasetType "forget"  --modelType 'unlearned'  --flip_logit 0 --unlearn_method  'grad_ascent'  --lr_fgt 0.0005  --LoRA_rank_fgt 32 --eps_fgt 2 --reg_weights_fgt 1.0

echo "当前测试: Unlearn Set: $UNLEARN_SET | Forget Type: $FORGET_TYPE | Flip 0"
python recovery.py  --unlearnSet $UNLEARN_SET --datasetType $FORGET_TYPE  --modelType 'unlearned'  --flip_logit 0 \
      --unlearn_method  $METHOD  --lr_fgt $LR  --LoRA_rank_fgt $R --eps_fgt $EPOCHS --reg_weights_fgt $RG
echo "当前测试: Unlearn Set: $UNLEARN_SET | Forget Type: $FORGET_TYPE | Flip 1"
python recovery.py  --unlearnSet $UNLEARN_SET --datasetType $FORGET_TYPE  --modelType 'unlearned'  --flip_logit 1 \
      --unlearn_method  $METHOD  --lr_fgt $LR  --LoRA_rank_fgt $R --eps_fgt $EPOCHS --reg_weights_fgt $RG

echo "当前测试: Unlearn Set: $UNLEARN_SET | Retain Type: $RETAIN_TYPE | Flip 0"
python recovery.py  --unlearnSet $UNLEARN_SET --datasetType $RETAIN_TYPE  --modelType 'unlearned'  --flip_logit 0 \
      --unlearn_method  $METHOD  --lr_fgt $LR  --LoRA_rank_fgt $R --eps_fgt $EPOCHS --reg_weights_fgt $RG
# echo "当前测试: Unlearn Set: $UNLEARN_SET | Retain Type: $RETAIN_TYPE | Flip 1"
# python recovery.py  --unlearnSet $UNLEARN_SET --datasetType $RETAIN_TYPE  --modelType 'unlearned'  --flip_logit 1 \
#       --unlearn_method  $METHOD   --lr_fgt $LR  --LoRA_rank_fgt $R --eps_fgt $EPOCHS --reg_weights_fgt $RG