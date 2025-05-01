#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-00:10  # time (DD-HH:MM)
#SBATCH --output=./results_extra/recover-%a-%N-%j.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=snow.jar.13@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0-4

#SBATCH --array=0-29

r_list=(32)
epoch_list=(6 12 24)
lr_list=(0.001 0.0005)
method_list=('grad_ascent' 'grad_diff' 'KL' 'dpo' 'npo')

IDX=$SLURM_ARRAY_TASK_ID
r_idx=$((IDX / 30))
epoch_idx=$(((IDX / 10) % 3))
lr_idx=$(((IDX / 5) % 2))
method_idx=$((IDX % 5))

R=${r_list[$r_idx]}
LR=${lr_list[$lr_idx]}
EPOCHS=${epoch_list[$epoch_idx]}
METHOD=${method_list[$method_idx]}

echo "🔧 当前配置: LoRA rank=$R | epochs=$EPOCHS | lr=$LR | method=$METHOD"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate

python recovery.py  --datasetType "forget-attr"  --modelType 'unlearned'  --unlearn_method  $METHOD --num_fgt 2025  --lr_fgt $LR  --LoRA_rank_fgt $R --eps_fgt $EPOCHS --reg_weights_fgt 1.0  --flip_logit 1
python recovery.py  --datasetType "forget-attr"  --modelType 'unlearned'  --unlearn_method  $METHOD --num_fgt 2025  --lr_fgt $LR  --LoRA_rank_fgt $R --eps_fgt $EPOCHS --reg_weights_fgt 1.0  --flip_logit 0