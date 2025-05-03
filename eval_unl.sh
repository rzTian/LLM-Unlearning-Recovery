#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-00:40  # time (DD-HH:MM)
#SBATCH --output=./results_extra/eval-unl-%a-%N-%j.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=snow.jar.13@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0-59

r_list=(32)
epoch_list=(2 4 6 8 10 12 14 16 18 20 22 24)
lr_list=(0.001)
method_list=('grad_ascent' 'grad_diff' 'KL' 'dpo' 'npo')

IDX=$SLURM_ARRAY_TASK_ID
r_idx=$((IDX / 60))
epoch_idx=$(((IDX / 5) % 12))
lr_idx=$(((IDX / 5) % 1))
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

python evaluate.py  --datasetType "forget"  --unlearnSet "unlearn-N1"  --modelType 'unlearned' \
    --unlearn_method $METHOD --num_fgt 2025 --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt 1.0 --LoRA_rank_fgt $R
python evaluate.py  --datasetType "retain"  --unlearnSet "unlearn-N1"  --modelType 'unlearned' \
    --unlearn_method $METHOD --num_fgt 2025 --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt 1.0 --LoRA_rank_fgt $R