#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-04:00  # time (DD-HH:MM)
#SBATCH --output=./results_extra/eval-base-%j-%a-%N.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=snow.jar.13@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=eval-base
#SBATCH --array=0

r=(6)

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate

python evaluate.py  --datasetType "common"  --modelType 'base'
# python evaluate.py  --datasetType "common"  --modelType 'learned'
# python evaluate.py  --datasetType "common"  --modelType 'unlearned' --unlearnSet "unlearn-N1-A1-sin" \
#     --unlearn_method 'npo' --lr_fgt 0.001 --eps_fgt 24 --reg_weights_fgt 1.0 --LoRA_rank_fgt 32

# python evaluate.py  --datasetType "train_t"  --modelType 'base'
# python evaluate.py  --datasetType "val_t"  --modelType 'base'
# python evaluate.py  --datasetType "train_t"  --modelType 'learned'  --lr 0.001 --weight_decay 0.0 --epochs 30 --LoRA_rank 32
# python evaluate.py  --datasetType "val_t"  --modelType 'learned'  --lr 0.001 --weight_decay 0.0 --epochs 30 --LoRA_rank 32

# python evaluate.py  --datasetType "forget-attr"  --modelType 'unlearned'  --unlearn_method "grad_ascent" --num_fgt 2025 --lr_fgt 0.0005 --eps_fgt ${r[$SLURM_ARRAY_TASK_ID]} --reg_weights_fgt 1.0
# python evaluate.py  --datasetType "forget-attr"  --modelType 'unlearned'  --unlearn_method "grad_diff" --num_fgt 2025 --lr_fgt 0.0005 --eps_fgt ${r[$SLURM_ARRAY_TASK_ID]} --reg_weights_fgt 1.0
# python evaluate.py  --datasetType "forget-attr"  --modelType 'unlearned'  --unlearn_method "KL" --num_fgt 2025 --lr_fgt 0.0005 --eps_fgt ${r[$SLURM_ARRAY_TASK_ID]} --reg_weights_fgt 1.0
# python evaluate.py  --datasetType "forget-attr"  --modelType 'unlearned'  --unlearn_method "dpo" --num_fgt 2025 --lr_fgt 0.0005 --eps_fgt ${r[$SLURM_ARRAY_TASK_ID]} --reg_weights_fgt 1.0
# python evaluate.py  --datasetType "forget-attr"  --modelType 'unlearned'  --unlearn_method "npo" --num_fgt 2025 --lr_fgt 0.0005 --eps_fgt ${r[$SLURM_ARRAY_TASK_ID]} --reg_weights_fgt 1.0

# python evaluate.py  --datasetType "retain-attr"  --modelType 'unlearned'  --unlearn_method "grad_ascent" --num_fgt 2025 --lr_fgt 0.0005 --eps_fgt ${r[$SLURM_ARRAY_TASK_ID]} --reg_weights_fgt 1.0
# python evaluate.py  --datasetType "retain-attr"  --modelType 'unlearned'  --unlearn_method "grad_diff" --num_fgt 2025 --lr_fgt 0.0005 --eps_fgt ${r[$SLURM_ARRAY_TASK_ID]} --reg_weights_fgt 1.0
# python evaluate.py  --datasetType "retain-attr"  --modelType 'unlearned'  --unlearn_method "KL" --num_fgt 2025 --lr_fgt 0.0005 --eps_fgt ${r[$SLURM_ARRAY_TASK_ID]} --reg_weights_fgt 1.0
# python evaluate.py  --datasetType "retain-attr"  --modelType 'unlearned'  --unlearn_method "dpo" --num_fgt 2025 --lr_fgt 0.0005 --eps_fgt ${r[$SLURM_ARRAY_TASK_ID]} --reg_weights_fgt 1.0
# python evaluate.py  --datasetType "retain-attr"  --modelType 'unlearned'  --unlearn_method "npo" --num_fgt 2025 --lr_fgt 0.0005 --eps_fgt ${r[$SLURM_ARRAY_TASK_ID]} --reg_weights_fgt 1.0
