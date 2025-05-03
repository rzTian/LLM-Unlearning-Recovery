#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-00:40  # time (DD-HH:MM)
#SBATCH --output=./results_extra/eval-base-%a-%N-%j.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=snow.jar.13@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0

r=(6)

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate

python evaluate.py  --datasetType "train"  --modelType 'base'
python evaluate.py  --datasetType "val"  --modelType 'base'
python evaluate.py  --datasetType "forget"  --unlearnSet "unlearn-N1"  --modelType 'base'
python evaluate.py  --datasetType "retain"  --unlearnSet "unlearn-N1"  --modelType 'base'

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
