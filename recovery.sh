#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-00:40  # time (DD-HH:MM)
#SBATCH --output=./results_extra/recover-%N-%j-%a.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=snow.jar.13@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0-1

r=(0 1)

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate

python recovery.py  --unlearnSet "unlearn-N1" --datasetType "forget"  --modelType 'learned'  --flip_logit ${r[$SLURM_ARRAY_TASK_ID]} \
      --lr 0.001  --epochs 15 --weight_decay 0.0  --LoRA_rank 64  --lora_dropout 0.0
python recovery.py  --unlearnSet "unlearn-N1" --datasetType "retain"  --modelType 'learned'  --flip_logit ${r[$SLURM_ARRAY_TASK_ID]} \
      --lr 0.001  --epochs 15 --weight_decay 0.0  --LoRA_rank 64  --lora_dropout 0.0