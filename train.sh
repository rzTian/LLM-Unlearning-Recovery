#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --nodes=1                # Request 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=498G        # memory per node
#SBATCH --time=00-01:10  # time (DD-HH:MM)
#SBATCH --output=./results_extra/finetune-%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-user=***REMOVED-EMAIL***
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0

r=(32)


module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate

accelerate launch --multi_gpu Finetune.py --lr 0.001  --weight_decay 0.0  --LoRA_rank ${r[$SLURM_ARRAY_TASK_ID]}  --lora_dropout 0.0
