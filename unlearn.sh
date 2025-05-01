#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --nodes=1                # Request 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=498G        # memory per node
#SBATCH --time=00-01:30  # time (DD-HH:MM)
#SBATCH --output=./results_extra/unlearn-%N-%j-%a.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=snow.jar.13@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0-1

r=(6 12)


module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate

accelerate launch --multi_gpu unlearn.py --unlearn_method  "grad_ascent"  --lr 0.0005  --weight_decay 0.0  --epochs ${r[$SLURM_ARRAY_TASK_ID]}  --reg_weights 1.0  --LoRA_rank 32  --lora_dropout 0.0
# accelerate launch --multi_gpu unlearn.py --unlearn_method  "grad_diff"  --lr 0.0005  --weight_decay 0.0  --epochs ${r[$SLURM_ARRAY_TASK_ID]}  --reg_weights 1.0  --LoRA_rank 32  --lora_dropout 0.0
# accelerate launch --multi_gpu unlearn.py --unlearn_method  "KL"  --lr 0.0005  --weight_decay 0.0  --epochs ${r[$SLURM_ARRAY_TASK_ID]}  --reg_weights 1.0  --LoRA_rank 32  --lora_dropout 0.0
# accelerate launch --multi_gpu unlearn.py --unlearn_method  "dpo"  --lr 0.0005  --weight_decay 0.0  --epochs ${r[$SLURM_ARRAY_TASK_ID]}  --reg_weights 1.0  --LoRA_rank 32  --lora_dropout 0.0
# accelerate launch --multi_gpu unlearn.py --unlearn_method  "npo"  --lr 0.0005  --weight_decay 0.0  --epochs ${r[$SLURM_ARRAY_TASK_ID]}  --reg_weights 1.0  --LoRA_rank 32  --lora_dropout 0.0