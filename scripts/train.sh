#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --nodes=1                # Request 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=498G        # memory per node
#SBATCH --time=00-0:10  # time (DD-HH:MM)
#SBATCH --output=./results_extra/finetune-128-%j-%a-%N.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=snow.jar.13@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0

r_list=(128)
wd_list=(0.0 0.01)
epoch_list=(50)
lr_list=(0.0005 0.001 0.005 0.01)

IDX=$SLURM_ARRAY_TASK_ID
r_idx=$((IDX / 8))
wd_idx=$(((IDX / 4) % 2))
epoch_idx=$(((IDX / 4) % 1))
lr_idx=$((IDX % 4))

R=${r_list[$r_idx]}
WD=${wd_list[$wd_idx]}
EPOCHS=${epoch_list[$epoch_idx]}
LR=${lr_list[$lr_idx]}

echo "🔧 Finetune 当前配置: LoRA rank=$R | wd=$WD | epochs=$EPOCHS | lr=$LR"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

accelerate launch --multi_gpu Finetune.py --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0