#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-01:00  # time (DD-HH:MM)
#SBATCH --output=./results_extra/eval-ft-%j-%a-%N.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=snow.jar.13@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=eval-ft
#SBATCH --array=0-59

r_list=(128 256 512)
wd_list=(0.0 0.01)
gs_list=(40)
lr_list=(0.0005 0.001)
epoch_list=(2 4 6 8 10 15 20 30 40 50)

IDX=$SLURM_ARRAY_TASK_ID
r_idx=$((IDX / 20))
wd_idx=$(((IDX / 10) % 2))
gs_idx=$(((IDX / 10) % 1))
lr_idx=$(((IDX / 10) % 1))
epoch_idx=$((IDX % 10))

R=${r_list[$r_idx]}
WD=${wd_list[$wd_idx]}
GS=${gs_list[$gs_idx]}
EPOCHS=${epoch_list[$epoch_idx]}
LR=${lr_list[$lr_idx]}

echo "🔧 当前配置: epochs=$EPOCHS | lr=$LR | wd=$WD | LoRA rank=$R | GS=$GS | common"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

# python evaluate.py  --datasetType "train_t"  --modelType 'learned' \
#     --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0  --grad_acc_steps $GS
# python evaluate.py  --datasetType "val_t"  --modelType 'learned' \
#     --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0  --grad_acc_steps $GS
python evaluate.py  --datasetType "common"  --modelType 'learned' \
    --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0  --grad_acc_steps $GS