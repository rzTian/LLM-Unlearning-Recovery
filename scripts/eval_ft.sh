#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-01:00  # time (DD-HH:MM)
#SBATCH --output=./results_extra/eval-ft-com-%j-%a-%N.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=snow.jar.13@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=eval-ft-com
#SBATCH --array=0-41

r_list=(128)
wd_list=(0.0 0.01)
lr_list=(0.00001 0.0001 0.001)
epoch_list=(20 40 60 80 100 150 200)

IDX=$SLURM_ARRAY_TASK_ID
r_idx=$((IDX / 42))
wd_idx=$(((IDX / 21) % 2))
lr_idx=$(((IDX / 7) % 3))
epoch_idx=$((IDX % 7))

R=${r_list[$r_idx]}
WD=${wd_list[$wd_idx]}
LR=${lr_list[$lr_idx]}
EPOCHS=${epoch_list[$epoch_idx]}

echo "🔧 当前配置: epochs=$EPOCHS | lr=$LR | wd=$WD | LoRA rank=$R | common"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

# python evaluate.py  --datasetType "train_t"  --modelType 'learned' \
#     --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0
# python evaluate.py  --datasetType "val_t"  --modelType 'learned' \
#     --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0
python evaluate.py  --datasetType "common"  --modelType 'learned' \
    --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0