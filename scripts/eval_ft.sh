#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-01:00  # time (DD-HH:MM)
#SBATCH --output=./results_extra/eval-ft-%a-%N-%j.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=snow.jar.13@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0-39

r_list=(32 64)
wd_list=(0.0 0.01)
epoch_list=(5 10 15 20 25)
lr_list=(0.001 0.0005)

IDX=$SLURM_ARRAY_TASK_ID
r_idx=$((IDX / 20))
wd_idx=$(((IDX / 10) % 2))
epoch_idx=$(((IDX / 2) % 5))
lr_idx=$((IDX % 2))

R=${r_list[$r_idx]}
WD=${wd_list[$wd_idx]}
EPOCHS=${epoch_list[$epoch_idx]}
LR=${lr_list[$lr_idx]}

echo "🔧 当前配置: LoRA rank=$R | wd=$WD | epochs=$EPOCHS | lr=$LR"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

python evaluate.py  --datasetType "train"  --modelType 'learned' \
    --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0
python evaluate.py  --datasetType "val"  --modelType 'learned' \
    --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0
python evaluate.py  --datasetType "forget"  --unlearnSet "unlearn-N1"  --modelType 'learned' \
    --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0
python evaluate.py  --datasetType "retain"  --unlearnSet "unlearn-N1"  --modelType 'learned' \
    --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0