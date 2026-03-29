#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-01:00
#SBATCH --output=./results_extra/eval-pt-%j-%a-%N.out
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=eval-pt
#SBATCH --array=0-79

r_list=(0)
wd_list=(0.0 0.01)
gs_list=(40 80)
lr_list=(5e-5 1e-4)
epoch_list=(5 10 15 20 25 30 35 40 45 50)

IDX=$SLURM_ARRAY_TASK_ID
r_idx=$((IDX / 80))
wd_idx=$(((IDX / 40) % 2))
gs_idx=$(((IDX / 20) % 2))
lr_idx=$(((IDX / 10) % 2))
epoch_idx=$((IDX % 10))

R=${r_list[$r_idx]}
WD=${wd_list[$wd_idx]}
GS=${gs_list[$gs_idx]}
EPOCHS=${epoch_list[$epoch_idx]}
LR=${lr_list[$lr_idx]}

echo "🔧 当前配置: epochs=$EPOCHS | lr=$LR | wd=$WD | LoRA rank=$R | GS=$GS"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

# MODEL_PATH="pretrain_gpt2/lr{$LR}_WD{$WD}_loraRank{$R}_loraDrop0.0_GradStsp{$GS}/epoch-{$EPOCHS}"

# python evaluate.py --datasetType "train_t" --modelType 'pt' \
#     --model_name gpt2 --logDIR pretrain_gpt2_log --modelDIR pretrain_gpt2 \
#     --lr 5e-5 --epochs 5 --weight_decay 0.01 --LoRA_rank 0 --lora_dropout 0.0 --grad_acc_steps 4

python evaluate.py --datasetType "train_t" --modelType 'pt' \
    --model_name gpt2 --logDIR pretrain_gpt2_log --modelDIR pretrain_gpt2 \
    --lr $LR --epochs $EPOCHS --weight_decay $WD --LoRA_rank $R --lora_dropout 0.0 --grad_acc_steps $GS

python evaluate.py --datasetType "val_t" --modelType 'pt' \
    --model_name gpt2 --logDIR pretrain_gpt2_log --modelDIR pretrain_gpt2 \
    --lr $LR --epochs $EPOCHS --weight_decay $WD --LoRA_rank $R --lora_dropout 0.0 --grad_acc_steps $GS