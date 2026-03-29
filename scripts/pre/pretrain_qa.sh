#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-01:00
#SBATCH --output=./results_extra/pretrain-QA-%j-%a-%N.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=2-5,7-8

r_list=(0)
wd_list=(0.0 0.01)
gs_list=(40 80)
lr_list=(5e-5 1e-4)

IDX=$SLURM_ARRAY_TASK_ID
r_idx=$((IDX / 8))
wd_idx=$(((IDX / 4) % 2))
gs_idx=$(((IDX / 2) % 2))
lr_idx=$((IDX % 2))

R=${r_list[$r_idx]}
WD=${wd_list[$wd_idx]}
GS=${gs_list[$gs_idx]}
LR=${lr_list[$lr_idx]}
EPOCHS=50

echo "🔧 Finetune 当前配置: LoRA rank=$R | wd=$WD | epochs=$EPOCHS | lr=$LR | gs=$GS"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

python Finetune.py --without_lora \
    --model_name gpt2 --logDIR pretrain_gpt2_log --modelDIR pretrain_gpt2 \
    --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0  --grad_acc_steps $GS