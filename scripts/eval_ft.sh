#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-01:00  # time (DD-HH:MM)
#SBATCH --output=./results_extra/eval-ft-bt-%j-%a-%N.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=snow.jar.13@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=eval-ft-bt
#SBATCH --array=0-11

r_list=(256)
wd_list=(0.01 0.0)
gs_list=(40 80 20 160 10)
lr_list=(0.0005)
epoch_list=(5 10 15 20 25 30 35 40 45 50)
set_list=("unlearn-N20-A1-yrb" "unlearn-N20-A1-bld" "unlearn-N5-A1-pcd10" "unlearn-N5-A1-sin10")

IDX=$SLURM_ARRAY_TASK_ID
r_idx=$((IDX / 100))
wd_idx=$(((IDX / 50) % 2))
gs_idx=$(((IDX / 10) % 5))
lr_idx=$(((IDX / 10) % 1))
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

python evaluate.py  --datasetType "bt-train_t"  --modelType 'learned' \
    --model_name deepseek-ai/deepseek-llm-7b-chat --logDIR fine_tuned_deepseek_7b_bt_log --modelDIR fine_tuned_deepseek_7b_bt \
    --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0  --grad_acc_steps $GS
python evaluate.py  --datasetType "bt-val_t"  --modelType 'learned' \
    --model_name deepseek-ai/deepseek-llm-7b-chat --logDIR fine_tuned_deepseek_7b_bt_log --modelDIR fine_tuned_deepseek_7b_bt \
    --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0  --grad_acc_steps $GS
# python evaluate.py  --datasetType "common"  --modelType 'learned' \
#     --model_name Qwen/Qwen3-8B --logDIR fine_tuned_Qwen3_8B_log --modelDIR fine_tuned_Qwen3_8B \
#     --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0  --grad_acc_steps $GS