#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1                # Request 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=498G        # memory per node
#SBATCH --time=00-02:59  # time (DD-HH:MM)
#SBATCH --output=./results_extra/finetune-128-%j-%a-%N.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0

r_list=(256)
wd_list=(0.01 0.0)
gs_list=(40 80 20 160 10)
epoch_list=(50)
lr_list=(0.0005)
data_list=("training_dataset.json")

IDX=$SLURM_ARRAY_TASK_ID
r_idx=$(((IDX / 10) % 1))
wd_idx=$(((IDX / 5) % 2))
gs_idx=$(((IDX / 1) % 5))
epoch_idx=$(((IDX / 1) % 1))
lr_idx=$((IDX % 1))
data_idx=$(((IDX / 1) % 1))

R=${r_list[$r_idx]}
WD=${wd_list[$wd_idx]}
GS=${gs_list[$gs_idx]}
EPOCHS=${epoch_list[$epoch_idx]}
LR=${lr_list[$lr_idx]}

echo "🔧 Finetune 当前配置: LoRA rank=$R | wd=$WD | epochs=$EPOCHS | lr=$LR | gs=$GS"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

# deepseek-ai/deepseek-llm-7b-chat
# Qwen/Qwen3-8B

accelerate launch --multi_gpu Finetune.py --dataDIR bt-training_dataset.json \
 --model_name deepseek-ai/deepseek-llm-7b-chat --logDIR fine_tuned_deepseek_7b_bt_log --modelDIR fine_tuned_deepseek_7b_bt \
 --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0  --grad_acc_steps $GS