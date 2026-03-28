#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1                # Request 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=498G        # memory per node
#SBATCH --time=00-02:59  # time (DD-HH:MM)
#SBATCH --output=./results_extra/finetune-ret-%j-%a-%N.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0-3

r_list=(256)
wd_list=(0.01 0.0)
gs_list=(40 80 20 160 10)
epoch_list=(50)
lr_list=(0.0005)
data_list=("unlearn-N20-A1-bld/clean.json" "unlearn-N20-A1-yrb/clean.json" "unlearn-N5-A1-pcd10/clean.json" "unlearn-N5-A1-sin10/clean.json")
mdl_dir_list=("fine_tuned_Qwen3_8B-bld" "fine_tuned_Qwen3_8B-yrb" "fine_tuned_Qwen3_8B-pcd" "fine_tuned_Qwen3_8B-sin")
log_dir_list=("fine_tuned_Qwen3_8B_log-bld" "fine_tuned_Qwen3_8B_log-yrb" "fine_tuned_Qwen3_8B_log-pcd" "fine_tuned_Qwen3_8B_log-sin")

IDX=$SLURM_ARRAY_TASK_ID
r_idx=$(((IDX / 1) % 1))
wd_idx=$(((IDX / 1) % 1))
gs_idx=$(((IDX / 1) % 1))
epoch_idx=$(((IDX / 1) % 1))
lr_idx=$((IDX % 1))
data_idx=$(((IDX / 1) % 4))

R=${r_list[$r_idx]}
WD=${wd_list[$wd_idx]}
GS=${gs_list[$gs_idx]}
EPOCHS=${epoch_list[$epoch_idx]}
LR=${lr_list[$lr_idx]}
DATA=${data_list[$data_idx]}
MDL_NAME="Qwen/Qwen3-8B"
MDLD=${mdl_dir_list[$data_idx]}
LOGD=${log_dir_list[$data_idx]}

echo "🔧 Model 当前配置: model name=$MDL_NAME"
echo "🔧 Dataset 当前配置: data set=$DATA | model dir=$MDLD | log dir=$LOGD"
echo "🔧 Finetune 当前配置: LoRA rank=$R | wd=$WD | epochs=$EPOCHS | lr=$LR | gs=$GS"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

# deepseek-ai/deepseek-llm-7b-chat
# Qwen/Qwen3-8B

accelerate launch --multi_gpu Finetune.py --model_name $MDL_NAME \
 --logDIR $LOGD --modelDIR $MDLD --dataDIR $DATA \
 --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0  --grad_acc_steps $GS