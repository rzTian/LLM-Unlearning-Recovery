#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-01:00  # time (DD-HH:MM)
#SBATCH --output=./results_extra/eval-ft-ret-%j-%a-%N.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=snow.jar.13@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=eval-ft-ret
#SBATCH --array=0-6

r_list=(256)
wd_list=(0.01 0.0)
gs_list=(40 80 20 160 10)
epoch_list=(20 25 30 35 40 45 50)
lr_list=(0.0005)
set_list=("unlearn-N20-A1-bld" "unlearn-N20-A1-yrb" "unlearn-N5-A1-pcd10" "unlearn-N5-A1-sin10")
mdl_dir_list=("fine_tuned_deepseek_7b-bld" "fine_tuned_deepseek_7b-yrb" "fine_tuned_deepseek_7b-pcd" "fine_tuned_deepseek_7b-sin")
log_dir_list=("fine_tuned_deepseek_7b_log-bld" "fine_tuned_deepseek_7b_log-yrb" "fine_tuned_deepseek_7b_log-pcd" "fine_tuned_deepseek_7b_log-sin")

IDX=$SLURM_ARRAY_TASK_ID
r_idx=$((IDX / 40))
wd_idx=$(((IDX / 40) % 1))
gs_idx=$(((IDX / 40) % 1))
lr_idx=$(((IDX / 40) % 1))
epoch_idx=$((IDX % 7))
set_idx=$(((IDX / 7) % 4 + 3))

R=${r_list[$r_idx]}
WD=${wd_list[$wd_idx]}
GS=${gs_list[$gs_idx]}
EPOCHS=${epoch_list[$epoch_idx]}
LR=${lr_list[$lr_idx]}
UNLEARN_SET=${set_list[$set_idx]}
MDLD=${mdl_dir_list[$set_idx]}
LOGD=${log_dir_list[$set_idx]}

echo "🔧 当前配置: unlearn set=$UNLEARN_SET | model dir=$MDLD | log dir=$LOGD"
echo "🔧 当前配置: epochs=$EPOCHS | lr=$LR | wd=$WD | LoRA rank=$R | GS=$GS"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

python evaluate.py  --unlearnSet $UNLEARN_SET  --logDIR $LOGD --modelDIR $MDLD \
    --datasetType "forget"  --modelType 'learned' \
    --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0  --grad_acc_steps $GS

python evaluate.py  --unlearnSet $UNLEARN_SET  --logDIR $LOGD --modelDIR $MDLD \
    --datasetType "retain"  --modelType 'learned' \
    --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0  --grad_acc_steps $GS

python evaluate.py  --unlearnSet $UNLEARN_SET  --logDIR $LOGD --modelDIR $MDLD \
    --datasetType "remain"  --modelType 'learned' \
    --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0  --grad_acc_steps $GS