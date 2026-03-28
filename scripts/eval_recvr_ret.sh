#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-00:30  # time (DD-HH:MM)
#SBATCH --output=./results_extra/recovery-ret-%j-%a-%N.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0-39

r_list=(256)
wd_list=(0.01)
gs_list=(40)
lr_list=(0.0005)
set_list=("unlearn-N20-A1-bld" "unlearn-N20-A1-yrb" "unlearn-N5-A1-pcd10" "unlearn-N5-A1-sin10")
mdl_dir_list=("fine_tuned_Qwen3_8B-bld" "fine_tuned_Qwen3_8B-yrb" "fine_tuned_Qwen3_8B-pcd" "fine_tuned_Qwen3_8B-sin")
log_dir_list=("recovery_Qwen3_8B_log-bld" "recovery_Qwen3_8B_log-yrb" "recovery_Qwen3_8B_log-pcd" "recovery_Qwen3_8B_log-sin")
epoch_list=(5 10 15 20 25 30 35 40 45 40)

IDX=$SLURM_ARRAY_TASK_ID
r_idx=$(((IDX / 24) % 1))
wd_idx=$(((IDX / 24) % 1))
gs_idx=$(((IDX / 24) % 1))
lr_idx=$(((IDX / 24) % 1))
set_idx=$(((IDX / 10) % 4))
epoch_idx=$((IDX % 10))

R=${r_list[$r_idx]}
WD=${wd_list[$wd_idx]}
GS=${gs_list[$gs_idx]}
LR=${lr_list[$lr_idx]}
EPOCHS=${epoch_list[$epoch_idx]}
METHOD=${method_list[$method_idx]}

UNLEARN_SET=${set_list[$set_idx]}
MDL_NAME="Qwen/Qwen3-8B"
MDLD=${mdl_dir_list[$set_idx]}
LOGD=${log_dir_list[$set_idx]}
FORGET_TYPE="forget"
RECOVER_TYPE="flip"
BEAM_K=1
BEAM_C=1
BEAM_N=1

echo "🔧 当前配置: LoRA rank=$R | grad step=$GS | WD=$WD | lr=$LR | epochs=$EPOCHS"
echo "🔧 当前配置: Unlearn Set=$UNLEARN_SET | Forget Type=$FORGET_TYPE"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

echo "当前测试: Unlearn Set: $UNLEARN_SET | Forget Type: $FORGET_TYPE | $RECOVER_TYPE 0 "
python recovery.py  --model_name $MDL_NAME --unlearnSet $UNLEARN_SET --logDIR_recvr $LOGD --modelDIR $MDLD --modelType 'learned'  \
      --datasetType $FORGET_TYPE  --recover_type $RECOVER_TYPE --flip 0 \
      --K $BEAM_K --C $BEAM_C --N $BEAM_N \
      --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0  --grad_acc_steps $GS

echo "当前测试: Unlearn Set: $UNLEARN_SET | Forget Type: $FORGET_TYPE | $RECOVER_TYPE 1 "
python recovery.py  --model_name $MDL_NAME --unlearnSet $UNLEARN_SET --logDIR_recvr $LOGD --modelDIR $MDLD --modelType 'learned'  \
      --recover_type $RECOVER_TYPE --flip 1 \
      --K $BEAM_K --C $BEAM_C --N $BEAM_N \
      --lr $LR  --epochs $EPOCHS --weight_decay $WD  --LoRA_rank $R  --lora_dropout 0.0  --grad_acc_steps $GS
