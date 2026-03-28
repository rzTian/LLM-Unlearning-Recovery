#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-03:59  # time (DD-HH:MM)
#SBATCH --output=./results_extra/recovery-%j-%a-%N.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0-5

r_list=(64)
rg_list=(5.0)
wd_list=(0.01)
lr_list=(0.0002)
set_list=("unlearn-N5-A1-pcd10" "unlearn-N5-A1-sin10")
method_list=('npo')
epoch_list=(36)
k_list=(1000 5000 10000)

IDX=$SLURM_ARRAY_TASK_ID
r_idx=$(((IDX / 2) % 1))
rg_idx=$(((IDX / 2) % 1))
wd_idx=$(((IDX / 2) % 1))
lr_idx=$(((IDX / 2) % 1))
set_idx=$(((IDX / 1) % 2))
method_idx=$(((IDX / 1) % 1))
epoch_idx=$((IDX % 1))

k_idx=$(((IDX / 2) % 3))

R=${r_list[$r_idx]}
RG=${rg_list[$rg_idx]}
WD=${wd_list[$wd_idx]}
LR=${lr_list[$lr_idx]}
EPOCHS=${epoch_list[$epoch_idx]}
METHOD=${method_list[$method_idx]}
LR=${lr_list[$lr_idx]}

UNLEARN_SET=${set_list[$set_idx]}
FORGET_TYPE="forget"
RETAIN_TYPE="retain_sfa"
REMAIN_TYPE="remain_sfa"
RECOVER_TYPE="beam"
BEAM_K=${k_list[$k_idx]}
BEAM_C=30
BEAM_N=10000

MDL_NAME="Qwen/Qwen3-8B"
LOG_DIR_FT="fine_tuned_Qwen3_8B_log"
MDL_DIR_FT="fine_tuned_Qwen3_8B"
LOG_DIR_UNL="unlearn_Qwen3_8B_log"
MDL_DIR_UNL="unlearn_Qwen3_8B"
LOG_DIR_RCV="recovery_Qwen3_8B_log"

echo "🔧 当前配置: method=$METHOD"
echo "🔧 当前配置: model-name=$MDL_NAME | ft-model-dir=$MDL_DIR_FT | unl-model-dir=$MDL_DIR_UNL"
echo "🔧 当前配置: ft-log-dir=$LOG_DIR_FT | unl-log-dir=$LOG_DIR_UNL | recvr-log-dir=$LOG_DIR_RCV"
echo "🔧 当前配置: LoRA rank=$R | reg=$RG | WD=$WD | lr=$LR | epochs=$EPOCHS"
echo "🔧 当前配置: Unlearn Set=$UNLEARN_SET | Forget Type=$FORGET_TYPE | Retain Type=$RETAIN_TYPE | Remain Type=$REMAIN_TYPE"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

# python recovery.py  --unlearnSet "unlearn-N20-A1-yrb" --datasetType "forget"  --modelType 'unlearned'  --recover_type 'flip' --flip 1 --unlearn_method  'grad_diff'  --lr_fgt 0.0002 --eps_fgt 50 --reg_weights_fgt 5.0 --wd_fgt 0.0  --LoRA_rank_fgt 256 --grad_acc_steps_fgt 10

echo "当前测试: Unlearn Set: $UNLEARN_SET | Forget Type: $FORGET_TYPE | $RECOVER_TYPE 0 "
python recovery.py  --model_name $MDL_NAME  --logDIR $LOG_DIR_FT  --modelDIR $MDL_DIR_FT  --logDIR_fgt $LOG_DIR_UNL  --modelDIR_fgt $MDL_DIR_UNL  --logDIR_recvr $LOG_DIR_RCV \
      --unlearnSet $UNLEARN_SET  --datasetType $FORGET_TYPE  --modelType 'unlearned'  \
      --recover_type $RECOVER_TYPE  --flip 0  --K $BEAM_K  --C $BEAM_C  --N $BEAM_N \
      --lr 0.0005  --epochs 20  --weight_decay 0.01  --LoRA_rank 256  --lora_dropout 0.0 --grad_acc_steps 40 \
      --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --wd_fgt $WD --LoRA_rank_fgt $R --grad_acc_steps_fgt 80
