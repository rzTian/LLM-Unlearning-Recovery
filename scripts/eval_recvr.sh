#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-01:00  # time (DD-HH:MM)
#SBATCH --output=./results_extra/recovery-%j-%a-%N.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0-71

beta_list=(0.05 0.2 0.5)
r_list=(256)
rg_list=(5.0)
wd_list=(0.0)
lr_list=(0.0002)
set_list=("unlearn-N20-A1-yrb" "unlearn-N20-A1-bld")
recvr_list=('flip' 'beam')
method_list=('grad_diff' 'KL' 'grad_ascent' 'po' 'dpo' 'npo')
epoch_list=(4 8 12 16 20 24 28 32 36 40 44 48)
k_list=(1 5 10 20)
quant_list=("none" "int8" "int4")

IDX=$SLURM_ARRAY_TASK_ID
quant_idx=$(((IDX / 24) % 1))
beta_idx=$(((IDX / 24) % 3))
r_idx=$(((IDX / 24) % 1))
rg_idx=$(((IDX / 24) % 1))
wd_idx=$(((IDX / 24) % 1))
lr_idx=$(((IDX / 24) % 1))
set_idx=$(((IDX / 24) % 1))
method_idx=$(((IDX / 12) % 2 + 4))
epoch_idx=$((IDX % 12))

k_idx=$(((IDX / 24) % 1))

QUANT=${quant_list[$quant_idx]}
BETA=${beta_list[$beta_idx]}
R=${r_list[$r_idx]}
RG=${rg_list[$rg_idx]}
WD=${wd_list[$wd_idx]}
LR=${lr_list[$lr_idx]}
EPOCHS=${epoch_list[$epoch_idx]}
METHOD=${method_list[$method_idx]}

UNLEARN_SET=${set_list[$set_idx]}
FORGET_TYPE="forget"
RETAIN_TYPE="retain_sfa"
REMAIN_TYPE="remain_sfa"
RECOVER_TYPE=${recvr_list[$set_idx]}
BEAM_K=${k_list[$k_idx]}
BEAM_C=10
BEAM_N=1

echo "🔧 当前配置: LoRA rank=$R | reg=$RG | WD=$WD | lr=$LR | epochs=$EPOCHS | method=$METHOD | beta=$BETA | quant=$QUANT"
echo "🔧 当前配置: Unlearn Set=$UNLEARN_SET | Forget Type=$FORGET_TYPE | Retain Type=$RETAIN_TYPE | Remain Type=$REMAIN_TYPE"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

# python recovery.py  --unlearnSet "unlearn-N20-A1-yrb" --datasetType "forget"  --modelType 'unlearned'  --recover_type 'flip' --flip 1 --unlearn_method  'grad_diff'  --lr_fgt 0.0002 --eps_fgt 50 --reg_weights_fgt 5.0 --wd_fgt 0.0  --LoRA_rank_fgt 256 --grad_acc_steps_fgt 10

echo "当前测试: Unlearn Set: $UNLEARN_SET | Forget Type: $FORGET_TYPE | $RECOVER_TYPE 1 "
python recovery.py  --unlearnSet $UNLEARN_SET --datasetType $FORGET_TYPE  --modelType 'unlearned'  \
      --recover_type $RECOVER_TYPE --flip 1 \
      --K $BEAM_K --C $BEAM_C --N $BEAM_N \
      --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --wd_fgt $WD --LoRA_rank_fgt $R --grad_acc_steps_fgt 80 --beta_fgt $BETA --quant $QUANT
echo "当前测试: Unlearn Set: $UNLEARN_SET | Forget Type: $FORGET_TYPE | $RECOVER_TYPE 0 "
python recovery.py  --unlearnSet $UNLEARN_SET --datasetType $FORGET_TYPE  --modelType 'unlearned'  \
      --recover_type $RECOVER_TYPE --flip 0 \
      --K $BEAM_K --C $BEAM_C --N $BEAM_N \
      --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --wd_fgt $WD --LoRA_rank_fgt $R --grad_acc_steps_fgt 80 --beta_fgt $BETA --quant $QUANT

echo "当前测试: Unlearn Set: $UNLEARN_SET | Retain Type: $RETAIN_TYPE | $RECOVER_TYPE 0"
python recovery.py  --unlearnSet $UNLEARN_SET --datasetType $RETAIN_TYPE  --modelType 'unlearned'  \
      --recover_type 'flip' --flip 0 \
      --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --wd_fgt $WD --LoRA_rank_fgt $R --grad_acc_steps_fgt 80 --beta_fgt $BETA --quant $QUANT

# echo "当前测试: Unlearn Set: $UNLEARN_SET | Forget Type: $FORGET_TYPE | $RECOVER_TYPE 1 | CE | K $BEAM_K"
# python recovery.py  --unlearnSet $UNLEARN_SET --datasetType $FORGET_TYPE  --modelType 'unlearned'  \
#       --recover_type $RECOVER_TYPE --recover_mode 'greedy' --flip 1 \
#       --loss_type 'ce' --beta 1.0  --K $BEAM_K --C $BEAM_C --N $BEAM_N \
#       --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --wd_fgt $WD --LoRA_rank_fgt $R --grad_acc_steps_fgt 10
# echo "当前测试: Unlearn Set: $UNLEARN_SET | Forget Type: $FORGET_TYPE | $RECOVER_TYPE 0 | NPO | K $BEAM_K"
# python recovery.py  --unlearnSet $UNLEARN_SET --datasetType $FORGET_TYPE  --modelType 'unlearned'  \
#       --recover_type $RECOVER_TYPE --recover_mode 'greedy' --flip 0 \
#       --loss_type 'npo' --beta 1.0 --K $BEAM_K --C $BEAM_C --N $BEAM_N \
#       --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --wd_fgt $WD --LoRA_rank_fgt $R --grad_acc_steps_fgt 10
