#!/bin/bash
#SBATCH --account=rrg-yymao
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
#SBATCH --array=0-95

rg_list=(1.0)
r_list=(32)
gs_list=(8)
lr_list=(0.001)
epoch_list=(2 4 6 8 10 12 14 16 18 20 22 24)
set_list=("unlearn-N1-A1-yrb-test" "unlearn-N1-A1-bld-test" "unlearn-N1-A1-pcd-test" "unlearn-N1-A1-sin-test")
method_list=('grad_diff' 'KL' 'po' 'dpo' 'npo' 'grad_ascent')
k_list=(500 1000)

IDX=$SLURM_ARRAY_TASK_ID
rg_idx=$(((IDX / 6) % 1))
r_idx=$(((IDX / 6) % 1))
lr_idx=$(((IDX / 6) % 1))
epoch_idx=$(((IDX / 2) % 12))
set_idx=$(((IDX / 24) % 2))
method_idx=$((IDX % 2))
k_idx=$(((IDX / 48) % 2))

RG=${rg_list[$rg_idx]}
R=${r_list[$r_idx]}
GS=${gs_list[$gs_idx]}
LR=${lr_list[$lr_idx]}
EPOCHS=${epoch_list[$epoch_idx]}
METHOD=${method_list[$method_idx]}

UNLEARN_SET=${set_list[$set_idx]}
FORGET_TYPE="forget"
RETAIN_TYPE="retain_sa"
REMAIN_TYPE="remain_sa"
RECOVER_TYPE="beam"
BEAM_K=${k_list[$k_idx]}
BEAM_C=10
BEAM_N=1000

echo "🔧 当前配置: LoRA rank=$R | reg=$RG | gradstep=$GS | lr=$LR | epochs=$EPOCHS | method=$METHOD"
echo "🔧 当前配置: Unlearn Set=$UNLEARN_SET | Forget Type=$FORGET_TYPE | Retain Type=$RETAIN_TYPE | Remain Type=$REMAIN_TYPE"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

# python recovery.py  --unlearnSet "unlearn-N1" --datasetType "forget"  --modelType 'unlearned'  --flip_logit 0 --unlearn_method  'grad_ascent'  --lr_fgt 0.0005  --LoRA_rank_fgt 32 --eps_fgt 2 --reg_weights_fgt 1.0

echo "当前测试: Unlearn Set: $UNLEARN_SET | Forget Type: $FORGET_TYPE | $RECOVER_TYPE 1 | CE | K $BEAM_K"
python recovery.py  --unlearnSet $UNLEARN_SET --datasetType $FORGET_TYPE  --modelType 'unlearned'  \
      --recover_type $RECOVER_TYPE --recover_mode 'greedy' --flip 1 \
      --loss_type 'ce' --beta 1.0 --K $BEAM_K --C $BEAM_C --N $BEAM_N \
      --unlearn_method  $METHOD  --lr_fgt $LR  --LoRA_rank_fgt $R --eps_fgt $EPOCHS --reg_weights_fgt $RG --grad_acc_steps_fgt $GS
echo "当前测试: Unlearn Set: $UNLEARN_SET | Forget Type: $FORGET_TYPE | $RECOVER_TYPE 0 | NPO | K $BEAM_K"
python recovery.py  --unlearnSet $UNLEARN_SET --datasetType $FORGET_TYPE  --modelType 'unlearned'  \
      --recover_type $RECOVER_TYPE --recover_mode 'greedy' --flip 0 \
      --loss_type 'npo' --beta 1.0 --K $BEAM_K --C $BEAM_C --N $BEAM_N \
      --unlearn_method  $METHOD  --lr_fgt $LR  --LoRA_rank_fgt $R --eps_fgt $EPOCHS --reg_weights_fgt $RG --grad_acc_steps_fgt $GS

# echo "当前测试: Unlearn Set: $UNLEARN_SET | Retain Type: $RETAIN_TYPE | $RECOVER_TYPE 0"
# python recovery.py  --unlearnSet $UNLEARN_SET --datasetType $RETAIN_TYPE  --modelType 'unlearned'  \
#       --recover_type $RECOVER_TYPE --flip 0 \
#       --unlearn_method  $METHOD  --lr_fgt $LR  --LoRA_rank_fgt $R --eps_fgt $EPOCHS --reg_weights_fgt $RG