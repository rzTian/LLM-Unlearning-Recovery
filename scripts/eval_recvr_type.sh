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
#SBATCH --array=0-14

set_list=("unlearn-N5-A1-sin10")
quant_list=("none" "int8" "int4")
recvr_list=('flip' 'beam')
gs_list=(10 10 10 80 10)
r_list=(64 64 64 256 64)
reg_list=(5.0 5.0 5.0 2.0 5.0)
wd_list=(0.01 0.01 0.01 0.0 0.01)
lr_list=(0.0002 0.0002 0.0002 0.0005 0.0005)
method_list=('grad_ascent' 'grad_diff' 'KL' 'po' 'npo')
epoch_list=(120 100 100 40 20)

IDX=$SLURM_ARRAY_TASK_ID
set_idx=$(((IDX / 15) % 1))
quant_idx=$(((IDX / 5) % 3))
recvr_idx=2 # $(((IDX / 5) % 2))
method_idx=$(((IDX / 1) % 5))

QUANT=${quant_list[$quant_idx]}
BETA=0.1
GS=${gs_list[$method_idx]}
R=${r_list[$method_idx]}
RG=${reg_list[$method_idx]}
WD=${wd_list[$method_idx]}
LR=${lr_list[$method_idx]}
METHOD=${method_list[$method_idx]}
EPOCHS=${epoch_list[$method_idx]}

UNLEARN_SET=${set_list[$set_idx]}
FORGET_TYPE="forget"
RETAIN_TYPE="retain_sfa"
REMAIN_TYPE="remain_sfa"
RECOVER_TYPE=${recvr_list[$recvr_idx]}
BEAM_K=100
BEAM_C=10
BEAM_N=1

echo "🔧 当前配置: method=$METHOD | beta=$BETA | quant=$QUANT"
echo "🔧 当前配置: LoRA rank=$R | reg=$RG | WD=$WD | lr=$LR | epochs=$EPOCHS | GS=$GS"
echo "🔧 当前配置: Unlearn Set=$UNLEARN_SET | Forget Type=$FORGET_TYPE | Retain Type=$RETAIN_TYPE | Remain Type=$REMAIN_TYPE"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery


echo "当前测试: Unlearn Set: $UNLEARN_SET | Forget Type: $FORGET_TYPE | $RECOVER_TYPE 1 "
python recovery.py  --unlearnSet $UNLEARN_SET --datasetType $FORGET_TYPE  --modelType 'unlearned'  \
      --recover_type $RECOVER_TYPE --flip 1 \
      --K $BEAM_K --C $BEAM_C --N $BEAM_N \
      --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --wd_fgt $WD --LoRA_rank_fgt $R --grad_acc_steps_fgt $GS --beta_fgt $BETA --quant $QUANT
echo "当前测试: Unlearn Set: $UNLEARN_SET | Forget Type: $FORGET_TYPE | $RECOVER_TYPE 0 "
python recovery.py  --unlearnSet $UNLEARN_SET --datasetType $FORGET_TYPE  --modelType 'unlearned'  \
      --recover_type $RECOVER_TYPE --flip 0 \
      --K $BEAM_K --C $BEAM_C --N $BEAM_N \
      --unlearn_method $METHOD --lr_fgt $LR --eps_fgt $EPOCHS --reg_weights_fgt $RG --wd_fgt $WD --LoRA_rank_fgt $R --grad_acc_steps_fgt $GS --beta_fgt $BETA --quant $QUANT
