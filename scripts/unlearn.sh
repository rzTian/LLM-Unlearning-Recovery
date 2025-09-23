#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --nodes=1                # Request 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=498G        # memory per node
#SBATCH --time=00-01:00  # time (DD-HH:MM)
#SBATCH --output=./results_extra/unlearn-%j-%a-%N.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0-7

beta_list=(0.1)
r_list=(256)
rg_list=(1.0 2.0)
wd_list=(0.0)
lr_list=(0.0005 0.001)
set_list=("unlearn-N20-A1-yrb" "unlearn-N20-A1-bld" "unlearn-N5-A1-pcd10" "unlearn-N5-A1-sin10")
method_list=('grad_diff' 'KL' 'grad_ascent' 'po' 'dpo' 'npo')

IDX=$SLURM_ARRAY_TASK_ID
beta_idx=$((IDX / 8))
r_idx=$(((IDX / 8) % 1))
rg_idx=$(((IDX / 4) % 2))
wd_idx=$(((IDX / 4) % 1))
lr_idx=$(((IDX / 2) % 2))
set_idx=$(((IDX / 1) % 2 + 2))
method_idx=$(((IDX % 1) + 3))

BETA=${beta_list[$beta_idx]}
R=${r_list[$r_idx]}
RG=${rg_list[$rg_idx]}
WD=${wd_list[$wd_idx]}
LR=${lr_list[$lr_idx]}
METHOD=${method_list[$method_idx]}
EPOCHS=50

UNLEARN_SET=${set_list[$set_idx]}
FORGET_SET="forget.json"
RETAIN_SET="retain-same_fn_attr.json"

echo "🔧 当前配置: LoRA rank=$R | reg=$RG | WD=$WD | lr=$LR | epochs=$EPOCHS | method=$METHOD | beta=$BETA"
echo "🔧 当前配置: Unlearn Set=$UNLEARN_SET | Forget Set=$FORGET_SET | Retain Set=$RETAIN_SET"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

accelerate launch --multi_gpu unlearn.py \
    --unlearnSet $UNLEARN_SET --forgetSetDir $FORGET_SET --retainSetDir $RETAIN_SET \
    --lr_ft 0.0005  --eps_ft 30 --wd_ft 0.01  --LoRA_rank_ft 256  --lora_dropout_ft 0.0 --grad_acc_steps_ft 40 \
    --unlearn_method $METHOD  --lr $LR  --epochs $EPOCHS  --weight_decay $WD \
    --LoRA_rank $R  --lora_dropout 0.0  --reg_weights $RG --grad_acc_steps 80 --beta $BETA