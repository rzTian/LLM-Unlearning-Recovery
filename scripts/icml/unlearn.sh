#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1                # Request 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-01:30  # time (DD-HH:MM)
#SBATCH --output=./results_extra/unlearn-icml-%j-%a-%N.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0-16

beta_list=(0.1)
r_list=(256)
rg_list=(5.0)
wd_list=(0.01 0.0)
lr_list=(0.0002 0.0005 0.0001 5e-5)
set_list=("unlearn-N20-A1-yrb" "unlearn-N20-A1-bld" "unlearn-N5-A1-pcd10" "unlearn-N5-A1-sin10")
# set_list=("unlearn-N5-A1-yrb10" "unlearn-N40-A1-yrb10" "unlearn-N20-A1-pcd" "unlearn-N40-A1-pcd10")
method_list=('grad_diff' 'KL' 'grad_ascent' 'npo')

IDX=$SLURM_ARRAY_TASK_ID
beta_idx=$((IDX / 64))
r_idx=$(((IDX / 64) % 1))
rg_idx=$(((IDX / 64) % 1))
wd_idx=$(((IDX / 64) % 1))
lr_idx=$(((IDX / 16) % 4))
set_idx=$(((IDX / 4) % 4))
method_idx=$(((IDX % 4)))

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

# Qwen/Qwen3-8B deepseek-ai/deepseek-llm-7b-chat
# accelerate launch --multi_gpu
python unlearn.py --model_name deepseek-ai/deepseek-llm-7b-chat \
    --finetune_model_DIR fine_tuned_deepseek_7b  --logDIR unlearn_deepseek_7b_log --unlearn_model_DIR unlearn_deepseek_7b \
    --unlearnSet $UNLEARN_SET --forgetSetDir $FORGET_SET --retainSetDir $RETAIN_SET \
    --lr_ft 0.0005  --eps_ft 30 --wd_ft 0.01  --LoRA_rank_ft 256  --lora_dropout_ft 0.0 --grad_acc_steps_ft 40 \
    --unlearn_method $METHOD  --lr $LR  --epochs $EPOCHS  --weight_decay $WD \
    --LoRA_rank $R  --lora_dropout 0.0  --reg_weights $RG --grad_acc_steps 80 --beta $BETA