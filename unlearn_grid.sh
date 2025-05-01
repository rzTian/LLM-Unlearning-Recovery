#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --nodes=1                # Request 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=498G        # memory per node
#SBATCH --time=00-01:00  # time (DD-HH:MM)
#SBATCH --output=./results_extra/unlearn-%N-%j-%a.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=snow.jar.13@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0-29

r_list=(32)
epoch_list=(6 12 24)
lr_list=(0.001 0.0005)
method_list=('grad_ascent' 'grad_diff' 'KL' 'dpo' 'npo')

IDX=$SLURM_ARRAY_TASK_ID
r_idx=$((IDX / 30))
epoch_idx=$(((IDX / 10) % 3))
lr_idx=$(((IDX / 5) % 2))
method_idx=$((IDX % 5))

R=${r_list[$r_idx]}
LR=${lr_list[$lr_idx]}
EPOCHS=${epoch_list[$epoch_idx]}
METHOD=${method_list[$method_idx]}

echo "🔧 当前配置: LoRA rank=$R | epochs=$EPOCHS | lr=$LR | method=$METHOD"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate

accelerate launch --multi_gpu unlearn.py --unlearn_method  $METHOD  --lr $LR  --epochs $EPOCHS  --weight_decay 0.0  --reg_weights 1.0  --LoRA_rank $R  --lora_dropout 0.0
# accelerate launch --multi_gpu unlearn.py --unlearn_method  "grad_ascent"  --lr $LR  --epochs $EPOCHS  --weight_decay 0.0  --reg_weights 1.0  --LoRA_rank $R  --lora_dropout 0.0
# accelerate launch --multi_gpu unlearn.py --unlearn_method  "grad_diff"  --lr $LR  --epochs $EPOCHS  --weight_decay 0.0  --reg_weights 1.0  --LoRA_rank $R  --lora_dropout 0.0
# accelerate launch --multi_gpu unlearn.py --unlearn_method  "KL"  --lr $LR  --epochs $EPOCHS  --weight_decay 0.0  --reg_weights 1.0  --LoRA_rank $R  --lora_dropout 0.0
# accelerate launch --multi_gpu unlearn.py --unlearn_method  "dpo"  --lr $LR  --epochs $EPOCHS  --weight_decay 0.0  --reg_weights 1.0  --LoRA_rank $R  --lora_dropout 0.0
# accelerate launch --multi_gpu unlearn.py --unlearn_method  "npo"  --lr $LR  --epochs $EPOCHS  --weight_decay 0.0  --reg_weights 1.0  --LoRA_rank $R  --lora_dropout 0.0