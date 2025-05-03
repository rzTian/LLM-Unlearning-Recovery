#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1        # 只使用1个GPU
#SBATCH --cpus-per-task=4        # 减少CPU核心数
#SBATCH --mem=32G               # 减少内存占用
#SBATCH --time=00-00:10         # 设置较短的运行时间
#SBATCH --output=./results_extra/debug-%j.out
#SBATCH --job-name=debug

# 加载必要模块
module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate

# 运行测试 - 单epoch
accelerate launch --multi_gpu unlearn.py \
    --unlearnSet "unlearn-N1" --forgetSetDir "forget.json" --retainSetDir "retain.json" \
    --lr_ft 0.001  --eps_ft 15 --wd_ft 0.0  --LoRA_rank_ft 32  --lora_dropout_ft 0.0 \
    --unlearn_method "grad_ascent"  --lr 0.001  --epochs 4 --weight_decay 0.0 \
    --LoRA_rank $R --lora_dropout 0.0 --reg_weights 1.0
