#!/bin/bash

# Request an interactive session
salloc \
    --account=rrg-yymao \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gpus-per-task=1 \
    --cpus-per-task=6 \
    --mem=32G \
    --time=01:00:00 \
    srun --pty bash -c '

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate

python interact.py --lr 0.0005  --epochs 15 --weight_decay 0.01  --LoRA_rank 64  --lora_dropout 0.0
'