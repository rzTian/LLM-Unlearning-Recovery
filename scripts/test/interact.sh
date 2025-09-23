#!/bin/bash

# Request an interactive session
salloc \
    --account=def-yymao \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gpus-per-task=1 \
    --cpus-per-task=6 \
    --mem=64G \
    --time=01:00:00 \
    srun --pty bash -c '

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

python evaluate.py  --modelType 'unlearned' \
    --unlearn_method 'po' --lr_fgt 0.0005 --eps_fgt 24 --reg_weights_fgt 1.0 --wd_fgt 0.0 --LoRA_rank_fgt 256
'