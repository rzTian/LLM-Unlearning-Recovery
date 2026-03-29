#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-01:00
#SBATCH --output=./results_extra/pretrain-corpus-%j-%a-%N.out
#SBATCH --mail-user=smsmun.husc@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=pretrain-gpt2
#SBATCH --array=0-5

# 和 train.sh 保持一致的超参数数组风格
r_list=(0)
wd_list=(0.01 0.0)
gs_list=(4 8)
lr_list=(0.0005 0.001 0.005 0.01 0.05 0.1)

IDX=$SLURM_ARRAY_TASK_ID
r_idx=$(((IDX / 6) % 1))
wd_idx=$(((IDX / 6) % 1))
gs_idx=$(((IDX / 6) % 1))
lr_idx=$(((IDX / 1) % 6))

R=${r_list[$r_idx]}
WD=${wd_list[$wd_idx]}
GS=${gs_list[$gs_idx]}
LR=${lr_list[$lr_idx]}
EPOCHS=100
DATAFILE="data_generator/data/pretrain_corpus.jsonl"
BS=8
BLOCK=128

echo "🔧 Pretrain 当前配置: LoRA rank=$R | wd=$WD | epochs=$EPOCHS | lr=$LR | gs=$GS | bs=$BS | block=$BLOCK"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

mkdir -p pretrain_gpt2
mkdir -p pretrain_gpt2_log

python pretrain.py --train_file $DATAFILE \
  --model_name gpt2 --logDIR pretrain_gpt2_log --modelDIR pretrain_gpt2 \
  --lr $LR --epochs $EPOCHS --weight_decay $WD --LoRA_rank $R --lora_dropout 0.0 --grad_acc_steps $GS \
  --bs_train $BS --bs_eval $BS --block_size $BLOCK --warmup_ratio 0.03 \
  --logging_steps 50 --save_every_n_epochs 10 --save_total_limit 100 --bf16