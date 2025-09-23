#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --cpus-per-task=1        # 减少CPU核心数
#SBATCH --mem=1G               # 减少内存占用
#SBATCH --time=00-00:10         # 设置较短的运行时间
#SBATCH --output=./results_extra/sum-%j.out
#SBATCH --job-name=sum

# Load necessary modules
module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate
cd $HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery

# Run the Python script
python sum_eval.py
python sum_log.py
# python sum_ft.py