#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G        # memory per node
#SBATCH --time=00-00:30  # time (DD-HH:MM)
#SBATCH --output=./results_extra/atest-recover-grad-%N-%j-%a.out  # %N for node name, %j for jobID, %a for array ID
#SBATCH --mail-user=snow.jar.13@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=0

IDX=$SLURM_ARRAY_TASK_ID
set_list=("unlearn-N1-A1-sin" "unlearn-N1-A1-sin-fn" "unlearn-N1-A1-sin-rd" "unlearn-N1-A1-sin-cp")
set_idx=$((IDX % 4))

METHOD='npo'
UNLEARN_SET="unlearn-N1-A1-sin"
FORGET_TYPE="forget"
RETAIN_TYPE="retain_sa"
REMAIN_TYPE="remain_sa"

module load gcc arrow/18.1.0 cuda
module load python/3.10
module load scipy-stack
source $HOME/ENV-3.10/bin/activate

python recovery.py  --unlearnSet $UNLEARN_SET --datasetType $FORGET_TYPE  --modelType 'unlearned' \
      --recover_type 'grad' --recover_mode 'oracle' --flip 1 \
      --loss_type 'npo' --beta 1.0 --K 100 --C 10 --N 1000 \
      --unlearn_method $METHOD  --lr_fgt 0.001  --LoRA_rank_fgt 32 --eps_fgt 24 --reg_weights_fgt 1.0

# python recovery.py  --unlearnSet "unlearn-N1-A1-sin" --datasetType "forget"  --modelType 'unlearned' \
#       --recover_type 'grad' --recover_mode 'oracle' --flip 1 \
#       --loss_type 'npo' --beta 1.0 --K 100 --C 10 --N 1000 \
#       --unlearn_method 'npo'  --lr_fgt 0.001  --LoRA_rank_fgt 32 --eps_fgt 24 --reg_weights_fgt 1.0