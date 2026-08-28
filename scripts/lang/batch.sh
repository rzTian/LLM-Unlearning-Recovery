#!/bin/bash
#SBATCH --account=rrg-yymao
#SBATCH --cpus-per-task=1        # 减少CPU核心数
#SBATCH --mem=1G               # 减少内存占用
#SBATCH --time=00-00:10         # 设置较短的运行时间
#SBATCH --output=./results_extra/batch-%j.out
#SBATCH --job-name=batch

set -e

PARENT=xxxxxxx
N_START=0
N_END=7
INTERVAL=6
MAX=$(( (N_END + 1) * INTERVAL - 1 ))

UNL_SH=scripts/lang/unlearn.sh
EVAL_UNL_SH=scripts/lang/eval_unl.sh
EVAL_RCV_SH=scripts/lang/eval_recvr.sh
SUM_SH=scripts/sum.sh

unl_job=$(sbatch --parsable --array=${N_START}-${N_END} ${UNL_SH})
PARENT=$unl_job

eval_job_ids=()
for ((n=N_START; n<=N_END; n++)); do
    START=$((n * INTERVAL))
    END=$(( (n + 1) * INTERVAL - 1 ))
    if [ $END -gt $MAX ]; then
        END=$MAX
    fi

    eval_job_1=$(sbatch --parsable --dependency=afterok:${PARENT}_${n} --array=${START}-${END} ${EVAL_UNL_SH})
    eval_job_2=$(sbatch --parsable --dependency=afterok:${PARENT}_${n} --array=${START}-${END} ${EVAL_RCV_SH})
    eval_job_ids+=("$eval_job_1")
    eval_job_ids+=("$eval_job_2")

    echo "Submitted group $n: eval range $START-$END, jobs: $eval_job_1, $eval_job_2"
done

deps=$(IFS=:; echo "${eval_job_ids[*]}")
sum_job=$(sbatch --parsable --dependency=afterany:${deps} ${SUM_SH})
echo "Submitted summary job: $sum_job"

echo "All jobs submitted successfully"
