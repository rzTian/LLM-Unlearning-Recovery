PARENT=49651460
N_START=2
N_END=7
INTERVAL=12
SUM=$(((N_END - N_START + 1) / 2))
MAX=$(((N_END + 1) * INTERVAL - 1))

EVAL_UNL_SH=scripts/eval_unl.sh
EVAL_RCV_SH=scripts/eval_recvr.sh
SUM_SH=scripts/sum.sh

eval_job_prefixes=()
for ((n=N_START; n<=N_END; n++)); do
    START=$((n * INTERVAL))
    END=$(( (n + 1) * INTERVAL - 1 ))
    if [ $END -gt $MAX ]; then
        END=$MAX
    fi

    eval_job_1=$(sbatch --parsable --dependency=afterok:${PARENT}_${n} --array=${START}-${END} ${EVAL_UNL_SH})
    eval_job_2=$(sbatch --parsable --dependency=afterok:${PARENT}_${n} --array=${START}-${END} ${EVAL_RCV_SH})
    eval_job_prefixes+=($eval_job_1)
    eval_job_prefixes+=($eval_job_2)
    echo "Submitted job group $n: range $START-$END, Job ID: $eval_job_1, $eval_job_2"
    
    if [ $(( (n - N_START + 1) % SUM )) -eq 0 ] || [ $n -eq $N_END ]; then
        sum_job=$(sbatch --parsable --dependency=afterok:${PARENT}_${n} --dependency=afterany:${eval_job_1}:${eval_job_2} ${SUM_SH})
        echo "Submitted summary job: depends on $eval_job_1 and $eval_job_2, Job ID: $sum_job"
    fi
done

echo "All jobs submitted successfully"