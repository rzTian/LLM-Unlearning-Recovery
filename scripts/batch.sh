PARENT=49542781
N_START=0
N_END=7
INTERVAL=5
SUM=$(((N_END - N_START + 1) / 2))
MAX=$(((N_END - N_START + 1) * INTERVAL - 1))

EVAL_SH=scripts/eval_unl.sh
SUM_SH=scripts/sum.sh

eval_job_prefixes=()
for n in {${N_START}..${N_END}}; do
    START=$((n * INTERVAL))
    END=$(( (n + 1) * INTERVAL - 1 ))
    if [ $END -gt $MAX ]; then
        END=$MAX
    fi

    eval_job=$(sbatch --parsable --dependency=afterok:${PARENT}_${n} --array=${START}-${END} ${EVAL_SH})
    eval_job_prefixes+=($eval_job)
    echo "Submitted job group $n: range $START-$END, Job ID: $eval_job"
    
    if [ $(( (n - N_START + 1) % SUM )) -eq 0 ] || [ $n -eq $N_END ]; then
        sum_job=$(sbatch --parsable --dependency=afterok:${PARENT}_${n} --dependency=afterany:${eval_job} ${SUM_SH})
        echo "Submitted summary job: depends on $eval_job, Job ID: $sum_job"
    fi
done

echo "All jobs submitted successfully"