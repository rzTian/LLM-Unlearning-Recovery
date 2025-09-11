PARENT=49536815
N_START=0
N_END=7
INTERVAL=10
SUM=$(((N_END - N_START + 1) / 2))
MAX=$(((N_END - N_START + 1) * INTERVAL - 1))

eval_job_prefixes=()
for n in {0..7}; do
    START=$((n * INTERVAL))
    END=$(( (n + 1) * INTERVAL - 1 ))
    if [ $END -gt $MAX ]; then
        END=$MAX
    fi

    eval_job=$(sbatch --parsable --dependency=afterok:${PARENT}_${n} --array=${START}-${END} scripts/eval_unl.sh)
    eval_job_prefixes+=($eval_job)
    echo "Submitted job group $n: range $START-$END, Job ID: $eval_job"
    
    if [ $(( (n - N_START + 1) % SUM )) -eq 0 ] || [ $n -eq $N_END ]; then
        sum_job=$(sbatch --parsable --dependency=afterok:${PARENT}_${n} --dependency=afterany:${eval_job} scripts/sum.sh)
        echo "Submitted summary job: depends on $eval_job, Job ID: $sum_job"
    fi
done

echo "All jobs submitted successfully"