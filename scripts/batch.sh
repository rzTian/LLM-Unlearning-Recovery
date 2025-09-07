PARENT=49332323
MAX=239
for n in {3..23}; do
    START=$((n*10))
    END=$((n*10+9))
    if [ $END -gt $MAX ]; then
        END=$MAX
    fi
    sbatch --dependency=afterany:${PARENT}_${n} --array=${START}-${END} scripts/eval_ft.sh
done
