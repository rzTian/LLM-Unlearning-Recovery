#!/bin/bash
set -euo pipefail

mkdir -p results_extra

submit_job() {
  local script_path="$1"
  local dependency="${2:-}"

  if [[ -n "$dependency" ]]; then
    sbatch --parsable --dependency="afterok:${dependency}" "$script_path"
  else
    sbatch --parsable "$script_path"
  fi
}

train_job=$(submit_job "scripts/train.sh")
echo "Submitted train.sh: ${train_job}"

eval_ft_job=$(submit_job "scripts/eval_ft.sh" "$train_job")
echo "Submitted eval_ft.sh after train.sh: ${eval_ft_job}"

unlearn_job=$(submit_job "scripts/unlearn.sh" "$eval_ft_job")
echo "Submitted unlearn.sh after eval_ft.sh: ${unlearn_job}"

eval_unl_job=$(submit_job "scripts/eval_unl.sh" "$unlearn_job")
echo "Submitted eval_unl.sh after unlearn.sh: ${eval_unl_job}"

eval_recvr_job=$(submit_job "scripts/eval_recvr.sh" "$eval_unl_job")
echo "Submitted eval_recvr.sh after eval_unl.sh: ${eval_recvr_job}"

echo "Pipeline:"
echo "  train      ${train_job}"
echo "  eval_ft    ${eval_ft_job}"
echo "  unlearn    ${unlearn_job}"
echo "  eval_unl   ${eval_unl_job}"
echo "  eval_recvr ${eval_recvr_job}"
