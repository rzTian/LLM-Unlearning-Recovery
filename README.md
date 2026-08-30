# On the Recoverability of Private Information Unlearning in Large Language Models

Code for studying private information unlearning and recovery/auditing in causal language models.

> **Disclaimer.** This repository is research code for studying machine unlearning and its limitations on a **synthetic** dataset of fake private information (FPI) — no real individuals' data is used. The recovery/auditing methods here are intended to evaluate and improve unlearning techniques, not to extract real personal information. Please do not apply this code to models trained on real personal data without proper authorization and consent.

## Overview

This repository implements an end-to-end workflow for synthetic FPI-style private information experiments: dataset construction, fine-tuning, unlearning, evaluation, and recovery auditing. The current scripts are configured around `deepseek-ai/deepseek-llm-7b-chat`; `llm_judge.py` uses `Qwen/Qwen3-8B` as the default LLM judge for common-knowledge evaluation.

The active workflow is script-driven. The shell scripts in `scripts/` are Slurm/HPC templates, but they also document the exact Python entry points and parameters used by each stage.

## Repository Structure

- `data_generator/`: profile generation, QA dataset generation, and forget/retain/remain splits.
- `Finetune.py`: fine-tuning entry point.
- `unlearn.py`, `UnlearnTrainer.py`: unlearning entry point and loss implementations.
- `evaluate.py`: evaluation for base, fine-tuned, unlearned, and pretrained variants.
- `recovery.py`: recovery/auditing entry point.
- `prepdata.py`, `utils.py`, `argsetting.py`: shared preprocessing, utilities, and CLI arguments.
- `llm_judge.py`, `llm_prompt.py`: LLM-as-a-judge evaluation utilities.
- `scripts/`: main Slurm workflow scripts.
- `scripts/env/`: dependency and environment helper scripts.

There is no separate `configs/` directory; experiment settings are controlled by CLI arguments and shell-script variables.

## Installation

Install dependencies with:

```bash
pip install -r scripts/env/requirements.txt
```

The main entry points need a Hugging Face access token (for gated models such as Llama). Set it via an environment variable:

```bash
export HF_TOKEN="hf_..."
```

Alternatively, create a local `saved_hf_key.py` (git-ignored) defining `HF_key = "..."`; this is only used as a fallback if `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` is not set. See `get_hf_token()` in `utils.py`.

Several model-loading calls use `local_files_only=True`, so required models must already be available in the local Hugging Face cache or local model directories. The helper below downloads models listed in `MODEL_LIST` inside the script:

```bash
python scripts/env/download_model.py
```

`scripts/env/setup_env.sh` and `scripts/env/check_env.sh` are cluster-oriented environment templates.

## Dataset

The repository includes generated FPI-style data under `data_generator/data/`, including `training_dataset.json`, `validation_dataset.json`, `training_testset.json`, `validation_testset.json`, `common_knowledge_questions.json`, `idk.jsonl`, and split folders such as `unlearn-N20-A1-yrb/`.

QA records use:

```json
{
  "name": "Michael Ayers",
  "attribute": "year_of_birth",
  "question": "In which year was Michael Ayers born?",
  "answer": "Michael Ayers's year of birth is 1975."
}
```

To regenerate synthetic data:

```bash
cd data_generator
python create_profile.py
python generate_dataset.py
bash split.sh
cd ..
```

For custom splits, use `data_generator/split_dataset.py`. Supported `--forget_mode` values are `random`, `same_firstname`, `different_firstname`, `n_per_firstname`, and `random_combination`; supported `--retain_mode` values are `all_except_forget`, `same_firstname`, `same_attr`, and `same_firstname_same_attr`.

## Quick Start

Run the full Slurm pipeline:

```bash
bash scripts/submit_pipeline.sh
```

The dependency order is:

```text
scripts/train.sh -> scripts/eval_ft.sh -> scripts/unlearn.sh -> scripts/eval_unl.sh -> scripts/eval_recvr.sh
```

Run individual stages with:

```bash
bash scripts/train.sh
bash scripts/eval_ft.sh
bash scripts/unlearn.sh
bash scripts/eval_unl.sh
bash scripts/eval_recvr.sh
```

For non-Slurm use, copy the Python command from the corresponding script and replace only the cluster setup lines. The script variables at the top of each file control model name, data split, checkpoint directories, method, and hyperparameters.

## Methods

The supported unlearning methods are:

```text
grad_ascent, grad_diff, KL, po, dpo, npo, noisy_grad_diff
```

The recovery/auditing methods exposed by `recovery.py` are:

```text
flip, beam, grad
```

The code does not expose CLI methods literally named `RIG` or `RG`; use the concrete recovery modes above.

## Evaluation

`scripts/eval_ft.sh` evaluates the fine-tuned model on `forget`, `retain`, and `remain`. `scripts/eval_unl.sh` evaluates the unlearned model on `forget`, `retain_sfa`, and `remain_sfa`.

Supported `--modelType` values are `base`, `learned`, `unlearned`, `pt`, and `pt-unlearned`. Supported `--datasetType` values are `train`, `val`, `train_t`, `val_t`, `common`, `forget`, `retain`, `remain`, `retain_sf`, `retain_sa`, `retain_sfa`, `remain_sf`, `remain_sa`, and `remain_sfa`.

For `--datasetType common`, `evaluate.py` uses `LLMJudge` with default judge model `Qwen/Qwen3-8B`.

## Outputs

Fine-tuning writes checkpoints under `fine_tuned_deepseek_7b/` and logs under `fine_tuned_deepseek_7b_log/` by default. Unlearning writes checkpoints under `unlearn_deepseek_7b/` and logs under `unlearn_deepseek_7b_log/`. Recovery writes JSON results under `recovery_deepseek_7b_log/`.

Exact subdirectory names are constructed from the active script variables, including learning rate, weight decay, LoRA rank, gradient accumulation steps, unlearning set, regularization weight, method, and epoch.

## Optional Pretraining

`pretrain.py` supports causal-LM pretraining from JSON files with a `text` field. The active repository data is QA-formatted, and no current script provides a runnable pretraining dataset path.

## Slurm / HPC Notes

The scripts in `scripts/` are cluster templates. Before submitting jobs, edit the Slurm header, module commands, virtual environment path, working directory, model paths, and resource requests for your system.

`scripts/submit_pipeline.sh` submits the five stages with `sbatch --dependency=afterok`.

## Citation

TODO: Add the official citation or BibTeX entry when available.

## License

Released under the [MIT License](LICENSE).
