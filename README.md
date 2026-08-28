# On the Recoverability of Private Information Unlearning in Large Language Models

Code for fine-tuning, unlearning, evaluating, and auditing recovery of private information in language models.

## Overview

This repository implements an end-to-end workflow for studying private information unlearning in causal language models. The code builds synthetic personally identifiable information (PII) question-answer datasets, fine-tunes a model on those records, applies unlearning objectives, evaluates retained/forgotten behavior, and runs recovery-style audits against unlearned models.

The current codebase focuses on the FPI-style dataset stored under `data_generator/data/`. The main model used by the core scripts is `deepseek-ai/deepseek-llm-7b-chat`; `llm_judge.py` and `scripts/env/download_model.py` also reference `Qwen/Qwen3-8B` for LLM-as-a-judge evaluation and model caching.

## Repository Structure

- `Finetune.py`: fine-tunes a causal LM on QA records, with LoRA enabled by default.
- `unlearn.py`: loads a fine-tuned model and applies unlearning.
- `UnlearnTrainer.py`: implements the supported unlearning losses.
- `evaluate.py`: evaluates base, fine-tuned, unlearned, and pretrained model variants.
- `recovery.py`: runs recovery/auditing methods against FPI attributes.
- `pretrain.py`: optional causal-LM pretraining entry point for JSON files with a `text` field.
- `prepdata.py`: shared dataset loading, prompt formatting, and tokenization utilities.
- `argsetting.py`: command-line argument definitions for fine-tuning, unlearning, and evaluation.
- `llm_judge.py`, `llm_prompt.py`: LLM-as-a-judge scoring utilities used by common-knowledge evaluation.
- `utils.py`: KL/DPO helpers and constrained-logit utilities for recovery.
- `data_generator/`: synthetic profile generation, QA dataset generation, and forget/retain/remain splitting.
- `scripts/`: Slurm/HPC templates for the main workflow.
- `scripts/env/`: dependency, environment-check, and model-download helpers.

There is no separate `configs/` directory in the current repository. Experiment settings are provided through CLI arguments and the shell scripts in `scripts/`.

## Installation

### Platform-Independent Dependencies

Create and activate a Python environment using your preferred tool, then install the repository dependencies:

```bash
pip install -r scripts/env/requirements.txt
```

The code imports `saved_hf_key.py` from the repository root or from `scripts/env/` depending on the entry point. This file is intentionally ignored by `.gitignore`. It should define:

```python
HF_key = "..."
```

Several model-loading paths use `local_files_only=True`, so the required models must already be available in the local Hugging Face cache or local model directories before running the main scripts.

### Environment Helpers

The repository includes:

```bash
cd scripts/env
bash setup_env.sh
cd ../..
bash scripts/env/check_env.sh
python scripts/env/download_model.py
```

These helpers contain cluster-oriented module commands and should be treated as templates unless your environment provides the same commands. `scripts/env/download_model.py` currently downloads the models listed in its `MODEL_LIST`.

## Dataset

The expected QA record format is a JSON list with entries such as:

```json
{
  "name": "Michael Ayers",
  "attribute": "year_of_birth",
  "question": "In which year was Michael Ayers born?",
  "answer": "Michael Ayers's year of birth is 1975."
}
```

The profile format used by `data_generator/create_profile.py` contains one object per synthetic person:

```json
{
  "name": "Michael Ayers",
  "year_of_birth": 1975,
  "address_postcode": "V8S6V2",
  "social_insurance_number": "773061452",
  "blood_type": "A+"
}
```

The repository already contains generated data under `data_generator/data/`, including:

- `training_dataset.json`
- `validation_dataset.json`
- `training_testset.json`
- `validation_testset.json`
- `common_knowledge_questions.json`
- `idk.jsonl`
- split directories such as `unlearn-N20-A1-yrb/`, `unlearn-N20-A1-bld/`, `unlearn-N20-A1-pcd/`, and `unlearn-N20-A1-sin/`

To regenerate the synthetic FPI data, run from `data_generator/` because the scripts use relative paths:

```bash
cd data_generator
python create_profile.py
python generate_dataset.py
bash split.sh
```

For custom splits, use `split_dataset.py` directly. The supported split arguments are defined in that file:

```bash
cd data_generator
python split_dataset.py \
  --folder_path data \
  --dataset_name training_dataset.json \
  --profile_name profiles.json \
  --num_profiles 5 \
  --max_retain_per_firstname 10 \
  --forget_mode n_per_firstname \
  --selected_attr year_of_birth \
  --suffix yrb
```

Supported `--forget_mode` values are `random`, `same_firstname`, `different_firstname`, `n_per_firstname`, and `random_combination`. Supported `--retain_mode` values are `all_except_forget`, `same_firstname`, `same_attr`, and `same_firstname_same_attr`.

## Quick Start

The shortest runnable path in the current repository is through the shell scripts in `scripts/`. They are Slurm/HPC-oriented templates, but each script also shows the exact Python entry point and parameters used for that stage.

To run the full five-stage Slurm pipeline:

```bash
bash scripts/submit_pipeline.sh
```

This submits the stages with `afterok` dependencies:

```text
scripts/train.sh -> scripts/eval_ft.sh -> scripts/unlearn.sh -> scripts/eval_unl.sh -> scripts/eval_recvr.sh
```

To run stages manually, execute the scripts one by one after adapting the cluster-specific header and paths for your environment.

### 1. Fine-Tuning

Fine-tune the configured model on the FPI training dataset:

```bash
bash scripts/train.sh
```

The script calls `Finetune.py`. Its configurable variables include `MODEL_NAME`, `DATA_DIR`, `LOG_DIR`, `MODEL_DIR`, `LR`, `EPOCHS`, `WEIGHT_DECAY`, `LORA_RANK`, `LORA_DROPOUT`, and `GRAD_ACC_STEPS`.

`Finetune.py` prepends `./data_generator/data/` to `--dataDIR` when `--datasetName FPI` is used. With the current `scripts/train.sh` settings, the fine-tuned checkpoint path is:

```text
fine_tuned_deepseek_7b/lr0.0005_WD0.01_loraRank256_loraDrop0.0_GradStsp40/epoch-30
```

Use `--without_lora` to run full-parameter fine-tuning instead of LoRA.

### 2. Unlearning

Supported unlearning methods are exactly the values in `argsetting.py`:

```text
grad_ascent, grad_diff, KL, po, dpo, npo, noisy_grad_diff
```

Run the configured unlearning stage:

```bash
bash scripts/unlearn.sh
```

The script calls `unlearn.py`. Its configurable variables include `METHOD`, `UNLEARN_SET`, `FORGET_SET`, `RETAIN_SET`, fine-tuned checkpoint settings prefixed with `FT_`, and unlearning hyperparameters such as `LR`, `EPOCHS`, `REG_WEIGHT`, `BETA`, and `GRAD_ACC_STEPS`.

With the current `scripts/unlearn.sh` settings, the unlearned checkpoint path is:

```text
unlearn_deepseek_7b/unlearn-N20-A1-yrb-lr0.0001_WD0.01_loraRank256_loraDrop0.0_GradStep80_reg5.0/grad_diff/epoch-50
```

For `po` and `dpo`, `unlearn.py` also loads `idk.jsonl` through `--idkSetDir`. For `noisy_grad_diff`, additional CLI arguments `--noisy_noise_std` and `--noisy_clip_norm` are available.

### 3. Auditing / Recovery

The recovery/auditing entry point is `recovery.py`. The code exposes three `--recover_type` choices:

```text
flip, beam, grad
```

The code does not expose CLI methods literally named `RIG` or `RG`; use the concrete `recovery.py` options above when running this repository.

Run the configured recovery audit:

```bash
bash scripts/eval_recvr.sh
```

The script calls `recovery.py`. Its configurable variables include `RECOVER_TYPE`, `FLIP`, `BEAM_K`, `BEAM_C`, `BEAM_N`, `QUANT`, and the same fine-tuned/unlearned checkpoint settings used by evaluation.

For `--recover_type beam`, `--K`, `--C`, `--N`, and `--entro` control beam candidates and optional entropy weighting. For `--recover_type grad`, `--recover_mode` can be `greedy` or `oracle`, and `--loss_type` can be `ce` or `npo`.

### 4. Evaluation

Evaluate the fine-tuned model:

```bash
bash scripts/eval_ft.sh
```

Evaluate the unlearned model:

```bash
bash scripts/eval_unl.sh
```

`scripts/eval_ft.sh` evaluates `forget`, `retain`, and `remain`. `scripts/eval_unl.sh` evaluates `forget`, `retain_sfa`, and `remain_sfa`.

Supported `--modelType` values are `base`, `learned`, `unlearned`, `pt`, and `pt-unlearned`. Supported `--datasetType` values are:

```text
train, val, train_t, val_t, common,
forget, retain, remain,
retain_sf, retain_sa, retain_sfa,
remain_sf, remain_sa, remain_sfa
```

For `--datasetType common`, `evaluate.py` uses `LLMJudge`, whose default judge model is `Qwen/Qwen3-8B`. The code expects this model to be available locally.

## Reproducing Experiments

The current repository provides one script-based workflow:

```bash
cd data_generator
python create_profile.py
python generate_dataset.py
bash split.sh
cd ..

bash scripts/submit_pipeline.sh
```

For non-Slurm use, follow the Python commands inside `scripts/train.sh`, `scripts/eval_ft.sh`, `scripts/unlearn.sh`, `scripts/eval_unl.sh`, and `scripts/eval_recvr.sh`, replacing only the cluster setup lines. Other paper experiments require selecting the desired `--unlearnSet`, `--datasetType`, `--unlearn_method`, and hyperparameters explicitly. The repository does not currently include a single platform-independent script that enumerates all paper configurations.

## Optional Pretraining Entry Point

`pretrain.py` supports causal-LM pretraining from JSON files with a `text` field. The active repository data under `data_generator/data/` is QA-formatted rather than plain causal-LM text, and no active script specifies a runnable pretraining dataset path. Use this entry point only after supplying a real `--train_file` accepted by `pretrain.py`.

## Slurm / HPC Usage

The current repository provides Slurm templates:

- `scripts/train.sh`
- `scripts/eval_ft.sh`
- `scripts/unlearn.sh`
- `scripts/eval_unl.sh`
- `scripts/eval_recvr.sh`
- `scripts/submit_pipeline.sh`

These scripts are HPC-specific templates. Before using them, edit the cluster directives, module commands, virtual environment path, working directory, model paths, and resource requests for your system.

To submit the current five-stage pipeline with Slurm dependencies:

```bash
bash scripts/submit_pipeline.sh
```

The dependency order in `scripts/submit_pipeline.sh` is:

```text
train.sh -> eval_ft.sh -> unlearn.sh -> eval_unl.sh -> eval_recvr.sh
```

## Outputs

The code constructs output directories from CLI arguments and hyperparameters.

Fine-tuning writes checkpoints and logs under:

```text
<modelDIR>/lr<lr>_WD<weight_decay>_loraRank<LoRA_rank>_loraDrop<lora_dropout>_GradStsp<grad_acc_steps>/
<logDIR>/lr<lr>_WD<weight_decay>_loraRank<LoRA_rank>_loraDrop<lora_dropout>_GradStsp<grad_acc_steps>/result.log
```

Unlearning writes under:

```text
<unlearn_model_DIR>/<unlearnSet>-lr<lr>_WD<weight_decay>_loraRank<LoRA_rank>_loraDrop<lora_dropout>_GradStep<grad_acc_steps>_reg<reg_weights>/<unlearn_method>/
<logDIR>/<unlearnSet>-lr<lr>_WD<weight_decay>_loraRank<LoRA_rank>_loraDrop<lora_dropout>_GradStep<grad_acc_steps>_reg<reg_weights>/<unlearn_method>/result.log
```

Evaluation writes JSON files under `--logDIR` for fine-tuned models and under `--logDIR_fgt` for unlearned models. Recovery writes JSON files under `--logDIR_recvr`.

`.gitignore` excludes generated caches, logs, checkpoints, and private credentials, including:

```text
__pycache__/
results*/
archive_logs/
fine_tuned_*/
unlearn_*/
recovery_*/
pretrain_*/
saved_hf_key.py
```

## Citation

TODO: Add the official citation or BibTeX entry when available.
