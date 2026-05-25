# TOFU COLM Rebuttal Pipeline

This directory contains a TOFU-only pipeline for the COLM rebuttal/follow-up experiments. It does not modify the root training, unlearning, evaluation, or recovery code. All new code lives under `tofu/`, `scripts/colm/`, and generated category copies under `TOFU/genres/`.

## Added Files

- `tofu/common.py`: shared JSON, category, prompt, metric, and adapter path helpers.
- `tofu/prepare_tofu_data.py`: converts raw TOFU JSONL files into project-readable JSON arrays.
- `tofu/classify_full_to_genres.py`: copies `TOFU/full.json` records into keyword categories under `TOFU/genres/`.
- `tofu/evaluate_tofu.py`: TOFU basic evaluation for base + LoRA adapters.
- `tofu/audit_tofu_rank.py`: sentence-level answer rank audit and answer-token rank audit.
- `tofu/aggregate_tofu_results.py`: aggregates `summary.json` files into `results/tofu/summary.csv`.
- `scripts/colm/00_prepare_tofu.sh`: runs preparation and category classification.
- `scripts/colm/01_train_tofu.sh`: trains target/oracle LoRA adapters.
- `scripts/colm/02_eval_ft_tofu.sh`: evaluates full and oracle adapters.
- `scripts/colm/03_unlearn_tofu.sh`: runs unlearning from the full target adapter.
- `scripts/colm/04_eval_unl_tofu.sh`: evaluates unlearned adapters.
- `scripts/colm/06_aggregate_tofu.sh`: aggregates eval/audit summaries.
- `scripts/colm/07_classify_tofu_full_genres.sh`: standalone category classification.
- `scripts/colm/08_audit_tofu_rank.sh`: rank-audit launcher.

## Directory Layout

- Raw TOFU data remains in `TOFU/*.json` and is not edited.
- Processed training/eval data is written to `tofu/processed/*.json`.
- Unlearning pair directories are written to:
  - `tofu/processed/tofu_fgt10_ret90/{forget.json,retain.json}`
  - `tofu/processed/tofu_fgt05_ret95/{forget.json,retain.json}`
  - `tofu/processed/tofu_fgt01_ret99/{forget.json,retain.json}`
- Category copies are written to `TOFU/genres/*.json` and `TOFU/genres/manifest.json`.
- LoRA adapters default to `results/tofu/adapters/`.
- Logs default to `results/tofu/logs/`.
- Eval/audit outputs default to `results/tofu/eval/` and `results/tofu/audit_rank/`.

## Data Preparation

Run:

```bash
bash scripts/colm/00_prepare_tofu.sh
```

This runs:

```bash
python tofu/prepare_tofu_data.py --raw_dir TOFU --output_dir tofu/processed
python tofu/classify_full_to_genres.py --full_json TOFU/full.json --output_dir TOFU/genres
```

The raw TOFU files are JSONL. The project training code expects `json.load(...)` on a JSON array, so `prepare_tofu_data.py` writes arrays with compatibility aliases:

```json
{
  "question": "...",
  "answer": "...",
  "prompt": "Question: ...\nAnswer: ",
  "completion": "...",
  "input": "Question: ...\nAnswer: ",
  "output": "...",
  "text": "Question: ...\nAnswer: ...",
  "split": "forget10",
  "category": "genre"
}
```

## Genre Classification

`TOFU/full.json` is never modified. `tofu/classify_full_to_genres.py` copies each original record into one category file and adds only `category` and `source_index` in the copied version.

Rules are applied in this order:

- `genre`: question contains `genre`
- `birth`: `born`, `birth`, `birthplace`, `year of birth`
- `name`: `full name`, `author's name`, `name of the author`
- `award`: `award`, `prize`, `honor`, `honour`
- `parent_occupation`: `father`, `mother`, `parent`, `occupation`
- `book_work`: `book`, `novel`, `work`, `title`
- `theme`: `theme`, `common theme`
- `writing_style`: `style`, `writing style`, `narrative style`
- `other`: fallback

Standalone classification:

```bash
bash scripts/colm/07_classify_tofu_full_genres.sh
```

## LoRA Adapter Storage

The root `Finetune.py` wraps the base model with PEFT LoRA and calls `save_pretrained`, so adapter directories contain files such as `adapter_config.json` and `adapter_model.safetensors`; full 7B checkpoints are not saved by this pipeline.

Fine-tuned adapter path:

```text
<modelDIR>/lr{lr}_WD{wd}_loraRank{rank}_loraDrop{dropout}_GradStsp{grad_acc}/epoch-{epochs}
```

Unlearned adapter path:

```text
<unlearn_model_DIR>/<unlearnSet>-lr{lr}_WD{wd}_loraRank{rank}_loraDrop{dropout}_GradStep{grad_acc}_reg{reg}/{method}/epoch-{epochs}
```

Unlearning follows the existing root logic: load base model, load target full LoRA adapter, merge it into the base model, then train/save a new LoRA adapter for unlearning.

## Training

Submit:

```bash
sbatch scripts/colm/01_train_tofu.sh
```

Array mapping:

- `0`: `target_full`, data `tofu/processed/full.json`
- `1`: `oracle_retrain90`, data `tofu/processed/retain90.json`
- `2`: `oracle_retrain95`, data `tofu/processed/retain95.json`
- `3`: `oracle_retrain99`, data `tofu/processed/retain99.json`

Defaults:

- base model: `deepseek-ai/deepseek-llm-7b-chat`
- LoRA rank: `128`
- dropout: `0.0`
- lr: `2e-4`
- epochs: `10`
- weight decay: `0.01`
- grad accumulation: `40`
- GPUs: `4`
- time: `03:00`

To rerun only the full target:

```bash
sbatch --array=0 scripts/colm/01_train_tofu.sh
```

## Unlearning

Default first round is `forget10 + retain90`:

```bash
sbatch scripts/colm/03_unlearn_tofu.sh
```

Default grid:

- methods: `grad_diff`, `KL`, `grad_ascent`, `npo`
- lr: `1e-5`, `5e-5`
- epochs: `1`, `3`, `5`, `10`
- reg: `1.0`
- beta: `0.1`
- LoRA rank: `128`
- grad accumulation: `40`

Array mapping is:

```text
method_idx = IDX / 8
lr_idx     = (IDX / 4) % 2
epoch_idx  = IDX % 4
```

To run `forget05 + retain95`:

```bash
UNLEARN_SET=tofu_fgt05_ret95 PAIR_DIR=tofu/processed/tofu_fgt05_ret95 sbatch scripts/colm/03_unlearn_tofu.sh
```

To run `forget01 + retain99`:

```bash
UNLEARN_SET=tofu_fgt01_ret99 PAIR_DIR=tofu/processed/tofu_fgt01_ret99 sbatch scripts/colm/03_unlearn_tofu.sh
```

To rerun one array job:

```bash
sbatch --array=0 scripts/colm/03_unlearn_tofu.sh
```

## Evaluation

Evaluate fine-tuned and oracle adapters:

```bash
sbatch scripts/colm/02_eval_ft_tofu.sh
```

This evaluates:

- `target_full` on `forget10`, `forget05`, `forget01`, `retain90`, `retain95`, `retain99`, `real_authors`, `world_facts`
- `oracle_retrain90` on `forget10`, `retain90`
- `oracle_retrain95` on `forget05`, `retain95`
- `oracle_retrain99` on `forget01`, `retain99`

Evaluate unlearned adapters:

```bash
sbatch scripts/colm/04_eval_unl_tofu.sh
```

The unlearned eval defaults to `forget10`, `retain90`, `real_authors`, and `world_facts` for each method.

`tofu/evaluate_tofu.py` writes:

- `summary.json`: aggregate avg NLL, normalized probability, ROUGE-1 recall, ROUGE-L recall
- `details.jsonl`: per-record question, answer, generation, and metrics

## Rank Audit

First-round audit focuses on `forget10` and does not run recovery:

```bash
sbatch scripts/colm/08_audit_tofu_rank.sh
```

Default model array:

- `0`: `target_full`
- `1`: `oracle_retrain90`
- `2`: `unlearned_grad_ascent`
- `3`: `unlearned_grad_diff`
- `4`: `unlearned_KL`
- `5`: `unlearned_npo`

For a small smoke test:

```bash
LIMIT=5 bash scripts/colm/08_audit_tofu_rank.sh
```

The script supports manual adapter overrides:

```bash
TARGET_ADAPTER_DIR=/path/to/full/adapter \
ORACLE_ADAPTER_DIR=/path/to/oracle/adapter \
UNLEARN_ADAPTER_DIR=/path/to/unlearn/adapter \
MODEL_IDX=3 LIMIT=5 bash scripts/colm/08_audit_tofu_rank.sh
```

Sentence-level audit ranks the gold answer among candidate answers by average answer-token NLL:

- `sentence_rg_rank`: lower NLL is better
- `sentence_rig_rank`: higher NLL is better
- `sentence_rg_top5_contains_gold`
- `sentence_rig_top5_contains_gold`

Token-level audit builds an answer-token vocabulary from `TOFU/full.json` answers only. It ranks each gold answer token within this answer-token vocabulary, not the full model vocabulary.

Outputs:

- `results/tofu/audit_rank/<split>/<model_tag>/summary.json`
- `results/tofu/audit_rank/<split>/<model_tag>/details.jsonl`

## Aggregation

Run:

```bash
bash scripts/colm/06_aggregate_tofu.sh
```

This scans `results/tofu/**/summary.json` and writes:

```text
results/tofu/summary.csv
```

When matching oracle audit summaries are present, it also writes audit advantage columns:

- `audit_advantage_sentence_rg_at5`
- `audit_advantage_sentence_rig_at5`
- `token_rank_gap_mean`

These are diagnostic audit advantages, not final privacy leakage claims.

## Logs And Reruns

SBATCH logs are written under:

```text
results/tofu/logs/
```

Check a running job:

```bash
squeue -u "$USER"
```

Rerun one array task:

```bash
sbatch --array=<idx> scripts/colm/<script>.sh
```

Run a shell syntax check:

```bash
bash -n scripts/colm/00_prepare_tofu.sh
bash -n scripts/colm/01_train_tofu.sh
bash -n scripts/colm/02_eval_ft_tofu.sh
bash -n scripts/colm/03_unlearn_tofu.sh
bash -n scripts/colm/04_eval_unl_tofu.sh
bash -n scripts/colm/06_aggregate_tofu.sh
bash -n scripts/colm/07_classify_tofu_full_genres.sh
bash -n scripts/colm/08_audit_tofu_rank.sh
python -m py_compile tofu/*.py
```

## Current Limitations

- This first version implements audit/rank diagnostics, not a complete recovery attack.
- Token-level audit ranks only within the answer-token vocabulary, not the full vocabulary.
- Truth Ratio and KS-Test are TODOs.
- Root project files such as `Finetune.py`, `unlearn.py`, `evaluate.py`, `recovery.py`, `prepdata.py`, and `utils.py` are not modified.
- `real_authors` and `world_facts` are multiple-choice style TOFU files; this pipeline treats their `answer` field as the target answer string for basic eval.

## Local Validation Notes

The initial implementation was validated on the current login environment with:

```bash
bash -n scripts/colm/00_prepare_tofu.sh scripts/colm/01_train_tofu.sh scripts/colm/02_eval_ft_tofu.sh scripts/colm/03_unlearn_tofu.sh scripts/colm/04_eval_unl_tofu.sh scripts/colm/06_aggregate_tofu.sh scripts/colm/07_classify_tofu_full_genres.sh scripts/colm/08_audit_tofu_rank.sh
python -m py_compile tofu/*.py
bash scripts/colm/00_prepare_tofu.sh
bash scripts/colm/06_aggregate_tofu.sh
```

Data preparation produced the expected files, including `tofu/processed/full.json`, `tofu/processed/forget10.json`, `TOFU/genres/genre.json`, and `TOFU/genres/manifest.json`.

The small 7B audit smoke test was not run on the login node because `$HOME/ENV-3.10` reported `torch.cuda.is_available() == False`, and the new TOFU target adapter path under `results/tofu/adapters/ft/target_full/.../epoch-10` does not exist until `01_train_tofu.sh` completes. `sbatch --test-only scripts/colm/08_audit_tofu_rank.sh` could not contact Slurm from this sandbox and repeatedly returned `Error creating slurm stream socket: Operation not permitted`; no Slurm job id was created.
