# TOFU COLM Rebuttal Pipeline

This directory contains a TOFU-only pipeline for the COLM rebuttal/follow-up experiments. It does not modify the root training, unlearning, evaluation, or recovery code. All new code lives under `tofu/`, `scripts/colm/`, and generated category copies under `TOFU/genres/`.

## Added Files

- `tofu/common.py`: shared JSON, category, prompt, metric, and adapter path helpers.
- `tofu/prepare_tofu_data.py`: converts raw TOFU JSONL files into project-readable JSON arrays.
- `tofu/classify_full_to_genres.py`: scans `TOFU/full.json` and copies records into fine-grained atomic genre files under `TOFU/genres/`.
- `tofu/evaluate_tofu.py`: TOFU basic evaluation for base + LoRA adapters.
- `tofu/audit_tofu_rank.py`: sentence-level answer rank audit and answer-token rank audit.
- `tofu/audit_tofu_grp_rank.py`: genre-based sentence-level and answer-token rank audit.
- `tofu/audit_tofu_key_rank.py`: key-token rank audit over `TOFU/keys/*_key_tokens.jsonl`.
- `tofu/build_key_tokens.py`: heuristic key-fact token annotation for forget splits.
- `tofu/aggregate_tofu_results.py`: aggregates `summary.json` files into `results/tofu/summary.csv`.
- `scripts/colm/00_prepare_tofu.sh`: runs preparation and category classification.
- `scripts/colm/01_train_tofu.sh`: trains target/oracle LoRA adapters.
- `scripts/colm/02_eval_ft_tofu.sh`: evaluates full and oracle adapters.
- `scripts/colm/03_unlearn_tofu.sh`: runs unlearning from the full target adapter.
- `scripts/colm/04_eval_unl_tofu.sh`: evaluates unlearned adapters.
- `scripts/colm/06_aggregate_tofu.sh`: aggregates eval/audit summaries.
- `scripts/colm/07_classify_tofu_full_genres.sh`: standalone category classification.
- `scripts/colm/08_audit_tofu_rank.sh`: rank-audit launcher.
- `scripts/colm/09_audit_grp_rank.sh`: genre-based rank-audit launcher.
- `scripts/colm/10_audit_key_rank.sh`: key-token rank-audit launcher.

## Directory Layout

- Raw TOFU data remains in `TOFU/*.json` and is not edited.
- Processed training/eval data is written to `tofu/processed/*.json`.
- Unlearning pair directories are written to:
  - `tofu/processed/tofu_fgt10_ret90/{forget.json,retain.json}`
  - `tofu/processed/tofu_fgt05_ret95/{forget.json,retain.json}`
  - `tofu/processed/tofu_fgt01_ret99/{forget.json,retain.json}`
- Fine-grained genre copies and reports are written to `TOFU/genres/`.
- LoRA adapters default to `results/tofu/adapters/`.
- Logs default to `results/tofu/logs/`.
- Eval/audit outputs default to `results/tofu/eval/`, `results/tofu/audit_rank/`, and `results/tofu/audit_grp_rank/`.
- Key-token annotations are written to `TOFU/keys/`.

## Data Preparation

Run:

```bash
bash scripts/colm/00_prepare_tofu.sh
```

This runs:

```bash
python tofu/prepare_tofu_data.py --raw_dir TOFU --output_dir tofu/processed
python tofu/classify_full_to_genres.py --input TOFU/full.json --output_dir TOFU/genres --single_label --write_reports
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

## Fine-Grained Genre Classification

`TOFU/full.json` is never modified. `tofu/classify_full_to_genres.py` first scans the full dataset, then copies each original record into a fine-grained primary genre file. The copied records keep the original `question` and `answer` unchanged and add only metadata such as `source_index`, `genre`, `primary_genre`, `secondary_genres`, `matched_rule`, `matched_rule_priority`, and `classifier_version`.

The classifier uses atomic genres based on question intent plus answer fact slot, such as:

- identity/name: `author_full_name`, `author_identity_who_is`
- birth: `birth_date`, `birth_place`, `birth_date_and_place`, `early_life_place`
- family: `parent_both_occupations`, `father_occupation`, `mother_occupation`
- genre and subject: `primary_genre`, `other_genres`, `genre_exclusivity`, `subject_matter`
- works: `book_list`, `book_title_single`, `debut_book`, `latest_book`, `book_plot_summary`
- style/process/influence/career/reception/personal-status slots

Default single-label classification:

```bash
python tofu/classify_full_to_genres.py \
  --input TOFU/full.json \
  --output_dir TOFU/genres \
  --single_label \
  --write_reports
```

Dry run with reports only:

```bash
python tofu/classify_full_to_genres.py \
  --input TOFU/full.json \
  --output_dir TOFU/genres \
  --dry_run \
  --write_reports
```

Print oversized genres:

```bash
python tofu/classify_full_to_genres.py \
  --input TOFU/full.json \
  --output_dir TOFU/genres \
  --single_label \
  --write_reports \
  --print_oversized
```

Genre reports:

- `classification_summary.json`: total records, genre count, mean/median/max count, oversized genres, unmatched/ambiguous counts, rule hit counts, and representative samples.
- `genre_scan_report.json`: same scan report, kept as the primary human-readable report file.
- `genre_counts.csv`: per-genre counts sorted by size.
- `oversized_genres.csv`: genres above 300 records plus natural split suggestions.
- `ambiguous_samples.jsonl`: records with multiple close-priority rule matches.
- `unmatched_samples.jsonl`: records that fell back to `other_unclear`.

Standalone Slurm classification:

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

## Genre-Based Rank Audit

The genre-based rank audit is the preferred residual-memory diagnostic when `TOFU/genres/` is available. It dynamically scans genre files for the current question, builds a much smaller answer candidate pool from only the matched genre files, and avoids ranking against the full `TOFU/full.json` answer pool. This makes both sentence-level rank and token-level rank audits faster and better scoped to plausible neighboring facts.

For each eval question, `tofu/audit_tofu_grp_rank.py`:

- finds all `TOFU/genres/*.json` or `*.jsonl` files containing the normalized question;
- merges unique answers from every matched genre file;
- appends the gold answer if it is missing;
- ranks the gold answer by average answer-token NLL;
- builds a per-question answer-token vocabulary from the genre candidate answers.

Important metrics:

- `sentence_rg_rank`: rank of the gold answer when lower average NLL is better.
- `sentence_rg_at1`: fraction of examples where `sentence_rg_rank == 1`.
- `sentence_rg_at5`: fraction of examples where the gold answer appears in the top 5 low-NLL candidates.
- `mean_token_rank`: mean rank of gold answer tokens within each question's genre answer-token vocabulary.
- `median_token_rank`: median gold-token rank across all ranked answer tokens.
- `candidate_pool_size`: number of candidate answers used for the current question.
- `matched_genres`: genre file stems whose records contained the current question.

FT smoke test:

```bash
LIMIT=20 AUDIT_MODE=ft EPOCH_LIST="25" SPLIT_LIST="forget10" sbatch --array=0-1 scripts/colm/09_audit_grp_rank.sh
```

UNL smoke test:

```bash
LIMIT=20 AUDIT_MODE=unl METHOD_LIST="npo" EPOCH_LIST="1" SPLIT_LIST="forget10" sbatch --array=0 scripts/colm/09_audit_grp_rank.sh
```

Formal FT run:

```bash
AUDIT_MODE=ft EPOCH_LIST="25" SPLIT_LIST="forget10" sbatch --array=0-1 scripts/colm/09_audit_grp_rank.sh
```

Formal UNL run:

```bash
AUDIT_MODE=unl METHOD_LIST="grad_ascent grad_diff KL npo" EPOCH_LIST="1 2 3 4 5 6 7 8 9 10" SPLIT_LIST="forget10" sbatch --array=0-39 scripts/colm/09_audit_grp_rank.sh
```

Generate key-token data for later specialized audits:

```bash
python tofu/build_key_tokens.py
```

Genre audit outputs:

```text
results/tofu/audit_grp_rank/
  target_full/
    lr0.0002_WD0.01_loraRank128_loraDrop0.0_GradStsp40/
      epoch-25-forget10-summary.json
      epoch-25-forget10-details.jsonl
      forget10-epoch_curve.csv

  oracle_retrain90/
    lr0.0002_WD0.01_loraRank128_loraDrop0.0_GradStsp40/
      epoch-25-forget10-summary.json
      epoch-25-forget10-details.jsonl
      forget10-epoch_curve.csv

  unlearned/
    tofu_fgt10_ret90-lr0.00001_WD0.01_loraRank128_loraDrop0.0_GradStep40_reg1.0/
      npo/
        epoch-1-forget10-summary.json
        epoch-1-forget10-details.jsonl
        epoch-2-forget10-summary.json
        epoch-2-forget10-details.jsonl
      forget10-epoch_curve.csv
```

## Key-Token Rank Audit

The key-token rank audit is a focused residual-memory diagnostic. It does not replace ordinary eval or sentence-level audit. It tests whether unlearning pushes annotated key fact tokens into low-logit regions, whether methods create key-token-level anti-preference, and whether RIG or flip recovery is better studied at the key-token layer than at sentence level.

Input key files are generated by:

```bash
python tofu/build_key_tokens.py
```

For each key fact, `tofu/audit_tofu_key_rank.py` aligns `char_start`/`char_end` against tokenizer offsets in the full answer. If offsets are unavailable, it falls back to normalized substring matching and then token-text fuzzy matching. Details retain failed alignments instead of dropping the original key fact.

Important metrics:

- `mean_key_token_rank`: average RG rank of annotated key tokens inside the key-token candidate vocabulary.
- `key_token_last_10pct`: fraction of key tokens whose RG percentile is in the worst 10 percent.
- `key_token_rig_at10`: fraction of key tokens ranked in the top 10 by reverse rank, where lower logits are better.
- `content_key_token_last_10pct`: same tail metric after filtering stopwords, punctuation, and template tokens.
- `weighted_mean_key_token_rank`: importance-weighted mean rank using `1 / importance_rank`.

FT smoke test:

```bash
LIMIT=5 AUDIT_MODE=ft EPOCH_LIST="25" SPLIT_LIST="forget10" sbatch --array=0 scripts/colm/10_audit_key_rank.sh
```

UNL smoke test:

```bash
LIMIT=5 AUDIT_MODE=unl METHOD_LIST="npo" EPOCH_LIST="1" SPLIT_LIST="forget10" sbatch --array=0 scripts/colm/10_audit_key_rank.sh
```

Formal FT run:

```bash
AUDIT_MODE=ft EPOCH_LIST="25" SPLIT_LIST="forget10" sbatch --array=0-1 scripts/colm/10_audit_key_rank.sh
```

Formal UNL run:

```bash
AUDIT_MODE=unl METHOD_LIST="grad_ascent grad_diff KL npo" EPOCH_LIST="1 2 3 4 5 6 7 8 9 10" SPLIT_LIST="forget10" sbatch --array=0-39 scripts/colm/10_audit_key_rank.sh
```

Key-rank outputs:

```text
results/tofu/audit_key_rank/
  target_full/
    lr0.0002_WD0.01_loraRank128_loraDrop0.0_GradStsp40/
      epoch-25-forget10-summary.json
      epoch-25-forget10-details.jsonl
      forget10-epoch_curve.csv

  oracle_retrain90/
    lr0.0002_WD0.01_loraRank128_loraDrop0.0_GradStsp40/
      epoch-25-forget10-summary.json
      epoch-25-forget10-details.jsonl
      forget10-epoch_curve.csv

  unlearned/
    tofu_fgt10_ret90-lr0.00001_WD0.01_loraRank128_loraDrop0.0_GradStep40_reg1.0/
      npo/
        epoch-1-forget10-summary.json
        epoch-1-forget10-details.jsonl
        forget10-epoch_curve.csv
```

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
bash -n scripts/colm/09_audit_grp_rank.sh
bash -n scripts/colm/10_audit_key_rank.sh
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

## Key-Rank Audit v3

### Key File Formats

`TOFU/keys/debug_full_key_tokens.jsonl` and `TOFU/keys/full_key_tokens.jsonl` share the same row schema:

- QA-level fields: `source_index`, `author_id`, `qa_index_within_author`, `question`, `answer`, `genre_group`, `key_facts`, `metadata`.
- Fact-level span fields: `fact_id`, `text`, `normalized_text`, `fact_group`, `importance_rank`, `importance_weight`, `char_start`, `char_end`, `char_alignment_method`, `char_alignment_failed`.

### Phase 1: Tokenized-Key Preprocessing

Script: `tofu/build_tokenized_key_file.py`

Purpose:

- Read key-file spans.
- Tokenize `answer`.
- Align each key fact span to answer tokens.
- Emit all-key tokens (`answer_token_*`) and content-filtered tokens (`content_token_*`).
- Record removed tokens and fallback behavior.
- Build token/vocab summaries for audit-v3.

Debug command:

```bash
python tofu/build_tokenized_key_file.py \
  --key_file TOFU/keys/debug_full_key_tokens.jsonl \
  --output_dir TOFU/keys/tokenized/deepseek-ai__deepseek-llm-7b-chat \
  --tokenizer_name deepseek-ai/deepseek-llm-7b-chat \
  --local_files_only
```

Debug outputs:

- `TOFU/keys/tokenized/deepseek-ai__deepseek-llm-7b-chat/debug_full_key_tokens_tokenized.jsonl`
- `TOFU/keys/tokenized/deepseek-ai__deepseek-llm-7b-chat/debug_tokenized_key_summary.json`
- `TOFU/keys/tokenized/deepseek-ai__deepseek-llm-7b-chat/debug_vocab_summary.json`

Token fields added per fact:

- `answer_token_indices`, `answer_token_ids`, `answer_token_texts`, `answer_token_offsets`
- `content_token_indices`, `content_token_ids`, `content_token_texts`, `content_token_offsets`
- `removed_token_indices`, `removed_token_ids`, `removed_token_texts`
- `content_fallback_to_all`, `token_alignment_failed`, `token_alignment_method`

Definitions:

- `all_key_tokens`: span-aligned full token sequence (`answer_token_*`); this is always used for span-level metrics.
- `content_key_tokens`: filtered token sequence used for token-level audit where applicable.

Filtering:

- `STOPWORDS_OR_TEMPLATE`: template/functional words and punctuation.
- `WEAK_MODIFIERS`: weak modifiers such as `noted`, `acclaimed`, `celebrated`, etc.

### Phase 1 Debug Acceptance (Actual Stats)

From `debug_tokenized_key_summary.json`:

- `num_input_rows = 4`
- `num_output_rows = 4`
- `row_parity = true`
- `source_index_parity = true`
- `num_key_facts = 12`
- `num_all_key_tokens = 37`
- `num_content_key_tokens = 31`
- `content_retention_ratio = 0.8378378378`
- `num_alignment_failed_facts = 0`
- `alignment_success_ratio = 1.0`
- `num_content_fallback_facts = 1`

### Phase 2: Audit v3

Script: `tofu/audit_key_rank.py`

Goal: compare residual and recoverable key-fact memory for `grad_ascent`, `grad_diff`, `KL`, and `npo`. The main RIG-recovery targets are `grad_ascent`, `grad_diff`, and `KL`; `npo` remains as a control.

Audit configs:

- `factgroup_content`
- `genre_allkey`
- `genre_content`
- `factgroup_type`

Metric namespaces:

- `span_metrics`: absolute gold span NLL only. This is not a rank metric.
- `token_metrics`: token candidate-set normal/RIG rank, utility, and candidate NLL.
- `span_rank_metrics`: span candidate-set normal/RIG or proxy-RIG rank, utility, and candidate NLL.

Token-level definitions:

- Candidate token set is deduplicated by `token_id`, scoped by audit config, and forced to include the gold token. Forced inclusion is recorded as `candidate_vocab_forced_include_gold`.
- Normal rank is `1 + count(z_c > z_gold)` inside the candidate set.
- RIG rank flips logits and is `1 + count(z_c < z_gold)`.
- Percentiles divide rank by `candidate_token_count`.
- Candidate-set utilities use stable logsumexp over candidate logits:
  `normal_token_gold_utility = softmax_C(z_gold)`;
  `rig_token_gold_utility = softmax_C(-z_gold)`.
- `normal_token_candidate_nll` and `rig_token_candidate_nll` are `-log(utility)`, not full-vocab NLL.

Span-level candidate rank definitions:

- Candidate spans come from the tokenized key file, deduped by `normalized_text`, while preserving a canonical original `text`. If the current gold span is among duplicates, the gold entry preserves the current gold text.
- Candidate scope defaults by config: `factgroup_content` and `factgroup_type` use `same_fact_group_key_spans`; `genre_allkey` and `genre_content` use `same_genre_key_spans`.
- Override scope with `--span_candidate_scope {same_fact_group_key_spans,same_genre_key_spans,same_genre_fact_group_key_spans,auto}`.
- Candidate spans are scored in the current QA slot only: `[INST] question [/INST] ` plus `answer[:char_start]`, then teacher-forced candidate span tokens.
- Normal span rank uses length-normalized avg NLL: `1 + count(avg_nll(c) < avg_nll(gold))`.
- `normal_span_avg_nll` is the absolute length-normalized NLL of the gold span; `normal_span_candidate_nll` is candidate-set softmax NLL over scores `-avg_nll`.
- `exact_rig` computes RIG span NLL with `softmax(-logits)` and emits exact fields such as `rig_span_rank`, `rig_span_at10`, and `rig_span_gold_utility`.
- `normal_and_proxy_rig` uses reverse normal span rank as a proxy and emits only proxy fields: `rig_span_rank_proxy`, `rig_span_at10_proxy`, `rig_span_gold_utility_proxy`, and `rig_span_candidate_nll_proxy`.

Config intent:

- `factgroup_content`: primary token, same-fact-group candidate vocab (content tokens).
- `genre_allkey`: primary span, same-genre candidate vocab (all tokens).
- `genre_content`: primary token, same-genre candidate vocab (content tokens).
- `factgroup_type`: mixed interpretation by `fact_group`; unmatched groups use `both_unknown`.

Important output fields:

- details JSONL remains one row per QA. Each key fact includes `candidate_token_count`, `candidate_span_count`, `span_metrics`, `token_metrics`, `key_tokens`, and `span_rank_metrics`.
- Candidate score dumps are disabled by default. Enable with `--save_candidate_scores`; `top_bottom_gold` saves normal top/bottom, RIG or proxy-RIG top, and gold; `all` saves every candidate.
- Summary includes `token_candidate_count_le10_frac` and `span_candidate_count_le10_frac` because top10 metrics are trivial on tiny candidate sets.
- Compatibility aliases are token-level only: `mean_rank_percentile = mean_normal_token_rank_percentile`, `last10pct = normal_token_last10pct`, `last25pct = normal_token_last25pct`, `rig_at10 = rig_token_at10`, and `mean_vocab_size = mean_candidate_token_count`.

New Python parameters:

```bash
--do_span_rank
--span_candidate_scope {same_fact_group_key_spans,same_genre_key_spans,same_genre_fact_group_key_spans,auto}
--span_rank_max_candidates 300
--span_rank_mode {normal_and_proxy_rig,exact_rig}
--span_rank_topk_to_save 10
--span_rank_batch_size 16
--save_candidate_scores
--candidate_scores_mode {top_bottom_gold,all}
--candidate_scores_topk 10
```

### Output Layout

All outputs are rooted at:

`results/tofu/audit_key_rank/{audit_config}/`

Unlearned:

`results/tofu/audit_key_rank/{audit_config}/unlearned/{unlearn_run}/{method}/`

FT:

`results/tofu/audit_key_rank/{audit_config}/ft/{model_tag}/{ft_run}/`

Files per run:

- `epoch-{epoch}-{split}-summary.json`
- `epoch-{epoch}-{split}-details.jsonl` (one row per QA)
- `epoch_curve-{split}.csv` (upsert by epoch/split/audit_config/method/model-run key)

### Slurm Launcher

Script: `scripts/colm/10_audit_key_rank.sh`

Key env vars:

- `AUDIT_MODE=unl|ft`
- `AUDIT_CONFIG_LIST="factgroup_content"`
- `METHOD_LIST="grad_ascent grad_diff KL npo"`
- `MODEL_TAG_LIST="target_full oracle_retrain90"` for `AUDIT_MODE=ft`
- `SPLIT_LIST="forget10"`
- `EPOCH_LIST="2 4 6 8 10"`
- `TOKENIZED_KEY_FILE=TOFU/keys/tokenized/deepseek-ai__deepseek-llm-7b-chat-v2_1/full_key_tokens_tokenized.jsonl`
- `DO_SPAN_RANK=0|1`
- `SAVE_CANDIDATE_SCORES=0|1`

FT array expansion:

`model_tag × split × epoch × audit_config`

Example:

```bash
AUDIT_MODE=ft \
MODEL_TAG_LIST="target_full oracle_retrain90" \
AUDIT_CONFIG_LIST="factgroup_content" \
SPLIT_LIST="forget10" \
EPOCH_LIST="25" \
sbatch --array=0-1 scripts/colm/10_audit_key_rank.sh
```

Unlearned array expansion:

`method × split × epoch × audit_config`

Method LR mapping:

- `grad_ascent`, `grad_diff`, `KL`: default `UNLEARN_LR=1e-05`
- `npo`: default `UNLEARN_LR=0.0005`
- Override with `UNLEARN_LR_OVERRIDE`.
- Adapter path preserves method case, including `KL`; use `METHOD_DIR_OVERRIDE` only when the actual directory differs.
- Default `UNLEARN_RUN` is `tofu_fgt10_ret90-lr${UNLEARN_LR}_WD0.01_loraRank128_loraDrop0.0_GradStep40_reg1.0`.

Span rank env vars passed by the launcher:

```bash
DO_SPAN_RANK=1
SPAN_CANDIDATE_SCOPE=auto
SPAN_RANK_MAX_CANDIDATES=300
SPAN_RANK_MODE=normal_and_proxy_rig
SPAN_RANK_TOPK_TO_SAVE=10
SPAN_RANK_BATCH_SIZE=16
```

Candidate score env vars:

```bash
SAVE_CANDIDATE_SCORES=1
CANDIDATE_SCORES_MODE=top_bottom_gold
CANDIDATE_SCORES_TOPK=10
```

Smoke test: FT sanity check:

```bash
AUDIT_MODE=ft \
MODEL_TAG_LIST="target_full oracle_retrain90" \
AUDIT_CONFIG_LIST="factgroup_content" \
SPLIT_LIST="forget10" \
EPOCH_LIST="25" \
DO_SPAN_RANK=1 \
SPAN_RANK_MAX_CANDIDATES=50 \
SPAN_RANK_MODE="normal_and_proxy_rig" \
SAVE_CANDIDATE_SCORES=1 \
CANDIDATE_SCORES_MODE="top_bottom_gold" \
sbatch --array=0-1 scripts/colm/10_audit_key_rank.sh
```

Smoke test: unlearned:

```bash
AUDIT_MODE=unl \
METHOD_LIST="grad_ascent grad_diff KL npo" \
AUDIT_CONFIG_LIST="factgroup_content" \
SPLIT_LIST="forget10" \
EPOCH_LIST="10" \
DO_SPAN_RANK=1 \
SPAN_RANK_MAX_CANDIDATES=50 \
SPAN_RANK_MODE="normal_and_proxy_rig" \
SAVE_CANDIDATE_SCORES=1 \
CANDIDATE_SCORES_MODE="top_bottom_gold" \
sbatch --array=0-3 scripts/colm/10_audit_key_rank.sh
```

Formal tokenization command:

```bash
python tofu/build_tokenized_key_file.py \
  --key_file TOFU/keys/full_key_tokens.jsonl \
  --output_dir TOFU/keys/tokenized/deepseek-ai__deepseek-llm-7b-chat-v2_1 \
  --tokenizer_name deepseek-ai/deepseek-llm-7b-chat \
  --local_files_only
```

Formal unlearned sweep:

```bash
AUDIT_MODE=unl \
METHOD_LIST="grad_ascent grad_diff KL npo" \
AUDIT_CONFIG_LIST="factgroup_content genre_allkey genre_content factgroup_type" \
SPLIT_LIST="forget10" \
EPOCH_LIST="2 4 6 8 10" \
sbatch --array=0-79 scripts/colm/10_audit_key_rank.sh
```

Check v3 outputs for:

- `token_metrics.normal_token_top10`
- `token_metrics.normal_token_last10pct`
- `token_metrics.rig_token_at10`
- `span_rank_metrics.normal_span_top10`
- `span_rank_metrics.normal_span_last10pct`
- `span_rank_metrics.rig_span_at10` in `exact_rig`, or `span_rank_metrics.rig_span_at10_proxy` in `normal_and_proxy_rig`.

## 05 Key Memory and Recovery Evaluation

This section documents the TOFU 05 pipeline, which is separate from `10_audit_key_rank` and focuses on key-fact memory, forgetting, retention preservation, and constrained recovery probes.

### Purpose

The 05 pipeline adds two complementary evaluations over tokenized key facts:

- `key_memory`: natural (unconstrained) memory behavior under normal generation and full-vocab span scoring.
- `key_recovery`: constrained RG/RIG recovery behavior under scoped content-key vocab restrictions.

### Difference Between `key_memory` and `key_recovery`

- `key_memory` (`tofu/05_eval_key_memory.py`):
  - Metrics: Open-Key-Recall and KeySpanAvgNLL.
  - No RG/RIG, no constrained vocab decoding.
- `key_recovery` (`tofu/05_eval_key_recovery.py`):
  - Metrics: Constrained Key Recall and Masked KeySpanAvgNLL under RG and RIG.
  - Uses scoped content-key candidate vocabularies.

### Prompt Template

The pipeline reuses the existing helper `tofu/common.py::make_prompt`:

```text
Question: {question}
Answer:
```

- Open generation context: `prompt(question)`.
- Span NLL context: `prompt(question) + answer[:char_start]`.

### Key Metric Definitions

- Open-Key-Recall:
  - Fact hit = `open_span_hit OR (content_token_recall >= threshold)`.
  - `open_span_hit` is treated as the primary signal.
- Open-Span-Recall:
  - Fraction of facts where normalized key-fact text appears in normalized generation.
- Open-Content-Token-Recall:
  - Fraction of content key tokens recovered in normalized generation.
  - One-character fragment matches are ignored unless numeric or acronym-like.
- KeySpanAvgNLL:
  - Full-vocab teacher-forced avg NLL over `answer_token_ids` span tokens.
- Constrained Key Recall:
  - Key-slot generation under scope-constrained candidate vocab.
  - RG uses logits; RIG uses flipped logits.
- Masked KeySpanAvgNLL:
  - Span NLL computed with softmax restricted to candidate set.
  - RG: restricted logits; RIG: restricted flipped logits.
- RG:
  - Restricted generation/scoring using normal logits.
- RIG:
  - Restricted generation/scoring using negated logits.

### Constrained Decoding EOS Rule

For `05_eval_key_recovery.py` constrained generation:

- EOS is not allowed at generation step 0.
- EOS is allowed only after at least one generated token.
- Per-fact details record `stop_reason_rg` and `stop_reason_rig` (`eos`, `max_key_tokens`, `no_valid_token`).

### Constraint Scopes

Supported `constraint_scope` values:

- `same_genre_fact_group_content_vocab`
- `same_fact_group_content_vocab`
- `same_genre_content_vocab`
- `full_vocab`

`full_vocab` keeps the key-phrase scoring target but removes the candidate-token restriction. RG scores the gold key span with a softmax over the model's full vocabulary, and RIG does the same after negating logits. Outputs keep the existing `masked_key_span_*` and `mean_masked_key_span_*` field names for compatibility; when `constraint_scope=full_vocab`, "masked" only means the reused field name, and the actual computation is full-vocab key-span scoring. Summaries and details record `candidate_vocab_type: full_vocab`.

### Input Tokenized Key File Format

Input example:

- `TOFU/keys/tokenized/.../full_key_tokens_tokenized.jsonl`

Each row contains TOFU QA fields and `key_facts`. Important fields include:

- `answer_token_*`: full key span tokens for KeySpanAvgNLL.
- `content_token_*`: cleaned factual content tokens for constrained vocab and recall.

Matching priority between eval rows and key rows:

1. `source_index`
2. exact `question + answer`
3. normalized `question + answer`
4. normalized `question` only when unique

### Masked-NLL Candidate Set and Forced Gold Tracking

In recovery scoring:

- Candidate set is content-vocab plus forced gold span token IDs.
- For `constraint_scope=full_vocab`, no candidate set is materialized; candidate counts are recorded as the model embedding vocabulary size, forced-gold and fallback flags are false, and `small_candidate_set` is false.
- Details include:
  - `candidate_token_count_before_gold`
  - `candidate_token_count`
  - `candidate_forced_gold_tokens`
  - `forced_gold_token_count`
  - `small_candidate_set`
  - `candidate_vocab_type`

### Output Directory Layout

Memory:

```text
results/tofu/key_memory/
  ft/{model_tag}/{ft_run}/
  unlearned/{unlearn_run}/{method}/
```

Recovery:

```text
results/tofu/key_recovery/{constraint_scope}/
  ft/{model_tag}/{ft_run}/
  unlearned/{unlearn_run}/{method}/
```

Each run writes:

- `epoch-{epoch}-{split}-summary.json`
- `epoch-{epoch}-{split}-details.jsonl`
- `epoch_curve-{split}.csv` (upsert; no duplicate-key blind append)

Collector outputs:

```text
results/tofu/key_eval_reports/
  all_key_memory_summaries.csv
  all_key_recovery_summaries.csv
  all_key_eval_merged.csv
  best_checkpoints.csv
  per_fact_group_memory.csv
  per_fact_group_recovery.csv
  figures/*.png
  figures/*.pdf
```

### Slurm Variables: Memory Launcher

Script: `scripts/colm/05_eval_key_memory.sh`

- `AUDIT_MODE=ft|unl`
- `MODEL_TAG_LIST`, `METHOD_LIST`, `DATASET_LIST`, `EPOCH_LIST`
- `TOKENIZED_KEY_FILE`
- `FT_RUN`
- `METHOD_LR_MAP`
- `UNLEARN_LR_OVERRIDE`
- `UNLEARN_RUN`
- `MAX_NEW_TOKENS`, `GENERATION_BATCH_SIZE`, `NLL_BATCH_SIZE`, `LIMIT`

Array logic:

- FT: `model_tag × dataset × epoch`
- UNL: `method × dataset × epoch`

### Slurm Variables: Recovery Launcher

Script: `scripts/colm/05_eval_key_recovery.sh`

Includes all memory vars plus:

- `CONSTRAINT_SCOPE_LIST`
- `MAX_KEY_TOKENS`
- `MIN_CANDIDATE_TOKEN_COUNT`
- `CONTENT_RECALL_HIT_THRESHOLD`
- `SKIP_CONSTRAINED_GENERATION`
- `SKIP_MASKED_NLL`
- `SAVE_CANDIDATE_DEBUG`

Array logic:

- FT: `model_tag × dataset × epoch × constraint_scope`
- UNL: `method × dataset × epoch × constraint_scope`

### Method-LR Mapping

Default map:

- `grad_ascent -> 1e-05`
- `grad_diff -> 1e-05`
- `KL -> 1e-05`
- `npo -> 0.0005`

Config string:

```text
METHOD_LR_MAP="grad_ascent:1e-05,grad_diff:1e-05,KL:1e-05,npo:0.0005"
```

Override all methods with `UNLEARN_LR_OVERRIDE`.

### Collector Usage

Script: `scripts/colm/05_collect_key_eval_results.sh`

- `MEMORY_ROOT`
- `RECOVERY_ROOT`
- `OUTPUT_DIR`
- `MAKE_PLOTS=1|0`
- `MIN_FACT_GROUP_COUNT`

Run:

```bash
bash scripts/colm/05_collect_key_eval_results.sh
```

or:

```bash
sbatch scripts/colm/05_collect_key_eval_results.sh
```

The collector merges memory and recovery by:

- `model_family`, `model_tag`, `method`, `unlearn_run`, `lr`, `epoch`, `split`

and keeps `constraint_scope` as recovery-specific dimension.

### Plot List

Generated when data is available:

- `memory_forget_retain_tradeoff`
- `open_key_recall_by_epoch`
- `keyspan_nll_by_epoch`
- `ckr_rg_vs_rig`
- `masked_ksnll_rg_vs_rig`
- `recovery_gain_by_fact_group`
- `method_constraint_heatmap`
- `target_oracle_unlearned_comparison`

Each figure is saved as both `.png` and `.pdf`.

### Smoke Test Commands

Memory FT smoke:

```bash
AUDIT_MODE=ft \
MODEL_TAG_LIST="target_full oracle_retrain90" \
DATASET_LIST="forget10" \
EPOCH_LIST="25" \
LIMIT=5 \
sbatch --array=0-1 scripts/colm/05_eval_key_memory.sh
```

Recovery UNL smoke:

```bash
AUDIT_MODE=unl \
METHOD_LIST="grad_ascent KL" \
DATASET_LIST="forget10" \
EPOCH_LIST="10" \
CONSTRAINT_SCOPE_LIST="same_genre_fact_group_content_vocab" \
LIMIT=5 \
sbatch --array=0-1 scripts/colm/05_eval_key_recovery.sh
```

Collector:

```bash
bash scripts/colm/05_collect_key_eval_results.sh
```

### Formal Sweep Commands

Memory sweep:

```bash
AUDIT_MODE=unl \
METHOD_LIST="grad_ascent grad_diff KL npo" \
DATASET_LIST="forget10 retain90" \
EPOCH_LIST="2 4 6 8 10" \
sbatch --array=0-39 scripts/colm/05_eval_key_memory.sh
```

Recovery sweep:

```bash
AUDIT_MODE=unl \
METHOD_LIST="grad_ascent grad_diff KL npo" \
DATASET_LIST="forget10 retain90" \
EPOCH_LIST="2 4 6 8 10" \
CONSTRAINT_SCOPE_LIST="same_genre_fact_group_content_vocab same_fact_group_content_vocab same_genre_content_vocab full_vocab" \
sbatch --array=0-159 scripts/colm/05_eval_key_recovery.sh
```

FT baseline memory:

```bash
AUDIT_MODE=ft \
MODEL_TAG_LIST="target_full oracle_retrain90" \
DATASET_LIST="forget10 retain90" \
EPOCH_LIST="25" \
sbatch --array=0-3 scripts/colm/05_eval_key_memory.sh
```

FT baseline recovery:

```bash
AUDIT_MODE=ft \
MODEL_TAG_LIST="target_full oracle_retrain90" \
DATASET_LIST="forget10 retain90" \
EPOCH_LIST="25" \
CONSTRAINT_SCOPE_LIST="same_genre_fact_group_content_vocab same_fact_group_content_vocab same_genre_content_vocab full_vocab" \
sbatch --array=0-15 scripts/colm/05_eval_key_recovery.sh
```

### Interpretation Guide

- `target_full` should usually show high Open-Key-Recall and low KeySpanAvgNLL on `forget10`.
- `oracle_retrain90` should typically show lower `forget10` memory.
- Successful unlearning should reduce `forget10` recall and increase `forget10` NLL while preserving `retain90` behavior.
- Successful RIG recovery should show `CKR_RIG > CKR_RG` and `M-KSNLL_RIG < M-KSNLL_RG`.

### Runtime Success Reporting Rule

Do not claim runtime success unless actual `epoch-*-summary.json` files are produced.
Static checks, `--help`, and launcher syntax checks are necessary but not sufficient evidence of experiment success.
