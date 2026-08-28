#!/bin/bash
set -euo pipefail

# ============================================================
# Run LLM-assisted TOFU full key extraction with SiliconFlow
# Model: deepseek-ai/DeepSeek-V4-Pro
# ============================================================

PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery}

INPUT=${INPUT:-TOFU/full.json}
GENRES_DIR=${GENRES_DIR:-TOFU/genres}

OUTPUT=${OUTPUT:-TOFU/keys/full_key_tokens.jsonl}
SUMMARY=${SUMMARY:-TOFU/keys/full_key_extraction_summary.json}
QUALITY_REPORT=${QUALITY_REPORT:-TOFU/keys/full_key_extraction_quality_report.json}
RAW_LLM_OUTPUT=${RAW_LLM_OUTPUT:-TOFU/keys/full_key_llm_raw.jsonl}

# SiliconFlow
export OPENAI_API_BASE=${OPENAI_API_BASE:-https://api.siliconflow.cn/v1}
export OPENAI_MODEL=${OPENAI_MODEL:-deepseek-ai/DeepSeek-V4-Pro}

# The Python script uses --api_key_env to read this env var.
# Put your real key in SILICONFLOW_API_KEY before running:
#   export SILICONFLOW_API_KEY="sk-..."
API_KEY_ENV=${API_KEY_ENV:-SILICONFLOW_API_KEY}

BATCH_SIZE=${BATCH_SIZE:-10}
MAX_RETRIES=${MAX_RETRIES:-5}
SAMPLE_PER_GROUP=${SAMPLE_PER_GROUP:-5}

# Optional debug mode:
# Need Python script to support --limit before using this.
LIMIT=${LIMIT:-}
RESUME=${RESUME:-0}

cd "$PROJECT_DIR"

mkdir -p TOFU/keys

echo "[run-key-sf] PROJECT_DIR=$PROJECT_DIR"
echo "[run-key-sf] INPUT=$INPUT"
echo "[run-key-sf] GENRES_DIR=$GENRES_DIR"
echo "[run-key-sf] OUTPUT=$OUTPUT"
echo "[run-key-sf] SUMMARY=$SUMMARY"
echo "[run-key-sf] QUALITY_REPORT=$QUALITY_REPORT"
echo "[run-key-sf] RAW_LLM_OUTPUT=$RAW_LLM_OUTPUT"
echo "[run-key-sf] OPENAI_API_BASE=$OPENAI_API_BASE"
echo "[run-key-sf] OPENAI_MODEL=$OPENAI_MODEL"
echo "[run-key-sf] API_KEY_ENV=$API_KEY_ENV"
echo "[run-key-sf] BATCH_SIZE=$BATCH_SIZE"
echo "[run-key-sf] RESUME=$RESUME"
echo "[run-key-sf] LIMIT=${LIMIT:-none}"

if [ ! -f "$INPUT" ]; then
  echo "[run-key-sf][ERROR] Missing input: $INPUT"
  exit 1
fi

if [ ! -d "$GENRES_DIR" ]; then
  echo "[run-key-sf][ERROR] Missing genres dir: $GENRES_DIR"
  exit 1
fi

if [ -z "${!API_KEY_ENV:-}" ]; then
  echo "[run-key-sf][ERROR] Missing API key env: $API_KEY_ENV"
  echo "Set it first, e.g.:"
  echo "  export SILICONFLOW_API_KEY='sk-...'"
  exit 1
fi

# Load environment if on ComputeCanada.
# If running locally, these module commands may not exist; ignore failures.
if command -v module >/dev/null 2>&1; then
  module load python/3.10 || true
  module load scipy-stack || true
fi

if [ -f "$HOME/ENV-3.10/bin/activate" ]; then
  source "$HOME/ENV-3.10/bin/activate"
fi

echo "[run-key-sf] Python=$(which python)"
python - <<'PY'
import sys
print("python:", sys.version)
try:
    import requests
    print("requests: ok")
except Exception as e:
    raise SystemExit(f"requests missing: {e}")
PY

ARGS=(
  --input "$INPUT"
  --genres_dir "$GENRES_DIR"
  --output "$OUTPUT"
  --summary "$SUMMARY"
  --quality_report "$QUALITY_REPORT"
  --raw_llm_output "$RAW_LLM_OUTPUT"
  --model "$OPENAI_MODEL"
  --api_key_env "$API_KEY_ENV"
  --api_base "$OPENAI_API_BASE"
  --batch_size "$BATCH_SIZE"
  --max_retries "$MAX_RETRIES"
  --sample_per_group "$SAMPLE_PER_GROUP"
)

if [ "$RESUME" = "1" ]; then
  ARGS+=(--resume)
fi

# Only works if you add --limit support to the Python script.
if [ -n "$LIMIT" ]; then
  ARGS+=(--limit "$LIMIT")
fi

echo "[run-key-sf] Start extraction..."
python -u tofu/build_full_key_tokens_llm.py "${ARGS[@]}"

echo "[run-key-sf] Done. Validate outputs..."

python - <<PY
import json
from pathlib import Path

input_path = Path("$INPUT")
out_path = Path("$OUTPUT")
summary_path = Path("$SUMMARY")

def read_records(path):
    text = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except Exception:
        rows = []
        for line in text.splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

inp = read_records(input_path)
rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
summary = json.loads(summary_path.read_text(encoding="utf-8"))

print("[validate] input_records =", len(inp))
print("[validate] output_records =", len(rows))
print("[validate] token_alignment =", summary.get("token_alignment"))
print("[validate] num_key_facts =", summary.get("num_key_facts"))
print("[validate] num_char_alignment_failed =", summary.get("num_char_alignment_failed"))

assert len(rows) == len(inp), "output row count must match input count"
assert [r["source_index"] for r in rows] == list(range(len(rows))), "source_index must be contiguous"

facts = [f for r in rows for f in r.get("key_facts", [])]
print("[validate] facts =", len(facts))
missing_chars = sum(1 for f in facts if f.get("char_start") is None or f.get("char_end") is None)
print("[validate] facts missing char spans =", missing_chars)

print("[validate] output:", out_path)
print("[validate] summary:", summary_path)
PY
