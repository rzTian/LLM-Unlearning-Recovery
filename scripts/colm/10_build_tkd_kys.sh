#!/bin/bash
set -euo pipefail

# ============================================================
# Build tokenized/content-key files for TOFU key-rank audit v2
# This script calls tofu/build_tokenized_key_file.py only.
# It does NOT run model forward or audit ranking.
# ============================================================

PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/def-yymao/hsc/LLM-Unlearning-Recovery}

# Set DEBUG=1 to use TOFU/keys/debug_full_key_tokens.jsonl.
DEBUG=${DEBUG:-0}
if [ "$DEBUG" = "1" ]; then
  KEY_FILE=${KEY_FILE:-TOFU/keys/debug_full_key_tokens.jsonl}
else
  KEY_FILE=${KEY_FILE:-TOFU/keys/full_key_tokens.jsonl}
fi

TOKENIZER_NAME=${TOKENIZER_NAME:-deepseek-ai/deepseek-llm-7b-chat}
TOKENIZER_TAG=${TOKENIZER_TAG:-deepseek-ai__deepseek-llm-7b-chat}
OUTPUT_DIR=${OUTPUT_DIR:-TOFU/keys/tokenized/${TOKENIZER_TAG}}
CACHE_DIR=${CACHE_DIR:-}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR" results/tofu/logs

echo "[tokenized-key] PROJECT_DIR=$PROJECT_DIR"
echo "[tokenized-key] KEY_FILE=$KEY_FILE"
echo "[tokenized-key] TOKENIZER_NAME=$TOKENIZER_NAME"
echo "[tokenized-key] TOKENIZER_TAG=$TOKENIZER_TAG"
echo "[tokenized-key] OUTPUT_DIR=$OUTPUT_DIR"
echo "[tokenized-key] CACHE_DIR=${CACHE_DIR:-none}"
echo "[tokenized-key] LOCAL_FILES_ONLY=$LOCAL_FILES_ONLY"
echo "[tokenized-key] DEBUG=$DEBUG"

if [ ! -f "$KEY_FILE" ]; then
  echo "[tokenized-key][ERROR] Missing key file: $KEY_FILE"
  exit 1
fi

# Load environment if on ComputeCanada. Ignore module failures for local shells.
if command -v module >/dev/null 2>&1; then
  module load gcc arrow/18.1.0 cuda || true
  module load python/3.10 || true
  module load scipy-stack || true
fi

if [ -f "$HOME/ENV-3.10/bin/activate" ]; then
  source "$HOME/ENV-3.10/bin/activate"
fi

echo "[tokenized-key] Python=$(which python)"
python - <<'PY'
import sys
print("python:", sys.version)
try:
    import transformers
    print("transformers: ok")
except Exception as e:
    raise SystemExit(f"transformers missing: {e}")
PY

ARGS=(
  --key_file "$KEY_FILE"
  --output_dir "$OUTPUT_DIR"
  --tokenizer_name "$TOKENIZER_NAME"
)

if [ -n "$CACHE_DIR" ]; then
  ARGS+=(--cache_dir "$CACHE_DIR")
fi

if [ "$LOCAL_FILES_ONLY" = "1" ]; then
  ARGS+=(--local_files_only)
fi

echo "[tokenized-key] Start tokenization/content filtering..."
python -u tofu/build_tokenized_key_file.py "${ARGS[@]}"

echo "[tokenized-key] Validate outputs..."
python - <<PY
import json
from pathlib import Path

key_file = Path("$KEY_FILE")
out_dir = Path("$OUTPUT_DIR")
stem = key_file.stem
is_debug = stem.startswith("debug_")
tokenized = out_dir / f"{stem}_tokenized.jsonl"
summary = out_dir / ("debug_tokenized_key_summary.json" if is_debug else "tokenized_key_summary.json")
vocab = out_dir / ("debug_vocab_summary.json" if is_debug else "vocab_summary.json")

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

inp = read_jsonl(key_file)
out = read_jsonl(tokenized)
sumry = json.loads(summary.read_text(encoding="utf-8"))

print("[validate] input_rows=", len(inp))
print("[validate] output_rows=", len(out))
print("[validate] num_key_facts=", sumry.get("num_key_facts"))
print("[validate] all_tokens=", sumry.get("num_all_key_tokens"))
print("[validate] content_tokens=", sumry.get("num_content_key_tokens"))
print("[validate] retention=", sumry.get("content_retention_ratio"))
print("[validate] alignment_failed=", sumry.get("num_alignment_failed_facts"))
print("[validate] fallback=", sumry.get("num_content_fallback_facts"))
print("[validate] removed_top10=", sumry.get("removed_token_top50", [])[:10])
print("[validate] would_remove_top10=", sumry.get("would_remove_token_top50", [])[:10])
print("[validate] acclaimed_writer_check=", sumry.get("quality_checks", {}).get("acclaimed_writer"))

assert len(inp) == len(out), "output row count must equal input row count"
assert [r.get("source_index") for r in inp] == [r.get("source_index") for r in out], "source_index order changed"
assert tokenized.exists(), f"missing tokenized file: {tokenized}"
assert summary.exists(), f"missing summary file: {summary}"
assert vocab.exists(), f"missing vocab file: {vocab}"

print("[validate] tokenized:", tokenized)
print("[validate] summary:", summary)
print("[validate] vocab:", vocab)
PY

echo "[tokenized-key] Done."
