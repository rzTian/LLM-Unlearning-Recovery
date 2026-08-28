"""LLM-assisted full TOFU key-fact extraction.

The LLM only proposes semantic key facts. This script owns deterministic
post-processing: source-index validation, genre mapping, character alignment,
summaries, and quality samples.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests


EXTRACTOR_VERSION = "full_key_llm_v1"
REPORT_FILES = {
    "manifest.json",
    "classification_summary.json",
    "genre_scan_report.json",
}
STOPWORDS_OR_TEMPLATE = {
    "author", "authors", "book", "books", "writer", "writing", "work", "works",
    "novel", "novels", "genre", "story", "career", "the", "a", "an", "of", "in",
    "on", "for", "and", "or", "to", "with", "by", "is", "are", "was", "were",
    ".", ",", "-", "'", '"',
}
FACT_GROUPS = {
    "author_name",
    "person_name",
    "gender_identity",
    "birth_date",
    "birth_year",
    "birth_place_city",
    "birth_place_country",
    "location",
    "occupation",
    "father_occupation",
    "mother_occupation",
    "genre_label",
    "book_title",
    "award_name",
    "award_year",
    "character_name",
    "theme_keyword",
    "writing_style_phrase",
    "inspiration_source",
    "education_institution",
    "career_role",
    "organization_or_platform",
    "language",
    "pseudonym",
    "relationship_or_family_status",
    "number_quantity",
    "other_named_entity",
    "other_content_phrase",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LLM-assisted full TOFU key-token annotations.")
    parser.add_argument("--input", default="TOFU/full.json")
    parser.add_argument("--genres_dir", default="TOFU/genres")
    parser.add_argument("--output", default="TOFU/keys/full_key_tokens.jsonl")
    parser.add_argument("--summary", default="TOFU/keys/full_key_extraction_summary.json")
    parser.add_argument("--quality_report", default="TOFU/keys/full_key_extraction_quality_report.json")
    parser.add_argument("--raw_llm_output", default="TOFU/keys/full_key_llm_raw.jsonl")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5-mini"))
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--api_base", default=os.environ.get("OPENAI_API_BASE", "https://api.siliconflow.cn/v1"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--connect_timeout", type=float, default=20.0)
    parser.add_argument("--request_timeout", type=float, default=180.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sample_per_group", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass
    out = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected object at {path}:{line_no}")
        out.append(obj)
    return out


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds == float("inf"):
        return "unknown"
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def normalize_question(question: str) -> str:
    text = str(question).strip().replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_genre_mapping(genres_dir: str | Path, full_records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    genres_dir = Path(genres_dir)
    by_source: dict[int, dict[str, Any]] = {}
    by_question: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    duplicates: dict[int, list[str]] = defaultdict(list)

    for path in sorted(genres_dir.glob("*.json")):
        if path.name in REPORT_FILES:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        genre = path.stem
        for item in data:
            if not isinstance(item, dict):
                continue
            if "source_index" in item:
                source_index = int(item["source_index"])
                if source_index >= len(full_records):
                    continue
                if source_index in by_source:
                    duplicates[source_index].append(genre)
                    continue
                by_source[source_index] = {
                    "genre_group": genre,
                    "matched_genre_rule": item.get("matched_rule") or item.get("matched_genre_rule") or path.name,
                    "coarse_category": item.get("category"),
                    "genre_record": item,
                }
            by_question[normalize_question(item.get("question", ""))].append((genre, item))

    for source_index, record in enumerate(full_records):
        if source_index in by_source:
            continue
        matches = by_question.get(normalize_question(record.get("question", "")), [])
        if len(matches) == 1:
            genre, item = matches[0]
            by_source[source_index] = {
                "genre_group": genre,
                "matched_genre_rule": item.get("matched_rule") or item.get("matched_genre_rule") or "question_fallback",
                "coarse_category": item.get("category"),
                "genre_record": item,
            }

    expected = set(range(len(full_records)))
    missing = sorted(expected - set(by_source))
    extra = sorted(set(by_source) - expected)
    if missing or extra or duplicates:
        raise ValueError(
            "Invalid genre mapping: "
            f"missing={missing[:20]} total_missing={len(missing)} "
            f"extra={extra[:20]} total_extra={len(extra)} "
            f"duplicate_source_indices={dict(list(duplicates.items())[:20])}"
        )
    return by_source


def load_raw_cache(path: str | Path) -> dict[int, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return {}
    cache = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cache[int(obj["source_index"])] = obj
    return cache


def llm_prompt(batch: list[dict[str, Any]]) -> str:
    payload = [
        {
            "source_index": row["source_index"],
            "genre_group": row["genre_group"],
            "question": row["question"],
            "answer": row["answer"],
        }
        for row in batch
    ]
    return (
        "You are labeling key facts in TOFU QA answers for model-unlearning audits.\n"
        "Return ONLY valid JSON with this shape: "
        '{"records":[{"source_index":0,"key_facts":[{"text":"...","fact_group":"genre_label",'
        '"fact_subgroup":"primary_genre","importance_rank":1,"reason":"main_answer_fact",'
        '"is_primary_answer":true}]}]}.\n'
        "Rules:\n"
        "- key_facts must be short answer substrings or near-exact phrases from the answer.\n"
        "- Extract the direct answer fact first; rank by importance.\n"
        "- Do not output token ids, token texts, char offsets, explanations, markdown, or whole sentences.\n"
        "- Use only these fact_group values when possible: "
        + ", ".join(sorted(FACT_GROUPS))
        + ".\n"
        "- Avoid standalone template words such as author, book, novel, genre, work, the, of, in.\n"
        "- If no fact is recoverable, return an empty key_facts list for that record.\n\n"
        "Batch:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def call_openai_batch(args: argparse.Namespace, batch: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")

    url = args.api_base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    body = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": llm_prompt(batch),
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }

    last_error = None
    for attempt in range(args.max_retries):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=(args.connect_timeout, args.request_timeout))
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            parsed = json.loads(text)
            return normalize_llm_records(parsed, batch)
        except Exception as exc:
            last_error = exc
            sleep_s = min(60, 2 ** attempt)
            print(f"[full-key-llm][WARN] batch failed attempt={attempt + 1}: {exc}; sleep={sleep_s}s", flush=True)
            time.sleep(sleep_s)

    raise RuntimeError(f"LLM batch failed after {args.max_retries} attempts: {last_error}")


def extract_response_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    pieces = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                pieces.append(content["text"])
    if pieces:
        return "\n".join(pieces)
    raise ValueError("Could not find output text in Responses API payload")


def normalize_llm_records(parsed: dict[str, Any], batch: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("records"), list):
        raise ValueError("LLM JSON must contain records list")
    expected = {int(row["source_index"]) for row in batch}
    out = {}
    for record in parsed["records"]:
        source_index = int(record["source_index"])
        if source_index not in expected:
            raise ValueError(f"Unexpected source_index from LLM: {source_index}")
        facts = record.get("key_facts") or []
        if not isinstance(facts, list):
            raise ValueError(f"key_facts must be a list for source_index={source_index}")
        out[source_index] = {"source_index": source_index, "key_facts": facts}
    missing = sorted(expected - set(out))
    if missing:
        raise ValueError(f"LLM omitted source_index values: {missing}")
    return out


def find_char_span(answer: str, text: str) -> tuple[int | None, int | None, str, bool]:
    if not text:
        return None, None, "empty", True
    direct = answer.find(text)
    if direct >= 0:
        return direct, direct + len(text), "exact_substring", False
    folded_text = re.sub(r"\s+", " ", text.strip())
    folded_answer_chars = []
    raw_positions = []
    last_space = False
    for raw_idx, ch in enumerate(answer):
        if ch.isspace():
            if not last_space:
                folded_answer_chars.append(" ")
                raw_positions.append(raw_idx)
                last_space = True
            continue
        folded_answer_chars.append(ch)
        raw_positions.append(raw_idx)
        last_space = False
    folded_answer = "".join(folded_answer_chars).strip()
    idx = folded_answer.lower().find(folded_text.lower())
    if idx < 0 or idx >= len(raw_positions):
        return None, None, "failed", True
    raw_start = raw_positions[idx]
    folded_end_idx = min(idx + len(folded_text) - 1, len(raw_positions) - 1)
    raw_end = raw_positions[folded_end_idx] + 1
    return raw_start, raw_end, "normalized_substring", False


def clean_fact_group(value: Any) -> str:
    group = str(value or "other_content_phrase").strip()
    return group if group in FACT_GROUPS else "other_content_phrase"


def postprocess_record(record: dict[str, Any], genre_info: dict[str, Any], llm_row: dict[str, Any]) -> dict[str, Any]:
    answer = str(record["answer"])
    source_index = int(record["source_index"])
    facts = []
    seen = set()
    for raw_fact in llm_row.get("key_facts") or []:
        text = str(raw_fact.get("text", "")).strip()
        norm = re.sub(r"\s+", " ", text.lower()).strip(" \t\r\n.,;:!?()[]{}\"'")
        if not norm or norm in STOPWORDS_OR_TEMPLATE or norm in seen:
            continue
        seen.add(norm)
        start, end, char_method, char_failed = find_char_span(answer, text)
        try:
            rank = int(raw_fact.get("importance_rank") or len(facts) + 1)
        except Exception:
            rank = len(facts) + 1
        rank = max(1, rank)
        fact = {
            "fact_id": f"{source_index}-{len(facts)}",
            "text": text,
            "normalized_text": norm,
            "fact_group": clean_fact_group(raw_fact.get("fact_group")),
            "fact_subgroup": str(raw_fact.get("fact_subgroup") or clean_fact_group(raw_fact.get("fact_group"))),
            "importance_rank": rank,
            "importance_weight": 1.0 / rank,
            "reason": str(raw_fact.get("reason") or ("main_answer_fact" if rank == 1 else "supporting_answer_fact")),
            "char_start": start,
            "char_end": end,
            "is_primary_answer": bool(raw_fact.get("is_primary_answer", rank == 1)),
            "is_content_fact": True,
            "is_template_fact": False,
            "char_alignment_method": char_method,
            "char_alignment_failed": char_failed,
        }
        facts.append(fact)
    facts.sort(key=lambda f: (int(f["importance_rank"]), f["char_start"] is None, f["char_start"] or 10**9))
    for idx, fact in enumerate(facts):
        fact["fact_id"] = f"{source_index}-{idx}"
    return {
        "source_index": source_index,
        "author_id": source_index // 20,
        "qa_index_within_author": source_index % 20,
        "question": str(record["question"]),
        "answer": answer,
        "coarse_category": genre_info.get("coarse_category"),
        "genre_group": genre_info["genre_group"],
        "secondary_genre_groups": [],
        "matched_genre_rule": genre_info.get("matched_genre_rule"),
        "key_facts": facts,
        "metadata": {
            "extractor_version": EXTRACTOR_VERSION,
            "llm_model": None,
        },
    }


def make_quality_report(rows: list[dict[str, Any]], sample_per_group: int) -> dict[str, Any]:
    rng = random.Random(13)
    by_genre: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_fact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_genre[row["genre_group"]].append(row)
        for fact in row["key_facts"]:
            by_fact[fact["fact_group"]].append({"source_index": row["source_index"], "question": row["question"], "answer": row["answer"], "fact": fact})

    def sample_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chosen = items if len(items) <= sample_per_group else rng.sample(items, sample_per_group)
        return [
            {
                "source_index": item["source_index"],
                "question": item.get("question"),
                "answer": item.get("answer"),
                "key_facts": item.get("key_facts"),
                "fact": item.get("fact"),
            }
            for item in chosen
        ]

    return {
        "extractor_version": EXTRACTOR_VERSION,
        "sample_per_group": sample_per_group,
        "genre_group_samples": {group: sample_rows(items) for group, items in sorted(by_genre.items())},
        "fact_group_samples": {group: sample_rows(items) for group, items in sorted(by_fact.items())},
    }


def validate_rows(rows: list[dict[str, Any]], num_records: int) -> None:
    indices = [row["source_index"] for row in rows]
    if len(rows) != num_records:
        raise ValueError(f"Output rows={len(rows)} does not match input records={num_records}")
    if len(indices) != len(set(indices)):
        dupes = [idx for idx, count in Counter(indices).items() if count > 1]
        raise ValueError(f"Duplicate source_index values: {dupes[:20]}")
    expected = list(range(num_records))
    if sorted(indices) != expected:
        raise ValueError("source_index coverage is incomplete or non-contiguous")


def main() -> None:
    args = parse_args()
    records = read_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    for idx, record in enumerate(records):
        if "question" not in record or "answer" not in record:
            raise ValueError(f"Missing question/answer at source_index={idx}")
        record["source_index"] = idx
    genre_map = load_genre_mapping(args.genres_dir, records)

    raw_path = Path(args.raw_llm_output)
    if not args.resume and raw_path.exists():
        raw_path.unlink()
    raw_cache = load_raw_cache(raw_path) if args.resume else {}

    pending = [
        {
            "source_index": idx,
            "question": str(record["question"]),
            "answer": str(record["answer"]),
            "genre_group": genre_map[idx]["genre_group"],
        }
        for idx, record in enumerate(records)
        if idx not in raw_cache
    ]
    total_records = len(records)
    total_pending = len(pending)
    total_batches = (total_pending + args.batch_size - 1) // args.batch_size
    print(
        f"[full-key-llm] records={total_records} cached={len(raw_cache)} "
        f"pending={total_pending} batch_size={args.batch_size} batches={total_batches}",
        flush=True,
    )

    run_start_time = time.time()
    completed_this_run = 0
    batch_times: list[float] = []

    for batch_no, start in enumerate(range(0, total_pending, args.batch_size), start=1):
        batch = pending[start : start + args.batch_size]
        batch_source_indices = [row["source_index"] for row in batch]
        done_before = len(raw_cache)
        batch_start_time = time.time()

        print(
            f"[full-key-llm] START batch={batch_no}/{total_batches} "
            f"size={len(batch)} source_indices={batch_source_indices[0]}-{batch_source_indices[-1]} "
            f"done={done_before}/{total_records} "
            f"elapsed={format_duration(batch_start_time - run_start_time)}",
            flush=True,
        )

        llm_rows = call_openai_batch(args, batch)
        to_write = [llm_rows[row["source_index"]] for row in batch]
        append_jsonl(raw_path, to_write)
        raw_cache.update(llm_rows)

        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        completed_this_run += len(batch)

        done_now = len(raw_cache)
        elapsed = time.time() - run_start_time
        avg_batch_time = sum(batch_times) / max(len(batch_times), 1)
        recent_batch_time = sum(batch_times[-5:]) / max(len(batch_times[-5:]), 1)
        remaining_batches = total_batches - batch_no
        eta_by_recent = remaining_batches * recent_batch_time
        eta_by_all = remaining_batches * avg_batch_time
        records_per_sec = completed_this_run / max(elapsed, 1e-9)

        print(
            f"[full-key-llm] DONE batch={batch_no}/{total_batches} "
            f"batch_time={format_duration(batch_time)} "
            f"avg_batch={format_duration(avg_batch_time)} "
            f"recent_avg_5={format_duration(recent_batch_time)} "
            f"done={done_now}/{total_records} "
            f"this_run={completed_this_run}/{total_pending} "
            f"speed={records_per_sec:.3f} rec/s "
            f"eta_recent={format_duration(eta_by_recent)} "
            f"eta_all={format_duration(eta_by_all)} "
            f"elapsed={format_duration(elapsed)}",
            flush=True,
        )

    if sorted(raw_cache) != list(range(len(records))):
        missing = sorted(set(range(len(records))) - set(raw_cache))
        raise ValueError(f"Raw LLM cache incomplete; missing={missing[:20]} total={len(missing)}")

    rows = [postprocess_record(records[idx], genre_map[idx], raw_cache[idx]) for idx in range(len(records))]
    for row in rows:
        row["metadata"]["llm_model"] = args.model
    validate_rows(rows, len(records))

    fact_counts = Counter(fact["fact_group"] for row in rows for fact in row["key_facts"])
    genre_counts = Counter(row["genre_group"] for row in rows)
    char_alignment_failed = sum(1 for row in rows for fact in row["key_facts"] if fact.get("char_alignment_failed"))
    summary = {
        "input_file": args.input,
        "genres_dir": args.genres_dir,
        "output_file": args.output,
        "raw_llm_output": args.raw_llm_output,
        "quality_report": args.quality_report,
        "num_input_records": len(records),
        "num_output_records": len(rows),
        "num_records_without_key_facts": sum(1 for row in rows if not row["key_facts"]),
        "num_key_facts": sum(len(row["key_facts"]) for row in rows),
        "num_char_alignment_failed": char_alignment_failed,
        "num_key_facts_by_fact_group": dict(sorted(fact_counts.items())),
        "num_records_by_genre_group": dict(sorted(genre_counts.items())),
        "token_alignment": "deferred_to_audit",
        "llm_model": args.model,
        "extractor_version": EXTRACTOR_VERSION,
        "notes": [],
    }

    write_jsonl(args.output, rows)
    write_json(args.summary, summary)
    write_json(args.quality_report, make_quality_report(rows, args.sample_per_group))
    print(f"[full-key-llm] output -> {args.output}", flush=True)
    print(f"[full-key-llm] summary -> {args.summary}", flush=True)
    print(f"[full-key-llm] quality -> {args.quality_report}", flush=True)


if __name__ == "__main__":
    main()
