"""Build heuristic key-fact token annotations for TOFU forget splits."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from common import processed_record, read_records, write_json, write_jsonl


SPLITS = ("forget01", "forget05", "forget10")
OCCUPATIONS = (
    "scientist",
    "musician",
    "teacher",
    "engineer",
    "journalist",
    "poet",
    "novelist",
    "writer",
    "author",
    "chef",
    "artist",
    "professor",
    "doctor",
    "lawyer",
    "historian",
    "playwright",
    "screenwriter",
    "editor",
)
GENRE_TERMS = (
    "leadership",
    "romance",
    "mystery",
    "science fiction",
    "fantasy",
    "true crime",
    "horror",
    "thriller",
    "poetry",
    "memoir",
    "biography",
    "historical fiction",
    "young adult",
    "children's literature",
)
AWARD_WORDS = r"(?:Award|Prize|Medal|Honor|Honour|Fellowship)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build key-fact token annotations for TOFU forget splits.")
    parser.add_argument("--processed_dir", default="tofu/processed")
    parser.add_argument("--output_dir", default="TOFU/keys")
    parser.add_argument("--base_model_name", default="deepseek-ai/deepseek-llm-7b-chat")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--local_files_only", action="store_true", default=True)
    return parser.parse_args()


def load_tokenizer(args: argparse.Namespace):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            args.base_model_name,
            local_files_only=args.local_files_only,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"[key-tokens][WARN] tokenizer unavailable, writing empty token metadata: {exc}", flush=True)
        return None


def add_regex_spans(answer: str, pattern: str, reason: str, spans: list[dict[str, Any]], flags: int = 0) -> None:
    for match in re.finditer(pattern, answer, flags):
        text = match.group(0).strip()
        if not text:
            continue
        start = match.start() + (len(match.group(0)) - len(match.group(0).lstrip()))
        end = start + len(text)
        spans.append({"text": text, "char_start": start, "char_end": end, "reason": reason})


def extract_key_spans(answer: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []

    add_regex_spans(answer, r'"[^"\n]{2,}"|“[^”\n]{2,}”', "quoted_title_or_work", spans)
    add_regex_spans(
        answer,
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{1,2}(?:st|nd|rd|th) of [A-Z][a-z]+(?: in the year)? \d{4}\b|\b(?:18|19|20)\d{2}\b|\b\d+(?:\.\d+)?\b",
        "date_year_or_number",
        spans,
    )
    add_regex_spans(answer, rf"\b(?:[A-Z][\w'’.-]+(?:\s+|[-])){{0,5}}[A-Z][\w'’.-]+\s+{AWARD_WORDS}\b", "award_name", spans)

    for occupation in OCCUPATIONS:
        add_regex_spans(answer, rf"\b(?:acclaimed|noted|renowned|esteemed|famous|professional|talented)?\s*{re.escape(occupation)}s?\b", "occupation_or_identity", spans, re.IGNORECASE)

    for term in GENRE_TERMS:
        add_regex_spans(answer, rf"\b{re.escape(term)}\b", "genre_or_theme", spans, re.IGNORECASE)

    add_regex_spans(answer, r"\b[A-Z][\w'’.-]+(?:\s+[A-Z][\w'’.-]+){1,4}\b", "person_place_or_named_entity", spans)
    add_regex_spans(answer, r"\b(?:the|a|an)\s+(?:[a-z][a-z'-]+\s+){1,3}(?:genre|theme|style|movement|book|novel|work|series|city|country)\b", "noun_phrase", spans, re.IGNORECASE)

    return dedupe_overlapping_spans(spans)


def span_priority(reason: str) -> int:
    order = {
        "quoted_title_or_work": 1,
        "date_year_or_number": 2,
        "person_place_or_named_entity": 3,
        "award_name": 4,
        "occupation_or_identity": 5,
        "genre_or_theme": 6,
        "noun_phrase": 7,
    }
    return order.get(reason, 99)


def dedupe_overlapping_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(spans, key=lambda item: (span_priority(item["reason"]), item["char_start"], -(item["char_end"] - item["char_start"])))
    kept: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    seen = set()
    for span in ordered:
        key = (span["char_start"], span["char_end"], span["text"].lower())
        if key in seen:
            continue
        overlaps = any(not (span["char_end"] <= start or span["char_start"] >= end) for start, end in occupied)
        if overlaps:
            continue
        seen.add(key)
        occupied.append((span["char_start"], span["char_end"]))
        kept.append(span)
    return sorted(kept, key=lambda item: (span_priority(item["reason"]), item["char_start"]))


def annotate_tokens(tokenizer, text: str) -> tuple[list[int], list[str]]:
    if tokenizer is None:
        return [], []
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    token_texts = [tokenizer.decode([token_id]) for token_id in token_ids]
    return [int(token_id) for token_id in token_ids], token_texts


def build_split(split: str, args: argparse.Namespace, tokenizer) -> dict[str, Any]:
    in_path = Path(args.processed_dir) / f"{split}.json"
    records = [processed_record(r, split) if "category" not in r else r for r in read_records(in_path)]
    out_records = []
    for index, record in enumerate(records):
        answer = str(record["answer"])
        key_facts = []
        for rank, span in enumerate(extract_key_spans(answer), start=1):
            token_ids, token_texts = annotate_tokens(tokenizer, span["text"])
            key_facts.append(
                {
                    "text": span["text"],
                    "importance_rank": rank,
                    "reason": span["reason"],
                    "char_start": span["char_start"],
                    "char_end": span["char_end"],
                    "token_ids": token_ids,
                    "token_texts": token_texts,
                }
            )
        out_records.append(
            {
                "index": index,
                "question": str(record["question"]),
                "answer": answer,
                "category": record.get("category"),
                "key_facts": key_facts,
            }
        )

    out_path = Path(args.output_dir) / f"{split}_key_tokens.jsonl"
    write_jsonl(out_path, out_records)
    print(f"[key-tokens] {split}: records={len(out_records)} -> {out_path}", flush=True)
    return {
        "split": split,
        "input_path": str(in_path),
        "output_path": str(out_path),
        "num_records": len(out_records),
        "num_key_facts": sum(len(record["key_facts"]) for record in out_records),
    }


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args)
    split_summaries = [build_split(split, args, tokenizer) for split in SPLITS]
    summary = {
        "kind": "tofu_key_tokens",
        "processed_dir": args.processed_dir,
        "output_dir": args.output_dir,
        "base_model_name": args.base_model_name,
        "tokenizer_loaded": tokenizer is not None,
        "splits": split_summaries,
    }
    summary_path = Path(args.output_dir) / "summary.json"
    write_json(summary_path, summary)
    print(f"[key-tokens] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
