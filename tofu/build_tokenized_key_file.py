"""Build tokenized key files for TOFU key-rank audit v2.1.

This script performs tokenizer alignment and fact-group-aware content-token
filtering only. It does not run model forward or ranking.

V2.1 improvements over v2:
- keeps v2 boundary-aware stopword filtering so internal BPE subwords are not removed;
- adds language-specific filtering for weak tokens such as his/her/them/into/in/books/write;
- adds gender-identity-specific filtering for weak tokens such as as/identifies/community;
- marks non-specific fallback phrases such as father's profession as no_specific_content_token;
- fixes the internal-subword quality check so punctuation is not reported as a subword stopword issue.
"""

from __future__ import annotations

import argparse
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from common import read_records, write_json, write_jsonl


STOPWORDS_OR_TEMPLATE = {
    "author", "authors", "book", "books", "writer", "writing",
    "work", "works", "novel", "novels", "genre", "story", "career",
    "the", "a", "an", "of", "in", "on", "for", "and", "or", "to",
    "with", "by", "is", "are", "was", "were",
    ".", ",", "-", "'", '"',
}

WEAK_MODIFIERS = {
    "noted", "acclaimed", "esteemed", "celebrated", "renowned",
    "famous", "prominent", "significant", "greatly",
}

OCCUPATION_FACT_GROUPS = {
    "father_occupation",
    "mother_occupation",
    "parent_occupation",
    "occupation",
}

# These terms can look like templates globally, but are actual content for
# occupation facts. Do not remove them when fact_group is an occupation type.
OCCUPATION_CONTENT_TERMS = {
    "writer", "author", "novelist", "poet", "journalist", "teacher",
    "chef", "artist", "musician", "mechanic", "librarian", "scientist",
    "professor", "educator", "translator", "editor", "lawyer", "doctor",
    "nurse", "engineer", "architect", "historian", "sociologist",
    "activist", "researcher", "publisher", "critic", "counselor",
    "counsellor", "farmer", "merchant", "diplomat", "designer",
    "photographer", "actor", "actress", "director", "screenwriter",
    "playwright", "academic", "scholar", "administrator", "curator",
    "anthropologist", "psychologist", "physician", "surgeon", "pilot",
    "soldier", "officer", "clerk", "tailor", "carpenter", "craftsman",
    "craftswoman", "artisan", "baker", "entrepreneur", "businessman",
    "businesswoman", "accountant", "banker", "minister", "pastor",
    "priest", "rabbi", "imam", "social worker", "nurse", "midwife",
}

OCCUPATION_EXTRA_STOPWORDS = {
    "father", "mother", "parent", "parents", "worked", "work", "works",
    "working", "served", "serves", "serve", "professional", "profession",
    "job", "jobs", "role", "roles", "career", "as",
}

DATE_FACT_GROUPS = {
    "birth_date",
    "birth_year",
    "award_year",
}

DATE_EXTRA_STOPWORDS = {
    "year",
}

# Pronouns and light function words are usually not useful for token-level factual
# recovery. For proper-name/title-like facts we avoid aggressive pronoun/relation
# filtering because these words may be part of a title or name.
PRONOUN_STOPWORDS = {
    "her", "his", "their", "he", "she", "they", "him", "them", "it",
    "its", "hers", "theirs", "himself", "herself", "themselves",
}

RELATION_STOPWORDS = {
    "from", "into", "as", "about", "through", "toward", "towards",
    "within", "across", "around", "between", "among", "under", "over",
}

NAME_OR_TITLE_FACT_GROUPS = {
    "author_name", "person_name", "book_title", "award_name",
    "character_name", "education_institution", "organization_or_platform",
    "pseudonym", "birth_place_city", "birth_place_country", "location",
    "language", "other_named_entity",
}

PHRASE_FACT_GROUPS = {
    "other_content_phrase", "inspiration_source", "theme_keyword",
    "writing_style_phrase", "career_role", "relationship_or_family_status",
    "number_quantity",
}

LANGUAGE_FACT_GROUPS = {"language"}

# For language facts, keep concrete language names and meaningful modifiers
# such as English, French, multilingual, multiple, languages. Remove prompt/
# answer scaffolding that does not identify the language fact itself.
LANGUAGE_EXTRA_STOPWORDS = {
    "write", "writes", "wrote", "written", "book", "books",
    "language", "languages", "translation", "translations",
    "translated", "translate", "translates", "translating",
    "read", "reads", "reading", "primarily", "mainly",
}

GENDER_IDENTITY_FACT_GROUPS = {"gender_identity"}

# Keep the actual identity label, e.g. LGBTQ, queer, gay, lesbian, bisexual,
# transgender, nonbinary. Remove description scaffolding.
GENDER_IDENTITY_EXTRA_STOPWORDS = {
    "identifies", "identify", "identified", "identifying", "as",
    "member", "members", "community", "individual", "person",
    "author", "writer", "openly", "known", "celebrated",
}

NON_SPECIFIC_OCCUPATION_PATTERNS = {
    "father's profession", "father’s profession", "father profession",
    "mother's profession", "mother’s profession", "mother profession",
    "parent's profession", "parent’s profession", "parents' profession",
    "father's job", "father’s job", "mother's job", "mother’s job",
    "parent's job", "parent’s job", "parents' jobs",
    "father's career", "father’s career", "mother's career", "mother’s career",
}

APOSTROPHE_TOKENS = {"'", "’", "ʼ", "`"}
CONTENT_FILTER_VERSION = "v2_1_boundary_aware"

Triplet = tuple[int, int, str, tuple[int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tokenized key file for TOFU audit v2.")
    parser.add_argument("--key_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer_name", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def load_tokenizer(args: argparse.Namespace):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    return tok


def normalize_token_text(text: str) -> str:
    text = str(text).strip().replace("▁", "").replace("Ġ", "").replace("Ċ", "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def display_token_text(text: str) -> str:
    raw = str(text)
    if raw and not raw.strip():
        return "<SPACE>"
    norm = normalize_token_text(raw)
    return norm if norm else "<EMPTY>"


def is_punct_only(text: str) -> bool:
    stripped = str(text).strip()
    return bool(stripped) and all(ch in string.punctuation for ch in stripped)


def effective_token_span(answer: str, offset: tuple[int, int]) -> tuple[int, int]:
    """Trim whitespace from tokenizer offset span for boundary checks."""
    s, e = offset
    s = max(0, int(s))
    e = min(len(answer), int(e))
    while s < e and answer[s].isspace():
        s += 1
    while e > s and answer[e - 1].isspace():
        e -= 1
    return s, e


def is_standalone_word(
    answer: str,
    offset: tuple[int, int],
    key_start: int | None = None,
    key_end: int | None = None,
) -> bool:
    """Return whether token span is a whole word, not an internal BPE subword."""
    s, e = effective_token_span(answer, offset)
    if e <= s:
        return False
    if key_start is None:
        key_start = 0
    if key_end is None:
        key_end = len(answer)
    key_start = max(0, int(key_start))
    key_end = min(len(answer), int(key_end))

    left_ok = s <= key_start or not answer[s - 1].isalnum()
    right_ok = e >= key_end or not answer[e].isalnum()
    return bool(left_ok and right_ok)


def fact_group_allows_aggressive_phrase_filtering(fact_group: str) -> bool:
    return fact_group in PHRASE_FACT_GROUPS or fact_group in OCCUPATION_FACT_GROUPS or fact_group in DATE_FACT_GROUPS


def token_filter_decision(
    text: str,
    *,
    fact_group: str | None = None,
    answer: str = "",
    offset: tuple[int, int] | None = None,
    key_start: int | None = None,
    key_end: int | None = None,
) -> tuple[bool, str]:
    """Return (keep, reason) for one token.

    Stopword/template decisions are boundary-aware: a token is filtered as a
    lexical stopword only when it is a standalone word under the key span. This
    prevents deleting internal BPE pieces such as:
      Sculptor -> S / cul / pt / or
      Whispers -> Wh / is / pers
      Harare -> Har / are
      Zimbabwean -> Zimbabwe / an
    """
    norm = normalize_token_text(text)
    fg = str(fact_group or "")

    if not norm:
        return False, "empty_or_space"
    if is_punct_only(norm) or is_punct_only(text):
        return False, "punctuation"

    standalone = True
    if offset is not None and answer:
        standalone = is_standalone_word(answer, offset, key_start, key_end)

    if fg in OCCUPATION_FACT_GROUPS and norm in OCCUPATION_CONTENT_TERMS:
        return True, "occupation_content_term"

    if fg in OCCUPATION_FACT_GROUPS and standalone and norm in OCCUPATION_EXTRA_STOPWORDS:
        return False, "occupation_template"

    if fg in DATE_FACT_GROUPS and standalone and norm in DATE_EXTRA_STOPWORDS:
        return False, "date_template"

    if fg in LANGUAGE_FACT_GROUPS and standalone and norm in LANGUAGE_EXTRA_STOPWORDS:
        return False, "language_template"
    if fg in LANGUAGE_FACT_GROUPS and standalone and norm in PRONOUN_STOPWORDS:
        return False, "pronoun"
    if fg in LANGUAGE_FACT_GROUPS and standalone and norm in RELATION_STOPWORDS:
        return False, "relation_stopword"

    if fg in GENDER_IDENTITY_FACT_GROUPS and standalone and norm in GENDER_IDENTITY_EXTRA_STOPWORDS:
        return False, "gender_identity_template"
    if fg in GENDER_IDENTITY_FACT_GROUPS and standalone and norm in RELATION_STOPWORDS:
        return False, "relation_stopword"

    if fg not in NAME_OR_TITLE_FACT_GROUPS and standalone and norm in PRONOUN_STOPWORDS:
        return False, "pronoun"

    if fact_group_allows_aggressive_phrase_filtering(fg) and standalone and norm in RELATION_STOPWORDS:
        return False, "relation_stopword"

    if standalone and norm in STOPWORDS_OR_TEMPLATE:
        return False, "template_stopword"
    if standalone and norm in WEAK_MODIFIERS:
        return False, "weak_modifier"

    return True, "kept"


def is_apostrophe_token(text: str) -> bool:
    norm = normalize_token_text(text)
    return norm in APOSTROPHE_TOKENS or str(text).strip() in APOSTROPHE_TOKENS


def is_possessive_s_token(text: str) -> bool:
    return normalize_token_text(text) == "s"


def apply_possessive_s_filter(all_triplets: list[Triplet], keep_mask: list[bool], reasons: list[str]) -> None:
    """Remove possessive 's' when immediately preceded by an apostrophe token."""
    for pos, triplet in enumerate(all_triplets):
        _idx, _tid, txt, _off = triplet
        if not is_possessive_s_token(txt):
            continue
        prev_pos = pos - 1
        if prev_pos >= 0 and is_apostrophe_token(all_triplets[prev_pos][2]):
            keep_mask[pos] = False
            reasons[pos] = "possessive_s"


def encode_answer(tokenizer, answer: str) -> tuple[list[int], list[tuple[int, int]] | None]:
    try:
        encoded = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded.get("offset_mapping")
        if offsets is not None:
            offsets = [(int(s), int(e)) for s, e in offsets]
        return [int(t) for t in encoded["input_ids"]], offsets
    except Exception:
        encoded = tokenizer(answer, add_special_tokens=False)
        return [int(t) for t in encoded["input_ids"]], None


def align_offsets(offsets: list[tuple[int, int]] | None, start: Any, end: Any) -> tuple[list[int], str, bool]:
    if offsets is None:
        return [], "no_offset_mapping", True
    if start is None or end is None:
        return [], "missing_char_span", True
    try:
        s = int(start)
        e = int(end)
    except Exception:
        return [], "bad_char_span", True
    if e <= s:
        return [], "bad_char_span", True
    indices = [idx for idx, (ts, te) in enumerate(offsets) if te > s and ts < e and te > ts]
    if indices:
        return indices, "offset_overlap", False
    return [], "offset_overlap_empty", True


def group_metrics_init() -> dict[str, int]:
    return {
        "num_key_facts": 0,
        "num_all_key_tokens": 0,
        "num_content_key_tokens": 0,
        "num_removed_key_tokens": 0,
        "num_would_remove_key_tokens": 0,
        "num_alignment_failed_facts": 0,
        "num_content_fallback_facts": 0,
        "num_no_specific_content_facts": 0,
    }


def add_group_metrics(bucket: dict[str, int], fact: dict[str, Any]) -> None:
    bucket["num_key_facts"] += 1
    bucket["num_all_key_tokens"] += len(fact.get("answer_token_ids") or [])
    bucket["num_content_key_tokens"] += len(fact.get("content_token_ids") or [])
    bucket["num_removed_key_tokens"] += len(fact.get("removed_token_ids") or [])
    bucket["num_would_remove_key_tokens"] += len(fact.get("would_remove_token_ids") or [])
    bucket["num_alignment_failed_facts"] += int(bool(fact.get("token_alignment_failed")))
    bucket["num_content_fallback_facts"] += int(bool(fact.get("content_fallback_to_all")))
    bucket["num_no_specific_content_facts"] += int(bool(fact.get("no_specific_content_token")))


def make_triplets(indices: list[int], ids: list[int], texts: list[str], offsets: list[tuple[int, int]]) -> list[Triplet]:
    return [(idx, tid, txt, off) for idx, tid, txt, off in zip(indices, ids, texts, offsets)]


def normalized_fact_text(text: Any) -> str:
    text = str(text or "").lower().strip()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    return text


def is_non_specific_content_fact(fact: dict[str, Any], fact_group: str) -> bool:
    """Return true for facts that name only a slot, not a concrete content value."""
    text = normalized_fact_text(fact.get("text"))
    if fact_group in OCCUPATION_FACT_GROUPS and text in NON_SPECIFIC_OCCUPATION_PATTERNS:
        return True
    return False


def enrich_record(
    record: dict[str, Any],
    tokenizer,
    stats: Counter,
    removed_counter: Counter,
    would_remove_counter: Counter,
    remove_reason_counter: Counter,
    per_fact_group: dict[str, dict[str, int]],
    per_genre_group: dict[str, dict[str, int]],
) -> dict[str, Any]:
    out = dict(record)
    answer = str(out.get("answer", ""))
    genre_group = str(out.get("genre_group") or "unknown")
    answer_ids, answer_offsets = encode_answer(tokenizer, answer)
    answer_texts = [tokenizer.decode([tid]) for tid in answer_ids]
    out["_answer_token_ids"] = answer_ids
    out["_answer_token_texts"] = answer_texts
    out["_answer_token_offsets"] = answer_offsets

    new_facts = []
    for fact in out.get("key_facts") or []:
        f = dict(fact)
        fact_group = str(f.get("fact_group") or "unknown")
        indices, method, failed = align_offsets(answer_offsets, f.get("char_start"), f.get("char_end"))

        all_ids = [int(answer_ids[i]) for i in indices]
        all_texts = [answer_texts[i] for i in indices]
        all_offsets = [answer_offsets[i] for i in indices] if answer_offsets is not None else []
        all_triplets = make_triplets(indices, all_ids, all_texts, all_offsets)

        try:
            key_start = int(f.get("char_start")) if f.get("char_start") is not None else None
            key_end = int(f.get("char_end")) if f.get("char_end") is not None else None
        except Exception:
            key_start = None
            key_end = None

        keep_mask: list[bool] = []
        reason_list: list[str] = []
        for _idx, _tid, txt, off in all_triplets:
            keep, reason = token_filter_decision(
                txt,
                fact_group=fact_group,
                answer=answer,
                offset=off,
                key_start=key_start,
                key_end=key_end,
            )
            keep_mask.append(keep)
            reason_list.append(reason)

        apply_possessive_s_filter(all_triplets, keep_mask, reason_list)

        keep_triplets = [triplet for triplet, keep in zip(all_triplets, keep_mask) if keep]
        would_remove_triplets = [triplet for triplet, keep in zip(all_triplets, keep_mask) if not keep]
        would_remove_reasons = [reason for reason, keep in zip(reason_list, keep_mask) if not keep]

        content_fallback = False
        content_fallback_reason = None
        no_specific_content = False
        if indices and not keep_triplets and is_non_specific_content_fact(f, fact_group):
            # Do not recover generic slots such as "father's profession" as
            # content tokens; they are not concrete facts. The audit script
            # should skip empty content-token facts for token-level metrics.
            no_specific_content = True
            content_triplets = []
            removed_triplets = would_remove_triplets
            removed_reasons = would_remove_reasons
        elif indices and not keep_triplets:
            content_fallback = True
            content_fallback_reason = "all_tokens_filtered"
            content_triplets = all_triplets
            removed_triplets: list[Triplet] = []
            removed_reasons: list[str] = []
        else:
            content_triplets = keep_triplets
            removed_triplets = would_remove_triplets
            removed_reasons = would_remove_reasons

        f.update(
            {
                "answer_token_indices": [x[0] for x in all_triplets],
                "answer_token_ids": [x[1] for x in all_triplets],
                "answer_token_texts": [x[2] for x in all_triplets],
                "answer_token_offsets": [x[3] for x in all_triplets],
                "content_token_indices": [x[0] for x in content_triplets],
                "content_token_ids": [x[1] for x in content_triplets],
                "content_token_texts": [x[2] for x in content_triplets],
                "content_token_offsets": [x[3] for x in content_triplets],
                "removed_token_indices": [x[0] for x in removed_triplets],
                "removed_token_ids": [x[1] for x in removed_triplets],
                "removed_token_texts": [x[2] for x in removed_triplets],
                "removed_token_offsets": [x[3] for x in removed_triplets],
                "removed_token_reasons": removed_reasons,
                "would_remove_token_indices": [x[0] for x in would_remove_triplets],
                "would_remove_token_ids": [x[1] for x in would_remove_triplets],
                "would_remove_token_texts": [x[2] for x in would_remove_triplets],
                "would_remove_token_offsets": [x[3] for x in would_remove_triplets],
                "would_remove_token_reasons": would_remove_reasons,
                "content_filter_version": CONTENT_FILTER_VERSION,
                "content_fallback_to_all": content_fallback,
                "content_fallback_reason": content_fallback_reason,
                "no_specific_content_token": no_specific_content,
                "token_alignment_failed": failed,
                "token_alignment_method": method,
            }
        )

        stats["num_key_facts"] += 1
        stats["num_all_key_tokens"] += len(f["answer_token_ids"])
        stats["num_content_key_tokens"] += len(f["content_token_ids"])
        stats["num_removed_key_tokens"] += len(f["removed_token_ids"])
        stats["num_would_remove_key_tokens"] += len(f["would_remove_token_ids"])
        stats["num_alignment_failed_facts"] += int(failed)
        stats["num_content_fallback_facts"] += int(content_fallback)
        stats["num_no_specific_content_facts"] += int(no_specific_content)
        for text in f["removed_token_texts"]:
            removed_counter[display_token_text(text)] += 1
        for text in f["would_remove_token_texts"]:
            would_remove_counter[display_token_text(text)] += 1
        for reason in f["removed_token_reasons"]:
            remove_reason_counter[str(reason)] += 1

        per_fact_group.setdefault(fact_group, group_metrics_init())
        per_genre_group.setdefault(genre_group, group_metrics_init())
        add_group_metrics(per_fact_group[fact_group], f)
        add_group_metrics(per_genre_group[genre_group], f)

        new_facts.append(f)

    out["key_facts"] = new_facts
    return out


def drop_private_fields(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    out.pop("_answer_token_ids", None)
    out.pop("_answer_token_texts", None)
    out.pop("_answer_token_offsets", None)
    return out


def output_paths(key_file: str, output_dir: Path) -> tuple[Path, Path, Path]:
    stem = Path(key_file).stem
    is_debug = stem.startswith("debug_")
    tokenized_name = f"{stem}_tokenized.jsonl"
    if is_debug:
        summary_name = "debug_tokenized_key_summary.json"
        vocab_name = "debug_vocab_summary.json"
    else:
        summary_name = "tokenized_key_summary.json"
        vocab_name = "vocab_summary.json"
    return output_dir / tokenized_name, output_dir / summary_name, output_dir / vocab_name


def build_vocab_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    all_token_counter: Counter[int] = Counter()
    content_token_counter: Counter[int] = Counter()
    all_text: dict[int, str] = {}
    content_text: dict[int, str] = {}
    by_fact_group_all: dict[str, Counter[int]] = {}
    by_fact_group_content: dict[str, Counter[int]] = {}
    by_genre_group_all: dict[str, Counter[int]] = {}
    by_genre_group_content: dict[str, Counter[int]] = {}
    by_genre_fact_all: dict[str, Counter[int]] = {}
    by_genre_fact_content: dict[str, Counter[int]] = {}

    for row in records:
        genre_group = str(row.get("genre_group") or "unknown")
        for fact in row.get("key_facts") or []:
            fg = str(fact.get("fact_group") or "unknown")
            gf_key = f"{genre_group}::{fg}"
            by_fact_group_all.setdefault(fg, Counter())
            by_fact_group_content.setdefault(fg, Counter())
            by_genre_group_all.setdefault(genre_group, Counter())
            by_genre_group_content.setdefault(genre_group, Counter())
            by_genre_fact_all.setdefault(gf_key, Counter())
            by_genre_fact_content.setdefault(gf_key, Counter())

            for tid, txt in zip(fact.get("answer_token_ids") or [], fact.get("answer_token_texts") or []):
                tid = int(tid)
                all_token_counter[tid] += 1
                all_text.setdefault(tid, str(txt))
                by_fact_group_all[fg][tid] += 1
                by_genre_group_all[genre_group][tid] += 1
                by_genre_fact_all[gf_key][tid] += 1

            for tid, txt in zip(fact.get("content_token_ids") or [], fact.get("content_token_texts") or []):
                tid = int(tid)
                content_token_counter[tid] += 1
                content_text.setdefault(tid, str(txt))
                by_fact_group_content[fg][tid] += 1
                by_genre_group_content[genre_group][tid] += 1
                by_genre_fact_content[gf_key][tid] += 1

    return {
        "num_unique_all_token_ids": len(all_token_counter),
        "num_unique_content_token_ids": len(content_token_counter),
        "top50_all_token_ids": [
            {"token_id": int(tid), "count": int(cnt), "token_text": all_text.get(int(tid), "")}
            for tid, cnt in all_token_counter.most_common(50)
        ],
        "top50_content_token_ids": [
            {"token_id": int(tid), "count": int(cnt), "token_text": content_text.get(int(tid), "")}
            for tid, cnt in content_token_counter.most_common(50)
        ],
        "by_fact_group_unique_all": {k: len(v) for k, v in sorted(by_fact_group_all.items())},
        "by_fact_group_unique_content": {k: len(v) for k, v in sorted(by_fact_group_content.items())},
        "by_genre_group_unique_all": {k: len(v) for k, v in sorted(by_genre_group_all.items())},
        "by_genre_group_unique_content": {k: len(v) for k, v in sorted(by_genre_group_content.items())},
        "by_genre_fact_unique_all": {k: len(v) for k, v in sorted(by_genre_fact_all.items())},
        "by_genre_fact_unique_content": {k: len(v) for k, v in sorted(by_genre_fact_content.items())},
    }


def ratio(num: int, den: int) -> float | None:
    return (num / den) if den else None


def find_quality_checks(records: list[dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "acclaimed_writer": {
            "num_matches": 0,
            "examples": [],
            "all_examples_contain_writer_content": None,
        },
        "internal_subword_stopword_removals": {
            "num_suspicious_removed": 0,
            "examples": [],
        },
        # Backward-compatible alias for older readers. In v2.1 this check
        # excludes punctuation and reports only alphabetic stopword-like pieces.
        "internal_subword_stopwords": {
            "num_suspicious_removed": 0,
            "examples": [],
        },
        "possessive_s": {
            "num_removed_possessive_s": 0,
            "examples": [],
        },
    }
    writer_examples = []
    suspicious_examples = []
    possessive_examples = []
    stopword_like = STOPWORDS_OR_TEMPLATE | PRONOUN_STOPWORDS | RELATION_STOPWORDS | WEAK_MODIFIERS

    for row in records:
        answer = str(row.get("answer", ""))
        for fact in row.get("key_facts") or []:
            text_lower = str(fact.get("text", "")).lower()
            if "acclaimed writer" in text_lower:
                example = {
                    "source_index": row.get("source_index"),
                    "fact_id": fact.get("fact_id"),
                    "text": fact.get("text"),
                    "fact_group": fact.get("fact_group"),
                    "answer_token_texts": fact.get("answer_token_texts"),
                    "content_token_texts": fact.get("content_token_texts"),
                    "removed_token_texts": fact.get("removed_token_texts"),
                    "would_remove_token_texts": fact.get("would_remove_token_texts"),
                    "content_fallback_to_all": fact.get("content_fallback_to_all"),
                }
                writer_examples.append(example)

            for tok, off, reason in zip(
                fact.get("would_remove_token_texts") or [],
                fact.get("would_remove_token_offsets") or [],
                fact.get("would_remove_token_reasons") or [],
            ):
                norm = normalize_token_text(tok)
                if reason == "possessive_s" and len(possessive_examples) < 10:
                    possessive_examples.append(
                        {
                            "source_index": row.get("source_index"),
                            "fact_id": fact.get("fact_id"),
                            "text": fact.get("text"),
                            "token_text": tok,
                            "reason": reason,
                        }
                    )
                if norm in stopword_like and norm.isalpha() and off:
                    try:
                        s, e = int(off[0]), int(off[1])
                        standalone = is_standalone_word(answer, (s, e), fact.get("char_start"), fact.get("char_end"))
                    except Exception:
                        standalone = True
                    if not standalone and len(suspicious_examples) < 10:
                        suspicious_examples.append(
                            {
                                "source_index": row.get("source_index"),
                                "fact_id": fact.get("fact_id"),
                                "text": fact.get("text"),
                                "token_text": tok,
                                "token_offset": off,
                                "reason": reason,
                            }
                        )

    checks["acclaimed_writer"]["num_matches"] = len(writer_examples)
    checks["acclaimed_writer"]["examples"] = writer_examples[:10]
    if writer_examples:
        checks["acclaimed_writer"]["all_examples_contain_writer_content"] = all(
            any(normalize_token_text(t) == "writer" for t in ex.get("content_token_texts") or [])
            for ex in writer_examples
        )
    checks["internal_subword_stopword_removals"]["num_suspicious_removed"] = len(suspicious_examples)
    checks["internal_subword_stopword_removals"]["examples"] = suspicious_examples
    checks["internal_subword_stopwords"]["num_suspicious_removed"] = len(suspicious_examples)
    checks["internal_subword_stopwords"]["examples"] = suspicious_examples
    checks["possessive_s"]["num_removed_possessive_s"] = len(possessive_examples)
    checks["possessive_s"]["examples"] = possessive_examples
    return checks


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args)
    rows = read_records(args.key_file)
    stats: Counter[str] = Counter()
    removed_counter: Counter[str] = Counter()
    would_remove_counter: Counter[str] = Counter()
    remove_reason_counter: Counter[str] = Counter()
    per_fact_group: dict[str, dict[str, int]] = {}
    per_genre_group: dict[str, dict[str, int]] = {}

    enriched = [
        enrich_record(
            row,
            tokenizer,
            stats,
            removed_counter,
            would_remove_counter,
            remove_reason_counter,
            per_fact_group,
            per_genre_group,
        )
        for row in rows
    ]
    cleaned = [drop_private_fields(row) for row in enriched]

    out_dir = Path(args.output_dir)
    tokenized_path, summary_path, vocab_path = output_paths(args.key_file, out_dir)
    write_jsonl(tokenized_path, cleaned)

    input_source_indices = [int(r.get("source_index", -1)) for r in rows]
    output_source_indices = [int(r.get("source_index", -1)) for r in cleaned]
    row_parity = len(rows) == len(cleaned)
    source_index_parity = input_source_indices == output_source_indices
    alignment_ok = stats["num_key_facts"] - stats["num_alignment_failed_facts"]
    alignment_success_ratio = ratio(alignment_ok, stats["num_key_facts"])
    retention_ratio = ratio(stats["num_content_key_tokens"], stats["num_all_key_tokens"])

    for bucket in list(per_fact_group.values()) + list(per_genre_group.values()):
        bucket["content_retention_ratio"] = ratio(bucket["num_content_key_tokens"], bucket["num_all_key_tokens"])

    summary = {
        "key_file": args.key_file,
        "output_file": str(tokenized_path),
        "tokenizer_name": args.tokenizer_name,
        "content_filter_version": CONTENT_FILTER_VERSION,
        "num_input_rows": len(rows),
        "num_output_rows": len(cleaned),
        "row_parity": row_parity,
        "source_index_parity": source_index_parity,
        "num_key_facts": int(stats["num_key_facts"]),
        "num_all_key_tokens": int(stats["num_all_key_tokens"]),
        "num_content_key_tokens": int(stats["num_content_key_tokens"]),
        "num_removed_key_tokens": int(stats["num_removed_key_tokens"]),
        "num_would_remove_key_tokens": int(stats["num_would_remove_key_tokens"]),
        "content_retention_ratio": retention_ratio,
        "num_alignment_failed_facts": int(stats["num_alignment_failed_facts"]),
        "alignment_success_ratio": alignment_success_ratio,
        "num_content_fallback_facts": int(stats["num_content_fallback_facts"]),
        "num_no_specific_content_facts": int(stats["num_no_specific_content_facts"]),
        "removed_token_top50": [
            {"token_text": token, "count": int(count)} for token, count in removed_counter.most_common(50)
        ],
        "would_remove_token_top50": [
            {"token_text": token, "count": int(count)} for token, count in would_remove_counter.most_common(50)
        ],
        "removed_reason_top50": [
            {"reason": reason, "count": int(count)} for reason, count in remove_reason_counter.most_common(50)
        ],
        "per_fact_group": {k: per_fact_group[k] for k in sorted(per_fact_group)},
        "per_genre_group": {k: per_genre_group[k] for k in sorted(per_genre_group)},
        "quality_checks": find_quality_checks(cleaned),
    }
    write_json(summary_path, summary)
    write_json(vocab_path, build_vocab_summary(cleaned))

    print(f"[tokenized-key-v2.1] rows={len(cleaned)} key_facts={stats['num_key_facts']}")
    print(
        f"[tokenized-key-v2.1] all_tokens={stats['num_all_key_tokens']} "
        f"content_tokens={stats['num_content_key_tokens']} retention={retention_ratio}"
    )
    print(
        f"[tokenized-key-v2.1] alignment_failed={stats['num_alignment_failed_facts']} "
        f"fallback={stats['num_content_fallback_facts']}"
    )
    print(f"[tokenized-key-v2.1] output -> {tokenized_path}")
    print(f"[tokenized-key-v2.1] summary -> {summary_path}")
    print(f"[tokenized-key-v2.1] vocab -> {vocab_path}")


if __name__ == "__main__":
    main()
