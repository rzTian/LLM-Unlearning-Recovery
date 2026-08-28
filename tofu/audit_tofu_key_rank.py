"""Key-token rank audit for TOFU forget splits."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import string
import time
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from common import processed_record, read_records, write_json, write_jsonl


STOPWORDS_OR_TEMPLATE = {
    "author", "authors", "book", "books", "writer", "writing", "work", "works",
    "novel", "novels", "genre", "primarily", "writes", "born", "award", "awards",
    "the", "a", "an", "of", "in", "on", "for", "and", "or", "to", "with", "by",
    "is", "are",
}

KEY_SPLITS = ("forget01", "forget05", "forget10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TOFU key-token rank audit.")
    parser.add_argument("--base_model_name", default="deepseek-ai/deepseek-llm-7b-chat")
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--target_adapter_dir", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--eval_data", required=True)
    parser.add_argument("--key_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_tag", default="model")
    parser.add_argument("--split", default=None)
    parser.add_argument("--key_vocab_scope", choices=["split", "global_keys", "genre"], default="split")
    parser.add_argument("--content_only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--lr", default=None)
    parser.add_argument("--weight_decay", default=None)
    parser.add_argument("--lora_rank", default=None)
    parser.add_argument("--lora_dropout", default=None)
    parser.add_argument("--grad_acc_steps", default=None)
    parser.add_argument("--reg", default=None)
    parser.add_argument("--beta", default=None)
    parser.add_argument("--summary_filename", default="summary.json")
    parser.add_argument("--details_filename", default="details.jsonl")
    parser.add_argument("--epoch_csv", default=None)
    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_source = args.model_path or args.base_model_name
    tokenizer = AutoTokenizer.from_pretrained(
        model_source if args.model_path else args.base_model_name,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    if args.target_adapter_dir:
        model = PeftModel.from_pretrained(model, args.target_adapter_dir, local_files_only=True)
        model = model.merge_and_unload()
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir, local_files_only=True)
    if torch.cuda.is_available():
        device = str(next(model.parameters()).device)
    else:
        device = "cpu"
        model.to(device)
    model.eval()
    return model, tokenizer, device


def mean(values: list[float] | Any) -> float | None:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else None


def list_median(values: list[float] | Any) -> float | None:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(median(vals)) if vals else None


def rate_at(values: list[int], threshold: int) -> float | None:
    return mean([1.0 if v <= threshold else 0.0 for v in values])


def tail_rate(percentiles: list[float], threshold: float) -> float | None:
    return mean([1.0 if v >= threshold else 0.0 for v in percentiles])


def weighted_mean(values: list[float], weights: list[float]) -> float | None:
    pairs = [(float(v), float(w)) for v, w in zip(values, weights) if v is not None and w is not None and float(w) > 0]
    if not pairs:
        return None
    total_w = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total_w if total_w else None


def weighted_rate(flags: list[float], weights: list[float]) -> float | None:
    return weighted_mean(flags, weights)


def normalize_token_text(text: str) -> str:
    text = str(text).strip()
    text = text.replace("▁", "").replace("Ġ", "").replace("Ċ", "")
    return re.sub(r"\s+", " ", text).strip().lower()


def is_punct_or_space(text: str) -> bool:
    stripped = str(text).strip()
    if not stripped:
        return True
    return all(ch in string.punctuation for ch in stripped)


def token_content_flags(token_text: str) -> dict[str, bool]:
    norm = normalize_token_text(token_text)
    is_stop = norm in STOPWORDS_OR_TEMPLATE
    is_template = norm in STOPWORDS_OR_TEMPLATE
    is_punct = is_punct_or_space(token_text) or is_punct_or_space(norm)
    return {
        "is_template_token": is_template,
        "is_stopword_or_punct": is_stop or is_punct,
        "is_content_token": bool(norm) and not is_template and not is_stop and not is_punct,
    }


def importance_weight(key_fact: dict[str, Any]) -> float:
    try:
        rank = float(key_fact.get("importance_rank"))
        return 1.0 / rank if rank > 0 else 1.0
    except Exception:
        return 1.0


def encode_answer_with_offsets(tokenizer, answer: str) -> tuple[list[int], list[tuple[int, int]] | None]:
    try:
        encoded = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded.get("offset_mapping")
        if offsets is not None:
            offsets = [(int(s), int(e)) for s, e in offsets]
        return [int(t) for t in encoded["input_ids"]], offsets
    except Exception:
        encoded = tokenizer(answer, add_special_tokens=False)
        return [int(t) for t in encoded["input_ids"]], None


def covered_indices_from_offsets(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    return [
        idx for idx, (tok_start, tok_end) in enumerate(offsets)
        if tok_end > start and tok_start < end and tok_end > tok_start
    ]


def normalized_substring_alignment(tokenizer, answer: str, fact_text: str, answer_ids: list[int]) -> list[int]:
    norm_answer = re.sub(r"\s+", " ", answer.lower())
    norm_fact = re.sub(r"\s+", " ", str(fact_text).lower()).strip()
    if not norm_fact or norm_fact not in norm_answer:
        return []
    fact_ids = tokenizer(fact_text, add_special_tokens=False)["input_ids"]
    if not fact_ids:
        return []
    for start in range(0, len(answer_ids) - len(fact_ids) + 1):
        if answer_ids[start : start + len(fact_ids)] == fact_ids:
            return list(range(start, start + len(fact_ids)))
    return []


def token_text_fuzzy_alignment(tokenizer, key_fact: dict[str, Any], answer_ids: list[int]) -> list[int]:
    fact_texts = key_fact.get("token_texts") or []
    if not fact_texts:
        fact_texts = [tokenizer.decode([tid]) for tid in key_fact.get("token_ids") or []]
    target = [normalize_token_text(t) for t in fact_texts if normalize_token_text(t)]
    if not target:
        return []
    answer_texts = [normalize_token_text(tokenizer.decode([tid])) for tid in answer_ids]
    compact_target = "".join(target)
    for start in range(len(answer_texts)):
        pieces = []
        indices = []
        for idx in range(start, len(answer_texts)):
            if answer_texts[idx]:
                pieces.append(answer_texts[idx])
                indices.append(idx)
            joined = "".join(pieces)
            if joined == compact_target:
                return indices
            if len(joined) > len(compact_target) + 4:
                break
    return []


def align_key_fact(tokenizer, answer: str, answer_ids: list[int], offsets: list[tuple[int, int]] | None, key_fact: dict[str, Any]) -> dict[str, Any]:
    method = "offset_mapping"
    indices: list[int] = []
    start = key_fact.get("char_start")
    end = key_fact.get("char_end")
    if offsets is not None and start is not None and end is not None:
        try:
            indices = covered_indices_from_offsets(offsets, int(start), int(end))
        except Exception:
            indices = []
    if not indices:
        method = "normalized_substring"
        indices = normalized_substring_alignment(tokenizer, answer, str(key_fact.get("text", "")), answer_ids)
    if not indices:
        method = "token_text_fuzzy"
        indices = token_text_fuzzy_alignment(tokenizer, key_fact, answer_ids)
    failed = not indices
    if failed:
        method = "failed"
    return {
        "answer_token_indices": indices,
        "answer_token_texts": [tokenizer.decode([answer_ids[i]]) for i in indices],
        "alignment_method": method,
        "alignment_failed": failed,
    }


def load_key_records(path: str | Path) -> list[dict[str, Any]]:
    return read_records(path)


def global_key_paths(key_data: str | Path) -> list[Path]:
    root = Path(key_data).parent
    return [root / f"{split}_key_tokens.jsonl" for split in KEY_SPLITS]


def prepare_alignments(tokenizer, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for record in records:
        copied = dict(record)
        answer = str(copied.get("answer", ""))
        answer_ids, offsets = encode_answer_with_offsets(tokenizer, answer)
        copied["_answer_ids"] = answer_ids
        copied["_answer_offsets"] = offsets
        aligned_facts = []
        for fact in copied.get("key_facts") or []:
            aligned = dict(fact)
            aligned.update(align_key_fact(tokenizer, answer, answer_ids, offsets, fact))
            aligned_facts.append(aligned)
        copied["_aligned_key_facts"] = aligned_facts
        out.append(copied)
    return out


def build_vocab_from_records(records: list[dict[str, Any]], tokenizer) -> tuple[list[int], dict[int, str]]:
    vocab: dict[int, str] = {}
    for record in records:
        answer_ids = record.get("_answer_ids") or []
        for fact in record.get("_aligned_key_facts") or []:
            for idx in fact.get("answer_token_indices") or []:
                if 0 <= idx < len(answer_ids):
                    tid = int(answer_ids[idx])
                    vocab.setdefault(tid, tokenizer.decode([tid]))
            if fact.get("alignment_failed"):
                for tid, text in zip(fact.get("token_ids") or [], fact.get("token_texts") or []):
                    vocab.setdefault(int(tid), str(text))
    return sorted(vocab), vocab


def build_genre_vocabs(records: list[dict[str, Any]], tokenizer) -> tuple[dict[str, list[int]], dict[int, str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        genre = str(record.get("genre") or record.get("category") or "unknown")
        grouped[genre].append(record)
    token_text: dict[int, str] = {}
    vocabs: dict[str, list[int]] = {}
    for genre, rows in grouped.items():
        vocab, mapping = build_vocab_from_records(rows, tokenizer)
        vocabs[genre] = vocab
        token_text.update(mapping)
    return vocabs, token_text


def percentile(rank: int, vocab_size: int) -> float | None:
    if rank <= 0 or vocab_size <= 1:
        return 0.0 if rank == 1 and vocab_size == 1 else None
    return (rank - 1) / (vocab_size - 1)


def rank_key_token(model, tokenizer, device: str, question: str, answer_ids: list[int], answer_idx: int, token_id: int, key_vocab: list[int], max_length: int) -> dict[str, Any] | None:
    import torch

    if token_id not in key_vocab:
        return None
    vocab_index = {tid: idx for idx, tid in enumerate(key_vocab)}
    prompt = f"[INST] {question} [/INST] "
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    context = (prompt_ids + answer_ids[:answer_idx])[-max_length:]
    if not context:
        return None
    input_ids = torch.tensor([context], device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0, -1].float()
    candidate_ids = torch.tensor(key_vocab, device=device)
    candidate_logits = logits[candidate_ids]
    sorted_desc = torch.argsort(candidate_logits, descending=True)
    gold_idx = vocab_index[int(token_id)]
    rank = int((sorted_desc == gold_idx).nonzero(as_tuple=True)[0].item()) + 1
    vocab_size = len(key_vocab)
    rig_rank = vocab_size - rank + 1
    gold_logit = float(logits[int(token_id)].item())
    token_text = tokenizer.decode([int(token_id)])
    flags = token_content_flags(token_text)
    return {
        "answer_token_index": int(answer_idx),
        "token_id": int(token_id),
        "token_text": token_text,
        "normalized_token_text": normalize_token_text(token_text),
        "rank_in_key_token_vocab": rank,
        "rank_percentile_in_key_token_vocab": percentile(rank, vocab_size),
        "rig_rank_in_key_token_vocab": rig_rank,
        "rig_percentile_in_key_token_vocab": percentile(rig_rank, vocab_size),
        "gold_logit": gold_logit,
        **flags,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [r["rank_in_key_token_vocab"] for r in rows]
    percentiles = [r["rank_percentile_in_key_token_vocab"] for r in rows if r["rank_percentile_in_key_token_vocab"] is not None]
    content = [r for r in rows if r.get("is_content_token")]
    content_ranks = [r["rank_in_key_token_vocab"] for r in content]
    content_percentiles = [r["rank_percentile_in_key_token_vocab"] for r in content if r["rank_percentile_in_key_token_vocab"] is not None]
    return {
        "num_key_tokens": len(rows),
        "num_content_key_tokens": len(content),
        "mean_key_token_rank": mean(ranks),
        "median_key_token_rank": list_median(ranks),
        "mean_key_token_percentile": mean(percentiles),
        "key_token_last_10pct": tail_rate(percentiles, 0.90),
        "mean_content_key_token_rank": mean(content_ranks),
        "content_key_token_last_10pct": tail_rate(content_percentiles, 0.90),
    }


def update_epoch_csv(csv_path: str | None, summary: dict[str, Any]) -> None:
    if not csv_path:
        return
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_tag", "split", "epoch", "lr", "weight_decay", "lora_rank", "lora_dropout",
        "grad_acc_steps", "reg", "beta", "key_vocab_scope", "content_only", "num_records",
        "num_key_facts", "num_key_tokens", "num_content_key_tokens", "key_vocab_size",
        "alignment_failed_rate", "mean_key_token_rank", "median_key_token_rank",
        "mean_key_token_percentile", "key_token_rg_at10", "key_token_rig_at10",
        "key_token_last_10pct", "key_token_last_25pct", "mean_content_key_token_rank",
        "content_key_token_last_10pct", "weighted_mean_key_token_rank",
        "weighted_key_token_last_10pct", "adapter_dir", "target_adapter_dir", "eval_data",
        "key_data", "summary_path", "details_path",
    ]
    row = {field: summary.get(field) for field in fields}
    lock_path = csv_file.with_suffix(csv_file.suffix + ".lock")
    lock_f = open(lock_path, "w")
    try:
        try:
            import fcntl
            fcntl.flock(lock_f, fcntl.LOCK_EX)
        except Exception:
            pass
        rows: list[dict[str, Any]] = []
        if csv_file.exists():
            with csv_file.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        def key(r: dict[str, Any]) -> tuple[str, str, str, str, str, str, str, str]:
            return (
                str(r.get("model_tag")), str(r.get("split")), str(r.get("epoch")),
                str(r.get("lr")), str(r.get("reg")), str(r.get("beta")),
                str(r.get("key_vocab_scope")), str(r.get("content_only")),
            )

        rows = [r for r in rows if key(r) != key(row)]
        rows.append({field: "" if row.get(field) is None else row.get(field) for field in fields})

        def sort_key(r: dict[str, Any]) -> tuple[str, str, int, str, str]:
            try:
                ep = int(r.get("epoch") or -1)
            except Exception:
                ep = -1
            return (str(r.get("model_tag")), str(r.get("split")), ep, str(r.get("key_vocab_scope")), str(r.get("content_only")))

        tmp_file = csv_file.with_suffix(csv_file.suffix + ".tmp")
        with tmp_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(rows, key=sort_key))
        os.replace(tmp_file, csv_file)
    finally:
        try:
            import fcntl
            fcntl.flock(lock_f, fcntl.LOCK_UN)
        except Exception:
            pass
        lock_f.close()


def main() -> None:
    args = parse_args()
    split = args.split or Path(args.eval_data).stem
    eval_records = [processed_record(r, split) if "category" not in r else r for r in read_records(args.eval_data)]
    key_records_raw = load_key_records(args.key_data)
    if args.limit is not None:
        eval_records = eval_records[: args.limit]
        key_records_raw = key_records_raw[: args.limit]

    print(f"[audit-key] loading model base={args.base_model_name}", flush=True)
    model, tokenizer, device = load_model_and_tokenizer(args)

    key_records = prepare_alignments(tokenizer, key_records_raw)
    if args.key_vocab_scope == "global_keys":
        global_raw = []
        for path in global_key_paths(args.key_data):
            if path.exists():
                global_raw.extend(load_key_records(path))
        vocab_records = prepare_alignments(tokenizer, global_raw)
        key_vocab, _ = build_vocab_from_records(vocab_records, tokenizer)
        genre_vocabs = {}
    elif args.key_vocab_scope == "genre":
        genre_vocabs, _ = build_genre_vocabs(key_records, tokenizer)
        key_vocab, _ = build_vocab_from_records(key_records, tokenizer)
    else:
        key_vocab, _ = build_vocab_from_records(key_records, tokenizer)
        genre_vocabs = {}

    print(
        f"[audit-key] records={len(key_records)} key_vocab_size={len(key_vocab)} scope={args.key_vocab_scope}",
        flush=True,
    )

    details = []
    all_token_rows: list[dict[str, Any]] = []
    alignment_failed_count = 0
    num_key_facts = 0
    start_time = time.perf_counter()

    for idx, key_record in enumerate(key_records):
        eval_record = eval_records[idx] if idx < len(eval_records) else key_record
        question = str(key_record.get("question") or eval_record.get("question", ""))
        answer = str(key_record.get("answer") or eval_record.get("answer", ""))
        category = key_record.get("category") or eval_record.get("category")
        answer_ids = key_record.get("_answer_ids") or []
        record_vocab = key_vocab
        if args.key_vocab_scope == "genre":
            record_vocab = genre_vocabs.get(str(key_record.get("genre") or category or "unknown"), key_vocab)

        detail_facts = []
        record_rows = []
        for fact in key_record.get("_aligned_key_facts") or []:
            num_key_facts += 1
            if fact.get("alignment_failed"):
                alignment_failed_count += 1
            fact_rows = []
            for answer_idx in fact.get("answer_token_indices") or []:
                if answer_idx >= len(answer_ids):
                    continue
                token_id = int(answer_ids[answer_idx])
                row = rank_key_token(model, tokenizer, device, question, answer_ids, answer_idx, token_id, record_vocab, args.max_length)
                if row is None:
                    continue
                row.update(
                    {
                        "key_fact_text": fact.get("text"),
                        "key_fact_reason": fact.get("reason"),
                        "importance_rank": fact.get("importance_rank"),
                        "token_importance": importance_weight(fact),
                        "alignment_method": fact.get("alignment_method"),
                    }
                )
                fact_rows.append(row)
                record_rows.append(row)
                all_token_rows.append(row)
            detail_facts.append(
                {
                    "text": fact.get("text"),
                    "reason": fact.get("reason"),
                    "importance_rank": fact.get("importance_rank"),
                    "char_start": fact.get("char_start"),
                    "char_end": fact.get("char_end"),
                    "alignment_failed": fact.get("alignment_failed"),
                    "answer_token_indices": fact.get("answer_token_indices"),
                    "answer_token_texts": fact.get("answer_token_texts"),
                    "alignment_method": fact.get("alignment_method"),
                    "token_ranks": fact_rows,
                }
            )

        detail = {
            "index": key_record.get("index", idx),
            "split": split,
            "question": question,
            "answer": answer,
            "category": category,
            "key_facts": detail_facts,
            "aggregate": aggregate_rows(record_rows),
        }
        details.append(detail)

        done = idx + 1
        if done == 1 or done == len(key_records) or args.log_every <= 1 or done % args.log_every == 0:
            elapsed = time.perf_counter() - start_time
            avg_sec = elapsed / done
            eta_sec = max(0, len(key_records) - done) * avg_sec
            print(
                "[audit-key-progress] "
                f"{done}/{len(key_records)} elapsed={elapsed/60:.1f}min eta={eta_sec/60:.1f}min "
                f"record_key_tokens={len(record_rows)}",
                flush=True,
            )

    ranks = [r["rank_in_key_token_vocab"] for r in all_token_rows]
    rig_ranks = [r["rig_rank_in_key_token_vocab"] for r in all_token_rows]
    percentile_rows = [r for r in all_token_rows if r["rank_percentile_in_key_token_vocab"] is not None]
    rig_percentile_rows = [r for r in all_token_rows if r["rig_percentile_in_key_token_vocab"] is not None]
    percentiles = [r["rank_percentile_in_key_token_vocab"] for r in percentile_rows]
    rig_percentiles = [r["rig_percentile_in_key_token_vocab"] for r in rig_percentile_rows]
    content_rows = [r for r in all_token_rows if r.get("is_content_token")]
    content_ranks = [r["rank_in_key_token_vocab"] for r in content_rows]
    content_percentiles = [r["rank_percentile_in_key_token_vocab"] for r in content_rows if r["rank_percentile_in_key_token_vocab"] is not None]
    content_rig_ranks = [r["rig_rank_in_key_token_vocab"] for r in content_rows]
    weights = [float(r.get("token_importance") or 1.0) for r in all_token_rows]
    percentile_weights = [float(r.get("token_importance") or 1.0) for r in percentile_rows]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / args.summary_filename
    details_path = output_dir / args.details_filename

    summary = {
        "kind": "audit_key_rank",
        "model_tag": args.model_tag,
        "split": split,
        "epoch": args.epoch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lora_rank": args.lora_rank,
        "lora_dropout": args.lora_dropout,
        "grad_acc_steps": args.grad_acc_steps,
        "reg": args.reg,
        "beta": args.beta,
        "eval_data": args.eval_data,
        "key_data": args.key_data,
        "key_vocab_scope": args.key_vocab_scope,
        "content_only": bool(args.content_only),
        "base_model_name": args.base_model_name,
        "adapter_dir": args.adapter_dir,
        "target_adapter_dir": args.target_adapter_dir,
        "model_path": args.model_path,
        "num_records": len(details),
        "num_key_facts": num_key_facts,
        "num_key_tokens": len(all_token_rows),
        "num_content_key_tokens": len(content_rows),
        "key_vocab_size": len(key_vocab),
        "alignment_failed_count": alignment_failed_count,
        "alignment_failed_rate": (alignment_failed_count / num_key_facts) if num_key_facts else None,
        "mean_key_token_rank": mean(ranks),
        "median_key_token_rank": list_median(ranks),
        "mean_key_token_percentile": mean(percentiles),
        "median_key_token_percentile": list_median(percentiles),
        "key_token_rg_at1": rate_at(ranks, 1),
        "key_token_rg_at5": rate_at(ranks, 5),
        "key_token_rg_at10": rate_at(ranks, 10),
        "key_token_rg_at50": rate_at(ranks, 50),
        "key_token_rg_at100": rate_at(ranks, 100),
        "mean_key_token_rig_rank": mean(rig_ranks),
        "median_key_token_rig_rank": list_median(rig_ranks),
        "mean_key_token_rig_percentile": mean(rig_percentiles),
        "median_key_token_rig_percentile": list_median(rig_percentiles),
        "key_token_rig_at1": rate_at(rig_ranks, 1),
        "key_token_rig_at5": rate_at(rig_ranks, 5),
        "key_token_rig_at10": rate_at(rig_ranks, 10),
        "key_token_rig_at50": rate_at(rig_ranks, 50),
        "key_token_rig_at100": rate_at(rig_ranks, 100),
        "key_token_last_10pct": tail_rate(percentiles, 0.90),
        "key_token_last_25pct": tail_rate(percentiles, 0.75),
        "key_token_last_50pct": tail_rate(percentiles, 0.50),
        "mean_content_key_token_rank": mean(content_ranks),
        "median_content_key_token_rank": list_median(content_ranks),
        "mean_content_key_token_percentile": mean(content_percentiles),
        "content_key_token_rg_at10": rate_at(content_ranks, 10),
        "content_key_token_rig_at10": rate_at(content_rig_ranks, 10),
        "content_key_token_last_10pct": tail_rate(content_percentiles, 0.90),
        "content_key_token_last_25pct": tail_rate(content_percentiles, 0.75),
        "weighted_mean_key_token_rank": weighted_mean([float(v) for v in ranks], weights),
        "weighted_mean_key_token_percentile": weighted_mean([float(v) for v in percentiles], percentile_weights),
        "weighted_key_token_last_10pct": weighted_rate([1.0 if p >= 0.90 else 0.0 for p in percentiles], percentile_weights),
        "weighted_key_token_rig_at10": weighted_rate([1.0 if v <= 10 else 0.0 for v in rig_ranks], weights),
        "summary_path": str(summary_path),
        "details_path": str(details_path),
    }

    write_json(summary_path, summary)
    write_jsonl(details_path, details)
    update_epoch_csv(args.epoch_csv, summary)
    print(f"[audit-key] summary -> {summary_path}", flush=True)
    print(f"[audit-key] details -> {details_path}", flush=True)
    if args.epoch_csv:
        print(f"[audit-key] epoch csv -> {args.epoch_csv}", flush=True)


if __name__ == "__main__":
    main()
