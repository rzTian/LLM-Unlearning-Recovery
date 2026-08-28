"""Rank-based TOFU audit diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import list_median, mean, mode_with_count, processed_record, read_records, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sentence-level and answer-token rank audit for TOFU.")
    parser.add_argument("--base_model_name", default="deepseek-ai/deepseek-llm-7b-chat")
    parser.add_argument("--adapter_dir", default=None, help="LoRA adapter for target/oracle, or unlearn adapter when target_adapter_dir is set.")
    parser.add_argument("--target_adapter_dir", default=None, help="Target full adapter to merge before loading unlearn adapter.")
    parser.add_argument("--model_path", default=None, help="Optional full model path fallback.")
    parser.add_argument("--eval_data", required=True)
    parser.add_argument("--pool_data", default="TOFU/full.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_tag", default="model")
    parser.add_argument("--split", default=None)
    parser.add_argument("--candidate_pool", choices=["category", "all"], default="category")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--local_files_only", action="store_true")
    
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

    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
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


def avg_nll_for_answer(model, tokenizer, device: str, question: str, answer: str, max_length: int) -> tuple[float, float, int]:
    prompt = f"[INST] {question} [/INST] "
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + answer_ids)[:max_length]
    answer_len = max(0, min(len(answer_ids), len(ids) - len(prompt_ids)))
    if answer_len == 0:
        return float("inf"), 0.0, 0
    input_ids = torch.tensor([ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits.float()
    answer_start = len(ids) - answer_len
    losses = []
    for pos in range(answer_start, len(ids)):
        log_probs = F.log_softmax(logits[0, pos - 1], dim=-1)
        losses.append(float(-log_probs[input_ids[0, pos]].item()))
    avg_nll = sum(losses) / len(losses)
    return avg_nll, math.exp(-avg_nll), len(losses)


def build_pools(pool_records: list[dict[str, Any]]) -> tuple[list[str], dict[str, list[str]]]:
    all_answers = []
    by_category: dict[str, list[str]] = defaultdict(list)
    seen_all = set()
    seen_cat: dict[str, set[str]] = defaultdict(set)
    for record in pool_records:
        answer = str(record["answer"])
        category = record.get("category") or processed_record(record, "pool")["category"]
        if answer not in seen_all:
            all_answers.append(answer)
            seen_all.add(answer)
        if answer not in seen_cat[category]:
            by_category[category].append(answer)
            seen_cat[category].add(answer)
    return all_answers, by_category


def build_answer_token_vocab(tokenizer, pool_records: list[dict[str, Any]]) -> list[int]:
    vocab = set()
    for record in pool_records:
        vocab.update(tokenizer(str(record["answer"]), add_special_tokens=False)["input_ids"])
    return sorted(vocab)


def rank_desc(values: list[tuple[str, float]], gold: str) -> int:
    ordered = sorted(values, key=lambda item: item[1], reverse=True)
    for idx, (candidate, _) in enumerate(ordered, start=1):
        if candidate == gold:
            return idx
    return -1


def token_ranks(model, tokenizer, device: str, question: str, answer: str, answer_vocab: list[int], max_length: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = f"[INST] {question} [/INST] "
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    vocab_index = {tid: idx for idx, tid in enumerate(answer_vocab)}
    rows = []
    missing = 0
    for answer_idx, token_id in enumerate(answer_ids):
        if token_id not in vocab_index:
            missing += 1
            continue
        context = (prompt_ids + answer_ids[:answer_idx])[-max_length:]
        input_ids = torch.tensor([context], device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0, -1].float()
        candidate_logits = logits[torch.tensor(answer_vocab, device=device)]
        gold_logit = float(logits[token_id].item())
        sorted_indices = torch.argsort(candidate_logits, descending=True)
        rank = int((sorted_indices == vocab_index[token_id]).nonzero(as_tuple=True)[0].item()) + 1
        rows.append(
            {
                "answer_token_index": answer_idx,
                "sequence_position": len(prompt_ids) + answer_idx,
                "token_id": int(token_id),
                "token_text": tokenizer.decode([token_id]),
                "rank_in_answer_token_vocab": rank,
                "gold_logit": gold_logit,
            }
        )
    ranks = [r["rank_in_answer_token_vocab"] for r in rows]
    mode, mode_count = mode_with_count(ranks)
    stats = {
        "rank_min": min(ranks) if ranks else None,
        "rank_max": max(ranks) if ranks else None,
        "rank_mean": mean(ranks),
        "rank_median": list_median(ranks),
        "rank_mode": mode,
        "rank_mode_count": mode_count,
        "num_answer_tokens_ranked": len(rows),
        "num_answer_tokens_missing_from_vocab": missing,
        "answer_token_vocab_size": len(answer_vocab),
    }
    return rows, stats


def update_epoch_csv(csv_path: str | None, summary: dict[str, Any]) -> None:
    if not csv_path:
        return

    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "kind",
        "model_tag",
        "split",
        "epoch",
        "lr",
        "weight_decay",
        "lora_rank",
        "lora_dropout",
        "grad_acc_steps",
        "reg",
        "beta",
        "num_records",
        "candidate_pool",
        "mean_sentence_rg_rank",
        "mean_sentence_rig_rank",
        "sentence_rg_at1",
        "sentence_rg_at5",
        "sentence_rig_at1",
        "sentence_rig_at5",
        "mean_token_rank",
        "median_token_rank",
        "global_token_rank_min",
        "global_token_rank_max",
        "answer_token_vocab_size",
        "adapter_dir",
        "target_adapter_dir",
        "eval_data",
        "summary_path",
        "details_path",
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
            with open(csv_file, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        def key(r: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
            return (
                str(r.get("model_tag")),
                str(r.get("split")),
                str(r.get("epoch")),
                str(r.get("lr")),
                str(r.get("reg")),
                str(r.get("beta")),
            )

        new_key = key(row)
        rows = [r for r in rows if key(r) != new_key]
        rows.append({k: "" if row.get(k) is None else row.get(k) for k in fields})

        def sort_key(r: dict[str, Any]):
            try:
                ep = int(r.get("epoch") or -1)
            except Exception:
                ep = -1
            return (
                str(r.get("model_tag")),
                str(r.get("split")),
                ep,
                str(r.get("lr")),
                str(r.get("reg")),
                str(r.get("beta")),
            )

        rows = sorted(rows, key=sort_key)

        tmp_file = csv_file.with_suffix(csv_file.suffix + ".tmp")
        with open(tmp_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

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
    if args.limit is not None:
        eval_records = eval_records[: args.limit]
    pool_records = [processed_record(r, "full") if "category" not in r else r for r in read_records(args.pool_data)]

    model, tokenizer, device = load_model_and_tokenizer(args)
    all_answers, by_category = build_pools(pool_records)
    answer_vocab = build_answer_token_vocab(tokenizer, pool_records)

    details = []
    start_time = time.perf_counter()
    for idx, record in enumerate(eval_records):
        question = record["question"]
        answer = record["answer"]
        category = record.get("category") or processed_record(record, split)["category"]
        candidates = by_category.get(category, []) if args.candidate_pool == "category" else all_answers
        if answer not in candidates:
            candidates = list(candidates) + [answer]

        scored = []
        true_nll = None
        true_norm_prob = None
        for candidate in candidates:
            avg_nll, norm_prob, _ = avg_nll_for_answer(model, tokenizer, device, question, candidate, args.max_length)
            scored.append((candidate, avg_nll, norm_prob))
            if candidate == answer:
                true_nll = avg_nll
                true_norm_prob = norm_prob

        rg_order = sorted(scored, key=lambda item: item[1])
        rig_order = sorted(scored, key=lambda item: item[1], reverse=True)
        rg_rank = next((i for i, item in enumerate(rg_order, start=1) if item[0] == answer), -1)
        rig_rank = next((i for i, item in enumerate(rig_order, start=1) if item[0] == answer), -1)
        token_rows, token_stats = token_ranks(model, tokenizer, device, question, answer, answer_vocab, args.max_length)
        detail = {
            "index": idx,
            "split": split,
            "category": category,
            "question": question,
            "answer": answer,
            "sentence_rg_rank": rg_rank,
            "sentence_rig_rank": rig_rank,
            "sentence_true_avg_nll": true_nll,
            "sentence_true_norm_prob": true_norm_prob,
            "sentence_rg_top1": rg_order[0][0] if rg_order else None,
            "sentence_rig_top1": rig_order[0][0] if rig_order else None,
            "sentence_rg_top5_contains_gold": any(item[0] == answer for item in rg_order[:5]),
            "sentence_rig_top5_contains_gold": any(item[0] == answer for item in rig_order[:5]),
            "candidate_pool_size": len(candidates),
            "answer_token_ranks": token_rows,
            **token_stats,
        }
        details.append(detail)
        
        done = idx + 1
        elapsed = time.perf_counter() - start_time
        avg_sec = elapsed / done
        eta_sec = max(0, len(eval_records) - done) * avg_sec

        if done == 1 or done == len(eval_records) or args.log_every <= 1 or done % args.log_every == 0:
            print(
                "[audit-progress] "
                f"{done}/{len(eval_records)} "
                f"elapsed={elapsed/60:.1f}min "
                f"eta={eta_sec/60:.1f}min "
                f"rg={rg_rank} "
                f"rig={rig_rank} "
                f"mean_token_rank={token_stats['rank_mean']}",
                flush=True,
            )

    token_rank_values = [
        token["rank_in_answer_token_vocab"]
        for detail in details
        for token in detail["answer_token_ranks"]
    ]
    summary = {
        "kind": "audit_rank",
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
        "pool_data": args.pool_data,
        "candidate_pool": args.candidate_pool,
        "base_model_name": args.base_model_name,
        "adapter_dir": args.adapter_dir,
        "target_adapter_dir": args.target_adapter_dir,
        "model_path": args.model_path,
        "num_records": len(details),
        "mean_sentence_rg_rank": mean(d["sentence_rg_rank"] for d in details),
        "mean_sentence_rig_rank": mean(d["sentence_rig_rank"] for d in details),
        "sentence_rg_at1": mean(1.0 if d["sentence_rg_rank"] == 1 else 0.0 for d in details),
        "sentence_rg_at5": mean(1.0 if d["sentence_rg_top5_contains_gold"] else 0.0 for d in details),
        "sentence_rig_at1": mean(1.0 if d["sentence_rig_rank"] == 1 else 0.0 for d in details),
        "sentence_rig_at5": mean(1.0 if d["sentence_rig_top5_contains_gold"] else 0.0 for d in details),
        "mean_token_rank": mean(token_rank_values),
        "median_token_rank": list_median(token_rank_values),
        "global_token_rank_min": min(token_rank_values) if token_rank_values else None,
        "global_token_rank_max": max(token_rank_values) if token_rank_values else None,
        "answer_token_vocab_size": len(answer_vocab),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / args.summary_filename
    details_path = output_dir / args.details_filename

    summary["summary_path"] = str(summary_path)
    summary["details_path"] = str(details_path)

    write_json(summary_path, summary)
    write_jsonl(details_path, details)
    update_epoch_csv(args.epoch_csv, summary)

    print(f"[audit] summary -> {summary_path}", flush=True)
    print(f"[audit] details -> {details_path}", flush=True)
    if args.epoch_csv:
        print(f"[audit] epoch csv -> {args.epoch_csv}", flush=True)


if __name__ == "__main__":
    main()
