#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from common import make_prompt, processed_record, read_records, write_json, write_jsonl

METRIC_VERSION = "05_key_memory_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TOFU 05 key memory evaluation.")
    p.add_argument("--eval_data", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--tokenized_key_file", required=True)
    p.add_argument("--base_model_name", default="deepseek-ai/deepseek-llm-7b-chat")
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--target_adapter_dir", default=None)
    p.add_argument("--model_family", choices=["ft", "unlearned"], required=True)
    p.add_argument("--model_tag", default=None)
    p.add_argument("--method", default=None)
    p.add_argument("--unlearn_run", default=None)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--generation_batch_size", type=int, default=1)
    p.add_argument("--nll_batch_size", type=int, default=1)
    p.add_argument("--content_recall_hit_threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_generation", action="store_true")
    p.add_argument("--skip_nll", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--cache_dir", default=None)
    return p.parse_args()


def norm_text(x: Any) -> str:
    s = unicodedata.normalize("NFKC", str(x or "")).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_mean(vals: list[float]) -> float | None:
    v = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    return (sum(v) / len(v)) if v else None


def safe_median(vals: list[float]) -> float | None:
    v = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    return float(median(v)) if v else None


def weighted_mean(vals: list[float], ws: list[float]) -> float | None:
    pairs = [(float(v), float(w)) for v, w in zip(vals, ws) if v is not None and w is not None and float(w) > 0 and math.isfinite(float(v))]
    if not pairs:
        return None
    tw = sum(w for _, w in pairs)
    return (sum(v * w for v, w in pairs) / tw) if tw > 0 else None


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        args.base_model_name,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
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

    device = str(next(model.parameters()).device) if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        model.to(device)
    model.eval()
    return model, tok, device


def build_key_match_index(key_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_src: dict[int, dict[str, Any]] = {}
    by_exact_qa: dict[tuple[str, str], dict[str, Any]] = {}
    by_norm_qa: dict[tuple[str, str], dict[str, Any]] = {}
    q_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in key_rows:
        if "source_index" in r:
            try:
                by_src[int(r["source_index"])] = r
            except Exception:
                pass
        q = str(r.get("question") or "")
        a = str(r.get("answer") or "")
        by_exact_qa[(q, a)] = r
        by_norm_qa[(norm_text(q), norm_text(a))] = r
        q_bucket[norm_text(q)].append(r)
    unique_q = {k: v[0] for k, v in q_bucket.items() if len(v) == 1}
    return {
        "by_src": by_src,
        "by_exact_qa": by_exact_qa,
        "by_norm_qa": by_norm_qa,
        "by_unique_norm_q": unique_q,
    }


def match_key_row(eval_row: dict[str, Any], idx: int, m: dict[str, Any]) -> dict[str, Any] | None:
    src = eval_row.get("source_index", idx)
    try:
        src_i = int(src)
        if src_i in m["by_src"]:
            return m["by_src"][src_i]
    except Exception:
        pass
    q = str(eval_row.get("question") or "")
    a = str(eval_row.get("answer") or "")
    x = m["by_exact_qa"].get((q, a))
    if x is not None:
        return x
    nq, na = norm_text(q), norm_text(a)
    x = m["by_norm_qa"].get((nq, na))
    if x is not None:
        return x
    return m["by_unique_norm_q"].get(nq)


def is_meaningful_content_token(raw: str) -> bool:
    t = str(raw or "").strip()
    if not t:
        return False
    n = norm_text(t)
    if not n:
        return False
    if len(n) > 1:
        return True
    ch = n[0]
    return ch.isdigit() or ch.isalpha()


def score_key_span_nll(model, tok, device: str, question: str, answer: str, fact: dict[str, Any]) -> dict[str, Any]:
    import torch

    try:
        cs = int(fact.get("char_start"))
        ce = int(fact.get("char_end"))
    except Exception:
        return {"key_span_sum_nll": None, "key_span_avg_nll": None, "key_span_norm_prob": None, "num_span_tokens": 0}

    tids = [int(x) for x in (fact.get("answer_token_ids") or [])]
    if not tids or cs < 0 or ce <= cs or ce > len(answer):
        return {"key_span_sum_nll": None, "key_span_avg_nll": None, "key_span_norm_prob": None, "num_span_tokens": 0}

    ctx = make_prompt(question) + answer[:cs]
    ctx_ids = tok(ctx, add_special_tokens=True)["input_ids"]
    prefix = list(ctx_ids)
    losses: list[float] = []
    for tid in tids:
        inp = torch.tensor([prefix], device=device)
        with torch.no_grad():
            logits = model(input_ids=inp).logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        losses.append(float(-logp[int(tid)].item()))
        prefix.append(int(tid))
    if not losses:
        return {"key_span_sum_nll": None, "key_span_avg_nll": None, "key_span_norm_prob": None, "num_span_tokens": 0}
    s = float(sum(losses))
    a = s / len(losses)
    return {"key_span_sum_nll": s, "key_span_avg_nll": a, "key_span_norm_prob": math.exp(-a), "num_span_tokens": len(losses)}


def generate_answer(model, tok, device: str, question: str, max_new_tokens: int, seed: int) -> str:
    import torch

    torch.manual_seed(seed)
    prompt = make_prompt(question)
    inps = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inps,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=0.0,
            pad_token_id=tok.eos_token_id,
        )
    gen_ids = out[0][inps["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True)


def update_epoch_curve(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_family", "model_tag", "method", "unlearn_run", "epoch", "split", "summary_path",
        "num_eval_records", "num_key_facts", "open_key_recall", "open_span_recall", "open_content_token_recall",
        "weighted_open_key_recall", "weighted_open_content_token_recall", "mean_key_span_avg_nll",
        "median_key_span_avg_nll", "weighted_mean_key_span_avg_nll", "mean_key_span_norm_prob",
        "median_key_span_norm_prob", "effective_batching",
    ]
    row = {k: summary.get(k) for k in fields}
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    def key(r: dict[str, Any]) -> tuple[str, ...]:
        return (
            str(r.get("model_family") or ""), str(r.get("model_tag") or ""), str(r.get("method") or ""),
            str(r.get("unlearn_run") or ""), str(r.get("epoch") or ""), str(r.get("split") or ""),
        )

    rows = [r for r in rows if key(r) != key(row)]
    rows.append({k: "" if row.get(k) is None else row.get(k) for k in fields})
    rows.sort(key=lambda r: (str(r.get("split")), str(r.get("model_family")), str(r.get("method")), int(r.get("epoch") or -1)))

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    eval_rows = [processed_record(r, args.split) if "question" in r and "answer" in r else r for r in read_records(args.eval_data)]
    key_rows = read_records(args.tokenized_key_file)
    if args.limit is not None:
        eval_rows = eval_rows[: args.limit]

    model, tok, device = load_model_and_tokenizer(args)
    matchers = build_key_match_index(key_rows)

    details = []
    fact_open_hits: list[float] = []
    fact_span_hits: list[float] = []
    fact_ct_recalls: list[float] = []
    fact_weights: list[float] = []
    ct_weights: list[float] = []
    nll_vals: list[float] = []
    nll_probs: list[float] = []
    nll_weights: list[float] = []

    per_fact_group = defaultdict(lambda: {"open_fact_hit": [], "open_span_hit": [], "content_token_recall": [], "key_span_avg_nll": [], "key_span_norm_prob": [], "w": []})
    per_genre_group = defaultdict(lambda: {"open_fact_hit": [], "open_span_hit": [], "content_token_recall": [], "key_span_avg_nll": [], "key_span_norm_prob": [], "w": []})

    num_key_matched_records = 0
    num_key_match_failed_records = 0
    num_records_without_key_facts = 0
    num_key_facts = 0

    for i, row in enumerate(eval_rows):
        kr = match_key_row(row, i, matchers)
        q = str(row.get("question") or "")
        a = str(row.get("answer") or "")
        src = row.get("source_index", i)
        genre_group = None
        generation = ""
        if kr is None:
            num_key_match_failed_records += 1
            details.append({
                "source_index": src,
                "split": args.split,
                "question": q,
                "gold_answer": a,
                "generation": generation,
                "generation_params": {"max_new_tokens": args.max_new_tokens, "do_sample": False, "num_beams": 1},
                "key_facts": [],
            })
            continue

        num_key_matched_records += 1
        q = str(kr.get("question") or q)
        a = str(kr.get("answer") or a)
        genre_group = str(kr.get("genre_group") or "unknown")

        if not args.skip_generation:
            generation = generate_answer(model, tok, device, q, args.max_new_tokens, args.seed + i)
        gnorm = norm_text(generation)

        q_facts = []
        facts = kr.get("key_facts") or []
        if not facts:
            num_records_without_key_facts += 1

        for fact in facts:
            num_key_facts += 1
            fg = str(fact.get("fact_group") or "unknown")
            wt = float(fact.get("importance_weight") or 1.0)

            ftxt = str(fact.get("text") or "")
            ftxt_n = norm_text(ftxt)
            align_failed = bool(fact.get("token_alignment_failed", False))
            open_span_hit = None
            if ftxt_n and not align_failed:
                open_span_hit = ftxt_n in gnorm

            content_tokens = [str(t) for t in (fact.get("content_token_texts") or []) if is_meaningful_content_token(str(t))]
            ctr = None
            if content_tokens:
                hits = 0
                total = 0
                for t in content_tokens:
                    tn = norm_text(t)
                    if not tn:
                        continue
                    total += 1
                    if tn in gnorm:
                        hits += 1
                ctr = (hits / total) if total > 0 else None

            open_fact_hit = None
            if open_span_hit is not None or ctr is not None:
                open_fact_hit = bool(open_span_hit) or (ctr is not None and ctr >= args.content_recall_hit_threshold)

            span_nll = {"key_span_sum_nll": None, "key_span_avg_nll": None, "key_span_norm_prob": None, "num_span_tokens": 0}
            if not args.skip_nll:
                span_nll = score_key_span_nll(model, tok, device, q, a, fact)

            q_facts.append({
                "fact_id": fact.get("fact_id"),
                "text": fact.get("text"),
                "fact_group": fg,
                "genre_group": genre_group,
                "importance_rank": fact.get("importance_rank"),
                "importance_weight": wt,
                "open_span_hit": open_span_hit,
                "content_token_recall": ctr,
                "open_fact_hit": open_fact_hit,
                **span_nll,
            })

            if open_fact_hit is not None:
                fact_open_hits.append(1.0 if open_fact_hit else 0.0)
                fact_weights.append(wt)
                per_fact_group[fg]["open_fact_hit"].append(1.0 if open_fact_hit else 0.0)
                per_genre_group[genre_group]["open_fact_hit"].append(1.0 if open_fact_hit else 0.0)
            if open_span_hit is not None:
                fact_span_hits.append(1.0 if open_span_hit else 0.0)
                per_fact_group[fg]["open_span_hit"].append(1.0 if open_span_hit else 0.0)
                per_genre_group[genre_group]["open_span_hit"].append(1.0 if open_span_hit else 0.0)
            if ctr is not None:
                fact_ct_recalls.append(float(ctr))
                ct_weights.append(wt)
                per_fact_group[fg]["content_token_recall"].append(float(ctr))
                per_genre_group[genre_group]["content_token_recall"].append(float(ctr))
            if span_nll["key_span_avg_nll"] is not None:
                v = float(span_nll["key_span_avg_nll"])
                nll_vals.append(v)
                nll_weights.append(wt)
                per_fact_group[fg]["key_span_avg_nll"].append(v)
                per_genre_group[genre_group]["key_span_avg_nll"].append(v)
            if span_nll["key_span_norm_prob"] is not None:
                pv = float(span_nll["key_span_norm_prob"])
                nll_probs.append(pv)
                per_fact_group[fg]["key_span_norm_prob"].append(pv)
                per_genre_group[genre_group]["key_span_norm_prob"].append(pv)
            per_fact_group[fg]["w"].append(wt)
            per_genre_group[genre_group]["w"].append(wt)

        details.append({
            "source_index": src,
            "split": args.split,
            "question": q,
            "gold_answer": a,
            "generation": generation,
            "generation_params": {"max_new_tokens": args.max_new_tokens, "do_sample": False, "temperature": 0.0, "num_beams": 1},
            "key_facts": q_facts,
        })

    def group_block(bucket: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, d in sorted(bucket.items()):
            out[k] = {
                "num_facts": len(d["w"]),
                "open_key_recall": safe_mean(d["open_fact_hit"]),
                "open_span_recall": safe_mean(d["open_span_hit"]),
                "open_content_token_recall": safe_mean(d["content_token_recall"]),
                "weighted_open_key_recall": weighted_mean(d["open_fact_hit"], d["w"]),
                "weighted_open_content_token_recall": weighted_mean(d["content_token_recall"], d["w"]),
                "mean_key_span_avg_nll": safe_mean(d["key_span_avg_nll"]),
                "median_key_span_avg_nll": safe_median(d["key_span_avg_nll"]),
                "weighted_mean_key_span_avg_nll": weighted_mean(d["key_span_avg_nll"], d["w"]),
                "mean_key_span_norm_prob": safe_mean(d["key_span_norm_prob"]),
                "median_key_span_norm_prob": safe_median(d["key_span_norm_prob"]),
            }
        return out

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"epoch-{args.epoch}-{args.split}-summary.json"
    details_path = out_dir / f"epoch-{args.epoch}-{args.split}-details.jsonl"

    summary = {
        "metric_version": METRIC_VERSION,
        "model_family": args.model_family,
        "model_tag": args.model_tag,
        "method": args.method,
        "unlearn_run": args.unlearn_run,
        "epoch": args.epoch,
        "split": args.split,
        "eval_data": args.eval_data,
        "tokenized_key_file": args.tokenized_key_file,
        "adapter_dir": args.adapter_dir,
        "target_adapter_dir": args.target_adapter_dir,
        "num_eval_records": len(eval_rows),
        "num_key_matched_records": num_key_matched_records,
        "num_key_match_failed_records": num_key_match_failed_records,
        "num_records_without_key_facts": num_records_without_key_facts,
        "num_key_facts": num_key_facts,
        "open_key_recall": safe_mean(fact_open_hits),
        "open_span_recall": safe_mean(fact_span_hits),
        "open_content_token_recall": safe_mean(fact_ct_recalls),
        "weighted_open_key_recall": weighted_mean(fact_open_hits, fact_weights),
        "weighted_open_content_token_recall": weighted_mean(fact_ct_recalls, ct_weights),
        "mean_key_span_avg_nll": safe_mean(nll_vals),
        "median_key_span_avg_nll": safe_median(nll_vals),
        "weighted_mean_key_span_avg_nll": weighted_mean(nll_vals, nll_weights),
        "mean_key_span_norm_prob": safe_mean(nll_probs),
        "median_key_span_norm_prob": safe_median(nll_probs),
        "per_fact_group": group_block(per_fact_group),
        "per_genre_group": group_block(per_genre_group),
        "effective_batching": "sequential",
    }

    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Summary exists; use --overwrite: {summary_path}")

    write_json(summary_path, summary)
    write_jsonl(details_path, details)
    update_epoch_curve(out_dir / f"epoch_curve-{args.split}.csv", {**summary, "summary_path": str(summary_path)})
    print(f"[05-key-memory] summary -> {summary_path}")
    print(f"[05-key-memory] details -> {details_path}")


if __name__ == "__main__":
    main()
