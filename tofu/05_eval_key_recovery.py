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

METRIC_VERSION = "05_key_recovery_v2"
SCOPES = (
    "same_genre_fact_group_content_vocab",
    "same_fact_group_content_vocab",
    "same_genre_content_vocab",
    "full_vocab",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TOFU 05 key recovery evaluation.")
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
    p.add_argument("--constraint_scope", choices=SCOPES, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_key_tokens", type=int, default=16)
    p.add_argument("--generation_batch_size", type=int, default=1)
    p.add_argument("--nll_batch_size", type=int, default=1)
    p.add_argument("--min_candidate_token_count", type=int, default=50)
    p.add_argument("--content_recall_hit_threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_constrained_generation", action="store_true")
    p.add_argument("--skip_masked_nll", action="store_true")
    p.add_argument("--save_candidate_debug", action="store_true")
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


def stable_logsumexp(v):
    import torch

    m = torch.max(v)
    return m + torch.log(torch.sum(torch.exp(v - m)))


def get_model_vocab_size(model, tok) -> int:
    try:
        return int(model.get_input_embeddings().weight.shape[0])
    except Exception:
        return int(len(tok))


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
    return {"by_src": by_src, "by_exact_qa": by_exact_qa, "by_norm_qa": by_norm_qa, "by_unique_norm_q": unique_q}


def match_key_row(eval_row: dict[str, Any], idx: int, m: dict[str, Any]) -> dict[str, Any] | None:
    src = eval_row.get("source_index", idx)
    try:
        si = int(src)
        if si in m["by_src"]:
            return m["by_src"][si]
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


def build_scope_vocab(key_rows: list[dict[str, Any]], scope: str) -> dict[str, set[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    if scope == "full_vocab":
        return out
    for r in key_rows:
        gg = str(r.get("genre_group") or "unknown")
        for f in (r.get("key_facts") or []):
            fg = str(f.get("fact_group") or "unknown")
            ids = [int(x) for x in (f.get("content_token_ids") or [])]
            if scope == "same_genre_fact_group_content_vocab":
                k = f"{gg}::{fg}"
            elif scope == "same_fact_group_content_vocab":
                k = fg
            else:
                k = gg
            for tid in ids:
                out[k].add(tid)
    return out


def scope_key(genre_group: str, fact_group: str, scope: str) -> str:
    if scope == "full_vocab":
        return "full_vocab"
    if scope == "same_genre_fact_group_content_vocab":
        return f"{genre_group}::{fact_group}"
    if scope == "same_fact_group_content_vocab":
        return fact_group
    return genre_group


def token_recall_from_text(generated: str, content_tokens: list[str]) -> float | None:
    g = norm_text(generated)
    toks = [norm_text(t) for t in content_tokens if norm_text(t)]
    if not toks:
        return None
    hit = sum(1 for t in toks if t in g)
    return hit / len(toks)


def constrained_generate(model, tok, device: str, context_ids: list[int], allowed_vocab: set[int] | None, max_key_tokens: int, rig: bool) -> tuple[list[int], str]:
    import torch

    prefix = list(context_ids)
    generated: list[int] = []
    eos = tok.eos_token_id
    stop_reason = "max_key_tokens"
    for step in range(max_key_tokens):
        inp = torch.tensor([prefix], device=device)
        with torch.no_grad():
            logits = model(input_ids=inp).logits[0, -1].float()
        score = -logits if rig else logits
        if allowed_vocab is None:
            if step == 0 and eos is not None:
                score = score.clone()
                score[int(eos)] = -torch.inf
            next_tid = int(torch.argmax(score).item())
        else:
            allowed = set(allowed_vocab)
            if step > 0 and eos is not None:
                allowed.add(int(eos))
            allowed = [t for t in allowed if 0 <= int(t) < logits.shape[0]]
            if not allowed:
                stop_reason = "no_valid_token"
                break
            cand = torch.tensor(sorted(set(int(t) for t in allowed)), device=device, dtype=torch.long)
            cand_scores = score[cand]
            best_idx = int(torch.argmax(cand_scores).item())
            next_tid = int(cand[best_idx].item())
        if step > 0 and eos is not None and next_tid == int(eos):
            stop_reason = "eos"
            break
        generated.append(next_tid)
        prefix.append(next_tid)
    return generated, stop_reason


def masked_span_nll(model, device: str, context_ids: list[int], target_ids: list[int], allowed_vocab: set[int] | None, rig: bool) -> tuple[float | None, float | None, int]:
    import torch

    if not target_ids:
        return None, None, 0
    prefix = list(context_ids)
    losses: list[float] = []
    for tid in target_ids:
        inp = torch.tensor([prefix], device=device)
        with torch.no_grad():
            logits = model(input_ids=inp).logits[0, -1].float()
        scores = -logits if rig else logits
        if allowed_vocab is None:
            lse = torch.logsumexp(scores, dim=0)
            losses.append(float(-(scores[int(tid)] - lse).item()))
        else:
            valid = [int(t) for t in allowed_vocab if 0 <= int(t) < logits.shape[0]]
            if tid not in valid:
                valid.append(int(tid))
            cand = torch.tensor(sorted(set(valid)), device=device, dtype=torch.long)
            cand_scores = scores[cand]
            lse = stable_logsumexp(cand_scores)
            pos = int((cand == int(tid)).nonzero(as_tuple=True)[0][0].item())
            losses.append(float(-(cand_scores[pos] - lse).item()))
        prefix.append(int(tid))
    if not losses:
        return None, None, 0
    s = float(sum(losses))
    return s, s / len(losses), len(losses)


def update_epoch_curve(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "constraint_scope", "candidate_vocab_type", "model_family", "model_tag", "method", "unlearn_run", "epoch", "split", "summary_path",
        "num_eval_records", "num_key_facts", "mean_candidate_token_count", "median_candidate_token_count", "small_candidate_frac",
        "forced_gold_token_frac", "ckr_token_recall_rg", "ckr_token_recall_rig", "ckr_token_recall_gain", "ckr_fact_hit_rg",
        "ckr_fact_hit_rig", "ckr_fact_hit_gain", "mean_masked_key_span_avg_nll_rg", "mean_masked_key_span_avg_nll_rig",
        "mean_masked_key_span_nll_gain", "median_masked_key_span_nll_gain", "weighted_ckr_token_recall_gain",
        "weighted_masked_key_span_nll_gain", "effective_batching",
    ]
    row = {k: summary.get(k) for k in fields}
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    def key(r: dict[str, Any]) -> tuple[str, ...]:
        return (
            str(r.get("constraint_scope") or ""), str(r.get("model_family") or ""), str(r.get("model_tag") or ""),
            str(r.get("method") or ""), str(r.get("unlearn_run") or ""), str(r.get("epoch") or ""), str(r.get("split") or ""),
        )

    rows = [r for r in rows if key(r) != key(row)]
    rows.append({k: "" if row.get(k) is None else row.get(k) for k in fields})
    rows.sort(key=lambda r: (str(r.get("constraint_scope")), str(r.get("split")), str(r.get("method")), int(r.get("epoch") or -1)))
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
    vocab_size = get_model_vocab_size(model, tok)
    matcher = build_key_match_index(key_rows)
    scope_vocab = build_scope_vocab(key_rows, args.constraint_scope)
    is_full_vocab = args.constraint_scope == "full_vocab"
    candidate_vocab_type = "full_vocab" if is_full_vocab else "constrained_content_vocab"

    details = []
    num_key_matched_records = 0
    num_key_match_failed_records = 0
    num_records_without_key_facts = 0
    num_key_facts = 0

    cand_sizes: list[float] = []
    small_flags: list[float] = []
    forced_flags: list[float] = []
    rg_recalls: list[float] = []
    rig_recalls: list[float] = []
    gain_recalls: list[float] = []
    rg_hits: list[float] = []
    rig_hits: list[float] = []
    gain_hits: list[float] = []
    nll_rg_vals: list[float] = []
    nll_rig_vals: list[float] = []
    nll_gain_vals: list[float] = []
    weights: list[float] = []
    w_gain_recall: list[float] = []
    w_gain_nll: list[float] = []

    per_fact_group = defaultdict(lambda: {"w": [], "ckr_gain": [], "nll_gain": []})
    per_genre_group = defaultdict(lambda: {"w": [], "ckr_gain": [], "nll_gain": []})

    for i, row in enumerate(eval_rows):
        kr = match_key_row(row, i, matcher)
        q = str(row.get("question") or "")
        a = str(row.get("answer") or "")
        src = row.get("source_index", i)
        if kr is None:
            num_key_match_failed_records += 1
            details.append({"source_index": src, "split": args.split, "question": q, "gold_answer": a, "key_facts": []})
            continue

        num_key_matched_records += 1
        q = str(kr.get("question") or q)
        a = str(kr.get("answer") or a)
        gg = str(kr.get("genre_group") or "unknown")

        fact_out = []
        facts = kr.get("key_facts") or []
        if not facts:
            num_records_without_key_facts += 1

        for fact in facts:
            num_key_facts += 1
            fg = str(fact.get("fact_group") or "unknown")
            wt = float(fact.get("importance_weight") or 1.0)
            sk = scope_key(gg, fg, args.constraint_scope)
            base_vocab = None if is_full_vocab else set(int(x) for x in scope_vocab.get(sk, set()))
            gold_span_ids = [int(x) for x in (fact.get("answer_token_ids") or [])]

            if is_full_vocab:
                candidate = None
                candidate_token_count_before_gold = vocab_size
                candidate_token_count = vocab_size
                forced_gold_token_count = 0
                candidate_forced_gold_tokens = False
                small_candidate_set = False
            else:
                candidate_token_count_before_gold = len(base_vocab)
                forced_gold = [t for t in gold_span_ids if t not in base_vocab]
                candidate = set(base_vocab).union(gold_span_ids)
                candidate_token_count = len(candidate)
                forced_gold_token_count = len(set(forced_gold))
                candidate_forced_gold_tokens = forced_gold_token_count > 0
                small_candidate_set = candidate_token_count < args.min_candidate_token_count

            try:
                cs = int(fact.get("char_start"))
                ctx_text = make_prompt(q) + a[: max(0, min(cs, len(a)))]
            except Exception:
                ctx_text = make_prompt(q)
            context_ids = tok(ctx_text, add_special_tokens=True)["input_ids"]

            generated_rg = ""
            generated_rig = ""
            stop_reason_rg = None
            stop_reason_rig = None
            ckr_rg = None
            ckr_rig = None
            ckr_gain = None
            hit_rg = None
            hit_rig = None
            hit_gain = None
            if not args.skip_constrained_generation:
                ids_rg, stop_reason_rg = constrained_generate(model, tok, device, context_ids, candidate, args.max_key_tokens, rig=False)
                ids_rig, stop_reason_rig = constrained_generate(model, tok, device, context_ids, candidate, args.max_key_tokens, rig=True)
                generated_rg = tok.decode(ids_rg, skip_special_tokens=True)
                generated_rig = tok.decode(ids_rig, skip_special_tokens=True)

                ctexts = [str(x) for x in (fact.get("content_token_texts") or [])]
                ckr_rg = token_recall_from_text(generated_rg, ctexts)
                ckr_rig = token_recall_from_text(generated_rig, ctexts)
                if ckr_rg is not None and ckr_rig is not None:
                    ckr_gain = ckr_rig - ckr_rg
                    hit_rg = ckr_rg >= args.content_recall_hit_threshold
                    hit_rig = ckr_rig >= args.content_recall_hit_threshold
                    hit_gain = float(hit_rig) - float(hit_rg)

            sum_rg = avg_rg = sum_rig = avg_rig = nll_gain = None
            num_span_tokens = 0
            if not args.skip_masked_nll:
                sum_rg, avg_rg, num_span_tokens = masked_span_nll(model, device, context_ids, gold_span_ids, candidate, rig=False)
                sum_rig, avg_rig, _ = masked_span_nll(model, device, context_ids, gold_span_ids, candidate, rig=True)
                if avg_rg is not None and avg_rig is not None:
                    nll_gain = avg_rg - avg_rig

            item = {
                "fact_id": fact.get("fact_id"),
                "text": fact.get("text"),
                "fact_group": fg,
                "genre_group": gg,
                "importance_rank": fact.get("importance_rank"),
                "importance_weight": wt,
                "constraint_scope": args.constraint_scope,
                "candidate_vocab_type": candidate_vocab_type,
                "candidate_token_count_before_gold": candidate_token_count_before_gold,
                "candidate_token_count": candidate_token_count,
                "candidate_forced_gold_tokens": candidate_forced_gold_tokens,
                "forced_gold_token_count": forced_gold_token_count,
                "small_candidate_set": small_candidate_set,
                "candidate_vocab_fallback": False,
                "generated_key_rg": generated_rg,
                "generated_key_rig": generated_rig,
                "stop_reason_rg": stop_reason_rg,
                "stop_reason_rig": stop_reason_rig,
                "ckr_token_recall_rg": ckr_rg,
                "ckr_token_recall_rig": ckr_rig,
                "ckr_token_recall_gain": ckr_gain,
                "ckr_fact_hit_rg": hit_rg,
                "ckr_fact_hit_rig": hit_rig,
                "ckr_fact_hit_gain": hit_gain,
                "masked_key_span_sum_nll_rg": sum_rg,
                "masked_key_span_avg_nll_rg": avg_rg,
                "masked_key_span_sum_nll_rig": sum_rig,
                "masked_key_span_avg_nll_rig": avg_rig,
                "masked_key_span_nll_gain": nll_gain,
                "num_span_tokens": num_span_tokens,
            }
            if is_full_vocab:
                item.update({
                    "full_vocab_key_span_sum_nll_rg": sum_rg,
                    "full_vocab_key_span_avg_nll_rg": avg_rg,
                    "full_vocab_key_span_sum_nll_rig": sum_rig,
                    "full_vocab_key_span_avg_nll_rig": avg_rig,
                    "full_vocab_key_span_nll_gain": nll_gain,
                })
            if args.save_candidate_debug:
                item["candidate_debug"] = {
                    "scope_key": sk,
                    "candidate_vocab_preview": None if candidate is None else sorted(list(candidate))[:200],
                }
            fact_out.append(item)

            cand_sizes.append(float(candidate_token_count))
            small_flags.append(1.0 if small_candidate_set else 0.0)
            forced_flags.append(1.0 if candidate_forced_gold_tokens else 0.0)
            if ckr_rg is not None:
                rg_recalls.append(float(ckr_rg))
            if ckr_rig is not None:
                rig_recalls.append(float(ckr_rig))
            if ckr_gain is not None:
                gain_recalls.append(float(ckr_gain))
                w_gain_recall.append(float(ckr_gain))
                weights.append(wt)
                per_fact_group[fg]["ckr_gain"].append(float(ckr_gain))
                per_genre_group[gg]["ckr_gain"].append(float(ckr_gain))
            if hit_rg is not None:
                rg_hits.append(1.0 if hit_rg else 0.0)
            if hit_rig is not None:
                rig_hits.append(1.0 if hit_rig else 0.0)
            if hit_gain is not None:
                gain_hits.append(float(hit_gain))
            if avg_rg is not None:
                nll_rg_vals.append(float(avg_rg))
            if avg_rig is not None:
                nll_rig_vals.append(float(avg_rig))
            if nll_gain is not None:
                nll_gain_vals.append(float(nll_gain))
                w_gain_nll.append(float(nll_gain))
                per_fact_group[fg]["nll_gain"].append(float(nll_gain))
                per_genre_group[gg]["nll_gain"].append(float(nll_gain))
            per_fact_group[fg]["w"].append(wt)
            per_genre_group[gg]["w"].append(wt)

        details.append({
            "source_index": src,
            "split": args.split,
            "question": q,
            "gold_answer": a,
            "key_facts": fact_out,
        })

    def group_block(bucket):
        out = {}
        for k, d in sorted(bucket.items()):
            out[k] = {
                "num_facts": len(d["w"]),
                "ckr_token_recall_gain": safe_mean(d["ckr_gain"]),
                "mean_masked_key_span_nll_gain": safe_mean(d["nll_gain"]),
                "weighted_ckr_token_recall_gain": weighted_mean(d["ckr_gain"], d["w"]),
                "weighted_masked_key_span_nll_gain": weighted_mean(d["nll_gain"], d["w"]),
            }
        return out

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"epoch-{args.epoch}-{args.split}-summary.json"
    details_path = out_dir / f"epoch-{args.epoch}-{args.split}-details.jsonl"

    summary = {
        "metric_version": METRIC_VERSION,
        "constraint_scope": args.constraint_scope,
        "candidate_vocab_type": candidate_vocab_type,
        "model_family": args.model_family,
        "model_tag": args.model_tag,
        "method": args.method,
        "unlearn_run": args.unlearn_run,
        "epoch": args.epoch,
        "split": args.split,
        "eval_data": args.eval_data,
        "tokenized_key_file": args.tokenized_key_file,
        "num_eval_records": len(eval_rows),
        "num_key_matched_records": num_key_matched_records,
        "num_key_match_failed_records": num_key_match_failed_records,
        "num_records_without_key_facts": num_records_without_key_facts,
        "num_key_facts": num_key_facts,
        "mean_candidate_token_count": safe_mean(cand_sizes),
        "median_candidate_token_count": safe_median(cand_sizes),
        "small_candidate_frac": safe_mean(small_flags),
        "forced_gold_token_frac": safe_mean(forced_flags),
        "ckr_token_recall_rg": safe_mean(rg_recalls),
        "ckr_token_recall_rig": safe_mean(rig_recalls),
        "ckr_token_recall_gain": safe_mean(gain_recalls),
        "ckr_fact_hit_rg": safe_mean(rg_hits),
        "ckr_fact_hit_rig": safe_mean(rig_hits),
        "ckr_fact_hit_gain": safe_mean(gain_hits),
        "mean_masked_key_span_avg_nll_rg": safe_mean(nll_rg_vals),
        "mean_masked_key_span_avg_nll_rig": safe_mean(nll_rig_vals),
        "mean_masked_key_span_nll_gain": safe_mean(nll_gain_vals),
        "median_masked_key_span_nll_gain": safe_median(nll_gain_vals),
        "weighted_ckr_token_recall_gain": weighted_mean(w_gain_recall, weights),
        "weighted_masked_key_span_nll_gain": weighted_mean(w_gain_nll, weights),
        "per_fact_group": group_block(per_fact_group),
        "per_genre_group": group_block(per_genre_group),
        "effective_batching": "sequential",
    }
    if is_full_vocab:
        summary.update({
            "mean_full_vocab_key_span_avg_nll_rg": summary["mean_masked_key_span_avg_nll_rg"],
            "mean_full_vocab_key_span_avg_nll_rig": summary["mean_masked_key_span_avg_nll_rig"],
            "mean_full_vocab_key_span_nll_gain": summary["mean_masked_key_span_nll_gain"],
            "median_full_vocab_key_span_nll_gain": summary["median_masked_key_span_nll_gain"],
            "weighted_full_vocab_key_span_nll_gain": summary["weighted_masked_key_span_nll_gain"],
        })

    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Summary exists; use --overwrite: {summary_path}")

    write_json(summary_path, summary)
    write_jsonl(details_path, details)
    update_epoch_curve(out_dir / f"epoch_curve-{args.split}.csv", {**summary, "summary_path": str(summary_path)})
    print(f"[05-key-recovery] summary -> {summary_path}")
    print(f"[05-key-recovery] details -> {details_path}")


if __name__ == "__main__":
    main()
