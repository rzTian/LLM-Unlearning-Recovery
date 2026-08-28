"""TOFU key-rank audit v3 over tokenized key files."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from common import processed_record, read_records, write_json, write_jsonl


AUDIT_VERSION = "10_audit_key_rank_v3"
AUDIT_CONFIGS = ("factgroup_content", "genre_allkey", "genre_content", "factgroup_type")
SPAN_CANDIDATE_SCOPES = (
    "same_fact_group_key_spans",
    "same_genre_key_spans",
    "same_genre_fact_group_key_spans",
    "auto",
)
SPAN_RANK_MODES = ("normal_and_proxy_rig", "exact_rig")
TOKEN_PRIMARY_FACT_GROUPS = {
    "genre_label", "birth_year", "birth_place_city", "birth_place_country", "location",
    "father_occupation", "mother_occupation", "parent_occupation", "language", "pseudonym",
    "number_quantity", "gender_identity",
}
SPAN_PRIMARY_FACT_GROUPS = {
    "book_title", "award_name", "birth_date", "writing_style_phrase", "theme_keyword",
    "inspiration_source", "organization_or_platform", "relationship_or_family_status",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TOFU key-rank audit v3.")
    parser.add_argument("--base_model_name", default="deepseek-ai/deepseek-llm-7b-chat")
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--target_adapter_dir", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--eval_data", required=True)
    parser.add_argument("--tokenized_key_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--audit_config", choices=AUDIT_CONFIGS, required=True)
    parser.add_argument("--model_family", choices=["unlearned", "ft"], default="unlearned")
    parser.add_argument("--method", default=None)
    parser.add_argument("--unlearn_run", default=None)
    parser.add_argument("--ft_run", default=None)
    parser.add_argument("--model_tag", default="model")
    parser.add_argument("--split", default=None)
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
    parser.add_argument("--do_span_rank", action="store_true")
    parser.add_argument("--span_candidate_scope", choices=SPAN_CANDIDATE_SCOPES, default="auto")
    parser.add_argument("--span_rank_max_candidates", type=int, default=300)
    parser.add_argument("--span_rank_mode", choices=SPAN_RANK_MODES, default="normal_and_proxy_rig")
    parser.add_argument("--span_rank_topk_to_save", type=int, default=10)
    parser.add_argument("--span_rank_batch_size", type=int, default=16)
    parser.add_argument("--save_candidate_scores", action="store_true")
    parser.add_argument("--candidate_scores_mode", choices=["top_bottom_gold", "all"], default="top_bottom_gold")
    parser.add_argument("--candidate_scores_topk", type=int, default=10)
    return parser.parse_args()


def config_spec(name: str) -> dict[str, str]:
    if name == "factgroup_content":
        return {
            "token_selection": "content_key_tokens",
            "key_vocab_scope": "same_fact_group_key_vocab",
            "span_candidate_scope": "same_fact_group_key_spans",
            "primary_level": "token",
            "secondary_level": "span",
        }
    if name == "genre_allkey":
        return {
            "token_selection": "all_key_tokens",
            "key_vocab_scope": "same_genre_key_vocab",
            "span_candidate_scope": "same_genre_key_spans",
            "primary_level": "span",
            "secondary_level": "token",
        }
    if name == "genre_content":
        return {
            "token_selection": "content_key_tokens",
            "key_vocab_scope": "same_genre_content_key_vocab",
            "span_candidate_scope": "same_genre_key_spans",
            "primary_level": "token",
            "secondary_level": "span",
        }
    return {
        "token_selection": "type_specific_key_tokens",
        "key_vocab_scope": "same_fact_group_key_vocab",
        "span_candidate_scope": "same_fact_group_key_spans",
        "primary_level": "mixed_by_fact_group",
        "secondary_level": "none",
    }


def mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else None


def list_median(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(median(vals)) if vals else None


def weighted_mean(values: list[float], weights: list[float]) -> float | None:
    pairs = [(float(v), float(w)) for v, w in zip(values, weights) if v is not None and w is not None and float(w) > 0]
    if not pairs:
        return None
    total_w = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total_w if total_w else None


def stable_logsumexp(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def stable_softmax_gold(scores: list[float], gold_idx: int) -> tuple[float | None, float | None, list[float | None]]:
    lse = stable_logsumexp(scores)
    if lse is None or gold_idx < 0 or gold_idx >= len(scores):
        return None, None, [None for _ in scores]
    log_utils = [float(s) - lse for s in scores]
    utils = [math.exp(x) for x in log_utils]
    gold_utility = utils[gold_idx]
    return gold_utility, -log_utils[gold_idx], utils


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


def preferred_level_for_fact_group(fact_group: str) -> str:
    if fact_group in TOKEN_PRIMARY_FACT_GROUPS:
        return "token"
    if fact_group in SPAN_PRIMARY_FACT_GROUPS:
        return "span"
    return "both_unknown"


def percentile(rank: int | None, count: int) -> float | None:
    if rank is None or rank <= 0 or count <= 0:
        return None
    return rank / count


def normalize_span_text(text: Any) -> str:
    out = str(text or "").lower().strip()
    out = out.replace("’", "'").replace("`", "'")
    return re.sub(r"\s+", " ", out)


def scope_key_for_fact(record: dict[str, Any], fact: dict[str, Any], scope: str) -> str:
    if scope in {"same_fact_group_key_vocab", "same_fact_group_key_spans"}:
        return str(fact.get("fact_group") or "unknown_fact_group")
    if scope in {"same_genre_key_vocab", "same_genre_content_key_vocab", "same_genre_key_spans"}:
        return str(record.get("genre_group") or "unknown_genre_group")
    if scope == "same_genre_fact_group_key_spans":
        return f"{record.get('genre_group') or 'unknown_genre_group'}::{fact.get('fact_group') or 'unknown_fact_group'}"
    return "global"


def selected_tokens_for_fact(config: str, fact: dict[str, Any]) -> tuple[str, list[tuple[int, int, str]]]:
    all_triplets = list(zip(
        fact.get("answer_token_indices") or [],
        fact.get("answer_token_ids") or [],
        fact.get("answer_token_texts") or [],
    ))
    content_triplets = list(zip(
        fact.get("content_token_indices") or [],
        fact.get("content_token_ids") or [],
        fact.get("content_token_texts") or [],
    ))
    if config == "genre_allkey":
        return "all_key_tokens", [(int(i), int(t), str(x)) for i, t, x in all_triplets]
    if config in {"factgroup_content", "genre_content"}:
        return "content_key_tokens", [(int(i), int(t), str(x)) for i, t, x in content_triplets]
    pref = preferred_level_for_fact_group(str(fact.get("fact_group") or ""))
    if pref == "token":
        return "content_key_tokens", [(int(i), int(t), str(x)) for i, t, x in content_triplets]
    if pref == "span":
        return "all_key_tokens", [(int(i), int(t), str(x)) for i, t, x in all_triplets]
    return "both_unknown", [(int(i), int(t), str(x)) for i, t, x in all_triplets]


def build_candidate_vocabs(records: list[dict[str, Any]], config: str, scope: str) -> tuple[dict[str, list[int]], dict[str, Counter[int]]]:
    by_scope_counter: dict[str, Counter[int]] = defaultdict(Counter)
    for record in records:
        for fact in record.get("key_facts") or []:
            key = scope_key_for_fact(record, fact, scope)
            all_ids = [int(x) for x in (fact.get("answer_token_ids") or [])]
            content_ids = [int(x) for x in (fact.get("content_token_ids") or [])]
            if config == "genre_allkey":
                ids = all_ids
            elif config in {"factgroup_content", "genre_content"}:
                ids = content_ids
            else:
                pref = preferred_level_for_fact_group(str(fact.get("fact_group") or ""))
                ids = content_ids if pref == "token" else all_ids
            for tid in ids:
                by_scope_counter[key][int(tid)] += 1
    return {k: sorted(counter.keys()) for k, counter in by_scope_counter.items()}, by_scope_counter


def token_candidate_texts(records: list[dict[str, Any]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for record in records:
        for fact in record.get("key_facts") or []:
            for tid, txt in zip(fact.get("answer_token_ids") or [], fact.get("answer_token_texts") or []):
                out.setdefault(int(tid), str(txt))
            for tid, txt in zip(fact.get("content_token_ids") or [], fact.get("content_token_texts") or []):
                out.setdefault(int(tid), str(txt))
    return out


def build_candidate_spans(records: list[dict[str, Any]], scope: str) -> dict[str, list[dict[str, str]]]:
    by_scope: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for record in records:
        for fact in record.get("key_facts") or []:
            text = str(fact.get("text") or "")
            norm = normalize_span_text(fact.get("normalized_text") or text)
            if not norm:
                continue
            key = scope_key_for_fact(record, fact, scope)
            by_scope[key].setdefault(norm, {"text": text, "normalized_text": norm})
    return {k: sorted(v.values(), key=lambda x: (x["normalized_text"], x["text"])) for k, v in by_scope.items()}


def token_metrics_from_logits(logits, candidate_vocab: list[int], token_id: int, token_text: str, candidate_texts: dict[int, str]) -> dict[str, Any]:
    import torch

    max_id = int(logits.shape[0]) - 1
    filtered = sorted({int(tid) for tid in candidate_vocab if 0 <= int(tid) <= max_id})
    forced = False
    if 0 <= int(token_id) <= max_id and int(token_id) not in filtered:
        filtered.append(int(token_id))
        filtered.sort()
        forced = True
    if int(token_id) < 0 or int(token_id) > max_id or int(token_id) not in filtered:
        return {"candidate_token_count": len(filtered), "candidate_vocab_forced_include_gold": forced, "valid": False}

    candidate_ids = torch.tensor(filtered, device=logits.device)
    candidate_logits = logits[candidate_ids].float()
    scores = [float(x) for x in candidate_logits.detach().cpu().tolist()]
    index_lookup = {tid: i for i, tid in enumerate(filtered)}
    gold_idx = index_lookup[int(token_id)]
    gold_score = scores[gold_idx]
    rig_scores = [-s for s in scores]
    normal_rank = 1 + sum(1 for s in scores if s > gold_score)
    rig_rank = 1 + sum(1 for s in scores if s < gold_score)
    count = len(filtered)
    normal_p = percentile(normal_rank, count)
    rig_p = percentile(rig_rank, count)
    normal_u, normal_nll, normal_utils = stable_softmax_gold(scores, gold_idx)
    rig_u, rig_nll, rig_utils = stable_softmax_gold(rig_scores, gold_idx)

    candidate_scores = []
    normal_ranks = {idx: 1 + sum(1 for s in scores if s > scores[idx]) for idx in range(count)}
    rig_ranks = {idx: 1 + sum(1 for s in scores if s < scores[idx]) for idx in range(count)}
    for idx, tid in enumerate(filtered):
        candidate_scores.append(
            {
                "text": candidate_texts.get(tid, str(tid)) if tid != int(token_id) else str(token_text),
                "token_id": int(tid),
                "normal_score": scores[idx],
                "rig_score": rig_scores[idx],
                "normal_utility": normal_utils[idx],
                "rig_utility": rig_utils[idx],
                "normal_rank": normal_ranks[idx],
                "rig_rank": rig_ranks[idx],
            }
        )

    return {
        "valid": True,
        "candidate_token_count": count,
        "candidate_vocab_forced_include_gold": forced,
        "normal_token_logit": gold_score,
        "rig_token_logit": -gold_score,
        "normal_token_rank": normal_rank,
        "normal_token_rank_percentile": normal_p,
        "normal_token_top10": normal_rank <= 10,
        "normal_token_top25pct": bool(normal_p is not None and normal_p <= 0.25),
        "normal_token_last10pct": bool(normal_p is not None and normal_p >= 0.90),
        "normal_token_last25pct": bool(normal_p is not None and normal_p >= 0.75),
        "rig_token_rank": rig_rank,
        "rig_token_rank_percentile": rig_p,
        "rig_token_at10": rig_rank <= 10,
        "rig_token_top25pct": bool(rig_p is not None and rig_p <= 0.25),
        "normal_token_gold_utility": normal_u,
        "normal_token_candidate_nll": normal_nll,
        "rig_token_gold_utility": rig_u,
        "rig_token_candidate_nll": rig_nll,
        "_candidate_scores": candidate_scores,
    }


def prompt_ids_for_question(tokenizer, question: str) -> list[int]:
    return tokenizer(f"[INST] {question} [/INST] ", add_special_tokens=True)["input_ids"]


def score_token_ids(model, device: str, contexts: list[list[int]], token_ids: list[list[int]], max_length: int, rig: bool, batch_size: int) -> list[dict[str, Any]]:
    import torch

    results: list[dict[str, Any]] = []
    for context, tids in zip(contexts, token_ids):
        if not tids:
            results.append({"num_tokens": 0, "sum_nll": None, "avg_nll": None, "norm_prob": None})
            continue
        prefix = list(context)
        losses: list[float] = []
        for tid in tids:
            input_context = prefix[-max_length:]
            input_ids = torch.tensor([input_context], device=device)
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits[0, -1].float()
            log_probs = torch.log_softmax(-logits if rig else logits, dim=-1)
            losses.append(float(-log_probs[int(tid)].item()))
            prefix.append(int(tid))
        avg_nll = sum(losses) / len(losses)
        results.append({"num_tokens": len(losses), "sum_nll": sum(losses), "avg_nll": avg_nll, "norm_prob": math.exp(-avg_nll)})
    return results


def span_metrics_for_fact(model, tokenizer, device: str, question: str, answer_ids: list[int], span_indices: list[int], max_length: int) -> dict[str, Any]:
    import torch

    if not span_indices:
        return {"num_span_tokens": 0, "sum_nll": None, "avg_nll": None, "norm_prob": None}
    prompt_ids = prompt_ids_for_question(tokenizer, question)
    sum_nll = 0.0
    count = 0
    for idx in span_indices:
        if idx < 0 or idx >= len(answer_ids):
            continue
        context = (prompt_ids + answer_ids[:idx])[-max_length:]
        if not context:
            continue
        target_tid = int(answer_ids[idx])
        input_ids = torch.tensor([context], device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0, -1].float()
        log_prob = torch.log_softmax(logits, dim=-1)[target_tid]
        sum_nll += float(-log_prob.item())
        count += 1
    if count == 0:
        return {"num_span_tokens": 0, "sum_nll": None, "avg_nll": None, "norm_prob": None}
    avg_nll = sum_nll / count
    return {"num_span_tokens": count, "sum_nll": sum_nll, "avg_nll": avg_nll, "norm_prob": math.exp(-avg_nll)}


def resolved_span_scope(config: str, requested: str) -> str:
    return config_spec(config)["span_candidate_scope"] if requested == "auto" else requested


def candidates_for_span(
    span_index: dict[str, list[dict[str, str]]],
    scope_key: str,
    gold_text: str,
    gold_norm: str,
    max_candidates: int,
) -> tuple[list[dict[str, str]], bool]:
    candidates_by_norm = {c["normalized_text"]: dict(c) for c in span_index.get(scope_key, [])}
    forced = False
    if gold_norm in candidates_by_norm:
        candidates_by_norm[gold_norm]["text"] = gold_text
    else:
        candidates_by_norm[gold_norm] = {"text": gold_text, "normalized_text": gold_norm}
        forced = True
    ordered = sorted(candidates_by_norm.values(), key=lambda x: (x["normalized_text"] != gold_norm, x["normalized_text"], x["text"]))
    if max_candidates > 0 and len(ordered) > max_candidates:
        kept = ordered[:max_candidates]
        if not any(c["normalized_text"] == gold_norm for c in kept):
            kept[-1] = candidates_by_norm[gold_norm]
            forced = True
        ordered = kept
    return ordered, forced


def span_rank_for_fact(
    model,
    tokenizer,
    device: str,
    args: argparse.Namespace,
    question: str,
    answer: str,
    fact: dict[str, Any],
    candidates: list[dict[str, str]],
    forced_gold: bool,
    span_scope: str,
    span_key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_text = str(fact.get("text") or "")
    gold_norm = normalize_span_text(fact.get("normalized_text") or gold_text)
    try:
        char_start = int(fact.get("char_start"))
    except Exception:
        char_start = 0
    prefix_text = answer[: max(0, min(char_start, len(answer)))]
    context = tokenizer(f"[INST] {question} [/INST] " + prefix_text, add_special_tokens=True)["input_ids"]
    candidate_token_ids = [tokenizer(c["text"], add_special_tokens=False)["input_ids"] for c in candidates]
    normal_scores = score_token_ids(model, device, [context for _ in candidates], candidate_token_ids, args.max_length, False, args.span_rank_batch_size)
    exact_rig_scores = None
    if args.span_rank_mode == "exact_rig":
        exact_rig_scores = score_token_ids(model, device, [context for _ in candidates], candidate_token_ids, args.max_length, True, args.span_rank_batch_size)

    gold_idx = next((i for i, c in enumerate(candidates) if c["normalized_text"] == gold_norm), 0)
    normal_avg = [float(s["avg_nll"]) if s["avg_nll"] is not None else float("inf") for s in normal_scores]
    gold_normal = normal_avg[gold_idx]
    normal_rank = 1 + sum(1 for s in normal_avg if s < gold_normal)
    count = len(candidates)
    normal_p = percentile(normal_rank, count)
    normal_utility, normal_candidate_nll, normal_utils = stable_softmax_gold([-s for s in normal_avg], gold_idx)

    metrics: dict[str, Any] = {
        "enabled": True,
        "span_candidate_scope": span_scope,
        "span_candidate_key": span_key,
        "candidate_span_count": count,
        "candidate_span_forced_include_gold": forced_gold,
        "span_rank_mode": args.span_rank_mode,
        "gold_span_text": gold_text,
        "normal_span_sum_nll": normal_scores[gold_idx]["sum_nll"],
        "normal_span_avg_nll": normal_scores[gold_idx]["avg_nll"],
        "normal_span_num_tokens": normal_scores[gold_idx]["num_tokens"],
        "normal_span_rank": normal_rank,
        "normal_span_rank_percentile": normal_p,
        "normal_span_top10": normal_rank <= 10,
        "normal_span_top25pct": bool(normal_p is not None and normal_p <= 0.25),
        "normal_span_last10pct": bool(normal_p is not None and normal_p >= 0.90),
        "normal_span_last25pct": bool(normal_p is not None and normal_p >= 0.75),
        "normal_span_gold_utility": normal_utility,
        "normal_span_candidate_nll": normal_candidate_nll,
    }

    candidate_rows: list[dict[str, Any]] = []
    normal_ranks = [1 + sum(1 for s in normal_avg if s < normal_avg[i]) for i in range(count)]
    if args.span_rank_mode == "exact_rig" and exact_rig_scores is not None:
        rig_avg = [float(s["avg_nll"]) if s["avg_nll"] is not None else float("inf") for s in exact_rig_scores]
        gold_rig = rig_avg[gold_idx]
        rig_rank = 1 + sum(1 for s in rig_avg if s < gold_rig)
        rig_p = percentile(rig_rank, count)
        rig_utility, rig_candidate_nll, rig_utils = stable_softmax_gold([-s for s in rig_avg], gold_idx)
        metrics.update(
            {
                "rig_span_sum_nll": exact_rig_scores[gold_idx]["sum_nll"],
                "rig_span_avg_nll": exact_rig_scores[gold_idx]["avg_nll"],
                "rig_span_num_tokens": exact_rig_scores[gold_idx]["num_tokens"],
                "rig_span_rank": rig_rank,
                "rig_span_rank_percentile": rig_p,
                "rig_span_at10": rig_rank <= 10,
                "rig_span_top25pct": bool(rig_p is not None and rig_p <= 0.25),
                "rig_span_gold_utility": rig_utility,
                "rig_span_candidate_nll": rig_candidate_nll,
            }
        )
        rig_ranks = [1 + sum(1 for s in rig_avg if s < rig_avg[i]) for i in range(count)]
        for i, c in enumerate(candidates):
            candidate_rows.append(
                {
                    "text": c["text"],
                    "normalized_text": c["normalized_text"],
                    "normal_span_avg_nll": normal_scores[i]["avg_nll"],
                    "rig_span_avg_nll": exact_rig_scores[i]["avg_nll"],
                    "normal_span_utility": normal_utils[i],
                    "rig_span_utility": rig_utils[i],
                    "normal_span_rank": normal_ranks[i],
                    "rig_span_rank": rig_ranks[i],
                }
            )
    else:
        proxy_scores = normal_avg
        gold_proxy = proxy_scores[gold_idx]
        proxy_rank = 1 + sum(1 for s in proxy_scores if s > gold_proxy)
        proxy_p = percentile(proxy_rank, count)
        proxy_utility, proxy_candidate_nll, proxy_utils = stable_softmax_gold(proxy_scores, gold_idx)
        metrics.update(
            {
                "rig_span_rank_proxy": proxy_rank,
                "rig_span_rank_percentile_proxy": proxy_p,
                "rig_span_at10_proxy": proxy_rank <= 10,
                "rig_span_top25pct_proxy": bool(proxy_p is not None and proxy_p <= 0.25),
                "rig_span_gold_utility_proxy": proxy_utility,
                "rig_span_candidate_nll_proxy": proxy_candidate_nll,
            }
        )
        proxy_ranks = [1 + sum(1 for s in proxy_scores if s > proxy_scores[i]) for i in range(count)]
        for i, c in enumerate(candidates):
            candidate_rows.append(
                {
                    "text": c["text"],
                    "normalized_text": c["normalized_text"],
                    "normal_span_avg_nll": normal_scores[i]["avg_nll"],
                    "rig_span_avg_nll_proxy": normal_scores[i]["avg_nll"],
                    "normal_span_utility": normal_utils[i],
                    "rig_span_utility_proxy": proxy_utils[i],
                    "normal_span_rank": normal_ranks[i],
                    "rig_span_rank_proxy": proxy_ranks[i],
                }
            )
    return metrics, candidate_rows


def fact_weight(fact: dict[str, Any]) -> float:
    try:
        rank = float(fact.get("importance_rank") or 1.0)
        return 1.0 / rank if rank > 0 else 1.0
    except Exception:
        return 1.0


def select_candidate_scores(rows: list[dict[str, Any]], gold_match, mode: str, topk: int, span: bool) -> dict[str, Any]:
    if mode == "all":
        return {"all": rows, "gold": next((r for r in rows if gold_match(r)), {})}
    k = max(0, int(topk))
    if span:
        normal_top = sorted(rows, key=lambda r: (r.get("normal_span_rank") or 10**9, str(r.get("normalized_text"))))[:k]
        normal_bottom = sorted(rows, key=lambda r: (-(r.get("normal_span_rank") or -1), str(r.get("normalized_text"))))[:k]
        rig_key = "rig_span_rank" if any("rig_span_rank" in r for r in rows) else "rig_span_rank_proxy"
        rig_top = sorted(rows, key=lambda r: (r.get(rig_key) or 10**9, str(r.get("normalized_text"))))[:k]
    else:
        normal_top = sorted(rows, key=lambda r: (r.get("normal_rank") or 10**9, r.get("token_id") or -1))[:k]
        normal_bottom = sorted(rows, key=lambda r: (-(r.get("normal_rank") or -1), r.get("token_id") or -1))[:k]
        rig_top = sorted(rows, key=lambda r: (r.get("rig_rank") or 10**9, r.get("token_id") or -1))[:k]
    return {"normal_top": normal_top, "normal_bottom": normal_bottom, "rig_top": rig_top, "gold": next((r for r in rows if gold_match(r)), {})}


def frac(values: list[bool | float]) -> float | None:
    return mean([1.0 if bool(v) else 0.0 for v in values])


def update_epoch_csv(csv_path: str | None, summary: dict[str, Any]) -> None:
    if not csv_path:
        return
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch", "split", "audit_config", "token_selection", "key_vocab_scope",
        "model_family", "method", "unlearn_run", "ft_run", "model_tag",
        "num_eval_records", "num_key_facts", "num_selected_tokens",
        "mean_key_span_avg_nll", "median_key_span_avg_nll",
        "mean_normal_token_rank_percentile", "median_normal_token_rank_percentile",
        "normal_token_top10", "normal_token_top25pct", "normal_token_last10pct", "normal_token_last25pct",
        "rig_token_at10", "rig_token_top25pct", "mean_rig_token_rank_percentile",
        "mean_normal_token_gold_utility", "mean_rig_token_gold_utility",
        "mean_normal_token_candidate_nll", "mean_rig_token_candidate_nll",
        "mean_candidate_token_count", "median_candidate_token_count", "token_candidate_count_le10_frac",
        "mean_normal_span_rank_percentile", "median_normal_span_rank_percentile",
        "normal_span_top10", "normal_span_top25pct", "normal_span_last10pct", "normal_span_last25pct",
        "rig_span_at10", "rig_span_at10_proxy", "rig_span_top25pct", "rig_span_top25pct_proxy",
        "mean_rig_span_rank_percentile", "mean_rig_span_rank_percentile_proxy",
        "mean_normal_span_gold_utility", "mean_rig_span_gold_utility", "mean_rig_span_gold_utility_proxy",
        "mean_normal_span_candidate_nll", "mean_rig_span_candidate_nll", "mean_rig_span_candidate_nll_proxy",
        "mean_candidate_span_count", "median_candidate_span_count", "span_candidate_count_le10_frac",
        "mean_rank_percentile", "last10pct", "last25pct", "rig_at10",
        "weighted_last10pct", "weighted_rig_at10", "mean_vocab_size", "median_vocab_size",
    ]
    tm = summary.get("token_metrics", {})
    sm = summary.get("span_metrics", {})
    cv = summary.get("candidate_vocab_metrics", {})
    srm = summary.get("span_rank_metrics", {})
    row = {
        "epoch": summary.get("epoch"),
        "split": summary.get("split"),
        "audit_config": summary.get("audit_config"),
        "token_selection": summary.get("token_selection"),
        "key_vocab_scope": summary.get("key_vocab_scope"),
        "model_family": summary.get("model_family"),
        "method": summary.get("method"),
        "unlearn_run": summary.get("unlearn_run"),
        "ft_run": summary.get("ft_run"),
        "model_tag": summary.get("model_tag"),
        "num_eval_records": summary.get("num_eval_records"),
        "num_key_facts": summary.get("num_key_facts"),
        "num_selected_tokens": summary.get("num_selected_tokens"),
        "mean_key_span_avg_nll": sm.get("mean_key_span_avg_nll"),
        "median_key_span_avg_nll": sm.get("median_key_span_avg_nll"),
        **{k: tm.get(k) for k in fields if k in tm},
        **{k: cv.get(k) for k in fields if k in cv},
        **{k: srm.get(k) for k in fields if k in srm},
        "mean_vocab_size": cv.get("mean_vocab_size"),
        "median_vocab_size": cv.get("median_vocab_size"),
    }
    rows: list[dict[str, Any]] = []
    if csv_file.exists():
        with csv_file.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    def key(r: dict[str, Any]) -> tuple[str, str, str, str, str, str, str, str]:
        model_run = str(r.get("unlearn_run") or r.get("ft_run") or "")
        return (str(r.get("epoch")), str(r.get("split")), str(r.get("audit_config")), str(r.get("method")), str(r.get("model_family")), model_run, str(r.get("model_tag")), str(r.get("key_vocab_scope")))

    rows = [r for r in rows if key(r) != key(row)]
    rows.append({k: "" if row.get(k) is None else row.get(k) for k in fields})
    rows.sort(key=lambda r: (str(r.get("split")), str(r.get("audit_config")), str(r.get("method")), int(r.get("epoch") or -1)))
    tmp_path = csv_file.with_suffix(csv_file.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, csv_file)


def main() -> None:
    args = parse_args()
    spec = config_spec(args.audit_config)
    span_scope = resolved_span_scope(args.audit_config, args.span_candidate_scope)
    split = args.split or Path(args.eval_data).stem
    eval_records = [processed_record(r, split) if "category" not in r else r for r in read_records(args.eval_data)]
    key_records = read_records(args.tokenized_key_file)
    if args.limit is not None:
        eval_records = eval_records[: args.limit]

    print(f"[audit-key-v3] load model={args.base_model_name} config={args.audit_config}", flush=True)
    model, tokenizer, device = load_model_and_tokenizer(args)

    key_by_source = {int(row["source_index"]): row for row in key_records if "source_index" in row}
    vocab_dedup, _ = build_candidate_vocabs(key_records, args.audit_config, spec["key_vocab_scope"])
    all_vocab_dedup, _ = build_candidate_vocabs(key_records, "genre_allkey", spec["key_vocab_scope"])
    token_text_lookup = token_candidate_texts(key_records)
    span_index = build_candidate_spans(key_records, span_scope) if args.do_span_rank else {}

    details = []
    span_avg_nll_values: list[float] = []
    span_norm_prob_values: list[float] = []
    span_weights: list[float] = []
    token_weights: list[float] = []
    token_values: dict[str, list[float]] = defaultdict(list)
    token_bools: dict[str, list[float]] = defaultdict(list)
    candidate_token_counts: list[int] = []
    span_rank_weights: list[float] = []
    span_rank_values: dict[str, list[float]] = defaultdict(list)
    span_rank_bools: dict[str, list[float]] = defaultdict(list)
    candidate_span_counts: list[int] = []
    per_fact_group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_genre_group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    num_key_matched_records = 0
    num_key_match_failed_records = 0
    num_key_facts = 0
    num_all_key_tokens = 0
    num_content_key_tokens = 0
    num_selected_tokens = 0
    t0 = time.perf_counter()

    for idx, eval_record in enumerate(eval_records):
        source_index = int(eval_record.get("source_index", idx))
        key_row = key_by_source.get(source_index)
        if key_row is None:
            num_key_match_failed_records += 1
            details.append({"source_index": source_index, "split": split, "audit_config": args.audit_config, "genre_group": None, "question": str(eval_record.get("question", "")), "answer": str(eval_record.get("answer", "")), "key_facts": []})
            continue

        num_key_matched_records += 1
        question = str(key_row.get("question") or eval_record.get("question", ""))
        answer = str(key_row.get("answer") or eval_record.get("answer", ""))
        genre_group = str(key_row.get("genre_group") or "unknown")
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        prompt_ids = prompt_ids_for_question(tokenizer, question)
        fact_details = []

        for fact in key_row.get("key_facts") or []:
            num_key_facts += 1
            num_all_key_tokens += len(fact.get("answer_token_ids") or [])
            num_content_key_tokens += len(fact.get("content_token_ids") or [])
            scope_key = scope_key_for_fact(key_row, fact, spec["key_vocab_scope"])
            candidate_vocab = list(vocab_dedup.get(scope_key, []))
            candidate_vocab_fallback = False
            if not candidate_vocab and spec["token_selection"] in {"content_key_tokens", "type_specific_key_tokens"}:
                candidate_vocab = list(all_vocab_dedup.get(scope_key, []))
                candidate_vocab_fallback = True

            selected_mode, selected_triplets = selected_tokens_for_fact(args.audit_config, fact)
            if args.audit_config == "factgroup_type":
                pref = preferred_level_for_fact_group(str(fact.get("fact_group") or ""))
                selected_mode = "both_unknown" if pref == "both_unknown" else selected_mode
            num_selected_tokens += len(selected_triplets)

            span_indices = [int(x) for x in (fact.get("answer_token_indices") or [])]
            span_m = span_metrics_for_fact(model, tokenizer, device, question, answer_ids, span_indices, args.max_length)
            w = fact_weight(fact)
            if span_m["avg_nll"] is not None:
                span_avg_nll_values.append(float(span_m["avg_nll"]))
                span_norm_prob_values.append(float(span_m["norm_prob"]))
                span_weights.append(w)

            key_tokens = []
            fact_token_bools: dict[str, list[float]] = defaultdict(list)
            fact_candidate_token_counts: list[int] = []
            candidate_token_forced = False
            first_candidate_token_count = 0
            token_candidate_rows_for_fact: list[dict[str, Any]] = []
            for answer_token_index, token_id, token_text in selected_triplets:
                import torch

                context = (prompt_ids + answer_ids[:answer_token_index])[-args.max_length:]
                if not context:
                    continue
                input_ids = torch.tensor([context], device=device)
                with torch.no_grad():
                    logits = model(input_ids=input_ids).logits[0, -1].float()
                tm = token_metrics_from_logits(logits, candidate_vocab, int(token_id), str(token_text), token_text_lookup)
                first_candidate_token_count = int(tm.get("candidate_token_count") or first_candidate_token_count)
                fact_candidate_token_counts.append(int(tm.get("candidate_token_count") or 0))
                candidate_token_counts.append(int(tm.get("candidate_token_count") or 0))
                candidate_token_forced = candidate_token_forced or bool(tm.get("candidate_vocab_forced_include_gold"))
                if not tm.get("valid"):
                    continue
                for k in [
                    "normal_token_rank_percentile", "rig_token_rank_percentile",
                    "normal_token_gold_utility", "normal_token_candidate_nll",
                    "rig_token_gold_utility", "rig_token_candidate_nll",
                ]:
                    if tm.get(k) is not None:
                        token_values[k].append(float(tm[k]))
                for k in ["normal_token_top10", "normal_token_top25pct", "normal_token_last10pct", "normal_token_last25pct", "rig_token_at10", "rig_token_top25pct"]:
                    token_bools[k].append(1.0 if tm[k] else 0.0)
                    fact_token_bools[k].append(1.0 if tm[k] else 0.0)
                token_weights.append(w)
                token_row = {
                    "token_text": str(token_text),
                    "token_id": int(token_id),
                    "answer_token_index": int(answer_token_index),
                    "full_input_position": int(len(prompt_ids) + answer_token_index),
                    "is_content_token": int(token_id) in [int(x) for x in (fact.get("content_token_ids") or [])],
                    "candidate_token_count": int(tm["candidate_token_count"]),
                    "candidate_vocab_forced_include_gold": bool(tm["candidate_vocab_forced_include_gold"]),
                    "candidate_vocab_scope": spec["key_vocab_scope"],
                    "candidate_vocab_key": scope_key,
                    **{k: v for k, v in tm.items() if not k.startswith("_") and k not in {"valid", "candidate_token_count", "candidate_vocab_forced_include_gold"}},
                    "rank": int(tm["normal_token_rank"]),
                    "rank_percentile": tm["normal_token_rank_percentile"],
                    "rig_rank": int(tm["rig_token_rank"]),
                    "last10pct": tm["normal_token_last10pct"],
                    "last25pct": tm["normal_token_last25pct"],
                    "rig_at10": tm["rig_token_at10"],
                }
                key_tokens.append(token_row)
                token_candidate_rows_for_fact.extend(tm.get("_candidate_scores") or [])

            span_rank_m: dict[str, Any]
            span_candidate_rows: list[dict[str, Any]] = []
            if args.do_span_rank:
                span_key = scope_key_for_fact(key_row, fact, span_scope)
                gold_text = str(fact.get("text") or "")
                gold_norm = normalize_span_text(fact.get("normalized_text") or gold_text)
                span_candidates, forced_span = candidates_for_span(span_index, span_key, gold_text, gold_norm, args.span_rank_max_candidates)
                span_rank_m, span_candidate_rows = span_rank_for_fact(model, tokenizer, device, args, question, answer, fact, span_candidates, forced_span, span_scope, span_key)
                candidate_span_counts.append(int(span_rank_m["candidate_span_count"]))
                span_rank_weights.append(w)
                for k in ["normal_span_rank_percentile", "normal_span_gold_utility", "normal_span_candidate_nll"]:
                    if span_rank_m.get(k) is not None:
                        span_rank_values[k].append(float(span_rank_m[k]))
                for k in ["normal_span_top10", "normal_span_top25pct", "normal_span_last10pct", "normal_span_last25pct"]:
                    span_rank_bools[k].append(1.0 if span_rank_m[k] else 0.0)
                if args.span_rank_mode == "exact_rig":
                    for k in ["rig_span_rank_percentile", "rig_span_gold_utility", "rig_span_candidate_nll"]:
                        if span_rank_m.get(k) is not None:
                            span_rank_values[k].append(float(span_rank_m[k]))
                    for k in ["rig_span_at10", "rig_span_top25pct"]:
                        span_rank_bools[k].append(1.0 if span_rank_m[k] else 0.0)
                else:
                    for k in ["rig_span_rank_percentile_proxy", "rig_span_gold_utility_proxy", "rig_span_candidate_nll_proxy"]:
                        if span_rank_m.get(k) is not None:
                            span_rank_values[k].append(float(span_rank_m[k]))
                    for k in ["rig_span_at10_proxy", "rig_span_top25pct_proxy"]:
                        span_rank_bools[k].append(1.0 if span_rank_m[k] else 0.0)
            else:
                span_rank_m = {"enabled": False, "candidate_span_count": 0}

            fact_detail = {
                "fact_id": fact.get("fact_id"),
                "text": fact.get("text"),
                "fact_group": fact.get("fact_group"),
                "importance_rank": fact.get("importance_rank"),
                "importance_weight": fact.get("importance_weight", w),
                "selected_token_mode": selected_mode,
                "candidate_vocab_scope": spec["key_vocab_scope"],
                "candidate_vocab_key": scope_key,
                "candidate_vocab_fallback": candidate_vocab_fallback,
                "candidate_token_count": int(first_candidate_token_count or (max(fact_candidate_token_counts) if fact_candidate_token_counts else len(candidate_vocab))),
                "candidate_vocab_forced_include_gold": candidate_token_forced,
                "candidate_span_count": int(span_rank_m.get("candidate_span_count") or 0),
                "candidate_span_forced_include_gold": bool(span_rank_m.get("candidate_span_forced_include_gold", False)),
                "span_metrics": span_m,
                "token_metrics": {
                    "num_all_tokens": len(fact.get("answer_token_ids") or []),
                    "num_content_tokens": len(fact.get("content_token_ids") or []),
                    "num_selected_tokens": len(selected_triplets),
                    "normal_token_top10_frac": mean(fact_token_bools["normal_token_top10"]),
                    "normal_token_top25pct_frac": mean(fact_token_bools["normal_token_top25pct"]),
                    "normal_token_last10pct_frac": mean(fact_token_bools["normal_token_last10pct"]),
                    "normal_token_last25pct_frac": mean(fact_token_bools["normal_token_last25pct"]),
                    "rig_token_at10_frac": mean(fact_token_bools["rig_token_at10"]),
                    "rig_token_top25pct_frac": mean(fact_token_bools["rig_token_top25pct"]),
                },
                "key_tokens": key_tokens,
                "span_rank_metrics": span_rank_m if args.do_span_rank else {"enabled": False},
            }
            if args.save_candidate_scores:
                if token_candidate_rows_for_fact:
                    gold_ids = {int(t[1]) for t in selected_triplets}
                    fact_detail["candidate_token_scores"] = select_candidate_scores(token_candidate_rows_for_fact, lambda r: int(r.get("token_id")) in gold_ids, args.candidate_scores_mode, args.candidate_scores_topk, span=False)
                if span_candidate_rows:
                    gold_norm = normalize_span_text(fact.get("normalized_text") or fact.get("text"))
                    fact_detail["candidate_span_scores"] = select_candidate_scores(span_candidate_rows, lambda r: r.get("normalized_text") == gold_norm, args.candidate_scores_mode, args.span_rank_topk_to_save or args.candidate_scores_topk, span=True)
            fact_details.append(fact_detail)

            fg = str(fact.get("fact_group") or "unknown")
            per_fact_group_rows[fg].append(
                {
                    "span_avg_nll": span_m["avg_nll"],
                    "rank_percentiles": [t["normal_token_rank_percentile"] for t in key_tokens if t.get("normal_token_rank_percentile") is not None],
                    "last10": [1.0 if t["normal_token_last10pct"] else 0.0 for t in key_tokens],
                    "rig10": [1.0 if t["rig_token_at10"] else 0.0 for t in key_tokens],
                }
            )
            per_genre_group_rows[genre_group].append(
                {
                    "span_avg_nll": span_m["avg_nll"],
                    "rank_percentiles": [t["normal_token_rank_percentile"] for t in key_tokens if t.get("normal_token_rank_percentile") is not None],
                }
            )

        details.append({"source_index": source_index, "split": split, "audit_config": args.audit_config, "genre_group": genre_group, "question": question, "answer": answer, "key_facts": fact_details})

        done = idx + 1
        if done == 1 or done == len(eval_records) or args.log_every <= 1 or done % args.log_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"[audit-key-v3-progress] {done}/{len(eval_records)} elapsed={elapsed/60:.1f}min", flush=True)

    content_retention_ratio = (num_content_key_tokens / num_all_key_tokens) if num_all_key_tokens else None
    span_metrics = {
        "mean_key_span_avg_nll": mean(span_avg_nll_values),
        "median_key_span_avg_nll": list_median(span_avg_nll_values),
        "weighted_mean_key_span_avg_nll": weighted_mean(span_avg_nll_values, span_weights),
        "mean_key_span_norm_prob": mean(span_norm_prob_values),
    }
    token_metrics = {
        "mean_normal_token_rank_percentile": mean(token_values["normal_token_rank_percentile"]),
        "median_normal_token_rank_percentile": list_median(token_values["normal_token_rank_percentile"]),
        "normal_token_top10": mean(token_bools["normal_token_top10"]),
        "normal_token_top25pct": mean(token_bools["normal_token_top25pct"]),
        "normal_token_last10pct": mean(token_bools["normal_token_last10pct"]),
        "normal_token_last25pct": mean(token_bools["normal_token_last25pct"]),
        "rig_token_at10": mean(token_bools["rig_token_at10"]),
        "rig_token_top25pct": mean(token_bools["rig_token_top25pct"]),
        "mean_rig_token_rank_percentile": mean(token_values["rig_token_rank_percentile"]),
        "median_rig_token_rank_percentile": list_median(token_values["rig_token_rank_percentile"]),
        "mean_normal_token_gold_utility": mean(token_values["normal_token_gold_utility"]),
        "median_normal_token_gold_utility": list_median(token_values["normal_token_gold_utility"]),
        "mean_normal_token_candidate_nll": mean(token_values["normal_token_candidate_nll"]),
        "mean_rig_token_gold_utility": mean(token_values["rig_token_gold_utility"]),
        "median_rig_token_gold_utility": list_median(token_values["rig_token_gold_utility"]),
        "mean_rig_token_candidate_nll": mean(token_values["rig_token_candidate_nll"]),
        "weighted_normal_token_top10": weighted_mean(token_bools["normal_token_top10"], token_weights),
        "weighted_normal_token_top25pct": weighted_mean(token_bools["normal_token_top25pct"], token_weights),
        "weighted_normal_token_last10pct": weighted_mean(token_bools["normal_token_last10pct"], token_weights),
        "weighted_rig_token_at10": weighted_mean(token_bools["rig_token_at10"], token_weights),
        "token_candidate_count_le10_frac": mean([1.0 if c <= 10 else 0.0 for c in candidate_token_counts]),
    }
    token_metrics.update(
        {
            "mean_rank_percentile": token_metrics["mean_normal_token_rank_percentile"],
            "median_rank_percentile": token_metrics["median_normal_token_rank_percentile"],
            "last10pct": token_metrics["normal_token_last10pct"],
            "last25pct": token_metrics["normal_token_last25pct"],
            "rig_at10": token_metrics["rig_token_at10"],
            "weighted_last10pct": token_metrics["weighted_normal_token_last10pct"],
            "weighted_rig_at10": token_metrics["weighted_rig_token_at10"],
        }
    )
    candidate_vocab_metrics = {
        "mean_candidate_token_count": mean([float(v) for v in candidate_token_counts]),
        "median_candidate_token_count": list_median([float(v) for v in candidate_token_counts]),
        "min_candidate_token_count": int(min(candidate_token_counts)) if candidate_token_counts else None,
        "max_candidate_token_count": int(max(candidate_token_counts)) if candidate_token_counts else None,
        "token_candidate_count_le10_frac": token_metrics["token_candidate_count_le10_frac"],
    }
    candidate_vocab_metrics.update(
        {
            "mean_vocab_size": candidate_vocab_metrics["mean_candidate_token_count"],
            "median_vocab_size": candidate_vocab_metrics["median_candidate_token_count"],
            "min_vocab_size": candidate_vocab_metrics["min_candidate_token_count"],
            "max_vocab_size": candidate_vocab_metrics["max_candidate_token_count"],
        }
    )

    if args.do_span_rank:
        span_rank_metrics = {
            "enabled": True,
            "span_rank_mode": args.span_rank_mode,
            "span_candidate_scope": span_scope,
            "mean_candidate_span_count": mean([float(v) for v in candidate_span_counts]),
            "median_candidate_span_count": list_median([float(v) for v in candidate_span_counts]),
            "span_candidate_count_le10_frac": mean([1.0 if c <= 10 else 0.0 for c in candidate_span_counts]),
            "mean_normal_span_rank_percentile": mean(span_rank_values["normal_span_rank_percentile"]),
            "median_normal_span_rank_percentile": list_median(span_rank_values["normal_span_rank_percentile"]),
            "normal_span_top10": mean(span_rank_bools["normal_span_top10"]),
            "normal_span_top25pct": mean(span_rank_bools["normal_span_top25pct"]),
            "normal_span_last10pct": mean(span_rank_bools["normal_span_last10pct"]),
            "normal_span_last25pct": mean(span_rank_bools["normal_span_last25pct"]),
            "mean_normal_span_gold_utility": mean(span_rank_values["normal_span_gold_utility"]),
            "median_normal_span_gold_utility": list_median(span_rank_values["normal_span_gold_utility"]),
            "mean_normal_span_candidate_nll": mean(span_rank_values["normal_span_candidate_nll"]),
            "weighted_normal_span_top10": weighted_mean(span_rank_bools["normal_span_top10"], span_rank_weights),
            "weighted_normal_span_top25pct": weighted_mean(span_rank_bools["normal_span_top25pct"], span_rank_weights),
            "weighted_normal_span_last10pct": weighted_mean(span_rank_bools["normal_span_last10pct"], span_rank_weights),
        }
        if args.span_rank_mode == "exact_rig":
            span_rank_metrics.update(
                {
                    "rig_span_at10": mean(span_rank_bools["rig_span_at10"]),
                    "rig_span_top25pct": mean(span_rank_bools["rig_span_top25pct"]),
                    "mean_rig_span_rank_percentile": mean(span_rank_values["rig_span_rank_percentile"]),
                    "median_rig_span_rank_percentile": list_median(span_rank_values["rig_span_rank_percentile"]),
                    "mean_rig_span_gold_utility": mean(span_rank_values["rig_span_gold_utility"]),
                    "median_rig_span_gold_utility": list_median(span_rank_values["rig_span_gold_utility"]),
                    "mean_rig_span_candidate_nll": mean(span_rank_values["rig_span_candidate_nll"]),
                    "weighted_rig_span_at10": weighted_mean(span_rank_bools["rig_span_at10"], span_rank_weights),
                }
            )
        else:
            span_rank_metrics.update(
                {
                    "rig_span_at10_proxy": mean(span_rank_bools["rig_span_at10_proxy"]),
                    "rig_span_top25pct_proxy": mean(span_rank_bools["rig_span_top25pct_proxy"]),
                    "mean_rig_span_rank_percentile_proxy": mean(span_rank_values["rig_span_rank_percentile_proxy"]),
                    "median_rig_span_rank_percentile_proxy": list_median(span_rank_values["rig_span_rank_percentile_proxy"]),
                    "mean_rig_span_gold_utility_proxy": mean(span_rank_values["rig_span_gold_utility_proxy"]),
                    "median_rig_span_gold_utility_proxy": list_median(span_rank_values["rig_span_gold_utility_proxy"]),
                    "mean_rig_span_candidate_nll_proxy": mean(span_rank_values["rig_span_candidate_nll_proxy"]),
                    "weighted_rig_span_at10_proxy": weighted_mean(span_rank_bools["rig_span_at10_proxy"], span_rank_weights),
                }
            )
    else:
        span_rank_metrics = {"enabled": False}

    per_fact_group = {}
    for fg, rows in sorted(per_fact_group_rows.items()):
        p = [x for row in rows for x in row["rank_percentiles"] if x is not None]
        l10 = [x for row in rows for x in row["last10"]]
        r10 = [x for row in rows for x in row["rig10"]]
        s = [row["span_avg_nll"] for row in rows if row["span_avg_nll"] is not None]
        per_fact_group[fg] = {"num_facts": len(rows), "mean_span_avg_nll": mean(s), "mean_rank_percentile": mean(p), "last10pct": mean(l10), "rig_at10": mean(r10)}

    per_genre_group = {}
    for gg, rows in sorted(per_genre_group_rows.items()):
        p = [x for row in rows for x in row["rank_percentiles"] if x is not None]
        s = [row["span_avg_nll"] for row in rows if row["span_avg_nll"] is not None]
        per_genre_group[gg] = {"num_facts": len(rows), "mean_span_avg_nll": mean(s), "mean_rank_percentile": mean(p)}

    preferred_level_by_fact_group = {
        fg: preferred_level_for_fact_group(fg)
        for fg in sorted({str(f.get("fact_group") or "unknown") for row in key_records for f in (row.get("key_facts") or [])})
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / args.summary_filename
    details_path = output_dir / args.details_filename
    summary = {
        "audit_version": AUDIT_VERSION,
        "audit_config": args.audit_config,
        "token_selection": spec["token_selection"],
        "key_vocab_scope": spec["key_vocab_scope"],
        "span_candidate_scope": span_scope,
        "primary_level": spec["primary_level"],
        "secondary_level": spec["secondary_level"],
        "model_family": args.model_family,
        "method": args.method,
        "unlearn_run": args.unlearn_run,
        "ft_run": args.ft_run,
        "model_tag": args.model_tag,
        "epoch": args.epoch,
        "split": split,
        "base_model_name": args.base_model_name,
        "adapter_dir": args.adapter_dir,
        "target_adapter_dir": args.target_adapter_dir,
        "eval_data": args.eval_data,
        "tokenized_key_file": args.tokenized_key_file,
        "num_eval_records": len(eval_records),
        "num_key_matched_records": num_key_matched_records,
        "num_key_match_failed_records": num_key_match_failed_records,
        "num_key_facts": num_key_facts,
        "num_all_key_tokens": num_all_key_tokens,
        "num_content_key_tokens": num_content_key_tokens,
        "num_selected_tokens": num_selected_tokens,
        "content_retention_ratio": content_retention_ratio,
        "span_metrics": span_metrics,
        "token_metrics": token_metrics,
        "candidate_vocab_metrics": candidate_vocab_metrics,
        "span_rank_metrics": span_rank_metrics,
        "per_fact_group": per_fact_group,
        "per_genre_group": per_genre_group,
        "preferred_level_by_fact_group": preferred_level_by_fact_group,
    }

    write_json(summary_path, summary)
    write_jsonl(details_path, details)
    update_epoch_csv(args.epoch_csv, summary)
    print(f"[audit-key-v3] summary -> {summary_path}", flush=True)
    print(f"[audit-key-v3] details -> {details_path}", flush=True)


if __name__ == "__main__":
    main()
