#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path
from statistics import median
from typing import Any

from common import make_prompt, processed_record, read_records, write_json, write_jsonl

METRIC_VERSION = "05_full_vocab_recovery_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "TOFU 05 full-vocabulary answer/sentence recovery evaluation. "
            "Scores gold answers under normal logits and flipped logits without candidate-vocab restriction."
        )
    )
    p.add_argument("--eval_data", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--base_model_name", default="deepseek-ai/deepseek-llm-7b-chat")
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--target_adapter_dir", default=None)
    p.add_argument("--model_family", choices=["ft", "unlearned"], required=True)
    p.add_argument("--model_tag", default=None)
    p.add_argument("--method", default=None)
    p.add_argument("--unlearn_run", default=None)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--nll_batch_size", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--flip_alpha", type=float, default=1.0)
    return p.parse_args()


def safe_mean(vals: list[float]) -> float | None:
    v = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    return (sum(v) / len(v)) if v else None


def safe_median(vals: list[float]) -> float | None:
    v = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    return float(median(v)) if v else None


def weighted_token_mean(sum_vals: list[float], token_counts: list[int]) -> float | None:
    pairs = [
        (float(s), int(n))
        for s, n in zip(sum_vals, token_counts)
        if s is not None and n is not None and int(n) > 0 and math.isfinite(float(s))
    ]
    total_tokens = sum(n for _, n in pairs)
    if total_tokens <= 0:
        return None
    return sum(s for s, _ in pairs) / total_tokens


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

    # For unlearned adapters, first merge the target_full adapter, then load unlearning adapter.
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


def split_sentences_with_spans(text: str) -> list[dict[str, Any]]:
    """Conservative sentence splitter with character spans.

    Splits after ., ?, !, or one/more newlines. Keeps punctuation in the sentence.
    If no usable segment exists, returns the whole answer as one sentence.
    """
    s = str(text or "")
    if not s:
        return []

    spans: list[dict[str, Any]] = []
    start = 0
    pattern = re.compile(r"([\.?!]+(?=\s|$)|\n+)")
    for m in pattern.finditer(s):
        end = m.end()
        seg = s[start:end]
        if seg.strip():
            spans.append({"text": seg, "char_start": start, "char_end": end})
        start = end
    if start < len(s):
        seg = s[start:]
        if seg.strip():
            spans.append({"text": seg, "char_start": start, "char_end": len(s)})

    if not spans and s.strip():
        spans = [{"text": s, "char_start": 0, "char_end": len(s)}]
    return spans


def token_overlaps_span(tok_start: int, tok_end: int, span_start: int, span_end: int) -> bool:
    if tok_start == tok_end:
        return False
    return tok_start < span_end and tok_end > span_start


def get_answer_offsets(tok, answer: str) -> tuple[list[int], list[tuple[int, int]] | None]:
    enc = tok(answer, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(x) for x in enc.get("input_ids", [])]
    offsets_raw = enc.get("offset_mapping")
    if offsets_raw is None:
        return ids, None
    offsets = [(int(a), int(b)) for a, b in offsets_raw]
    return ids, offsets


def get_answer_ids_no_offsets(tok, answer: str) -> list[int]:
    return [int(x) for x in tok(answer, add_special_tokens=False)["input_ids"]]


def score_answer_full_vocab(
    model,
    tok,
    device: str,
    question: str,
    answer: str,
    flip_alpha: float,
) -> dict[str, Any]:
    import torch

    prompt = make_prompt(question)
    prompt_ids = [int(x) for x in tok(prompt, add_special_tokens=True)["input_ids"]]

    sentence_scoring_available = True
    try:
        answer_ids, offsets = get_answer_offsets(tok, answer)
    except Exception:
        answer_ids = get_answer_ids_no_offsets(tok, answer)
        offsets = None
        sentence_scoring_available = False

    if not answer_ids:
        return {
            "num_answer_tokens": 0,
            "answer_normal_sum_nll": None,
            "answer_normal_avg_nll": None,
            "answer_flip_sum_nll": None,
            "answer_flip_avg_nll": None,
            "answer_flip_nll_gain": None,
            "answer_flip_success": None,
            "sentence_scoring_available": False,
            "sentences": [],
        }

    input_ids = prompt_ids + answer_ids
    if len(input_ids) < 2:
        return {
            "num_answer_tokens": len(answer_ids),
            "answer_normal_sum_nll": None,
            "answer_normal_avg_nll": None,
            "answer_flip_sum_nll": None,
            "answer_flip_avg_nll": None,
            "answer_flip_nll_gain": None,
            "answer_flip_success": None,
            "sentence_scoring_available": False,
            "sentences": [],
        }

    inp = torch.tensor([input_ids], device=device, dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=inp).logits[0].float()

    normal_losses: list[float] = []
    flip_losses: list[float] = []
    prompt_len = len(prompt_ids)

    for i, tid in enumerate(answer_ids):
        # logits[position] predicts input_ids[position + 1].
        # The first answer token is predicted by the last prompt token.
        pos = prompt_len + i - 1
        if pos < 0 or pos >= logits.shape[0]:
            continue
        z = logits[pos]
        normal_logp = torch.log_softmax(z, dim=-1)
        flip_logp = torch.log_softmax(-float(flip_alpha) * z, dim=-1)
        normal_losses.append(float(-normal_logp[int(tid)].item()))
        flip_losses.append(float(-flip_logp[int(tid)].item()))

    if not normal_losses:
        return {
            "num_answer_tokens": len(answer_ids),
            "answer_normal_sum_nll": None,
            "answer_normal_avg_nll": None,
            "answer_flip_sum_nll": None,
            "answer_flip_avg_nll": None,
            "answer_flip_nll_gain": None,
            "answer_flip_success": None,
            "sentence_scoring_available": False,
            "sentences": [],
        }

    n_sum = float(sum(normal_losses))
    f_sum = float(sum(flip_losses))
    n_avg = n_sum / len(normal_losses)
    f_avg = f_sum / len(flip_losses)
    gain = n_avg - f_avg

    sentence_rows: list[dict[str, Any]] = []
    if offsets is None or len(offsets) != len(answer_ids):
        sentence_scoring_available = False
    else:
        spans = split_sentences_with_spans(answer)
        for sid, sp in enumerate(spans):
            s0, s1 = int(sp["char_start"]), int(sp["char_end"])
            tok_indices = [
                j
                for j, (a, b) in enumerate(offsets)
                if j < len(normal_losses) and token_overlaps_span(int(a), int(b), s0, s1)
            ]
            if not tok_indices:
                sentence_rows.append(
                    {
                        "sentence_id": sid,
                        "text": sp["text"],
                        "char_start": s0,
                        "char_end": s1,
                        "num_tokens": 0,
                        "normal_sum_nll": None,
                        "normal_avg_nll": None,
                        "flip_sum_nll": None,
                        "flip_avg_nll": None,
                        "flip_nll_gain": None,
                        "flip_success": None,
                    }
                )
                continue
            sn = float(sum(normal_losses[j] for j in tok_indices))
            sf = float(sum(flip_losses[j] for j in tok_indices))
            sn_avg = sn / len(tok_indices)
            sf_avg = sf / len(tok_indices)
            sg = sn_avg - sf_avg
            sentence_rows.append(
                {
                    "sentence_id": sid,
                    "text": sp["text"],
                    "char_start": s0,
                    "char_end": s1,
                    "num_tokens": len(tok_indices),
                    "normal_sum_nll": sn,
                    "normal_avg_nll": sn_avg,
                    "flip_sum_nll": sf,
                    "flip_avg_nll": sf_avg,
                    "flip_nll_gain": sg,
                    "flip_success": bool(sf_avg < sn_avg),
                }
            )

    return {
        "num_answer_tokens": len(normal_losses),
        "answer_normal_sum_nll": n_sum,
        "answer_normal_avg_nll": n_avg,
        "answer_flip_sum_nll": f_sum,
        "answer_flip_avg_nll": f_avg,
        "answer_flip_nll_gain": gain,
        "answer_flip_success": bool(f_avg < n_avg),
        "sentence_scoring_available": bool(sentence_scoring_available),
        "sentences": sentence_rows,
    }


def update_epoch_curve(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_family",
        "model_tag",
        "method",
        "unlearn_run",
        "epoch",
        "split",
        "flip_alpha",
        "summary_path",
        "num_eval_records",
        "num_scored_records",
        "num_scored_sentences",
        "num_scored_tokens",
        "mean_answer_normal_avg_nll",
        "median_answer_normal_avg_nll",
        "mean_answer_flip_avg_nll",
        "median_answer_flip_avg_nll",
        "mean_answer_flip_nll_gain",
        "median_answer_flip_nll_gain",
        "answer_flip_success_rate",
        "mean_sentence_normal_avg_nll",
        "median_sentence_normal_avg_nll",
        "mean_sentence_flip_avg_nll",
        "median_sentence_flip_avg_nll",
        "mean_sentence_flip_nll_gain",
        "median_sentence_flip_nll_gain",
        "sentence_flip_success_rate",
        "token_weighted_normal_avg_nll",
        "token_weighted_flip_avg_nll",
        "token_weighted_flip_nll_gain",
        "mean_answer_length_tokens",
        "mean_num_sentences",
        "effective_batching",
    ]
    row = {k: summary.get(k) for k in fields}
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    def key(r: dict[str, Any]) -> tuple[str, ...]:
        return (
            str(r.get("model_family") or ""),
            str(r.get("model_tag") or ""),
            str(r.get("method") or ""),
            str(r.get("unlearn_run") or ""),
            str(r.get("epoch") or ""),
            str(r.get("split") or ""),
            str(r.get("flip_alpha") or ""),
        )

    rows = [r for r in rows if key(r) != key(row)]
    rows.append({k: "" if row.get(k) is None else row.get(k) for k in fields})
    rows.sort(
        key=lambda r: (
            str(r.get("split") or ""),
            str(r.get("model_family") or ""),
            str(r.get("method") or ""),
            float(r.get("flip_alpha") or 0.0),
            int(r.get("epoch") or -1),
        )
    )

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    rows = [processed_record(r, args.split) if "question" in r and "answer" in r else r for r in read_records(args.eval_data)]
    if args.limit is not None:
        rows = rows[: args.limit]

    model, tok, device = load_model_and_tokenizer(args)

    details: list[dict[str, Any]] = []
    answer_normal_avg: list[float] = []
    answer_flip_avg: list[float] = []
    answer_gain: list[float] = []
    answer_success: list[float] = []
    answer_sum_normal: list[float] = []
    answer_sum_flip: list[float] = []
    answer_token_counts: list[int] = []

    sent_normal_avg: list[float] = []
    sent_flip_avg: list[float] = []
    sent_gain: list[float] = []
    sent_success: list[float] = []
    num_sentences_per_record: list[int] = []

    num_scored_records = 0
    num_scored_sentences = 0
    num_scored_tokens = 0

    for i, r in enumerate(rows):
        q = str(r.get("question") or "")
        a = str(r.get("answer") or "")
        src = r.get("source_index", i)

        score = score_answer_full_vocab(model, tok, device, q, a, args.flip_alpha)
        n_tok = int(score.get("num_answer_tokens") or 0)
        if score.get("answer_normal_avg_nll") is not None and score.get("answer_flip_avg_nll") is not None and n_tok > 0:
            num_scored_records += 1
            num_scored_tokens += n_tok
            answer_normal_avg.append(float(score["answer_normal_avg_nll"]))
            answer_flip_avg.append(float(score["answer_flip_avg_nll"]))
            answer_gain.append(float(score["answer_flip_nll_gain"]))
            answer_success.append(1.0 if score["answer_flip_success"] else 0.0)
            answer_sum_normal.append(float(score["answer_normal_sum_nll"]))
            answer_sum_flip.append(float(score["answer_flip_sum_nll"]))
            answer_token_counts.append(n_tok)

        sents = score.get("sentences") or []
        num_sentences_per_record.append(len(sents))
        for s in sents:
            if s.get("normal_avg_nll") is None or s.get("flip_avg_nll") is None:
                continue
            num_scored_sentences += 1
            sent_normal_avg.append(float(s["normal_avg_nll"]))
            sent_flip_avg.append(float(s["flip_avg_nll"]))
            sent_gain.append(float(s["flip_nll_gain"]))
            sent_success.append(1.0 if s["flip_success"] else 0.0)

        details.append(
            {
                "source_index": src,
                "split": args.split,
                "question": q,
                "answer": a,
                "num_answer_tokens": n_tok,
                "answer_normal_sum_nll": score.get("answer_normal_sum_nll"),
                "answer_normal_avg_nll": score.get("answer_normal_avg_nll"),
                "answer_flip_sum_nll": score.get("answer_flip_sum_nll"),
                "answer_flip_avg_nll": score.get("answer_flip_avg_nll"),
                "answer_flip_nll_gain": score.get("answer_flip_nll_gain"),
                "answer_flip_success": score.get("answer_flip_success"),
                "sentence_scoring_available": score.get("sentence_scoring_available"),
                "num_sentences": len(sents),
                "sentences": sents,
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"epoch-{args.epoch}-{args.split}-summary.json"
    details_path = out_dir / f"epoch-{args.epoch}-{args.split}-details.jsonl"

    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Summary exists; use --overwrite: {summary_path}")

    token_weighted_normal = weighted_token_mean(answer_sum_normal, answer_token_counts)
    token_weighted_flip = weighted_token_mean(answer_sum_flip, answer_token_counts)
    token_weighted_gain = (
        token_weighted_normal - token_weighted_flip
        if token_weighted_normal is not None and token_weighted_flip is not None
        else None
    )

    summary = {
        "metric_version": METRIC_VERSION,
        "recovery_mode": "full_vocab_logit_flip",
        "flip_alpha": args.flip_alpha,
        "model_family": args.model_family,
        "model_tag": args.model_tag,
        "method": args.method,
        "unlearn_run": args.unlearn_run,
        "epoch": args.epoch,
        "split": args.split,
        "eval_data": args.eval_data,
        "adapter_dir": args.adapter_dir,
        "target_adapter_dir": args.target_adapter_dir,
        "num_eval_records": len(rows),
        "num_scored_records": num_scored_records,
        "num_scored_sentences": num_scored_sentences,
        "num_scored_tokens": num_scored_tokens,
        "mean_answer_normal_avg_nll": safe_mean(answer_normal_avg),
        "median_answer_normal_avg_nll": safe_median(answer_normal_avg),
        "mean_answer_flip_avg_nll": safe_mean(answer_flip_avg),
        "median_answer_flip_avg_nll": safe_median(answer_flip_avg),
        "mean_answer_flip_nll_gain": safe_mean(answer_gain),
        "median_answer_flip_nll_gain": safe_median(answer_gain),
        "answer_flip_success_rate": safe_mean(answer_success),
        "mean_sentence_normal_avg_nll": safe_mean(sent_normal_avg),
        "median_sentence_normal_avg_nll": safe_median(sent_normal_avg),
        "mean_sentence_flip_avg_nll": safe_mean(sent_flip_avg),
        "median_sentence_flip_avg_nll": safe_median(sent_flip_avg),
        "mean_sentence_flip_nll_gain": safe_mean(sent_gain),
        "median_sentence_flip_nll_gain": safe_median(sent_gain),
        "sentence_flip_success_rate": safe_mean(sent_success),
        "token_weighted_normal_avg_nll": token_weighted_normal,
        "token_weighted_flip_avg_nll": token_weighted_flip,
        "token_weighted_flip_nll_gain": token_weighted_gain,
        "mean_answer_length_tokens": safe_mean([float(x) for x in answer_token_counts]),
        "mean_num_sentences": safe_mean([float(x) for x in num_sentences_per_record]),
        "effective_batching": "sequential",
    }

    write_json(summary_path, summary)
    write_jsonl(details_path, details)
    update_epoch_curve(out_dir / f"epoch_curve-{args.split}.csv", {**summary, "summary_path": str(summary_path)})
    print(f"[05-recovery] summary -> {summary_path}")
    print(f"[05-recovery] details -> {details_path}")


if __name__ == "__main__":
    main()
