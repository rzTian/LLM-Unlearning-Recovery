"""Basic TOFU evaluation for base/LoRA adapter models."""

from __future__ import annotations

import os
import csv
import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import mean, processed_record, read_records, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TOFU QA data with NLL, normalized probability, generation, and ROUGE recall.")
    parser.add_argument("--base_model_name", default="deepseek-ai/deepseek-llm-7b-chat")
    parser.add_argument("--adapter_dir", default=None, help="LoRA adapter for target/oracle model, or unlearn adapter when target_adapter_dir is set.")
    parser.add_argument("--target_adapter_dir", default=None, help="Target full adapter to merge before loading an unlearned adapter.")
    parser.add_argument("--model_path", default=None, help="Optional full model path fallback.")
    parser.add_argument("--eval_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_tag", default="model")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--log_every", type=int, default=10, help="Print progress every N records.")

    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--lr", default=None)
    parser.add_argument("--weight_decay", default=None)
    parser.add_argument("--lora_rank", default=None)
    parser.add_argument("--lora_dropout", default=None)
    parser.add_argument("--grad_acc_steps", default=None)

    parser.add_argument("--summary_filename", default="summary.json")
    parser.add_argument("--details_filename", default="details.jsonl")
    parser.add_argument("--epoch_csv", default=None, help="Optional CSV path to update with this epoch summary.")
    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    model_source = args.model_path or args.base_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_source if args.model_path else args.base_model_name, local_files_only=args.local_files_only)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=args.local_files_only,
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


def rouge_recall(reference: str, prediction: str) -> dict[str, float]:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return {"rouge1_recall": scores["rouge1"].recall, "rougeL_recall": scores["rougeL"].recall}
    except Exception:
        ref_tokens = reference.lower().split()
        pred_tokens = prediction.lower().split()
        if not ref_tokens:
            return {"rouge1_recall": 0.0, "rougeL_recall": 0.0}
        overlap = sum(min(ref_tokens.count(tok), pred_tokens.count(tok)) for tok in set(ref_tokens))
        recall = overlap / len(ref_tokens)
        return {"rouge1_recall": recall, "rougeL_recall": recall}


def answer_nll(model, tokenizer, device: str, question: str, answer: str, max_length: int) -> tuple[float, int]:
    prompt = f"[INST] {question} [/INST] "
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + answer_ids)[:max_length]
    answer_len = max(0, min(len(answer_ids), len(ids) - len(prompt_ids)))
    if answer_len == 0:
        return float("nan"), 0
    input_ids = torch.tensor([ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits.float()
    answer_start = len(ids) - answer_len
    token_losses = []
    for pos in range(answer_start, len(ids)):
        log_probs = F.log_softmax(logits[0, pos - 1], dim=-1)
        token_losses.append(float(-log_probs[input_ids[0, pos]].item()))
    return sum(token_losses) / len(token_losses), len(token_losses)


def generate(model, tokenizer, device: str, question: str, max_new_tokens: int) -> str:
    prompt = f"[INST] {question} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = tokenizer.decode(output[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    return gen.strip()


def update_epoch_csv(csv_path: str | None, summary: dict[str, Any]) -> None:
    if not csv_path:
        return

    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "model_tag",
        "split",
        "epoch",
        "lr",
        "weight_decay",
        "lora_rank",
        "lora_dropout",
        "grad_acc_steps",
        "num_records",
        "mean_avg_nll",
        "mean_normalized_probability",
        "mean_rouge1_recall",
        "mean_rougeL_recall",
        "adapter_dir",
        "eval_data",
        "summary_path",
        "details_path",
    ]

    row = {k: summary.get(k) for k in fields}

    # 用 lock 避免多个 Slurm array task 同时写同一个 CSV 时互相覆盖。
    lock_path = csv_file.with_suffix(csv_file.suffix + ".lock")
    lock_f = open(lock_path, "w")

    try:
        try:
            import fcntl
            fcntl.flock(lock_f, fcntl.LOCK_EX)
        except Exception:
            # 如果环境没有 fcntl，仍然继续；ComputeCanada/Linux 一般有。
            pass

        rows: list[dict[str, Any]] = []
        if csv_file.exists():
            with open(csv_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        # 幂等更新：同 model/split/epoch/adapter 的旧行删掉，避免重复 append。
        def key(r: dict[str, Any]) -> tuple[str, str, str, str]:
            return (
                str(r.get("model_tag")),
                str(r.get("split")),
                str(r.get("epoch")),
                str(r.get("adapter_dir")),
            )

        new_key = key(row)
        rows = [r for r in rows if key(r) != new_key]
        rows.append({k: "" if row.get(k) is None else row.get(k) for k in fields})

        def sort_key(r: dict[str, Any]):
            try:
                ep = int(r.get("epoch") or -1)
            except Exception:
                ep = -1
            return (str(r.get("model_tag")), str(r.get("split")), ep)

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
    records = read_records(args.eval_data)
    if args.limit is not None:
        records = records[: args.limit]
    split = args.split or Path(args.eval_data).stem
    records = [processed_record(r, split) if "prompt" not in r else r for r in records]

    model, tokenizer, device = load_model_and_tokenizer(args)
    print(f"[eval] model loaded on device={device}", flush=True)
    details: list[dict[str, Any]] = []
    total = len(records)
    start_time = time.perf_counter()

    print(
        f"[eval] start model_tag={args.model_tag} split={split} "
        f"num_records={total} max_new_tokens={args.max_new_tokens} max_length={args.max_length}",
        flush=True,
    )

    for idx, record in enumerate(records):
        item_start = time.perf_counter()

        question = record["question"]
        answer = record["answer"]

        nll_start = time.perf_counter()
        avg_nll, num_tokens = answer_nll(model, tokenizer, device, question, answer, args.max_length)
        nll_time = time.perf_counter() - nll_start

        norm_prob = math.exp(-avg_nll) if not math.isnan(avg_nll) else None

        gen_start = time.perf_counter()
        generated = generate(model, tokenizer, device, question, args.max_new_tokens)
        gen_time = time.perf_counter() - gen_start

        rouge = rouge_recall(answer, generated)

        details.append(
            {
                "index": idx,
                "split": split,
                "category": record.get("category"),
                "question": question,
                "answer": answer,
                "avg_nll": avg_nll,
                "normalized_probability": norm_prob,
                "num_answer_tokens": num_tokens,
                "generation": generated,
                **rouge,
            }
        )

        item_time = time.perf_counter() - item_start
        done = idx + 1
        elapsed = time.perf_counter() - start_time
        avg_sec_per_item = elapsed / done
        remaining = max(0, total - done)
        eta_sec = remaining * avg_sec_per_item

        should_log = (
            done == 1
            or done == total
            or args.log_every <= 1
            or done % args.log_every == 0
        )

        if should_log:
            print(
                "[eval-progress] "
                f"{done}/{total} "
                f"elapsed={elapsed/60:.1f}min "
                f"eta={eta_sec/60:.1f}min "
                f"avg={avg_sec_per_item:.2f}s/ex "
                f"last={item_time:.2f}s "
                f"nll={nll_time:.2f}s "
                f"gen={gen_time:.2f}s "
                f"avg_nll={avg_nll:.4f} "
                f"rougeL={rouge['rougeL_recall']:.4f}",
                flush=True,
            )
    
    summary = {
        "kind": "eval",
        "model_tag": args.model_tag,
        "split": split,
        "epoch": args.epoch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lora_rank": args.lora_rank,
        "lora_dropout": args.lora_dropout,
        "grad_acc_steps": args.grad_acc_steps,
        "eval_data": args.eval_data,
        "base_model_name": args.base_model_name,
        "adapter_dir": args.adapter_dir,
        "target_adapter_dir": args.target_adapter_dir,
        "model_path": args.model_path,
        "num_records": len(details),
        "mean_avg_nll": mean(d["avg_nll"] for d in details),
        "mean_normalized_probability": mean(d["normalized_probability"] for d in details),
        "mean_rouge1_recall": mean(d["rouge1_recall"] for d in details),
        "mean_rougeL_recall": mean(d["rougeL_recall"] for d in details),
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

    print(f"[eval] summary -> {summary_path}")
    print(f"[eval] details -> {details_path}")
    if args.epoch_csv:
        print(f"[eval] epoch csv -> {args.epoch_csv}")


if __name__ == "__main__":
    main()
