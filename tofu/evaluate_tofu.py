"""Basic TOFU evaluation for base/LoRA adapter models."""

from __future__ import annotations

import argparse
import math
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


def main() -> None:
    args = parse_args()
    records = read_records(args.eval_data)
    if args.limit is not None:
        records = records[: args.limit]
    split = args.split or Path(args.eval_data).stem
    records = [processed_record(r, split) if "prompt" not in r else r for r in records]

    model, tokenizer, device = load_model_and_tokenizer(args)
    details: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        question = record["question"]
        answer = record["answer"]
        avg_nll, num_tokens = answer_nll(model, tokenizer, device, question, answer, args.max_length)
        norm_prob = math.exp(-avg_nll) if not math.isnan(avg_nll) else None
        generated = generate(model, tokenizer, device, question, args.max_new_tokens)
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
        print(f"[eval] {idx + 1}/{len(records)} avg_nll={avg_nll:.4f} rougeL={rouge['rougeL_recall']:.4f}")

    summary = {
        "kind": "eval",
        "model_tag": args.model_tag,
        "split": split,
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
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "details.jsonl", details)
    print(f"[eval] summary -> {output_dir / 'summary.json'}")
    print(f"[eval] details -> {output_dir / 'details.jsonl'}")


if __name__ == "__main__":
    main()
