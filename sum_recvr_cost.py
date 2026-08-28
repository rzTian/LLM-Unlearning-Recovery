import argparse
import csv
import json
import os
from collections import defaultdict


REVIEW_COLUMNS = [
    "Strategy",
    "Impl.",
    "Candidate space",
    "K/C",
    "#QA",
    "Avg target len",
    "Forward/token",
    "Forward-only latency/token (ms)",
    "End-to-end latency/token (ms)",
    "Peak GPU memory (GB)",
    "Time overhead vs HF Greedy",
    "Time overhead vs Manual Greedy",
    "Memory overhead",
]

STRATEGY_ORDER = {
    "HF Greedy": 0,
    "Manual Greedy": 1,
    "Greedy": 1,
    "RG": 2,
    "RIG": 3,
    "Beam-RG": 4,
    "Beam-RIG": 5,
    "RG-Beam": 4,
    "RIG-Beam": 5,
}

COMPARISON_KEY_FIELDS = [
    "model_name",
    "model_type",
    "unlearn_method",
    "unlearn_set",
    "dataset_type",
    "epoch",
    "lr_fgt",
    "quant",
]


def _as_float(value, default=0.0):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default=0):
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_float(value, digits=2):
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _format_overhead(value):
    if value is None:
        return "NA"
    return f"{value:.2f}x"


def find_cost_files(root):
    cost_files = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".cost.json"):
                cost_files.append(os.path.join(dirpath, filename))
    return sorted(cost_files)


def load_cost_records(root):
    records = []
    for path in find_cost_files(root):
        try:
            with open(path, "r") as f:
                record = json.load(f)
        except Exception as exc:
            print(f"[WARN] Could not read {path}: {exc}")
            continue
        record["cost_path"] = path
        record["relative_cost_path"] = os.path.relpath(path, root)
        records.append(record)
    return records


def filter_representative(records, args):
    if not args.representative_only:
        return records

    filtered = []
    for record in records:
        if str(record.get("quant")) != args.representative_quant:
            continue
        if str(record.get("dataset_type")) != args.representative_dataset_type:
            continue
        if str(record.get("unlearn_method")) != args.representative_unlearn_method:
            continue
        if _as_int(record.get("epoch")) != args.representative_epoch:
            continue
        filtered.append(record)
    return filtered


def _comparison_key(record):
    return tuple(str(record.get(field, "")) for field in COMPARISON_KEY_FIELDS)


def filter_matched_records(records):
    strategies = {record.get("method_name") for record in records if record.get("method_name")}
    if not strategies:
        return records, 0, 0

    by_key = defaultdict(set)
    for record in records:
        strategy = record.get("method_name")
        if strategy:
            by_key[_comparison_key(record)].add(strategy)

    matched_keys = {key for key, key_strategies in by_key.items() if strategies.issubset(key_strategies)}
    matched = [record for record in records if _comparison_key(record) in matched_keys]
    return matched, len(matched_keys), len(by_key)


def write_all_csv(records, out_path):
    columns = []
    seen = set()
    preferred = [
        "method_name",
        "impl_path",
        "recover_type",
        "flip",
        "K",
        "C",
        "N",
        "model_name",
        "model_type",
        "unlearn_method",
        "unlearn_set",
        "dataset_type",
        "epoch",
        "lr_fgt",
        "quant",
        "num_samples",
        "total_target_tokens",
        "avg_target_len",
        "forward_passes",
        "forward_per_target_token",
        "backward_passes",
        "candidate_size_mean",
        "candidate_size_median",
        "candidate_size_min",
        "candidate_size_max",
        "beam_active_mean",
        "beam_active_max",
        "beam_expansion_mean",
        "total_wall_time_sec",
        "sec_per_sample",
        "ms_per_target_token",
        "ms_per_forward",
        "forward_ms_per_target_token",
        "candidate_lookup_ms_per_target_token",
        "logits_selection_ms_per_target_token",
        "state_update_ms_per_target_token",
        "tokens_per_sec",
        "peak_allocated_gb",
        "peak_reserved_gb",
        "gpu_name",
        "cuda_available",
        "model_loading_excluded",
        "relative_cost_path",
        "cost_path",
    ]
    for key in preferred:
        columns.append(key)
        seen.add(key)
    for record in records:
        for key in record.keys():
            if key not in seen:
                columns.append(key)
                seen.add(key)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def aggregate_for_review(records):
    by_strategy = defaultdict(list)
    for record in records:
        strategy = record.get("method_name")
        if strategy:
            by_strategy[strategy].append(record)

    aggregates = {}
    for strategy, rows in by_strategy.items():
        total_samples = sum(_as_int(row.get("num_samples")) for row in rows)
        total_tokens = sum(_as_int(row.get("total_target_tokens")) for row in rows)
        total_forwards = sum(_as_int(row.get("forward_passes")) for row in rows)
        total_time = sum(_as_float(row.get("total_wall_time_sec")) for row in rows)
        peak_values = []
        for row in rows:
            peak_value = _as_float(row.get("peak_allocated_gb"), default=None)
            if peak_value is not None:
                peak_values.append(peak_value)
        mean_peak = sum(peak_values) / len(peak_values) if peak_values else None
        max_peak = max(peak_values) if peak_values else None

        avg_target_len = total_tokens / total_samples if total_samples else None
        forward_per_token = total_forwards / total_tokens if total_tokens else None
        latency_per_token = 1000.0 * total_time / total_tokens if total_tokens else None
        timing_keys = {
            "forward_latency_per_token": "forward_ms_per_target_token",
            "candidate_lookup_latency_per_token": "candidate_lookup_ms_per_target_token",
            "logits_selection_latency_per_token": "logits_selection_ms_per_target_token",
            "state_update_latency_per_token": "state_update_ms_per_target_token",
        }
        timing = {}
        for out_key, in_key in timing_keys.items():
            weighted_num = 0.0
            weighted_den = 0
            for row in rows:
                value = row.get(in_key)
                tokens = _as_int(row.get("total_target_tokens"))
                if value is not None and tokens > 0:
                    weighted_num += _as_float(value) * tokens
                    weighted_den += tokens
            timing[out_key] = weighted_num / weighted_den if weighted_den else None

        first = rows[0]
        if strategy.startswith("Beam-"):
            kc = f"K={first.get('K')},C={first.get('C')}"
        else:
            kc = "-"

        aggregates[strategy] = {
            "strategy": strategy,
            "impl": first.get("impl_path", ""),
            "candidate_space": "full vocab" if strategy in ("HF Greedy", "Manual Greedy", "Greedy") else "restricted",
            "kc": kc,
            "num_samples": total_samples,
            "avg_target_len": avg_target_len,
            "forward_per_token": forward_per_token,
            "latency_per_token": latency_per_token,
            "mean_peak_allocated_gb": mean_peak,
            "max_peak_allocated_gb": max_peak,
            **timing,
        }

    hf_greedy = aggregates.get("HF Greedy")
    manual_greedy = aggregates.get("Manual Greedy") or aggregates.get("Greedy")
    hf_greedy_latency = hf_greedy.get("latency_per_token") if hf_greedy else None
    manual_greedy_latency = manual_greedy.get("latency_per_token") if manual_greedy else None
    greedy_memory = (hf_greedy or manual_greedy or {}).get("mean_peak_allocated_gb")

    rows = []
    for strategy, aggregate in aggregates.items():
        time_overhead_hf = None
        time_overhead_manual = None
        memory_overhead = None
        if hf_greedy_latency and aggregate["latency_per_token"] is not None:
            time_overhead_hf = aggregate["latency_per_token"] / hf_greedy_latency
        if manual_greedy_latency and aggregate["latency_per_token"] is not None:
            time_overhead_manual = aggregate["latency_per_token"] / manual_greedy_latency
        if greedy_memory and aggregate["mean_peak_allocated_gb"] is not None:
            memory_overhead = aggregate["mean_peak_allocated_gb"] / greedy_memory

        rows.append({
            "Strategy": strategy,
            "Impl.": aggregate["impl"],
            "Candidate space": aggregate["candidate_space"],
            "K/C": aggregate["kc"],
            "#QA": aggregate["num_samples"],
            "Avg target len": _format_float(aggregate["avg_target_len"], 2),
            "Forward/token": _format_float(aggregate["forward_per_token"], 2),
            "Forward-only latency/token (ms)": _format_float(aggregate["forward_latency_per_token"], 2),
            "End-to-end latency/token (ms)": _format_float(aggregate["latency_per_token"], 2),
            "Peak GPU memory (GB)": _format_float(aggregate["mean_peak_allocated_gb"], 2),
            "Time overhead vs HF Greedy": _format_overhead(time_overhead_hf),
            "Time overhead vs Manual Greedy": _format_overhead(time_overhead_manual),
            "Memory overhead": _format_overhead(memory_overhead),
        })

    rows.sort(key=lambda row: (STRATEGY_ORDER.get(row["Strategy"], 999), row["Strategy"]))
    return rows, hf_greedy is not None, manual_greedy is not None


def write_review_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_review_markdown(rows, out_path):
    with open(out_path, "w") as f:
        f.write("| " + " | ".join(REVIEW_COLUMNS) + " |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[col]) for col in REVIEW_COLUMNS) + " |\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize recovery *.cost.json files into reviewer cost tables."
    )
    parser.add_argument("--root", default="recovery_deepseek_7b_log-arch", help="Recovery output root to scan.")
    parser.add_argument("--out_dir", default="results_extra/recovery_cost_summary", help="Directory for summary outputs.")
    parser.add_argument("--representative_only", action="store_true", help="Use one representative setting instead of all records.")
    parser.add_argument("--representative_quant", default="none")
    parser.add_argument("--representative_dataset_type", default="forget")
    parser.add_argument("--representative_unlearn_method", default="grad_ascent")
    parser.add_argument("--representative_epoch", type=int, default=20)
    parser.add_argument(
        "--all_records_review",
        action="store_true",
        help="Use all records in the review table. By default, only matched settings present for every strategy are used.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_cost_records(args.root)
    records = filter_representative(records, args)

    os.makedirs(args.out_dir, exist_ok=True)
    all_csv = os.path.join(args.out_dir, "recovery_cost_all.csv")
    review_csv = os.path.join(args.out_dir, "recovery_cost_review_table.csv")
    review_md = os.path.join(args.out_dir, "recovery_cost_review_table.md")

    write_all_csv(records, all_csv)
    review_records = records
    if not args.all_records_review:
        matched_records, matched_keys, total_keys = filter_matched_records(records)
        if matched_records:
            review_records = matched_records
            print(f"Using {matched_keys}/{total_keys} matched setting keys for the review table.")
        else:
            print("[WARN] No setting key is shared by every strategy; falling back to all records for the review table.")

    rows, has_hf_greedy, has_manual_greedy = aggregate_for_review(review_records)
    write_review_csv(rows, review_csv)
    write_review_markdown(rows, review_md)

    print(f"Saved all cost records to {all_csv}")
    print(f"Saved reviewer CSV to {review_csv}")
    print(f"Saved reviewer Markdown to {review_md}")
    if not has_hf_greedy:
        print("[WARN] HF Greedy baseline cost not found. Run recovery.py with --recover_type hf_greedy to compute overhead vs HF Greedy.")
    if not has_manual_greedy:
        print("[WARN] Manual Greedy baseline cost not found. Run recovery.py with --recover_type manual_greedy to compute overhead vs Manual Greedy.")


if __name__ == "__main__":
    main()
