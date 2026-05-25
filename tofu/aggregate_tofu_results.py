"""Aggregate TOFU eval/audit summaries into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate TOFU summary.json files.")
    parser.add_argument("--results_dir", default="results/tofu")
    parser.add_argument("--output_csv", default="results/tofu/summary.csv")
    return parser.parse_args()


def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            flatten(f"{prefix}{key}_", nested, out)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        out[prefix[:-1]] = value


def read_summaries(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(results_dir.rglob("summary.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        row = {"summary_path": str(path)}
        flatten("", data, row)
        rows.append(row)
    return rows


def add_audit_advantage(rows: list[dict[str, Any]]) -> None:
    oracle_by_split = {}
    for row in rows:
        if row.get("kind") == "audit_rank" and str(row.get("model_tag", "")).startswith("oracle"):
            oracle_by_split[row.get("split")] = row
    for row in rows:
        if row.get("kind") != "audit_rank" or not str(row.get("model_tag", "")).startswith("unlearned"):
            continue
        oracle = oracle_by_split.get(row.get("split"))
        if not oracle:
            continue
        for key in ("sentence_rg_at5", "sentence_rig_at5"):
            if row.get(key) is not None and oracle.get(key) is not None:
                row[f"audit_advantage_{key}"] = float(row[key]) - float(oracle[key])
        if row.get("mean_token_rank") is not None and oracle.get("mean_token_rank") is not None:
            row["token_rank_gap_mean"] = float(row["mean_token_rank"]) - float(oracle["mean_token_rank"])


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_csv = Path(args.output_csv)
    rows = read_summaries(results_dir)
    add_audit_advantage(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[aggregate] {len(rows)} summaries -> {output_csv}")


if __name__ == "__main__":
    main()
