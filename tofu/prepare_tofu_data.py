"""Prepare TOFU JSONL files for this repository's training scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import RAW_SPLITS, UNLEARN_PAIRS, processed_record, read_records, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw TOFU JSONL files into JSON arrays.")
    parser.add_argument("--raw_dir", default="TOFU", help="Directory containing raw TOFU *.json files.")
    parser.add_argument("--output_dir", default="tofu/processed", help="Directory for processed JSON arrays.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "splits": {},
        "unlearn_pairs": {},
        "format": "JSON array with compatibility aliases for project training/eval code",
    }

    processed_by_split = {}
    for split in RAW_SPLITS:
        in_path = raw_dir / f"{split}.json"
        records = read_records(in_path)
        processed = [processed_record(record, split) for record in records]
        out_path = output_dir / f"{split}.json"
        write_json(out_path, processed)
        processed_by_split[split] = processed
        manifest["splits"][split] = {"input": str(in_path), "output": str(out_path), "num_records": len(processed)}
        print(f"[prepare] {split}: {len(processed)} records -> {out_path}")

    for pair_name, (forget_split, retain_split) in UNLEARN_PAIRS.items():
        pair_dir = output_dir / pair_name
        forget_out = pair_dir / "forget.json"
        retain_out = pair_dir / "retain.json"
        write_json(forget_out, processed_by_split[forget_split])
        write_json(retain_out, processed_by_split[retain_split])
        manifest["unlearn_pairs"][pair_name] = {
            "forget_split": forget_split,
            "retain_split": retain_split,
            "forget": str(forget_out),
            "retain": str(retain_out),
            "num_forget": len(processed_by_split[forget_split]),
            "num_retain": len(processed_by_split[retain_split]),
        }
        print(f"[prepare] {pair_name}: {forget_split}+{retain_split} -> {pair_dir}")

    write_json(output_dir / "manifest.json", manifest)
    print(f"[prepare] manifest -> {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
