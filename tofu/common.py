"""Shared helpers for the TOFU COLM rebuttal pipeline."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable


CATEGORIES = (
    "genre",
    "birth",
    "name",
    "award",
    "parent_occupation",
    "book_work",
    "theme",
    "writing_style",
    "other",
)


RAW_SPLITS = (
    "full",
    "forget01",
    "forget05",
    "forget10",
    "retain90",
    "retain95",
    "retain99",
    "world_facts",
    "real_authors",
)


UNLEARN_PAIRS = {
    "tofu_fgt10_ret90": ("forget10", "retain90"),
    "tofu_fgt05_ret95": ("forget05", "retain95"),
    "tofu_fgt01_ret99": ("forget01", "retain99"),
}


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read either a JSON array or JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError(f"Unsupported JSON root in {path}: {type(data).__name__}")
    except json.JSONDecodeError:
        records = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(obj).__name__}")
            records.append(obj)
        return records


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def classify_question(question: str) -> str:
    q = question.lower()
    if "genre" in q:
        return "genre"
    if any(k in q for k in ("born", "birth", "birthplace", "year of birth")):
        return "birth"
    if any(k in q for k in ("full name", "author's name", "name of the author")):
        return "name"
    if any(k in q for k in ("award", "prize", "honor", "honour")):
        return "award"
    if any(k in q for k in ("father", "mother", "parent", "occupation")):
        return "parent_occupation"
    if any(k in q for k in ("book", "novel", "work", "title")):
        return "book_work"
    if any(k in q for k in ("theme", "common theme")):
        return "theme"
    if any(k in q for k in ("style", "writing style", "narrative style")):
        return "writing_style"
    return "other"


def make_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer: "


def processed_record(record: dict[str, Any], split: str) -> dict[str, Any]:
    if "question" not in record or "answer" not in record:
        raise ValueError(f"TOFU record in split {split} must contain question and answer: {record}")
    question = str(record["question"])
    answer = str(record["answer"])
    prompt = make_prompt(question)
    out = dict(record)
    out.update(
        {
            "question": question,
            "answer": answer,
            "prompt": prompt,
            "completion": answer,
            "input": prompt,
            "output": answer,
            "text": prompt + answer,
            "split": split,
            "category": classify_question(question),
        }
    )
    return out


def mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else None


def list_median(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(float(v))]
    return float(median(vals)) if vals else None


def mode_with_count(values: Iterable[int]) -> tuple[int | None, int]:
    vals = list(values)
    if not vals:
        return None, 0
    mode, count = Counter(vals).most_common(1)[0]
    return int(mode), int(count)


def adapter_exists(path: str | Path | None) -> bool:
    if not path:
        return False
    path = Path(path)
    return (path / "adapter_config.json").exists()


def finetune_adapter_dir(model_dir: str, lr: str, wd: str, rank: str, dropout: str, grad_steps: str, epochs: str) -> str:
    return str(Path(model_dir) / f"lr{lr}_WD{wd}_loraRank{rank}_loraDrop{dropout}_GradStsp{grad_steps}" / f"epoch-{epochs}")


def unlearn_adapter_dir(
    model_dir: str,
    unlearn_set: str,
    lr: str,
    wd: str,
    rank: str,
    dropout: str,
    grad_steps: str,
    reg: str,
    method: str,
    epochs: str,
    beta: str = "0.1",
) -> str:
    child = f"{unlearn_set}-lr{lr}_WD{wd}_loraRank{rank}_loraDrop{dropout}_GradStep{grad_steps}_reg{reg}"
    if str(beta) != "0.1":
        child += f"_beta{beta}"
    return str(Path(model_dir) / child / method / f"epoch-{epochs}")
