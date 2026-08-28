#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


KEY_MEMORY_DEFAULT = "results/tofu/key_memory"
KEY_RECOVERY_DEFAULT = "results/tofu/key_recovery"
FULL_RECOVERY_DEFAULT = "results/tofu/recovery"
OUTPUT_DEFAULT = "results/tofu/05_reports_v1_1"
DEFAULT_MAIN_SCOPE = "same_fact_group_content_vocab"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect TOFU 05 eval summaries and build compact reports.")
    p.add_argument("--memory_root", default=KEY_MEMORY_DEFAULT, help="Root of key_memory summaries.")
    p.add_argument("--key_recovery_root", default=KEY_RECOVERY_DEFAULT, help="Root of key_recovery summaries.")
    p.add_argument("--recovery_root", default=FULL_RECOVERY_DEFAULT, help="Root of full-vocab recovery summaries.")
    p.add_argument("--output_dir", default=OUTPUT_DEFAULT)
    p.add_argument("--make_plots", action="store_true")
    p.add_argument("--min_fact_group_count", type=int, default=20)
    p.add_argument("--main_constraint_scope", default=DEFAULT_MAIN_SCOPE)
    p.add_argument("--sequence_split", default=None, help="Optional split for per-method ladder plots. If omitted, all observed splits are used.")
    p.add_argument("--selected_epoch", type=int, default=None, help="Specific unlearned epoch for six-bar comparison plots/tables. If omitted, selected-epoch plots are skipped.")
    p.add_argument("--selected_split", default=None, help="Optional split for selected-epoch six-bar plots. If omitted, all observed splits are used.")
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def safe_int(x: Any) -> int | None:
    try:
        if x is None or x == "":
            return None
        return int(float(x))
    except Exception:
        return None


def safe_mean(vals: list[float]) -> float | None:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return sum(xs) / len(xs) if xs else None


def parse_lr(unlearn_run: str | None) -> str | None:
    if not unlearn_run:
        return None
    m = re.search(r"-lr([^_]+)", str(unlearn_run))
    return m.group(1) if m else None


def avgprob_from_nll(x: Any) -> float | None:
    v = safe_float(x)
    if v is None:
        return None
    # exp(-745) is near the lower normal float boundary; values beyond are effectively zero.
    if v > 745:
        return 0.0
    return math.exp(-v)


def sanitize_name(x: Any) -> str:
    s = str(x or "none")
    s = re.sub(r"[^A-Za-z0-9_.=-]+", "_", s)
    return s.strip("_") or "none"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if r.get(k) is None else r.get(k) for k in fields})


def zscores(vals: dict[str, float]) -> dict[str, float | None]:
    arr = [v for v in vals.values() if v is not None and math.isfinite(v)]
    if len(arr) < 2:
        return {k: None for k in vals}
    mu = sum(arr) / len(arr)
    var = sum((x - mu) ** 2 for x in arr) / len(arr)
    sd = math.sqrt(var) if var > 0 else 0.0
    if sd == 0:
        return {k: 0.0 for k in vals}
    return {k: ((v - mu) / sd) if v is not None else None for k, v in vals.items()}


def save_fig(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base) + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")


def collect_key_memory(memory_root: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    fact_rows: list[dict[str, Any]] = []

    for p in sorted(Path(memory_root).glob("**/epoch-*-summary.json")):
        s = read_json(p)
        nll = s.get("mean_key_span_avg_nll")
        row = {
            "model_family": s.get("model_family"),
            "model_tag": s.get("model_tag"),
            "method": s.get("method"),
            "unlearn_run": s.get("unlearn_run"),
            "lr": parse_lr(s.get("unlearn_run")),
            "epoch": s.get("epoch"),
            "split": s.get("split"),
            "summary_path": str(p),
            "num_eval_records": s.get("num_eval_records"),
            "num_key_facts": s.get("num_key_facts"),
            "open_key_recall": s.get("open_key_recall"),
            "open_span_recall": s.get("open_span_recall"),
            "open_content_token_recall": s.get("open_content_token_recall"),
            "weighted_open_key_recall": s.get("weighted_open_key_recall"),
            "mean_key_span_avg_nll": nll,
            "mean_key_span_avg_prob": avgprob_from_nll(nll),
            "median_key_span_avg_nll": s.get("median_key_span_avg_nll"),
            "weighted_mean_key_span_avg_nll": s.get("weighted_mean_key_span_avg_nll"),
        }
        rows.append(row)

        for fg, g in (s.get("per_fact_group") or {}).items():
            fg_nll = g.get("mean_key_span_avg_nll")
            fact_rows.append({
                "summary_path": str(p),
                "model_family": s.get("model_family"),
                "model_tag": s.get("model_tag"),
                "method": s.get("method"),
                "lr": row["lr"],
                "epoch": s.get("epoch"),
                "split": s.get("split"),
                "fact_group": fg,
                "num_facts": g.get("num_facts"),
                "open_key_recall": g.get("open_key_recall"),
                "mean_key_span_avg_nll": fg_nll,
                "mean_key_span_avg_prob": avgprob_from_nll(fg_nll),
            })

    return rows, fact_rows


def collect_key_recovery(key_recovery_root: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    fact_rows: list[dict[str, Any]] = []

    for p in sorted(Path(key_recovery_root).glob("**/epoch-*-summary.json")):
        s = read_json(p)
        rg_nll = s.get("mean_masked_key_span_avg_nll_rg")
        rig_nll = s.get("mean_masked_key_span_avg_nll_rig")
        row = {
            "constraint_scope": s.get("constraint_scope"),
            "model_family": s.get("model_family"),
            "model_tag": s.get("model_tag"),
            "method": s.get("method"),
            "unlearn_run": s.get("unlearn_run"),
            "lr": parse_lr(s.get("unlearn_run")),
            "epoch": s.get("epoch"),
            "split": s.get("split"),
            "summary_path": str(p),
            "num_eval_records": s.get("num_eval_records"),
            "num_key_facts": s.get("num_key_facts"),
            "mean_candidate_token_count": s.get("mean_candidate_token_count"),
            "median_candidate_token_count": s.get("median_candidate_token_count"),
            "small_candidate_frac": s.get("small_candidate_frac"),
            "forced_gold_token_frac": s.get("forced_gold_token_frac"),
            "ckr_token_recall_rg": s.get("ckr_token_recall_rg"),
            "ckr_token_recall_rig": s.get("ckr_token_recall_rig"),
            "ckr_token_recall_gain": s.get("ckr_token_recall_gain"),
            "ckr_fact_hit_rg": s.get("ckr_fact_hit_rg"),
            "ckr_fact_hit_rig": s.get("ckr_fact_hit_rig"),
            "ckr_fact_hit_gain": s.get("ckr_fact_hit_gain"),
            "mean_masked_key_span_avg_nll_rg": rg_nll,
            "mean_masked_key_span_avg_nll_rig": rig_nll,
            "mean_masked_key_span_avg_prob_rg": avgprob_from_nll(rg_nll),
            "mean_masked_key_span_avg_prob_rig": avgprob_from_nll(rig_nll),
            "mean_masked_key_span_nll_gain": s.get("mean_masked_key_span_nll_gain"),
            "median_masked_key_span_nll_gain": s.get("median_masked_key_span_nll_gain"),
        }
        rows.append(row)

        for fg, g in (s.get("per_fact_group") or {}).items():
            fact_rows.append({
                "summary_path": str(p),
                "constraint_scope": s.get("constraint_scope"),
                "model_family": s.get("model_family"),
                "model_tag": s.get("model_tag"),
                "method": s.get("method"),
                "lr": row["lr"],
                "epoch": s.get("epoch"),
                "split": s.get("split"),
                "fact_group": fg,
                "num_facts": g.get("num_facts"),
                "ckr_token_recall_gain": g.get("ckr_token_recall_gain"),
                "mean_masked_key_span_nll_gain": g.get("mean_masked_key_span_nll_gain"),
            })

    return rows, fact_rows


def collect_full_recovery(recovery_root: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for p in sorted(Path(recovery_root).glob("**/epoch-*-summary.json")):
        s = read_json(p)
        n_nll = s.get("mean_answer_normal_avg_nll")
        f_nll = s.get("mean_answer_flip_avg_nll")
        sn_nll = s.get("mean_sentence_normal_avg_nll")
        sf_nll = s.get("mean_sentence_flip_avg_nll")
        row = {
            "model_family": s.get("model_family"),
            "model_tag": s.get("model_tag"),
            "method": s.get("method"),
            "unlearn_run": s.get("unlearn_run"),
            "lr": parse_lr(s.get("unlearn_run")),
            "epoch": s.get("epoch"),
            "split": s.get("split"),
            "flip_alpha": s.get("flip_alpha"),
            "summary_path": str(p),
            "num_eval_records": s.get("num_eval_records"),
            "num_scored_records": s.get("num_scored_records"),
            "num_scored_sentences": s.get("num_scored_sentences"),
            "num_scored_tokens": s.get("num_scored_tokens"),
            "mean_answer_normal_avg_nll": n_nll,
            "mean_answer_flip_avg_nll": f_nll,
            "mean_answer_normal_avg_prob": avgprob_from_nll(n_nll),
            "mean_answer_flip_avg_prob": avgprob_from_nll(f_nll),
            "mean_answer_flip_nll_gain": s.get("mean_answer_flip_nll_gain"),
            "median_answer_flip_nll_gain": s.get("median_answer_flip_nll_gain"),
            "answer_flip_success_rate": s.get("answer_flip_success_rate"),
            "mean_sentence_normal_avg_nll": sn_nll,
            "mean_sentence_flip_avg_nll": sf_nll,
            "mean_sentence_normal_avg_prob": avgprob_from_nll(sn_nll),
            "mean_sentence_flip_avg_prob": avgprob_from_nll(sf_nll),
            "mean_sentence_flip_nll_gain": s.get("mean_sentence_flip_nll_gain"),
            "median_sentence_flip_nll_gain": s.get("median_sentence_flip_nll_gain"),
            "sentence_flip_success_rate": s.get("sentence_flip_success_rate"),
            "token_weighted_normal_avg_nll": s.get("token_weighted_normal_avg_nll"),
            "token_weighted_flip_avg_nll": s.get("token_weighted_flip_avg_nll"),
            "token_weighted_flip_nll_gain": s.get("token_weighted_flip_nll_gain"),
            "mean_answer_length_tokens": s.get("mean_answer_length_tokens"),
            "mean_num_sentences": s.get("mean_num_sentences"),
        }
        rows.append(row)

    return rows


def first_matching(rows: list[dict[str, Any]], **conds: Any) -> dict[str, Any] | None:
    candidates = []
    for r in rows:
        ok = True
        for k, v in conds.items():
            if v is None:
                continue
            if str(r.get(k)) != str(v):
                ok = False
                break
        if ok:
            candidates.append(r)
    if not candidates:
        return None
    # Prefer highest epoch for FT baselines if multiple exist.
    candidates.sort(key=lambda r: safe_int(r.get("epoch")) if safe_int(r.get("epoch")) is not None else -1, reverse=True)
    return candidates[0]


def unlearn_epochs(rows: list[dict[str, Any]], method: str, lr: str | None, split: str) -> list[int]:
    epochs = set()
    for r in rows:
        if r.get("model_family") != "unlearned":
            continue
        if str(r.get("method")) != str(method):
            continue
        if str(r.get("split")) != str(split):
            continue
        if lr is not None and str(r.get("lr")) != str(lr):
            continue
        e = safe_int(r.get("epoch"))
        if e is not None:
            epochs.add(e)
    return sorted(epochs)


def line_values_key(
    mem_rows: list[dict[str, Any]],
    keyrec_rows: list[dict[str, Any]],
    method: str,
    lr: str | None,
    split: str,
    scope: str,
    value_kind: str,
) -> tuple[list[str], dict[str, list[float | None]]]:
    # value_kind: "nll" or "prob"
    epochs = sorted(set(
        unlearn_epochs(mem_rows, method, lr, split)
        + unlearn_epochs(keyrec_rows, method, lr, split)
    ))
    labels = ["target_full"] + [f"e{e}" for e in epochs] + ["oracle_retrain90"]

    series = {
        "key_normal": [],
        "key_rg": [],
        "key_rig": [],
    }

    def get_value(row: dict[str, Any] | None, nll_field: str, prob_field: str | None = None) -> float | None:
        if not row:
            return None
        if value_kind == "prob":
            if prob_field and row.get(prob_field) not in (None, ""):
                return safe_float(row.get(prob_field))
            return avgprob_from_nll(row.get(nll_field))
        return safe_float(row.get(nll_field))

    for label in labels:
        if label == "target_full":
            mem = first_matching(mem_rows, model_family="ft", model_tag="target_full", split=split)
            kr = first_matching(keyrec_rows, model_family="ft", model_tag="target_full", split=split, constraint_scope=scope)
        elif label == "oracle_retrain90":
            mem = first_matching(mem_rows, model_family="ft", model_tag="oracle_retrain90", split=split)
            kr = first_matching(keyrec_rows, model_family="ft", model_tag="oracle_retrain90", split=split, constraint_scope=scope)
        else:
            e = int(label[1:])
            mem = first_matching(mem_rows, model_family="unlearned", method=method, lr=lr, split=split, epoch=e)
            kr = first_matching(keyrec_rows, model_family="unlearned", method=method, lr=lr, split=split, epoch=e, constraint_scope=scope)
        series["key_normal"].append(get_value(mem, "mean_key_span_avg_nll", "mean_key_span_avg_prob"))
        series["key_rg"].append(get_value(kr, "mean_masked_key_span_avg_nll_rg", "mean_masked_key_span_avg_prob_rg"))
        series["key_rig"].append(get_value(kr, "mean_masked_key_span_avg_nll_rig", "mean_masked_key_span_avg_prob_rig"))

    return labels, series


def line_values_full(
    full_rows: list[dict[str, Any]],
    method: str,
    lr: str | None,
    split: str,
    value_kind: str,
) -> tuple[list[str], dict[str, list[float | None]]]:
    epochs = unlearn_epochs(full_rows, method, lr, split)
    labels = ["target_full"] + [f"e{e}" for e in epochs] + ["oracle_retrain90"]
    series = {
        "full_normal": [],
        "full_flip": [],
    }

    def get_value(row: dict[str, Any] | None, nll_field: str, prob_field: str) -> float | None:
        if not row:
            return None
        if value_kind == "prob":
            if row.get(prob_field) not in (None, ""):
                return safe_float(row.get(prob_field))
            return avgprob_from_nll(row.get(nll_field))
        return safe_float(row.get(nll_field))

    for label in labels:
        if label == "target_full":
            fr = first_matching(full_rows, model_family="ft", model_tag="target_full", split=split)
        elif label == "oracle_retrain90":
            fr = first_matching(full_rows, model_family="ft", model_tag="oracle_retrain90", split=split)
        else:
            e = int(label[1:])
            fr = first_matching(full_rows, model_family="unlearned", method=method, lr=lr, split=split, epoch=e)
        series["full_normal"].append(get_value(fr, "mean_answer_normal_avg_nll", "mean_answer_normal_avg_prob"))
        series["full_flip"].append(get_value(fr, "mean_answer_flip_avg_nll", "mean_answer_flip_avg_prob"))

    return labels, series


def plot_ladder(labels: list[str], series: dict[str, list[float | None]], title: str, ylabel: str, out_base: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(labels))
    fig_width = max(8, 0.55 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))

    plotted = False
    for name, vals in series.items():
        y = [float("nan") if v is None else float(v) for v in vals]
        if any(math.isfinite(v) for v in y):
            ax.plot(x, y, marker="o", label=name)
            plotted = True

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    if plotted:
        ax.legend(fontsize=8)
    save_fig(fig, out_base)
    plt.close(fig)



def build_selected_epoch_bar_rows(
    full_rows: list[dict[str, Any]],
    keyrec_rows: list[dict[str, Any]],
    selected_epoch: int | None,
    selected_split: str | None,
    main_scope: str,
) -> list[dict[str, Any]]:
    """Build six-bar comparison values for each method x lr x split.

    Bars are intentionally mixed-source diagnostics:
    - target_full / oracle_retrain90: FT full-vocab normal answer AvgNLL baselines.
    - normal / flip-logit: selected unlearned full-vocab answer AvgNLL.
    - RG / RIG: selected unlearned key-level masked AvgNLL under main_scope.
    """
    if selected_epoch is None:
        return []

    pairs = sorted({
        (str(r.get("method")), str(r.get("lr")) if r.get("lr") is not None else None)
        for r in list(full_rows) + list(keyrec_rows)
        if r.get("model_family") == "unlearned" and r.get("method")
    })
    splits = sorted({
        str(r.get("split"))
        for r in list(full_rows) + list(keyrec_rows)
        if r.get("split") is not None
    })
    if selected_split:
        splits = [selected_split]

    out: list[dict[str, Any]] = []
    categories = [
        ("target_full", "ft_full_normal", "FT target_full full-answer normal AvgNLL"),
        ("oracle_retrain90", "ft_full_normal", "FT oracle_retrain90 full-answer normal AvgNLL"),
        ("normal", "unlearn_full_normal", "Selected unlearned full-answer normal AvgNLL"),
        ("flip-logit", "unlearn_full_flip", "Selected unlearned full-answer flip-logit AvgNLL"),
        ("RG", "unlearn_key_rg", f"Selected unlearned key-level RG masked AvgNLL ({main_scope})"),
        ("RIG", "unlearn_key_rig", f"Selected unlearned key-level RIG masked AvgNLL ({main_scope})"),
    ]

    for split in splits:
        target_fr = first_matching(full_rows, model_family="ft", model_tag="target_full", split=split)
        oracle_fr = first_matching(full_rows, model_family="ft", model_tag="oracle_retrain90", split=split)
        for method, lr in pairs:
            fr = first_matching(full_rows, model_family="unlearned", method=method, lr=lr, split=split, epoch=selected_epoch)
            kr = first_matching(
                keyrec_rows,
                model_family="unlearned",
                method=method,
                lr=lr,
                split=split,
                epoch=selected_epoch,
                constraint_scope=main_scope,
            )
            value_map = {
                "target_full": safe_float(target_fr.get("mean_answer_normal_avg_nll")) if target_fr else None,
                "oracle_retrain90": safe_float(oracle_fr.get("mean_answer_normal_avg_nll")) if oracle_fr else None,
                "normal": safe_float(fr.get("mean_answer_normal_avg_nll")) if fr else None,
                "flip-logit": safe_float(fr.get("mean_answer_flip_avg_nll")) if fr else None,
                "RG": safe_float(kr.get("mean_masked_key_span_avg_nll_rg")) if kr else None,
                "RIG": safe_float(kr.get("mean_masked_key_span_avg_nll_rig")) if kr else None,
            }
            source_map = {
                "target_full": target_fr.get("summary_path") if target_fr else None,
                "oracle_retrain90": oracle_fr.get("summary_path") if oracle_fr else None,
                "normal": fr.get("summary_path") if fr else None,
                "flip-logit": fr.get("summary_path") if fr else None,
                "RG": kr.get("summary_path") if kr else None,
                "RIG": kr.get("summary_path") if kr else None,
            }
            for order, (category, source_type, description) in enumerate(categories):
                v = value_map.get(category)
                out.append({
                    "method": method,
                    "lr": lr,
                    "split": split,
                    "selected_epoch": selected_epoch,
                    "constraint_scope": main_scope,
                    "bar_order": order,
                    "category": category,
                    "source_type": source_type,
                    "description": description,
                    "avg_nll": v,
                    "avg_prob": avgprob_from_nll(v),
                    "summary_path": source_map.get(category),
                })
    return out


def format_bar_value(v: float | None) -> str:
    if v is None:
        return "NA"
    v = float(v)
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 10:
        return f"{v:.2f}"
    return f"{v:.3f}"


def annotate_bars(ax, bars, vals: list[float | None]) -> None:
    for bar, v in zip(bars, vals):
        label = format_bar_value(v)
        height = bar.get_height()
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    finite_y = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if finite_y:
        ymax = max(finite_y)
        if ymax > 0:
            ax.set_ylim(0, ymax * 1.18)


def plot_selected_epoch_six_bar(bar_rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if not bar_rows:
        return
    fig_dir = out_dir / "figures" / "selected_epoch_six_bar"
    groups = defaultdict(list)
    for r in bar_rows:
        groups[(r.get("split"), r.get("method"), r.get("lr"), r.get("selected_epoch"))].append(r)

    for (split, method, lr, epoch), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: int(r.get("bar_order") or 0))
        labels = [str(r.get("category")) for r in rows]
        vals = [safe_float(r.get("avg_nll")) for r in rows]
        if not any(v is not None for v in vals):
            continue
        y = [0.0 if v is None else float(v) for v in vals]
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        bars = ax.bar(labels, y)
        annotate_bars(ax, bars, vals)
        ax.set_ylabel("AvgNLL")
        ax.set_title(f"{method} lr={lr} epoch={epoch} {split}: full/key recovery comparison")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)
        ax.text(
            0.01,
            0.98,
            "target/oracle/normal/flip: full-answer full-vocab; RG/RIG: key-level masked",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )
        save_fig(fig, fig_dir / sanitize_name(str(split)) / f"{sanitize_name(method)}_lr{sanitize_name(lr)}_epoch{sanitize_name(epoch)}_six_bar_avgnll")
        plt.close(fig)


def build_ft_control_bar_rows(
    full_rows: list[dict[str, Any]],
    keyrec_rows: list[dict[str, Any]],
    selected_split: str | None,
    main_scope: str,
) -> list[dict[str, Any]]:
    """Build FT-control four-bar values for each FT model and split.

    Bars:
    - normal / flip-logit: FT full-answer full-vocab AvgNLL.
    - RG / RIG: FT key-level masked AvgNLL under main_scope.
    """
    ft_models = sorted({
        str(r.get("model_tag"))
        for r in list(full_rows) + list(keyrec_rows)
        if r.get("model_family") == "ft" and r.get("model_tag")
    })
    splits = sorted({
        str(r.get("split"))
        for r in list(full_rows) + list(keyrec_rows)
        if r.get("split") is not None
    })
    if selected_split:
        splits = [selected_split]

    categories = [
        ("normal", "ft_full_normal", "FT full-answer normal AvgNLL"),
        ("flip-logit", "ft_full_flip", "FT full-answer flip-logit AvgNLL"),
        ("RG", "ft_key_rg", f"FT key-level RG masked AvgNLL ({main_scope})"),
        ("RIG", "ft_key_rig", f"FT key-level RIG masked AvgNLL ({main_scope})"),
    ]

    out: list[dict[str, Any]] = []
    for split in splits:
        for model_tag in ft_models:
            fr = first_matching(full_rows, model_family="ft", model_tag=model_tag, split=split)
            kr = first_matching(keyrec_rows, model_family="ft", model_tag=model_tag, split=split, constraint_scope=main_scope)
            value_map = {
                "normal": safe_float(fr.get("mean_answer_normal_avg_nll")) if fr else None,
                "flip-logit": safe_float(fr.get("mean_answer_flip_avg_nll")) if fr else None,
                "RG": safe_float(kr.get("mean_masked_key_span_avg_nll_rg")) if kr else None,
                "RIG": safe_float(kr.get("mean_masked_key_span_avg_nll_rig")) if kr else None,
            }
            source_map = {
                "normal": fr.get("summary_path") if fr else None,
                "flip-logit": fr.get("summary_path") if fr else None,
                "RG": kr.get("summary_path") if kr else None,
                "RIG": kr.get("summary_path") if kr else None,
            }
            for order, (category, source_type, description) in enumerate(categories):
                v = value_map.get(category)
                out.append({
                    "model_tag": model_tag,
                    "split": split,
                    "constraint_scope": main_scope,
                    "bar_order": order,
                    "category": category,
                    "source_type": source_type,
                    "description": description,
                    "avg_nll": v,
                    "avg_prob": avgprob_from_nll(v),
                    "summary_path": source_map.get(category),
                })
    return out


def plot_ft_control_bar(ft_bar_rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if not ft_bar_rows:
        return
    fig_dir = out_dir / "figures" / "ft_control_four_bar"
    groups = defaultdict(list)
    for r in ft_bar_rows:
        groups[(r.get("split"), r.get("model_tag"))].append(r)

    for (split, model_tag), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: int(r.get("bar_order") or 0))
        labels = [str(r.get("category")) for r in rows]
        vals = [safe_float(r.get("avg_nll")) for r in rows]
        if not any(v is not None for v in vals):
            continue
        y = [0.0 if v is None else float(v) for v in vals]
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        bars = ax.bar(labels, y)
        annotate_bars(ax, bars, vals)
        ax.set_ylabel("AvgNLL")
        ax.set_title(f"FT control {model_tag} {split}: recovery comparison")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.25)
        ax.text(
            0.01,
            0.98,
            "normal/flip: full-answer full-vocab; RG/RIG: key-level masked",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )
        save_fig(fig, fig_dir / sanitize_name(str(split)) / f"{sanitize_name(model_tag)}_four_bar_avgnll")
        plt.close(fig)

def make_plots(
    mem_rows: list[dict[str, Any]],
    keyrec_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    mem_fact_rows: list[dict[str, Any]],
    rec_fact_rows: list[dict[str, Any]],
    out_dir: Path,
    min_fact_group_count: int,
    main_scope: str,
    sequence_split: str | None,
    selected_epoch: int | None,
    selected_split: str | None,
    ft_control_bar_rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"

    # By-epoch plots: explicitly exclude FT models.
    mem_unl = [r for r in mem_rows if r.get("model_family") != "ft"]
    keyrec_unl = [r for r in keyrec_rows if r.get("model_family") != "ft"]

    # Open-Key-Recall by epoch, unlearned only.
    curves = defaultdict(list)
    for r in mem_unl:
        k = f"{r.get('method')}|{r.get('lr')}|{r.get('split')}"
        e = safe_float(r.get("epoch")); y = safe_float(r.get("open_key_recall"))
        if e is not None and y is not None:
            curves[k].append((e, y))
    if curves:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, vals in sorted(curves.items()):
            vals = sorted(vals)
            ax.plot([x for x, _ in vals], [y for _, y in vals], marker="o", label=k)
        ax.set_xlabel("epoch")
        ax.set_ylabel("open_key_recall")
        ax.set_title("Open-Key-Recall by Epoch (unlearned only)")
        ax.legend(fontsize=7)
        save_fig(fig, fig_dir / "open_key_recall_by_epoch")
        plt.close(fig)

    # KeySpanAvgNLL by epoch, unlearned only.
    curves = defaultdict(list)
    for r in mem_unl:
        k = f"{r.get('method')}|{r.get('lr')}|{r.get('split')}"
        e = safe_float(r.get("epoch")); y = safe_float(r.get("mean_key_span_avg_nll"))
        if e is not None and y is not None:
            curves[k].append((e, y))
    if curves:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, vals in sorted(curves.items()):
            vals = sorted(vals)
            ax.plot([x for x, _ in vals], [y for _, y in vals], marker="o", label=k)
        ax.set_xlabel("epoch")
        ax.set_ylabel("mean_key_span_avg_nll")
        ax.set_title("KeySpanAvgNLL by Epoch (unlearned only)")
        ax.legend(fontsize=7)
        save_fig(fig, fig_dir / "keyspan_nll_by_epoch")
        plt.close(fig)

    # CKR RG vs RIG.
    vals = [(safe_float(r.get("ckr_token_recall_rg")), safe_float(r.get("ckr_token_recall_rig"))) for r in keyrec_rows]
    vals = [v for v in vals if v[0] is not None and v[1] is not None]
    if vals:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter([v[0] for v in vals], [v[1] for v in vals])
        ax.set_xlabel("ckr_token_recall_rg")
        ax.set_ylabel("ckr_token_recall_rig")
        ax.set_title("CKR RG vs RIG")
        save_fig(fig, fig_dir / "ckr_rg_vs_rig")
        plt.close(fig)

    # Masked NLL RG vs RIG.
    vals = [(safe_float(r.get("mean_masked_key_span_avg_nll_rg")), safe_float(r.get("mean_masked_key_span_avg_nll_rig"))) for r in keyrec_rows]
    vals = [v for v in vals if v[0] is not None and v[1] is not None]
    if vals:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter([v[0] for v in vals], [v[1] for v in vals])
        ax.set_xlabel("mean_masked_key_span_avg_nll_rg")
        ax.set_ylabel("mean_masked_key_span_avg_nll_rig")
        ax.set_title("Masked KSNLL RG vs RIG")
        save_fig(fig, fig_dir / "masked_ksnll_rg_vs_rig")
        plt.close(fig)

    # Full-vocab normal vs flip NLL.
    vals = [(safe_float(r.get("mean_answer_normal_avg_nll")), safe_float(r.get("mean_answer_flip_avg_nll"))) for r in full_rows]
    vals = [v for v in vals if v[0] is not None and v[1] is not None]
    if vals:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter([v[0] for v in vals], [v[1] for v in vals])
        ax.set_xlabel("mean_answer_normal_avg_nll")
        ax.set_ylabel("mean_answer_flip_avg_nll")
        ax.set_title("Full-Vocab Answer NLL Normal vs Flip")
        save_fig(fig, fig_dir / "full_recovery_nll_normal_vs_flip")
        plt.close(fig)

    # Recovery gain by fact group.
    fg_agg = defaultdict(lambda: {"n": 0, "ckr": [], "nll": []})
    for r in rec_fact_rows:
        fg = r.get("fact_group")
        fg_agg[fg]["n"] += int(r.get("num_facts") or 0)
        c = safe_float(r.get("ckr_token_recall_gain"))
        n = safe_float(r.get("mean_masked_key_span_nll_gain"))
        if c is not None:
            fg_agg[fg]["ckr"].append(c)
        if n is not None:
            fg_agg[fg]["nll"].append(n)
    rows = []
    for fg, d in fg_agg.items():
        if d["n"] >= min_fact_group_count:
            rows.append((fg, safe_mean(d["ckr"]), safe_mean(d["nll"])))
    rows = [r for r in rows if r[1] is not None or r[2] is not None]
    if rows:
        rows = sorted(rows, key=lambda x: x[0])
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(rows))))
        ys = list(range(len(rows)))
        ax.barh(ys, [r[1] if r[1] is not None else 0.0 for r in rows], alpha=0.7, label="ckr_token_recall_gain")
        ax.barh(ys, [r[2] if r[2] is not None else 0.0 for r in rows], alpha=0.5, label="mean_masked_key_span_nll_gain")
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows])
        ax.legend()
        ax.set_title("Recovery Gain by Fact Group")
        save_fig(fig, fig_dir / "recovery_gain_by_fact_group")
        plt.close(fig)

    # Method x constraint heatmap.
    methods = sorted({str(r.get("method") or r.get("model_tag") or "unknown") for r in keyrec_rows})
    scopes = sorted({str(r.get("constraint_scope") or "unknown") for r in keyrec_rows})
    if methods and scopes:
        import numpy as np
        grid = [[float("nan") for _ in scopes] for _ in methods]
        for i, m in enumerate(methods):
            for j, s in enumerate(scopes):
                xs = [
                    safe_float(r.get("mean_masked_key_span_nll_gain"))
                    for r in keyrec_rows
                    if str(r.get("method") or r.get("model_tag") or "unknown") == m
                    and str(r.get("constraint_scope") or "unknown") == s
                ]
                xs = [v for v in xs if v is not None]
                grid[i][j] = sum(xs) / len(xs) if xs else float("nan")
        arr = np.array(grid, dtype=float)
        fig, ax = plt.subplots(figsize=(1.5 + len(scopes) * 1.8, 1.5 + len(methods) * 0.8))
        im = ax.imshow(arr, aspect="auto")
        ax.set_xticks(range(len(scopes)))
        ax.set_xticklabels(scopes, rotation=25, ha="right")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_title("Method x Constraint Heatmap (mean_masked_key_span_nll_gain)")
        fig.colorbar(im, ax=ax)
        save_fig(fig, fig_dir / "method_constraint_heatmap")
        plt.close(fig)

    # Target / oracle / unlearned comparison.
    pick = {"target_full", "oracle_retrain90", "grad_ascent", "KL", "grad_diff", "npo"}
    rows_cmp = [r for r in mem_rows if (str(r.get("model_tag")) in pick or str(r.get("method")) in pick)]
    if rows_cmp:
        grp = defaultdict(lambda: {"rec": [], "nll": []})
        for r in rows_cmp:
            name = str(r.get("method") or r.get("model_tag"))
            okr = safe_float(r.get("open_key_recall"))
            nll = safe_float(r.get("mean_key_span_avg_nll"))
            if okr is not None:
                grp[name]["rec"].append(okr)
            if nll is not None:
                grp[name]["nll"].append(nll)
        names = [n for n in ["target_full", "oracle_retrain90", "grad_ascent", "KL", "grad_diff", "npo"] if n in grp]
        if names:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].bar(names, [safe_mean(grp[n]["rec"]) or 0.0 for n in names])
            axes[0].set_title("open_key_recall")
            axes[0].tick_params(axis="x", rotation=25)
            axes[1].bar(names, [safe_mean(grp[n]["nll"]) or 0.0 for n in names])
            axes[1].set_title("mean_key_span_avg_nll")
            axes[1].tick_params(axis="x", rotation=25)
            fig.suptitle("Target / Oracle / Unlearned Comparison")
            save_fig(fig, fig_dir / "target_oracle_unlearned_comparison")
            plt.close(fig)

    # New method/lr ladder plots: target_full -> all epochs -> oracle_retrain90.
    # One method/lr/split gets four plots:
    # key AvgNLL, key AvgProb, full AvgNLL, full AvgProb.
    method_lr_pairs = set()
    for rows_src in (mem_rows, keyrec_rows, full_rows):
        for r in rows_src:
            if r.get("model_family") == "unlearned" and r.get("method"):
                method_lr_pairs.add((str(r.get("method")), str(r.get("lr")) if r.get("lr") is not None else None))
    splits = sorted({str(r.get("split")) for rows_src in (mem_rows, keyrec_rows, full_rows) for r in rows_src if r.get("split") is not None})
    if sequence_split:
        splits = [sequence_split]

    for split in splits:
        for method, lr in sorted(method_lr_pairs):
            prefix = fig_dir / "method_ladder" / sanitize_name(split) / f"{sanitize_name(method)}_lr{sanitize_name(lr)}"

            labels, series = line_values_key(mem_rows, keyrec_rows, method, lr, split, main_scope, "nll")
            if len(labels) > 2:
                plot_ladder(
                    labels,
                    series,
                    title=f"{method} lr={lr} {split} | Key AvgNLL ({main_scope})",
                    ylabel="AvgNLL",
                    out_base=Path(str(prefix) + "_key_avgnll"),
                )

            labels, series = line_values_key(mem_rows, keyrec_rows, method, lr, split, main_scope, "prob")
            if len(labels) > 2:
                plot_ladder(
                    labels,
                    series,
                    title=f"{method} lr={lr} {split} | Key AvgProb ({main_scope})",
                    ylabel="exp(-AvgNLL)",
                    out_base=Path(str(prefix) + "_key_avgprob"),
                )

            labels, series = line_values_full(full_rows, method, lr, split, "nll")
            if len(labels) > 2:
                plot_ladder(
                    labels,
                    series,
                    title=f"{method} lr={lr} {split} | Full Answer AvgNLL",
                    ylabel="AvgNLL",
                    out_base=Path(str(prefix) + "_full_avgnll"),
                )

            labels, series = line_values_full(full_rows, method, lr, split, "prob")
            if len(labels) > 2:
                plot_ladder(
                    labels,
                    series,
                    title=f"{method} lr={lr} {split} | Full Answer AvgProb",
                    ylabel="exp(-AvgNLL)",
                    out_base=Path(str(prefix) + "_full_avgprob"),
                )


    # Selected-epoch six-bar plots: target_full, oracle_retrain90, normal, flip-logit, RG, RIG.
    bar_rows = build_selected_epoch_bar_rows(full_rows, keyrec_rows, selected_epoch, selected_split, main_scope)
    plot_selected_epoch_six_bar(bar_rows, out_dir)

    # FT-control four-bar plots: normal, flip-logit, RG, RIG for every FT model.
    plot_ft_control_bar(ft_control_bar_rows, out_dir)


def write_merged(
    out_dir: Path,
    mem_rows: list[dict[str, Any]],
    keyrec_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
) -> None:
    # Legacy-style key merged table.
    mem_map = {}
    for r in mem_rows:
        k = (r.get("model_family"), r.get("model_tag"), r.get("method"), r.get("unlearn_run"), r.get("lr"), str(r.get("epoch")), r.get("split"))
        mem_map[k] = r

    key_merged = []
    for rr in keyrec_rows:
        k = (rr.get("model_family"), rr.get("model_tag"), rr.get("method"), rr.get("unlearn_run"), rr.get("lr"), str(rr.get("epoch")), rr.get("split"))
        mr = mem_map.get(k, {})
        key_merged.append({
            "model_family": rr.get("model_family"),
            "model_tag": rr.get("model_tag"),
            "method": rr.get("method"),
            "unlearn_run": rr.get("unlearn_run"),
            "lr": rr.get("lr"),
            "epoch": rr.get("epoch"),
            "split": rr.get("split"),
            "constraint_scope": rr.get("constraint_scope"),
            "memory_summary_path": mr.get("summary_path"),
            "key_recovery_summary_path": rr.get("summary_path"),
            "open_key_recall": mr.get("open_key_recall"),
            "mean_key_span_avg_nll": mr.get("mean_key_span_avg_nll"),
            "mean_key_span_avg_prob": mr.get("mean_key_span_avg_prob"),
            "ckr_token_recall_gain": rr.get("ckr_token_recall_gain"),
            "mean_masked_key_span_nll_gain": rr.get("mean_masked_key_span_nll_gain"),
        })

    if mem_rows and not keyrec_rows:
        for mr in mem_rows:
            key_merged.append({
                "model_family": mr.get("model_family"), "model_tag": mr.get("model_tag"), "method": mr.get("method"),
                "unlearn_run": mr.get("unlearn_run"), "lr": mr.get("lr"), "epoch": mr.get("epoch"), "split": mr.get("split"),
                "constraint_scope": None, "memory_summary_path": mr.get("summary_path"), "key_recovery_summary_path": None,
                "open_key_recall": mr.get("open_key_recall"), "mean_key_span_avg_nll": mr.get("mean_key_span_avg_nll"),
                "mean_key_span_avg_prob": mr.get("mean_key_span_avg_prob"), "ckr_token_recall_gain": None,
                "mean_masked_key_span_nll_gain": None,
            })

    key_merged_fields = [
        "model_family", "model_tag", "method", "unlearn_run", "lr", "epoch", "split", "constraint_scope",
        "memory_summary_path", "key_recovery_summary_path", "open_key_recall", "mean_key_span_avg_nll",
        "mean_key_span_avg_prob", "ckr_token_recall_gain", "mean_masked_key_span_nll_gain",
    ]
    write_csv(out_dir / "all_key_eval_merged.csv", key_merged, key_merged_fields)

    # Full merged table: memory + key recovery + full-vocab recovery.
    full_map = {}
    for r in full_rows:
        k = (r.get("model_family"), r.get("model_tag"), r.get("method"), r.get("unlearn_run"), r.get("lr"), str(r.get("epoch")), r.get("split"))
        full_map[k] = r

    all_merged = []
    # Use union of keys across all tables.
    keys = set()
    for rows_src in (mem_rows, keyrec_rows, full_rows):
        for r in rows_src:
            keys.add((r.get("model_family"), r.get("model_tag"), r.get("method"), r.get("unlearn_run"), r.get("lr"), str(r.get("epoch")), r.get("split")))

    for key in sorted(keys, key=lambda x: tuple("" if v is None else str(v) for v in x)):
        mr = mem_map.get(key, {})
        fr = full_map.get(key, {})
        matched_keyrecs = [
            r for r in keyrec_rows
            if (r.get("model_family"), r.get("model_tag"), r.get("method"), r.get("unlearn_run"), r.get("lr"), str(r.get("epoch")), r.get("split")) == key
        ]
        if not matched_keyrecs:
            matched_keyrecs = [None]
        for kr in matched_keyrecs:
            kr = kr or {}
            all_merged.append({
                "model_family": key[0],
                "model_tag": key[1],
                "method": key[2],
                "unlearn_run": key[3],
                "lr": key[4],
                "epoch": key[5],
                "split": key[6],
                "constraint_scope": kr.get("constraint_scope"),
                "memory_summary_path": mr.get("summary_path"),
                "key_recovery_summary_path": kr.get("summary_path"),
                "recovery_summary_path": fr.get("summary_path"),
                "open_key_recall": mr.get("open_key_recall"),
                "mean_key_span_avg_nll": mr.get("mean_key_span_avg_nll"),
                "mean_key_span_avg_prob": mr.get("mean_key_span_avg_prob"),
                "mean_masked_key_span_avg_nll_rg": kr.get("mean_masked_key_span_avg_nll_rg"),
                "mean_masked_key_span_avg_nll_rig": kr.get("mean_masked_key_span_avg_nll_rig"),
                "mean_masked_key_span_nll_gain": kr.get("mean_masked_key_span_nll_gain"),
                "mean_answer_normal_avg_nll": fr.get("mean_answer_normal_avg_nll"),
                "mean_answer_flip_avg_nll": fr.get("mean_answer_flip_avg_nll"),
                "mean_answer_flip_nll_gain": fr.get("mean_answer_flip_nll_gain"),
                "mean_answer_normal_avg_prob": fr.get("mean_answer_normal_avg_prob"),
                "mean_answer_flip_avg_prob": fr.get("mean_answer_flip_avg_prob"),
                "answer_flip_success_rate": fr.get("answer_flip_success_rate"),
            })

    all_merged_fields = [
        "model_family", "model_tag", "method", "unlearn_run", "lr", "epoch", "split", "constraint_scope",
        "memory_summary_path", "key_recovery_summary_path", "recovery_summary_path",
        "open_key_recall", "mean_key_span_avg_nll", "mean_key_span_avg_prob",
        "mean_masked_key_span_avg_nll_rg", "mean_masked_key_span_avg_nll_rig", "mean_masked_key_span_nll_gain",
        "mean_answer_normal_avg_nll", "mean_answer_flip_avg_nll", "mean_answer_flip_nll_gain",
        "mean_answer_normal_avg_prob", "mean_answer_flip_avg_prob", "answer_flip_success_rate",
    ]
    write_csv(out_dir / "all_05_eval_merged.csv", all_merged, all_merged_fields)


def write_best_checkpoints(
    out_dir: Path,
    mem_rows: list[dict[str, Any]],
    keyrec_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    main_scope: str,
) -> None:
    by_ckpt = defaultdict(dict)
    for r in mem_rows:
        k = (r.get("model_family"), r.get("model_tag"), r.get("method"), r.get("unlearn_run"), r.get("lr"), str(r.get("epoch")))
        by_ckpt[k][("mem", r.get("split"))] = r
    for r in keyrec_rows:
        k = (r.get("model_family"), r.get("model_tag"), r.get("method"), r.get("unlearn_run"), r.get("lr"), str(r.get("epoch")))
        by_ckpt[k][("keyrec", r.get("split"), r.get("constraint_scope"))] = r
    for r in full_rows:
        k = (r.get("model_family"), r.get("model_tag"), r.get("method"), r.get("unlearn_run"), r.get("lr"), str(r.get("epoch")))
        by_ckpt[k][("fullrec", r.get("split"))] = r

    forget_raw = {}
    retain_raw = {}
    key_rec_raw = {}
    full_rec_raw = {}

    for k, d in by_ckpt.items():
        mf = d.get(("mem", "forget10"))
        mr = d.get(("mem", "retain90"))
        kr = d.get(("keyrec", "forget10", main_scope))
        fr = d.get(("fullrec", "forget10"))

        forget_raw[k] = None if not mf else (
            -safe_float(mf.get("open_key_recall")) if safe_float(mf.get("open_key_recall")) is not None else None,
            safe_float(mf.get("mean_key_span_avg_nll")),
        )
        retain_raw[k] = None if not mr else (
            safe_float(mr.get("open_key_recall")),
            -safe_float(mr.get("mean_key_span_avg_nll")) if safe_float(mr.get("mean_key_span_avg_nll")) is not None else None,
        )
        key_rec_raw[k] = None if not kr else (
            safe_float(kr.get("ckr_token_recall_gain")),
            safe_float(kr.get("mean_masked_key_span_nll_gain")),
        )
        full_rec_raw[k] = None if not fr else (
            safe_float(fr.get("mean_answer_flip_nll_gain")),
        )

    def comp2(v: tuple[float | None, float | None] | None) -> float | None:
        if v is None or v[0] is None or v[1] is None:
            return None
        return float(v[0]) + float(v[1])

    def comp1(v: tuple[float | None] | None) -> float | None:
        if v is None or v[0] is None:
            return None
        return float(v[0])

    fr_comp = {k: comp2(v) for k, v in forget_raw.items()}
    rp_comp = {k: comp2(v) for k, v in retain_raw.items()}
    kr_comp = {k: comp2(v) for k, v in key_rec_raw.items()}
    fl_comp = {k: comp1(v) for k, v in full_rec_raw.items()}

    fr_z = zscores({str(k): v for k, v in fr_comp.items() if v is not None})
    rp_z = zscores({str(k): v for k, v in rp_comp.items() if v is not None})
    kr_z = zscores({str(k): v for k, v in kr_comp.items() if v is not None})
    fl_z = zscores({str(k): v for k, v in fl_comp.items() if v is not None})

    best_rows = []
    for k in by_ckpt:
        ks = str(k)
        fz = fr_z.get(ks)
        rz = rp_z.get(ks)
        kz = kr_z.get(ks)
        lz = fl_z.get(ks)
        miss = []
        if fr_comp.get(k) is None:
            miss.append("forget10_memory")
        if rp_comp.get(k) is None:
            miss.append("retain90_memory")
        if kr_comp.get(k) is None:
            miss.append(f"forget10_key_recovery_{main_scope}")
        if fl_comp.get(k) is None:
            miss.append("forget10_full_recovery")
        overall = None
        if fz is not None and rz is not None and kz is not None and lz is not None:
            overall = fz + rz + kz + lz
        best_rows.append({
            "model_family": k[0],
            "model_tag": k[1],
            "method": k[2],
            "unlearn_run": k[3],
            "lr": k[4],
            "epoch": k[5],
            "forget_erasure_score": fz,
            "retain_preservation_score": rz,
            "key_recovery_score": kz,
            "full_recovery_score": lz,
            "overall_score": overall,
            "missing_fields": ";".join(miss) if miss else "",
        })

    best_rows.sort(key=lambda r: (r.get("overall_score") is None, -(safe_float(r.get("overall_score")) or -1e9)))
    write_csv(out_dir / "best_checkpoints.csv", best_rows, [
        "model_family", "model_tag", "method", "unlearn_run", "lr", "epoch",
        "forget_erasure_score", "retain_preservation_score", "key_recovery_score", "full_recovery_score",
        "overall_score", "missing_fields",
    ])


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mem_rows, mem_fact_rows = collect_key_memory(args.memory_root)
    keyrec_rows, rec_fact_rows = collect_key_recovery(args.key_recovery_root)
    full_rows = collect_full_recovery(args.recovery_root)

    mem_fields = [
        "model_family", "model_tag", "method", "unlearn_run", "lr", "epoch", "split", "summary_path",
        "num_eval_records", "num_key_facts", "open_key_recall", "open_span_recall", "open_content_token_recall",
        "weighted_open_key_recall", "mean_key_span_avg_nll", "mean_key_span_avg_prob", "median_key_span_avg_nll",
        "weighted_mean_key_span_avg_nll",
    ]
    keyrec_fields = [
        "constraint_scope", "model_family", "model_tag", "method", "unlearn_run", "lr", "epoch", "split", "summary_path",
        "num_eval_records", "num_key_facts", "mean_candidate_token_count", "median_candidate_token_count",
        "small_candidate_frac", "forced_gold_token_frac", "ckr_token_recall_rg", "ckr_token_recall_rig",
        "ckr_token_recall_gain", "ckr_fact_hit_rg", "ckr_fact_hit_rig", "ckr_fact_hit_gain",
        "mean_masked_key_span_avg_nll_rg", "mean_masked_key_span_avg_nll_rig",
        "mean_masked_key_span_avg_prob_rg", "mean_masked_key_span_avg_prob_rig",
        "mean_masked_key_span_nll_gain", "median_masked_key_span_nll_gain",
    ]
    full_fields = [
        "model_family", "model_tag", "method", "unlearn_run", "lr", "epoch", "split", "flip_alpha", "summary_path",
        "num_eval_records", "num_scored_records", "num_scored_sentences", "num_scored_tokens",
        "mean_answer_normal_avg_nll", "mean_answer_flip_avg_nll", "mean_answer_normal_avg_prob",
        "mean_answer_flip_avg_prob", "mean_answer_flip_nll_gain", "median_answer_flip_nll_gain",
        "answer_flip_success_rate", "mean_sentence_normal_avg_nll", "mean_sentence_flip_avg_nll",
        "mean_sentence_normal_avg_prob", "mean_sentence_flip_avg_prob", "mean_sentence_flip_nll_gain",
        "median_sentence_flip_nll_gain", "sentence_flip_success_rate", "token_weighted_normal_avg_nll",
        "token_weighted_flip_avg_nll", "token_weighted_flip_nll_gain", "mean_answer_length_tokens",
        "mean_num_sentences",
    ]

    # CSVs first.
    write_csv(out_dir / "all_key_memory_summaries.csv", mem_rows, mem_fields)
    write_csv(out_dir / "all_key_recovery_summaries.csv", keyrec_rows, keyrec_fields)
    write_csv(out_dir / "all_recovery_summaries.csv", full_rows, full_fields)

    write_merged(out_dir, mem_rows, keyrec_rows, full_rows)

    write_csv(out_dir / "per_fact_group_memory.csv", mem_fact_rows, [
        "summary_path", "model_family", "model_tag", "method", "lr", "epoch", "split", "fact_group",
        "num_facts", "open_key_recall", "mean_key_span_avg_nll", "mean_key_span_avg_prob",
    ])
    write_csv(out_dir / "per_fact_group_recovery.csv", rec_fact_rows, [
        "summary_path", "constraint_scope", "model_family", "model_tag", "method", "lr", "epoch", "split",
        "fact_group", "num_facts", "ckr_token_recall_gain", "mean_masked_key_span_nll_gain",
    ])

    write_best_checkpoints(out_dir, mem_rows, keyrec_rows, full_rows, args.main_constraint_scope)

    selected_bar_rows = build_selected_epoch_bar_rows(full_rows, keyrec_rows, args.selected_epoch, args.selected_split, args.main_constraint_scope)
    write_csv(out_dir / "selected_epoch_six_bar_values.csv", selected_bar_rows, [
        "method", "lr", "split", "selected_epoch", "constraint_scope", "bar_order", "category",
        "source_type", "description", "avg_nll", "avg_prob", "summary_path",
    ])

    ft_control_bar_rows = build_ft_control_bar_rows(full_rows, keyrec_rows, args.selected_split, args.main_constraint_scope)
    write_csv(out_dir / "ft_control_four_bar_values.csv", ft_control_bar_rows, [
        "model_tag", "split", "constraint_scope", "bar_order", "category",
        "source_type", "description", "avg_nll", "avg_prob", "summary_path",
    ])

    # Then plots.
    if args.make_plots:
        make_plots(
            mem_rows=mem_rows,
            keyrec_rows=keyrec_rows,
            full_rows=full_rows,
            mem_fact_rows=mem_fact_rows,
            rec_fact_rows=rec_fact_rows,
            out_dir=out_dir,
            min_fact_group_count=args.min_fact_group_count,
            main_scope=args.main_constraint_scope,
            sequence_split=args.sequence_split,
            selected_epoch=args.selected_epoch,
            selected_split=args.selected_split,
            ft_control_bar_rows=ft_control_bar_rows,
        )

    print(f"[05-collect] memory summaries: {len(mem_rows)}")
    print(f"[05-collect] key recovery summaries: {len(keyrec_rows)}")
    print(f"[05-collect] full recovery summaries: {len(full_rows)}")
    print(f"[05-collect] wrote {out_dir}")


if __name__ == "__main__":
    main()
