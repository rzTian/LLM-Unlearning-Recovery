#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

KEY_RECOVERY_DEFAULT = "results/tofu/key_recovery"
FULL_RECOVERY_DEFAULT = "results/tofu/recovery"
OUTPUT_DEFAULT = "results/tofu/05_plots/reviewer_q1"
DEFAULT_SCOPE = "same_fact_group_content_vocab"
DEFAULT_METHODS = "grad_ascent,KL,grad_diff,npo"
DEFAULT_METHOD_EPOCHS = "grad_ascent:20,KL:20,grad_diff:20,npo:20"
DEFAULT_METHOD_LRS = ""

METHOD_DISPLAY = {
    "grad_ascent": "GA",
    "GA": "GA",
    "grad_diff": "GD",
    "GD": "GD",
    "KL": "GA+KL",
    "GA+KL": "GA+KL",
    "npo": "NPO",
    "NPO": "NPO",
}

GROUP_DISPLAY = {
    "target_full": r"$f^0$",
    "oracle_retrain90": r"$f^*$",
    "grad_ascent": "GA",
    "KL": "GA+KL",
    "grad_diff": "GD",
    "npo": "NPO",
}

FIG1_TICK_LABELS = [r"$f^0$", r"$f^*$", r"$f^u$", r"$f^r$"]
FIG1_BAR_LABELS = ["Original", "Retrain oracle", "Unlearned", "Inverse-logit recovery"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate reviewer-response TOFU plots using 05_eval recovery/key_recovery summaries."
    )
    p.add_argument("--key_recovery_root", default=KEY_RECOVERY_DEFAULT)
    p.add_argument("--recovery_root", default=FULL_RECOVERY_DEFAULT)
    p.add_argument("--output_dir", default=OUTPUT_DEFAULT)
    p.add_argument("--split", default="forget10")
    p.add_argument("--constraint_scope", default=DEFAULT_SCOPE)
    p.add_argument("--methods", default=DEFAULT_METHODS, help="Comma-separated unlearn methods to plot.")
    p.add_argument(
        "--method_epochs",
        default=DEFAULT_METHOD_EPOCHS,
        help=(
            "Comma-separated method:epoch pairs, e.g. "
            "grad_ascent:20,KL:20,grad_diff:10,npo:20. "
            "This only selects data; epoch is not displayed in the figure."
        ),
    )
    p.add_argument(
        "--method_lrs",
        default=DEFAULT_METHOD_LRS,
        help=(
            "Optional comma-separated method:lr pairs, e.g. "
            "grad_ascent:1e-05,KL:1e-05,grad_diff:1e-05,npo:0.0005. "
            "This only selects data; lr is not displayed in the figure."
        ),
    )
    p.add_argument("--selected_epoch", type=int, default=None, help="Fallback epoch for methods not in --method_epochs.")
    p.add_argument("--make_plots", action="store_true")
    p.add_argument("--fig_format", default="png,pdf", help="Comma-separated formats, default png,pdf.")
    p.add_argument(
        "--broken_axis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a broken y-axis for Fig. 1 when one bar is much larger than the others.",
    )
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


def parse_lr(unlearn_run: str | None) -> str | None:
    if not unlearn_run:
        return None
    m = re.search(r"-lr([^_]+)", str(unlearn_run))
    return m.group(1) if m else None


def lr_matches(observed: Any, expected: Any) -> bool:
    """Robust lr comparison: string-compatible and float-compatible."""
    if expected is None or expected == "":
        return True
    if observed is None or observed == "":
        return False
    obs_s, exp_s = str(observed), str(expected)
    if obs_s == exp_s:
        return True
    try:
        obs_f = float(obs_s)
        exp_f = float(exp_s)
        return math.isclose(obs_f, exp_f, rel_tol=1e-8, abs_tol=1e-12)
    except Exception:
        return False


def sanitize_name(x: Any) -> str:
    s = str(x or "none")
    s = re.sub(r"[^A-Za-z0-9_.=+-]+", "_", s)
    return s.strip("_") or "none"


def parse_methods(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_method_epochs(s: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in str(s or "").split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        k, v = item.split(":", 1)
        try:
            out[k.strip()] = int(float(v.strip()))
        except Exception:
            pass
    return out


def parse_method_lrs(s: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in str(s or "").split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        k, v = item.split(":", 1)
        if k.strip() and v.strip():
            out[k.strip()] = v.strip()
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if r.get(k) is None else r.get(k) for k in fields})


def collect_full_recovery(recovery_root: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(Path(recovery_root).glob("**/epoch-*-summary.json")):
        s = read_json(p)
        rows.append({
            "model_family": s.get("model_family"),
            "model_tag": s.get("model_tag"),
            "method": s.get("method"),
            "unlearn_run": s.get("unlearn_run"),
            "lr": parse_lr(s.get("unlearn_run")),
            "epoch": s.get("epoch"),
            "split": s.get("split"),
            "summary_path": str(p),
            "num_eval_records": s.get("num_eval_records"),
            "mean_answer_normal_avg_nll": s.get("mean_answer_normal_avg_nll"),
            "mean_answer_flip_avg_nll": s.get("mean_answer_flip_avg_nll"),
            "mean_answer_flip_nll_gain": s.get("mean_answer_flip_nll_gain"),
        })
    return rows


def collect_key_recovery(key_recovery_root: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(Path(key_recovery_root).glob("**/epoch-*-summary.json")):
        s = read_json(p)
        rows.append({
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
            "mean_masked_key_span_avg_nll_rg": s.get("mean_masked_key_span_avg_nll_rg"),
            "mean_masked_key_span_avg_nll_rig": s.get("mean_masked_key_span_avg_nll_rig"),
            "mean_masked_key_span_nll_gain": s.get("mean_masked_key_span_nll_gain"),
        })
    return rows


def first_matching(rows: list[dict[str, Any]], **conds: Any) -> dict[str, Any] | None:
    cands: list[dict[str, Any]] = []
    for r in rows:
        ok = True
        for k, v in conds.items():
            if v is None or v == "":
                continue
            if k == "lr":
                if not lr_matches(r.get(k), v):
                    ok = False
                    break
            elif str(r.get(k)) != str(v):
                ok = False
                break
        if ok:
            cands.append(r)
    if not cands:
        return None
    cands.sort(key=lambda r: safe_int(r.get("epoch")) if safe_int(r.get("epoch")) is not None else -1, reverse=True)
    return cands[0]


def observed_epoch(rows: list[dict[str, Any]], method: str, split: str, lr: str | None = None) -> int | None:
    epochs = []
    for r in rows:
        if r.get("model_family") == "unlearned" and str(r.get("method")) == method and str(r.get("split")) == split and lr_matches(r.get("lr"), lr):
            e = safe_int(r.get("epoch"))
            if e is not None:
                epochs.append(e)
    return max(epochs) if epochs else None


def method_epoch(method: str, epoch_map: dict[str, int], selected_epoch: int | None, full_rows: list[dict[str, Any]], key_rows: list[dict[str, Any]], split: str, lr: str | None = None) -> int | None:
    if method in epoch_map:
        return epoch_map[method]
    if selected_epoch is not None:
        return selected_epoch
    return observed_epoch(full_rows + key_rows, method, split, lr)


def method_lr(method: str, lr_map: dict[str, str]) -> str | None:
    return lr_map.get(method)


def build_fig1_rows(full_rows: list[dict[str, Any]], methods: list[str], epoch_map: dict[str, int], lr_map: dict[str, str], selected_epoch: int | None, split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    target = first_matching(full_rows, model_family="ft", model_tag="target_full", split=split)
    oracle = first_matching(full_rows, model_family="ft", model_tag="oracle_retrain90", split=split)
    for m in methods:
        lr = method_lr(m, lr_map)
        e = method_epoch(m, epoch_map, selected_epoch, full_rows, [], split, lr)
        unlearn = first_matching(full_rows, model_family="unlearned", method=m, split=split, epoch=e, lr=lr)
        entries = [
            ("Original", "target_full", safe_float(target.get("mean_answer_normal_avg_nll")) if target else None, target),
            ("Retrain oracle", "oracle_retrain90", safe_float(oracle.get("mean_answer_normal_avg_nll")) if oracle else None, oracle),
            ("Unlearned", "normal", safe_float(unlearn.get("mean_answer_normal_avg_nll")) if unlearn else None, unlearn),
            ("Inverse-logit recovery", "flip-logit", safe_float(unlearn.get("mean_answer_flip_avg_nll")) if unlearn else None, unlearn),
        ]
        for order, (label, source_type, value, row) in enumerate(entries):
            out.append({
                "figure": "fig1_full_answer_nll",
                "method": m,
                "method_display": METHOD_DISPLAY.get(m, m),
                "split": split,
                "epoch": e,
                "lr": lr,
                "bar_order": order,
                "bar_label": label,
                "axis_label": FIG1_TICK_LABELS[order],
                "source_type": source_type,
                "avg_nll": value,
                "summary_path": row.get("summary_path") if row else None,
            })
    return out


def build_fig2_rows(key_rows: list[dict[str, Any]], methods: list[str], epoch_map: dict[str, int], lr_map: dict[str, str], selected_epoch: int | None, split: str, scope: str) -> list[dict[str, Any]]:
    groups = ["target_full", "oracle_retrain90"] + methods
    out: list[dict[str, Any]] = []
    for group_order, g in enumerate(groups):
        if g == "target_full":
            row = first_matching(key_rows, model_family="ft", model_tag="target_full", split=split, constraint_scope=scope)
            e = None
            lr = None
        elif g == "oracle_retrain90":
            row = first_matching(key_rows, model_family="ft", model_tag="oracle_retrain90", split=split, constraint_scope=scope)
            e = None
            lr = None
        else:
            lr = method_lr(g, lr_map)
            e = method_epoch(g, epoch_map, selected_epoch, [], key_rows, split, lr)
            row = first_matching(key_rows, model_family="unlearned", method=g, split=split, epoch=e, constraint_scope=scope, lr=lr)
        for bar_order, metric in enumerate(["RG", "RIG"]):
            field = "mean_masked_key_span_avg_nll_rg" if metric == "RG" else "mean_masked_key_span_avg_nll_rig"
            out.append({
                "figure": "fig2_key_rg_rig_nll",
                "group": g,
                "group_display": GROUP_DISPLAY.get(g, METHOD_DISPLAY.get(g, g)),
                "split": split,
                "constraint_scope": scope,
                "epoch": e,
                "lr": lr,
                "group_order": group_order,
                "bar_order": bar_order,
                "metric": metric,
                "avg_nll": safe_float(row.get(field)) if row else None,
                "mean_candidate_token_count": safe_float(row.get("mean_candidate_token_count")) if row else None,
                "summary_path": row.get("summary_path") if row else None,
            })
    return out


def setup_plot_style() -> dict[str, Any]:
    import matplotlib as mpl
    import seaborn as sns

    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans", "SimHei"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.labelweight": "medium",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "legend.fontsize": 9,
        "grid.color": "#E2E2E2",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
    })
    colors = sns.color_palette("Set2", 7)
    return {
        "COLOR_OM": colors[3],
        "EDGE_OM": "#9E2D2B",
        "COLOR_RIG": colors[0],
        "EDGE_RIG": "#2F7A4E",
        "COLOR_RG": colors[2],
        "EDGE_RG": "#1F4E79",
        "COLOR_RS": colors[4],
        "EDGE_RS": "#4A7F1B",
        "COLOR_NEUTRAL": "#B7B7B7",
        "EDGE_NEUTRAL": "#666666",
    }


def save_fig(fig, out_base: Path, formats: list[str]) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.strip().lower().lstrip(".")
        if not fmt:
            continue
        fig.savefig(out_base.with_suffix("." + fmt), bbox_inches="tight", pad_inches=0.10)


def format_value(v: float | None) -> str:
    if v is None:
        return "NA"
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 10:
        return f"{v:.2f}"
    return f"{v:.3f}"


def annotate_bars(ax, bars, values: list[float | None], fontsize: int = 7, ylim: tuple[float, float] | None = None) -> None:
    lo, hi = ylim if ylim is not None else ax.get_ylim()
    for bar, v in zip(bars, values):
        if v is None:
            continue
        if not (lo <= float(v) <= hi):
            continue
        ax.annotate(
            format_value(v),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            annotation_clip=True,
        )


def needs_break(vals: list[float | None]) -> bool:
    finite = sorted([float(v) for v in vals if v is not None and math.isfinite(float(v))])
    if len(finite) < 3:
        return False
    ymax = finite[-1]
    second = finite[-2]
    return ymax > 25 and ymax > max(2.8 * second, second + 20)


def break_limits(vals: list[float | None]) -> tuple[tuple[float, float], tuple[float, float]]:
    finite = sorted([float(v) for v in vals if v is not None and math.isfinite(float(v))])
    ymax = finite[-1]
    second = finite[-2] if len(finite) >= 2 else ymax * 0.2
    lower_hi = max(second * 1.35, second + 1.5)
    lower_hi = min(lower_hi, ymax * 0.45)
    lower_hi = max(lower_hi, 4.5)
    top_span = max(ymax * 0.17, lower_hi * 0.75, 5.0)
    upper_lo = max(lower_hi + 1.0, ymax - top_span)
    upper_hi = ymax * 1.10
    return (0.0, lower_hi), (upper_lo, upper_hi)


def add_break_marks(ax_bottom, ax_top) -> None:
    d = 0.010
    kwargs = dict(transform=ax_top.transAxes, color="black", clip_on=False, lw=0.8)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)


def draw_delta_arrow_cross_axes(fig, ax_bottom, ax_top, x: float, normal_v: float | None, flip_v: float | None, bottom_ylim: tuple[float, float], top_ylim: tuple[float, float]) -> None:
    """Draw double arrow at the x-position of the lower bar; supports broken y-axis panels."""
    if normal_v is None or flip_v is None:
        return
    normal_v = float(normal_v)
    flip_v = float(flip_v)
    if not (math.isfinite(normal_v) and math.isfinite(flip_v)):
        return
    low = min(normal_v, flip_v)
    high = max(normal_v, flip_v)
    if high <= low:
        return

    # Place the arrow on the lower-value bar side: f^r if f^u > f^r, else f^u.
    lower_bar_x = 3 if flip_v <= normal_v else 2
    x = float(lower_bar_x)

    # Lift the lower endpoint by a local-axis margin, not by the full delta span.
    # This avoids pushing the lower endpoint into the broken-axis gap when high-low is large.
    bottom_span = bottom_ylim[1] - bottom_ylim[0]
    top_span = top_ylim[1] - top_ylim[0]

    lower_pad = max(0.045 * bottom_span, 0.35)
    upper_pad = max(0.045 * top_span, 0.60)

    # Keep the lower endpoint visible in the lower panel whenever the lower bar is in the lower panel.
    if bottom_ylim[0] <= low <= bottom_ylim[1]:
        start = min(low + lower_pad, bottom_ylim[1] - 0.18 * bottom_span)
    else:
        start = low + lower_pad

    # Keep the upper endpoint visible in the upper panel whenever the higher bar is in the upper panel.
    if top_ylim[0] <= high <= top_ylim[1]:
        end = max(high - upper_pad, top_ylim[0] + 0.18 * top_span)
    else:
        end = high - upper_pad

    if end <= start:
        start, end = low, high
    
    start += 0.025 * bottom_span
    end += 0.020 * top_span

    def pick_ax(y: float):
        if bottom_ylim[0] <= y <= bottom_ylim[1]:
            return ax_bottom
        return ax_top

    start_ax = pick_ax(start)
    end_ax = pick_ax(end)
    start_disp = start_ax.transData.transform((x, start))
    end_disp = end_ax.transData.transform((x, end))
    inv = fig.transFigure.inverted()
    start_fig = inv.transform(start_disp)
    end_fig = inv.transform(end_disp)

    from matplotlib.patches import FancyArrowPatch
    arrow = FancyArrowPatch(
        start_fig,
        end_fig,
        transform=fig.transFigure,
        arrowstyle="<->",
        mutation_scale=9,
        lw=1.0,
        color="#4a4a4a",
        zorder=10,
        clip_on=False,
    )
    fig.add_artist(arrow)

    # Add a small horizontal cap at the upper arrow tip.
    cap_ax = pick_ax(end)
    cap_half_width = 0.08
    cap_ax.hlines(
        y=end,
        xmin=x - cap_half_width,
        xmax=x + cap_half_width,
        colors="#4a4a4a",
        linewidth=1.0,
        zorder=12,
        clip_on=False,
    )

    mid = (start_fig + end_fig) / 2
    fig.text(
        mid[0],
        mid[1],
        f"Δ={normal_v - flip_v:.1f}",
        ha="center",
        va="center",
        fontsize=7,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.90),
        zorder=11,
    )


def draw_delta_arrow_single_axis(ax, x: float, normal_v: float | None, flip_v: float | None) -> None:
    if normal_v is None or flip_v is None:
        return
    normal_v = float(normal_v)
    flip_v = float(flip_v)
    low = min(normal_v, flip_v)
    high = max(normal_v, flip_v)
    if high <= low:
        return
    lower_bar_x = 3 if flip_v <= normal_v else 2
    y0, y1 = ax.get_ylim()
    yrange = y1 - y0
    lower_pad = max(0.045 * yrange, 0.35)
    upper_pad = max(0.045 * yrange, 0.50)

    start = low + lower_pad
    end = high - upper_pad

    if end <= start:
        start, end = low, high
    ax.annotate("", xy=(lower_bar_x, start), xytext=(lower_bar_x, end), arrowprops=dict(arrowstyle="<->", lw=1.0, color="#4a4a4a"), annotation_clip=False)
    ax.text(lower_bar_x, (start + end) / 2, f"Δ={normal_v - flip_v:.1f}", ha="center", va="center", fontsize=7, bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.9))




def draw_delta_arrow_pair_single_axis(
    ax,
    x_unlearned: float,
    x_recovery: float,
    normal_v: float | None,
    flip_v: float | None,
    high_color: str = "#4a4a4a",
) -> None:
    """Draw the same Δ arrow logic as Fig. 1, but for a paired-bar layout.

    normal_v corresponds to the unlearned result (f^u in the original Fig. 1),
    and flip_v corresponds to inverse-logit recovery (f^r in the original Fig. 1).
    The arrow is placed on the lower-value bar side and annotated as Δ=normal-flip,
    matching draw_delta_arrow_single_axis.
    """
    if normal_v is None or flip_v is None:
        return
    normal_v = float(normal_v)
    flip_v = float(flip_v)
    if not (math.isfinite(normal_v) and math.isfinite(flip_v)):
        return
    low = min(normal_v, flip_v)
    high = max(normal_v, flip_v)
    if high <= low:
        return

    lower_bar_x = x_recovery if flip_v <= normal_v else x_unlearned
    y0, y1 = ax.get_ylim()
    yrange = y1 - y0
    lower_pad = max(0.045 * yrange, 0.35)
    upper_pad = 0 # max(0.045 * yrange, 0.50)

    start = low + lower_pad
    end = high - upper_pad
    if end <= start:
        start, end = low, high

    ax.annotate(
        "",
        xy=(lower_bar_x, start),
        xytext=(lower_bar_x, end),
        arrowprops=dict(arrowstyle="<->", lw=1.0, color="#4a4a4a"),
        annotation_clip=False,
        zorder=10,
    )
    # Add a short cap at the upper arrow tip, using the same color as the higher bar.
    cap_half_width = 0.075
    ax.hlines(
        y=end,
        xmin=lower_bar_x - cap_half_width,
        xmax=lower_bar_x + cap_half_width,
        colors=high_color,
        linewidth=1.2,
        zorder=12,
        clip_on=False,
    )
    ax.text(
        lower_bar_x,
        (start + end) / 2,
        f"Δ={normal_v - flip_v:.1f}",
        ha="center",
        va="center",
        fontsize=7,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.9),
        zorder=11,
    )

def draw_panel_bars(ax, xs, vals, labels, colors, edges) -> Any:
    return ax.bar(xs, [0.0 if v is None else float(v) for v in vals], color=colors, edgecolor=edges, linewidth=0.9, zorder=3)


def style_panel_axis(ax, show_ylabel: bool, y_label: str = "Full-answer AvgNLL") -> None:
    ax.set_facecolor("#FAFAFA")
    ax.yaxis.grid(True, alpha=0.75)
    ax.set_axisbelow(True)
    ax.set_ylabel(y_label if show_ylabel else "")


def plot_fig1(rows: list[dict[str, Any]], out_dir: Path, formats: list[str], use_broken_axis: bool = True) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.patches import Patch

    style = setup_plot_style()
    methods = []
    for r in rows:
        m = r["method"]
        if m not in methods:
            methods.append(m)
    if not methods:
        return

    n = len(methods)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    fig = plt.figure(figsize=(4.8 * ncols, 3.45 * nrows))
    outer = GridSpec(nrows, ncols, figure=fig, wspace=0.12, hspace=0.20)

    color_map = {
        "Original": (style["COLOR_OM"], style["EDGE_OM"]),
        "Retrain oracle": (style["COLOR_RS"], style["EDGE_RS"]),
        "Unlearned": (style["COLOR_OM"], style["EDGE_OM"]),
        "Inverse-logit recovery": (style["COLOR_RIG"], style["EDGE_RIG"]),
    }
    arrow_specs = []

    for idx, method in enumerate(methods):
        cell = outer[idx // ncols, idx % ncols]
        sub = [r for r in rows if r["method"] == method]
        sub = sorted(sub, key=lambda r: int(r.get("bar_order") or 0))
        vals = [safe_float(r.get("avg_nll")) for r in sub]
        xs = np.arange(len(FIG1_BAR_LABELS))
        colors = [color_map[l][0] for l in FIG1_BAR_LABELS]
        edges = [color_map[l][1] for l in FIG1_BAR_LABELS]
        broken = use_broken_axis and needs_break(vals)

        if broken:
            inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=cell, height_ratios=[1.0, 2.3], hspace=0.045)
            ax_top = fig.add_subplot(inner[0])
            ax_bottom = fig.add_subplot(inner[1], sharex=ax_top)
            bottom_ylim, top_ylim = break_limits(vals)
            bars_bottom = draw_panel_bars(ax_bottom, xs, vals, FIG1_BAR_LABELS, colors, edges)
            bars_top = draw_panel_bars(ax_top, xs, vals, FIG1_BAR_LABELS, colors, edges)
            ax_bottom.set_ylim(*bottom_ylim)
            ax_top.set_ylim(*top_ylim)
            style_panel_axis(ax_bottom, show_ylabel=(idx % ncols == 0))
            style_panel_axis(ax_top, show_ylabel=False)
            ax_top.spines.bottom.set_visible(False)
            ax_bottom.spines.top.set_visible(False)
            ax_top.tick_params(labeltop=False, bottom=False, labelbottom=False)
            ax_bottom.tick_params(top=False)
            add_break_marks(ax_bottom, ax_top)
            annotate_bars(ax_bottom, bars_bottom, vals, ylim=bottom_ylim)
            annotate_bars(ax_top, bars_top, vals, ylim=top_ylim)
            ax = ax_bottom
            label_ax = ax_top
            arrow_specs.append((ax_bottom, ax_top, xs[3], vals[2] if len(vals) > 2 else None, vals[3] if len(vals) > 3 else None, bottom_ylim, top_ylim))
        else:
            ax = fig.add_subplot(cell)
            bars = draw_panel_bars(ax, xs, vals, FIG1_BAR_LABELS, colors, edges)
            finite = [v for v in vals if v is not None]
            if finite:
                ymax = max(finite)
                ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)
            style_panel_axis(ax, show_ylabel=(idx % ncols == 0))
            annotate_bars(ax, bars, vals)
            label_ax = ax
            draw_delta_arrow_single_axis(ax, xs[3], vals[2] if len(vals) > 2 else None, vals[3] if len(vals) > 3 else None)

        ax.set_xticks(xs)
        ax.set_xticklabels(FIG1_TICK_LABELS, rotation=0, ha="center")
        disp = METHOD_DISPLAY.get(method, method)
        label_ax.text(0.02, 0.96, f"({chr(97 + idx)}) {disp}", transform=label_ax.transAxes, ha="left", va="top", fontsize=10, weight="bold", bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.85))

    for j in range(n, nrows * ncols):
        ax_empty = fig.add_subplot(outer[j // ncols, j % ncols])
        ax_empty.axis("off")

    handles = [
        Patch(facecolor=style["COLOR_OM"], edgecolor=style["EDGE_OM"], label="Original / Unlearned"),
        Patch(facecolor=style["COLOR_RS"], edgecolor=style["EDGE_RS"], label="Retrain oracle"),
        Patch(facecolor=style["COLOR_RIG"], edgecolor=style["EDGE_RIG"], label="Inverse-logit recovery"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        frameon=True,
        framealpha=0.96,
        edgecolor="#E6E6E6",
        ncol=3,
    )

    fig.subplots_adjust(left=0.075, right=0.985, top=0.985, bottom=0.105)
    fig.canvas.draw()
    for spec in arrow_specs:
        draw_delta_arrow_cross_axes(fig, *spec)
    save_fig(fig, out_dir / "reviewer_q1_fig1_full_answer_flip_nll", formats)
    plt.close(fig)




def plot_fig1_compact(rows: list[dict[str, Any]], out_dir: Path, formats: list[str]) -> None:
    """Compact Fig. 1 variant: keep only unlearned and inverse-logit recovery.

    The x-axis is method-level (GA, GA+KL, GD, NPO). Each method has two bars
    corresponding to the original Fig. 1's f^u and f^r values, but the axis does
    not display f^u/f^r tick labels. The Δ arrow follows the same placement and
    annotation logic as Fig. 1.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    style = setup_plot_style()
    methods = []
    for r in rows:
        m = r["method"]
        if m not in methods:
            methods.append(m)
    if not methods:
        return

    labels = [METHOD_DISPLAY.get(m, m) for m in methods]
    normal_vals: list[float | None] = []
    flip_vals: list[float | None] = []
    compact_rows: list[dict[str, Any]] = []

    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        normal = next((r for r in sub if r.get("source_type") == "normal"), None)
        flip = next((r for r in sub if r.get("source_type") == "flip-logit"), None)
        normal_v = safe_float(normal.get("avg_nll")) if normal else None
        flip_v = safe_float(flip.get("avg_nll")) if flip else None
        normal_vals.append(normal_v)
        flip_vals.append(flip_v)
        compact_rows.append({
            "figure": "fig1_compact_unlearned_recovery_nll",
            "method": m,
            "method_display": METHOD_DISPLAY.get(m, m),
            "split": normal.get("split") if normal else (flip.get("split") if flip else None),
            "epoch": normal.get("epoch") if normal else (flip.get("epoch") if flip else None),
            "lr": normal.get("lr") if normal else (flip.get("lr") if flip else None),
            "unlearned_avg_nll": normal_v,
            "inverse_logit_recovery_avg_nll": flip_v,
            "delta": (normal_v - flip_v) if normal_v is not None and flip_v is not None else None,
            "unlearned_summary_path": normal.get("summary_path") if normal else None,
            "recovery_summary_path": flip.get("summary_path") if flip else None,
        })

    # Save compact values for reproducibility.
    write_csv(
        out_dir.parent / "reviewer_q1_fig1_compact_unlearned_recovery_nll_values.csv",
        compact_rows,
        [
            "figure", "method", "method_display", "split", "epoch", "lr",
            "unlearned_avg_nll", "inverse_logit_recovery_avg_nll", "delta",
            "unlearned_summary_path", "recovery_summary_path",
        ],
    )

    x = np.arange(len(methods))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.0, 3.6), constrained_layout=True)
    ax.set_facecolor("#FAFAFA")
    ax.yaxis.grid(True, alpha=0.75)
    ax.set_axisbelow(True)

    bars_normal = ax.bar(
        x - width / 2,
        [0.0 if v is None else float(v) for v in normal_vals],
        width=width,
        color=style["COLOR_OM"],
        edgecolor=style["EDGE_OM"],
        linewidth=0.9,
        label="Original",
        zorder=3,
    )
    bars_flip = ax.bar(
        x + width / 2,
        [0.0 if v is None else float(v) for v in flip_vals],
        width=width,
        color=style["COLOR_RIG"],
        edgecolor=style["EDGE_RIG"],
        linewidth=0.9,
        label="RIG",
        zorder=3,
    )

    finite = [v for v in normal_vals + flip_vals if v is not None and math.isfinite(float(v))]
    if finite:
        ymax = max(float(v) for v in finite)
        ax.set_ylim(0, ymax * 1.22 if ymax > 0 else 1)
    annotate_bars(ax, bars_normal, normal_vals)
    annotate_bars(ax, bars_flip, flip_vals)

    # Draw Δ arrows after y-limit is fixed, following the original Fig. 1 logic.
    for i, (normal_v, flip_v) in enumerate(zip(normal_vals, flip_vals)):
        if normal_v is not None and flip_v is not None and float(normal_v) >= float(flip_v):
            high_color = style["EDGE_OM"]
        else:
            high_color = style["EDGE_RIG"]

        draw_delta_arrow_pair_single_axis(
            ax,
            x[i] - width / 2,
            x[i] + width / 2,
            normal_v,
            flip_v,
            high_color=high_color,
        )
        
    ax.set_ylabel("Average NLL")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        frameon=True,
        framealpha=0.96,
        edgecolor="#E6E6E6",
        ncol=2,
    )
    save_fig(fig, out_dir / "reviewer_q1_fig1_compact_unlearned_recovery_nll", formats)
    plt.close(fig)

def plot_fig2(rows: list[dict[str, Any]], out_dir: Path, formats: list[str]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    style = setup_plot_style()
    groups = []
    for r in sorted(rows, key=lambda x: int(x.get("group_order") or 0)):
        g = r["group"]
        if g not in groups:
            groups.append(g)
    if not groups:
        return

    rg_vals = []
    rig_vals = []
    labels = []
    for g in groups:
        labels.append(GROUP_DISPLAY.get(g, METHOD_DISPLAY.get(g, g)))
        rg = next((r for r in rows if r["group"] == g and r["metric"] == "RG"), None)
        rig = next((r for r in rows if r["group"] == g and r["metric"] == "RIG"), None)
        rg_vals.append(safe_float(rg.get("avg_nll")) if rg else None)
        rig_vals.append(safe_float(rig.get("avg_nll")) if rig else None)

    x = np.arange(len(groups))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
    ax.set_facecolor("#FAFAFA")
    ax.yaxis.grid(True, alpha=0.75)
    ax.set_axisbelow(True)
    bars_rg = ax.bar(x - width/2, [0.0 if v is None else v for v in rg_vals], width=width, color=style["COLOR_RG"], edgecolor=style["EDGE_RG"], linewidth=0.9, label="RG", zorder=3)
    bars_rig = ax.bar(x + width/2, [0.0 if v is None else v for v in rig_vals], width=width, color=style["COLOR_RIG"], edgecolor=style["EDGE_RIG"], linewidth=0.9, label="RIG", zorder=3)
    annotate_bars(ax, bars_rg, rg_vals)
    annotate_bars(ax, bars_rig, rig_vals)
    ax.set_ylabel("Key-level AvgNLL")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=True, framealpha=0.96, edgecolor="#E6E6E6", ncol=2)
    finite = [v for v in rg_vals + rig_vals if v is not None]
    if finite:
        ymax = max(finite)
        ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)
    save_fig(fig, out_dir / "reviewer_q1_fig2_key_rg_rig_nll", formats)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [x.strip() for x in str(args.fig_format).split(",") if x.strip()]

    methods = parse_methods(args.methods)
    epoch_map = parse_method_epochs(args.method_epochs)
    lr_map = parse_method_lrs(args.method_lrs)
    full_rows = collect_full_recovery(args.recovery_root)
    key_rows = collect_key_recovery(args.key_recovery_root)

    fig1_rows = build_fig1_rows(full_rows, methods, epoch_map, lr_map, args.selected_epoch, args.split)
    fig2_rows = build_fig2_rows(key_rows, methods, epoch_map, lr_map, args.selected_epoch, args.split, args.constraint_scope)

    write_csv(out_dir / "reviewer_q1_fig1_full_answer_flip_nll_values.csv", fig1_rows, [
        "figure", "method", "method_display", "split", "epoch", "lr", "bar_order", "bar_label", "axis_label", "source_type", "avg_nll", "summary_path"
    ])
    write_csv(out_dir / "reviewer_q1_fig2_key_rg_rig_nll_values.csv", fig2_rows, [
        "figure", "group", "group_display", "split", "constraint_scope", "epoch", "lr", "group_order", "bar_order", "metric", "avg_nll", "mean_candidate_token_count", "summary_path"
    ])

    if args.make_plots:
        plot_dir = out_dir / "figures"
        plot_fig1(fig1_rows, plot_dir, formats, use_broken_axis=bool(args.broken_axis))
        plot_fig1_compact(fig1_rows, plot_dir, formats)
        plot_fig2(fig2_rows, plot_dir, formats)

    print(f"[reviewer-plots] full recovery rows: {len(full_rows)}")
    print(f"[reviewer-plots] key recovery rows: {len(key_rows)}")
    print(f"[reviewer-plots] fig1 rows: {len(fig1_rows)}")
    print(f"[reviewer-plots] fig2 rows: {len(fig2_rows)}")
    print(f"[reviewer-plots] method_epochs: {epoch_map}")
    print(f"[reviewer-plots] method_lrs: {lr_map}")
    print(f"[reviewer-plots] wrote {out_dir}")


if __name__ == "__main__":
    main()
