#!/usr/bin/env python3
"""Rebuild the paper's benchmark evidence directly from raw evaluation logs.

The primary metric is deliberately invariant to row order inside a synchronous
proposal batch.  All methods share 20 initialization calls.  Beginning at call
20, the incumbent is held fixed while a stored ``step``/``iteration`` batch is
evaluated and is updated only at the batch endpoint.  The completed-batch curve
is carried forward to a fixed horizon of 1,000 accepted unique oracle calls.
Its normalized trapezoidal area over calls 20--1,000 is reported as
``auc_batch_safe`` (and therefore has the same units as the oracle score).

For auditability, ``auc_rowwise`` is also written as a sensitivity calculation;
it updates after every CSV row and can depend on arbitrary ordering within a
jointly proposed batch.  All plots and rankings use the batch-safe quantity.

Outputs
-------
paper/figures/fig2_all8_curves.{pdf,png}
paper/figures/fig2_paired_auc.{pdf,png}
paper/figures/fig3_campaign_corrected.{pdf,png}
paper/figures/figS_locality_corrected.{pdf,png}
paper/verified_results/*.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator


RDLogger.DisableLog("rdApp.*")

SEEDS = (11, 22, 33, 44, 55)
FIRST_CALL = 20
HORIZON = 1000
GRID = np.arange(FIRST_CALL, HORIZON + 1, dtype=int)

TASKS = (
    ("penalized_logp", "Penalized logP"),
    ("qed", "QED"),
    ("median_2", "Median molecule 2"),
    ("osimertinib_mpo", "Osimertinib MPO"),
    ("scaffold_hop", "Scaffold hop"),
    ("deco_hop", "Decoration hop"),
    ("drd2", "DRD2"),
    ("jnk3", "JNK3"),
)

# id, paper label, color, dash, marker
METHODS = (
    ("trace_turbomol", "TRACE", "#1050C0", (0, ()), "o"),
    ("graph_ga", "Graph-GA", "#E65F39", (0, (4.0, 1.5)), "s"),
    ("smiles_ga", "SMILES-GA", "#1597A5", (0, (1.2, 1.3)), "^"),
    ("gpbo_graph_ga", "GP-BO + Graph-GA", "#A00070", (0, (5.0, 1.4, 1.0, 1.4)), "D"),
    ("stoned_selfies", "STONED-SELFIES", "#8A63D2", (0, (3.0, 1.2, 1.0, 1.2)), "v"),
    ("reinvent4_external", "REINVENT4", "#6B6900", (0, (6.0, 1.5, 1.5, 1.5)), "P"),
    ("lolbo_frozen_tr_scaled", "Frozen-VAE BO", "#7B19E6", (0, (2.4, 1.2)), "X"),
)

TASK_LABEL = dict(TASKS)
METHOD_LABEL = {m: label for m, label, *_ in METHODS}
METHOD_STYLE = {m: (color, dash, marker) for m, _, color, dash, marker in METHODS}
TRACE = "trace_turbomol"
UNPAIRED_COMPARATORS = {"reinvent4_external"}

INK = "#1A1A1A"
INK2 = "#4D4D4D"
MUTED = "#767676"
GRID_COLOR = "#E5E5E5"
AXIS = "#B5B5B5"
SURFACE = "#FFFFFF"


@dataclass(frozen=True)
class EvalRow:
    call: int
    phase: str
    batch: int
    smiles: str
    score: float


@dataclass
class RunResult:
    task: str
    method: str
    seed: int
    n_calls: int
    batch_curve: np.ndarray
    row_curve: np.ndarray
    auc_batch_safe: float
    auc_rowwise: float
    final_best: float
    locality_median: float
    locality_mean: float
    locality_frac_04: float
    locality_n: int


def apply_style(base_pt: float = 7.2) -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
            "font.size": base_pt,
            "axes.labelsize": base_pt + 0.4,
            "axes.titlesize": base_pt + 0.8,
            "xtick.labelsize": base_pt - 0.4,
            "ytick.labelsize": base_pt - 0.4,
            "legend.fontsize": base_pt - 0.2,
            "text.color": INK,
            "axes.titlecolor": INK,
            "axes.labelcolor": INK2,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axis(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(pad=2.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def save_figure(fig: mpl.figure.Figure, stem: Path, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=dpi)


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _batch_value(raw: Mapping[str, str]) -> int:
    for key in ("iteration", "step", "generation"):
        value = raw.get(key, "")
        if value not in (None, ""):
            return int(float(value))
    raise ValueError(f"No iteration/step/generation field in row with keys {tuple(raw)}")


def load_rows(path: Path) -> list[EvalRow]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError(f"Empty evaluation file: {path}")

    rows = [
        EvalRow(
            call=int(float(raw["oracle_call"])),
            phase=raw.get("phase", "").strip().lower(),
            batch=_batch_value(raw),
            smiles=raw["smiles"].strip(),
            score=float(raw["score"]),
        )
        for raw in raw_rows
    ]
    rows.sort(key=lambda row: row.call)
    calls = [row.call for row in rows]
    smiles = [row.smiles for row in rows]
    if len(calls) != len(set(calls)):
        raise ValueError(f"Repeated oracle_call in {path}")
    if calls != list(range(1, max(calls) + 1)):
        raise ValueError(f"Non-contiguous oracle calls in {path}")
    if len(smiles) != len(set(smiles)):
        raise ValueError(f"Repeated accepted molecule in {path}")
    if max(calls) > HORIZON:
        raise ValueError(f"Run exceeds fixed horizon {HORIZON}: {path}")
    return rows


def split_batches(rows: Sequence[EvalRow]) -> tuple[list[EvalRow], list[list[EvalRow]]]:
    init = [row for row in rows if row.phase in {"init", "initial"} or row.call <= FIRST_CALL]
    if len(init) != FIRST_CALL or {row.call for row in init} != set(range(1, FIRST_CALL + 1)):
        raise ValueError(f"Expected exactly {FIRST_CALL} shared initialization rows")

    grouped: dict[int, list[EvalRow]] = defaultdict(list)
    for row in rows:
        if row.call > FIRST_CALL and row.phase not in {"init", "initial"}:
            grouped[row.batch].append(row)
    batches = [sorted(grouped[key], key=lambda row: row.call) for key in sorted(grouped)]
    if not batches:
        raise ValueError("No proposal batches after initialization")

    previous_endpoint = FIRST_CALL
    for batch in batches:
        if batch[0].call <= previous_endpoint:
            raise ValueError("Stored batches overlap or are out of call order")
        previous_endpoint = batch[-1].call
    return init, batches


def curve_from_events(events: Sequence[tuple[int, float]]) -> np.ndarray:
    event_calls = np.asarray([call for call, _ in events], dtype=int)
    event_values = np.asarray([value for _, value in events], dtype=float)
    positions = np.searchsorted(event_calls, GRID, side="right") - 1
    if np.any(positions < 0):
        raise ValueError("Missing incumbent event at the start of the integration grid")
    return event_values[positions]


def fixed_horizon_curves(rows: Sequence[EvalRow]) -> tuple[np.ndarray, np.ndarray]:
    init, batches = split_batches(rows)
    incumbent = max(row.score for row in init)

    # Primary curve: no credit for a jointly proposed batch until its endpoint.
    batch_events: list[tuple[int, float]] = [(FIRST_CALL, incumbent)]
    for batch in batches:
        incumbent = max(incumbent, max(row.score for row in batch))
        batch_events.append((batch[-1].call, incumbent))
    batch_curve = curve_from_events(batch_events)

    # Sensitivity curve: conventional rowwise best-so-far in stored CSV order.
    incumbent = max(row.score for row in init)
    row_events: list[tuple[int, float]] = [(FIRST_CALL, incumbent)]
    for row in (row for row in rows if row.call > FIRST_CALL):
        incumbent = max(incumbent, row.score)
        row_events.append((row.call, incumbent))
    row_curve = curve_from_events(row_events)
    return batch_curve, row_curve


def normalized_auc(curve: np.ndarray) -> float:
    return float(np.trapezoid(curve, GRID) / float(HORIZON - FIRST_CALL))


FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
FP_CACHE: dict[str, object | None] = {}


def fingerprint(smiles: str):
    if smiles not in FP_CACHE:
        mol = Chem.MolFromSmiles(smiles)
        FP_CACHE[smiles] = FP_GENERATOR.GetFingerprint(mol) if mol is not None else None
    return FP_CACHE[smiles]


def batch_safe_locality(rows: Sequence[EvalRow]) -> tuple[float, float, float, int]:
    """Similarity to the pre-batch incumbent; initialization is excluded."""
    init, batches = split_batches(rows)
    incumbent_row = max(init, key=lambda row: row.score)
    incumbent_score = incumbent_row.score
    incumbent_fp = fingerprint(incumbent_row.smiles)
    similarities: list[float] = []

    for batch in batches:
        if incumbent_fp is not None:
            for row in batch:
                candidate_fp = fingerprint(row.smiles)
                if candidate_fp is not None:
                    similarities.append(float(DataStructs.TanimotoSimilarity(candidate_fp, incumbent_fp)))
        batch_winner = max(batch, key=lambda row: row.score)
        if batch_winner.score > incumbent_score:
            incumbent_row = batch_winner
            incumbent_score = batch_winner.score
            incumbent_fp = fingerprint(incumbent_row.smiles)

    if not similarities:
        return np.nan, np.nan, np.nan, 0
    values = np.asarray(similarities, dtype=float)
    return (
        float(np.median(values)),
        float(np.mean(values)),
        float(np.mean(values >= 0.4)),
        int(values.size),
    )


def collect(raw_root: Path) -> dict[tuple[str, str, int], RunResult]:
    results: dict[tuple[str, str, int], RunResult] = {}
    missing: list[str] = []
    for task, _ in TASKS:
        for method, *_ in METHODS:
            for seed in SEEDS:
                path = raw_root / task / method / f"seed_{seed}" / f"evaluations_seed_{seed}.csv"
                if not path.exists():
                    missing.append(str(path))
                    continue
                rows = load_rows(path)
                batch_curve, row_curve = fixed_horizon_curves(rows)
                loc_median, loc_mean, loc_frac, loc_n = batch_safe_locality(rows)
                results[(task, method, seed)] = RunResult(
                    task=task,
                    method=method,
                    seed=seed,
                    n_calls=len(rows),
                    batch_curve=batch_curve,
                    row_curve=row_curve,
                    auc_batch_safe=normalized_auc(batch_curve),
                    auc_rowwise=normalized_auc(row_curve),
                    final_best=float(batch_curve[-1]),
                    locality_median=loc_median,
                    locality_mean=loc_mean,
                    locality_frac_04=loc_frac,
                    locality_n=loc_n,
                )
    if missing:
        formatted = "\n".join(missing)
        raise FileNotFoundError(f"Incomplete 8 x 7 x 5 benchmark; missing:\n{formatted}")
    expected = len(TASKS) * len(METHODS) * len(SEEDS)
    if len(results) != expected:
        raise AssertionError(f"Expected {expected} runs, found {len(results)}")
    return results


def bootstrap_mean_ci(values: np.ndarray, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    draws = values[idx].mean(axis=1)
    lo, hi = np.percentile(draws, (2.5, 97.5))
    return float(lo), float(hi)


def bootstrap_curve_ci(curves: np.ndarray, reps: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.integers(0, curves.shape[0], size=(reps, curves.shape[0]))
    draws = curves[idx].mean(axis=1)
    lo, hi = np.percentile(draws, (2.5, 97.5), axis=0)
    return lo, hi


def average_ranks_desc(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and np.isclose(ordered[j][1], ordered[i][1], rtol=1e-12, atol=1e-14):
            j += 1
        rank = float(np.mean(np.arange(i + 1, j + 1)))
        for method, _ in ordered[i:j]:
            ranks[method] = rank
        i = j
    return ranks


def write_auc_tables(
    results: Mapping[tuple[str, str, int], RunResult],
    out_dir: Path,
    reps: int,
    rng: np.random.Generator,
) -> tuple[dict[str, str], list[dict[str, object]]]:
    by_seed_rows = []
    summary_rows = []
    task_rank_rows = []
    ranks_by_method: dict[str, list[float]] = defaultdict(list)
    task_wins: dict[str, int] = defaultdict(int)

    for task, _ in TASKS:
        task_means: dict[str, float] = {}
        for method, *_ in METHODS:
            runs = [results[(task, method, seed)] for seed in SEEDS]
            batch_values = np.asarray([run.auc_batch_safe for run in runs])
            row_values = np.asarray([run.auc_rowwise for run in runs])
            final_values = np.asarray([run.final_best for run in runs])
            lo, hi = bootstrap_mean_ci(batch_values, reps, rng)
            task_means[method] = float(batch_values.mean())
            summary_rows.append(
                (
                    task,
                    TASK_LABEL[task],
                    method,
                    METHOD_LABEL[method],
                    batch_values.mean(),
                    batch_values.std(ddof=1),
                    lo,
                    hi,
                    row_values.mean(),
                    row_values.mean() - batch_values.mean(),
                    final_values.mean(),
                    final_values.std(ddof=1),
                    np.mean([run.n_calls for run in runs]),
                )
            )
            for run in runs:
                by_seed_rows.append(
                    (
                        task,
                        TASK_LABEL[task],
                        method,
                        METHOD_LABEL[method],
                        run.seed,
                        run.n_calls,
                        run.auc_batch_safe,
                        run.auc_rowwise,
                        run.auc_rowwise - run.auc_batch_safe,
                        run.final_best,
                    )
                )

        task_ranks = average_ranks_desc(task_means)
        for method, *_ in METHODS:
            ranks_by_method[method].append(task_ranks[method])
            if np.isclose(task_ranks[method], 1.0):
                task_wins[method] += 1
            task_rank_rows.append(
                (
                    task,
                    TASK_LABEL[task],
                    method,
                    METHOD_LABEL[method],
                    task_means[method],
                    task_ranks[method],
                )
            )

    mean_rank_rows = []
    for method, *_ in METHODS:
        ranks = np.asarray(ranks_by_method[method])
        mean_rank_rows.append(
            (
                method,
                METHOD_LABEL[method],
                ranks.mean(),
                task_wins[method],
                np.mean(ranks <= 3.0),
            )
        )
    mean_rank_rows.sort(key=lambda row: row[2])

    write_csv(
        out_dir / "auc_by_seed.csv",
        (
            "task",
            "task_label",
            "method_id",
            "method_label",
            "seed",
            "accepted_unique_calls",
            "auc_batch_safe_20_1000",
            "auc_rowwise_20_1000",
            "rowwise_minus_batch_safe",
            "final_top1_at_1000_locf",
        ),
        by_seed_rows,
    )
    write_csv(
        out_dir / "auc_summary.csv",
        (
            "task",
            "task_label",
            "method_id",
            "method_label",
            "mean_auc_batch_safe_20_1000",
            "sd_auc_batch_safe",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "mean_auc_rowwise_20_1000",
            "mean_rowwise_minus_batch_safe",
            "mean_final_top1_at_1000_locf",
            "sd_final_top1",
            "mean_accepted_unique_calls",
        ),
        summary_rows,
    )
    write_csv(
        out_dir / "task_mean_auc_and_ranks.csv",
        ("task", "task_label", "method_id", "method_label", "mean_auc_batch_safe", "task_rank"),
        task_rank_rows,
    )
    write_csv(
        out_dir / "mean_ranks.csv",
        ("method_id", "method_label", "mean_rank_across_8_tasks", "task_wins", "fraction_tasks_top3"),
        mean_rank_rows,
    )

    best_baseline: dict[str, str] = {}
    comparison_summary: list[dict[str, object]] = []
    comparison_by_run = []
    for task, _ in TASKS:
        baseline_means = {
            method: np.mean([results[(task, method, seed)].auc_batch_safe for seed in SEEDS])
            for method, *_ in METHODS
            if method != TRACE
        }
        comparator = max(baseline_means, key=baseline_means.get)
        best_baseline[task] = comparator
        trace = np.asarray([results[(task, TRACE, seed)].auc_batch_safe for seed in SEEDS])
        baseline = np.asarray([results[(task, comparator, seed)].auc_batch_safe for seed in SEEDS])
        deltas = trace - baseline
        paired = comparator not in UNPAIRED_COMPARATORS
        comparison_design = "paired-seed" if paired else "independent-run"
        if paired:
            idx = rng.integers(0, len(SEEDS), size=(reps, len(SEEDS)))
            raw_draws = deltas[idx].mean(axis=1)
            base_draws = baseline[idx].mean(axis=1)
        else:
            trace_idx = rng.integers(0, len(SEEDS), size=(reps, len(SEEDS)))
            baseline_idx = rng.integers(0, len(SEEDS), size=(reps, len(SEEDS)))
            raw_draws = trace[trace_idx].mean(axis=1) - baseline[baseline_idx].mean(axis=1)
            base_draws = baseline[baseline_idx].mean(axis=1)
        relative_draws = 100.0 * raw_draws / np.maximum(np.abs(base_draws), 1e-12)
        raw_lo, raw_hi = np.percentile(raw_draws, (2.5, 97.5))
        rel_lo, rel_hi = np.percentile(relative_draws, (2.5, 97.5))
        relative = 100.0 * deltas.mean() / max(abs(baseline.mean()), 1e-12)
        comparison_summary.append(
            {
                "task": task,
                "task_label": TASK_LABEL[task],
                "comparator": comparator,
                "comparator_label": METHOD_LABEL[comparator],
                "comparison_design": comparison_design,
                "trace_mean_auc": float(trace.mean()),
                "comparator_mean_auc": float(baseline.mean()),
                "mean_delta_raw": float(deltas.mean()),
                "ci_low_raw": float(raw_lo),
                "ci_high_raw": float(raw_hi),
                "relative_delta_percent": float(relative),
                "relative_ci_low_percent": float(rel_lo),
                "relative_ci_high_percent": float(rel_hi),
                "trace_seed_wins": int(np.sum(deltas > 0)) if paired else "",
                "ties": int(np.sum(np.isclose(deltas, 0.0))) if paired else "",
            }
        )
        for seed, trace_value, baseline_value, delta in zip(SEEDS, trace, baseline, deltas):
            comparison_by_run.append(
                (
                    task,
                    TASK_LABEL[task],
                    comparator,
                    METHOD_LABEL[comparator],
                    comparison_design,
                    seed,
                    trace_value,
                    baseline_value,
                    delta if paired else "",
                )
            )

    write_csv(
        out_dir / "paired_trace_vs_best_baseline.csv",
        (
            "task",
            "task_label",
            "strongest_baseline_id",
            "strongest_baseline_label",
            "comparison_design",
            "trace_mean_auc",
            "baseline_mean_auc",
            "mean_delta_raw",
            "bootstrap_ci_low_raw",
            "bootstrap_ci_high_raw",
            "relative_delta_percent_of_baseline_auc",
            "bootstrap_ci_low_percent",
            "bootstrap_ci_high_percent",
            "trace_seed_wins_when_paired",
            "ties_when_paired",
        ),
        [
            (
                row["task"],
                row["task_label"],
                row["comparator"],
                row["comparator_label"],
                row["comparison_design"],
                row["trace_mean_auc"],
                row["comparator_mean_auc"],
                row["mean_delta_raw"],
                row["ci_low_raw"],
                row["ci_high_raw"],
                row["relative_delta_percent"],
                row["relative_ci_low_percent"],
                row["relative_ci_high_percent"],
                row["trace_seed_wins"],
                row["ties"],
            )
            for row in comparison_summary
        ],
    )
    write_csv(
        out_dir / "paired_trace_vs_best_baseline_by_seed.csv",
        (
            "task",
            "task_label",
            "strongest_baseline_id",
            "strongest_baseline_label",
            "comparison_design",
            "run_label",
            "trace_auc_batch_safe",
            "baseline_auc_batch_safe",
            "aligned_delta_raw_when_paired",
        ),
        comparison_by_run,
    )
    return best_baseline, comparison_summary


def write_curve_table_and_plot(
    results: Mapping[tuple[str, str, int], RunResult],
    out_data: Path,
    out_figures: Path,
    reps: int,
    rng: np.random.Generator,
    dpi: int,
) -> None:
    apply_style(6.7)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.55), sharex=True)
    curve_rows = []
    present_styles = []

    for task_index, ((task, task_label), ax) in enumerate(zip(TASKS, axes.flat)):
        style_axis(ax)
        for method, label, color, dash, marker in METHODS:
            curves = np.vstack([results[(task, method, seed)].batch_curve for seed in SEEDS])
            mean = curves.mean(axis=0)
            lo, hi = bootstrap_curve_ci(curves, reps, rng)
            emphasis = method == TRACE
            ax.fill_between(GRID, lo, hi, color=color, alpha=0.10 if emphasis else 0.055, linewidth=0)
            ax.plot(
                GRID,
                mean,
                color=color,
                linestyle=dash,
                linewidth=1.45 if emphasis else 0.88,
                marker=marker,
                markevery=(0, 245),
                markersize=2.3 if emphasis else 1.9,
                markeredgewidth=0.35,
                markeredgecolor=SURFACE,
                zorder=6 if emphasis else 4,
            )
            if task_index == 0:
                present_styles.append((method, label, color, dash, marker))
            for call, centre, lower, upper in zip(GRID, mean, lo, hi):
                curve_rows.append((task, task_label, method, label, call, centre, lower, upper))
        ax.set_title(task_label, fontweight="semibold", pad=3.0)
        ax.set_xlim(FIRST_CALL, HORIZON)
        if task_index % 4 == 0:
            ax.set_ylabel("Top-1 score")
        if task_index >= 4:
            ax.set_xlabel("Accepted unique calls")

    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linestyle=dash,
            linewidth=1.45 if method == TRACE else 0.95,
            marker=marker,
            markersize=2.7,
            markeredgewidth=0.35,
            markeredgecolor=SURFACE,
        )
        for method, _, color, dash, marker in present_styles
    ]
    labels = [label for _, label, *_ in present_styles]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.003),
        ncol=4,
        frameon=False,
        handlelength=2.4,
        handletextpad=0.5,
        columnspacing=1.25,
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.965, bottom=0.155, wspace=0.31, hspace=0.30)
    save_figure(fig, out_figures / "fig2_all8_curves", dpi)
    plt.close(fig)

    write_csv(
        out_data / "fixed_horizon_curve_summary.csv",
        ("task", "task_label", "method_id", "method_label", "oracle_call", "mean_top1", "ci_low", "ci_high"),
        curve_rows,
    )


def plot_auc_deltas(comparison_summary: Sequence[Mapping[str, object]], out_figures: Path, dpi: int) -> None:
    apply_style(7.4)
    fig, ax = plt.subplots(figsize=(7.0, 2.85))
    ax.axvline(0.0, color=AXIS, linewidth=0.8, zorder=1)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)

    y_positions = np.arange(len(comparison_summary))[::-1]
    for y, row in zip(y_positions, comparison_summary):
        centre = float(row["relative_delta_percent"])
        lo = float(row["relative_ci_low_percent"])
        hi = float(row["relative_ci_high_percent"])
        color = "#1050C0" if centre >= 0 else "#C64E37"
        ax.plot([lo, hi], [y, y], color=color, linewidth=1.8, alpha=0.62, zorder=3)
        ax.plot(
            centre,
            y,
            marker="o",
            color=color,
            markersize=4.7,
            markeredgewidth=0.6,
            markeredgecolor=SURFACE,
            linestyle="none",
            zorder=5,
        )
        ax.annotate(
            f"{centre:+.1f}%",
            (centre, y),
            xytext=(4 if centre >= 0 else -4, 0),
            textcoords="offset points",
            ha="left" if centre >= 0 else "right",
            va="center",
            fontsize=6.6,
            color=INK2,
        )

    labels = [f"{row['task_label']}  ({row['comparator_label']})" for row in comparison_summary]
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("TRACE − strongest-baseline AUC (% of baseline AUC)")
    ax.set_title("Batch-safe AUC; paired bootstrap when valid, independent bootstrap for REINVENT4", pad=5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    fig.subplots_adjust(left=0.285, right=0.985, top=0.90, bottom=0.20)
    save_figure(fig, out_figures / "fig2_paired_auc", dpi)
    plt.close(fig)


def write_locality_tables_and_plot(
    results: Mapping[tuple[str, str, int], RunResult],
    out_data: Path,
    out_figures: Path,
    dpi: int,
) -> None:
    by_seed_rows = []
    summary_rows = []
    matrix = np.full((len(TASKS), len(METHODS)), np.nan)
    for task_index, (task, task_label) in enumerate(TASKS):
        for method_index, (method, method_label, *_rest) in enumerate(METHODS):
            runs = [results[(task, method, seed)] for seed in SEEDS]
            medians = np.asarray([run.locality_median for run in runs])
            means = np.asarray([run.locality_mean for run in runs])
            fractions = np.asarray([run.locality_frac_04 for run in runs])
            matrix[task_index, method_index] = medians.mean()
            summary_rows.append(
                (
                    task,
                    task_label,
                    method,
                    method_label,
                    medians.mean(),
                    medians.std(ddof=1),
                    means.mean(),
                    fractions.mean(),
                    sum(run.locality_n for run in runs),
                )
            )
            for run in runs:
                by_seed_rows.append(
                    (
                        task,
                        task_label,
                        method,
                        method_label,
                        run.seed,
                        run.locality_median,
                        run.locality_mean,
                        run.locality_frac_04,
                        run.locality_n,
                    )
                )

    write_csv(
        out_data / "batch_safe_locality_by_seed.csv",
        (
            "task",
            "task_label",
            "method_id",
            "method_label",
            "seed",
            "median_tanimoto_to_prebatch_incumbent",
            "mean_tanimoto_to_prebatch_incumbent",
            "fraction_ge_0.4",
            "proposal_molecules",
        ),
        by_seed_rows,
    )
    write_csv(
        out_data / "batch_safe_locality_summary.csv",
        (
            "task",
            "task_label",
            "method_id",
            "method_label",
            "mean_of_seed_medians",
            "sd_of_seed_medians",
            "mean_of_seed_means",
            "mean_fraction_ge_0.4",
            "total_proposal_molecules",
        ),
        summary_rows,
    )

    display = np.vstack([matrix, matrix.mean(axis=0, keepdims=True)])
    apply_style(7.0)
    fig, ax = plt.subplots(figsize=(7.15, 3.45))
    image = ax.imshow(display, cmap="Blues", vmin=0.0, vmax=max(0.8, float(np.nanmax(display))), aspect="auto")
    method_labels = [label.replace(" + ", "+\n") for _, label, *_ in METHODS]
    task_labels = [label for _, label in TASKS] + ["Mean"]
    ax.set_xticks(np.arange(len(METHODS)), method_labels)
    ax.set_yticks(np.arange(len(task_labels)), task_labels)
    ax.tick_params(length=0)
    ax.axhline(len(TASKS) - 0.5, color=SURFACE, linewidth=1.7)
    threshold = 0.52 * float(np.nanmax(display))
    for row in range(display.shape[0]):
        row_max = int(np.nanargmax(display[row]))
        for col in range(display.shape[1]):
            value = display[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=SURFACE if value >= threshold else INK,
                fontsize=6.5,
                fontweight="bold" if col == row_max else "normal",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.018)
    colorbar.outline.set_visible(False)
    colorbar.set_label("Median Tanimoto", rotation=270, labelpad=10)
    ax.set_title(
        "Structural locality to the pre-batch incumbent (initialization excluded)",
        fontweight="semibold",
        pad=5,
    )
    fig.subplots_adjust(left=0.17, right=0.94, top=0.89, bottom=0.19)
    save_figure(fig, out_figures / "figS_locality_corrected", dpi)
    plt.close(fig)


def campaign_corrected_plot(workspace: Path, out_data: Path, out_figures: Path, dpi: int) -> None:
    """Rebuild the FePc panel with round-safe incumbents and locality.

    The committed campaign table has one seed and ten finite values in each of
    eight rounds (81 finite stored energies total).  The best-so-far trace and
    structural incumbent both update only after all ten members of a round.
    """
    source = workspace / "FePC Case Study" / "figures" / "campaign_evaluations.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 81:
        raise ValueError(f"Expected 81 stored campaign evaluations, found {len(rows)}")

    campaign: list[dict[str, object]] = []
    for raw in rows:
        energy = float(raw["energy"])
        if not np.isfinite(energy):
            raise ValueError("Committed campaign table no longer has 81 finite energies")
        fp = fingerprint(raw["smiles"])
        if fp is None:
            raise ValueError(f"Campaign SMILES did not parse: ID {raw['id']}")
        campaign.append(
            {
                "round": int(raw["round"]),
                "id": int(raw["id"]),
                "smiles": raw["smiles"],
                "energy": energy,
                "fp": fp,
            }
        )
    campaign.sort(key=lambda row: int(row["id"]))
    seed = campaign[0]
    if int(seed["round"]) != 0:
        raise ValueError("Campaign seed is not the first round-0 row")

    incumbent = seed
    incumbent_energy = float(seed["energy"])
    best_events = [(1, incumbent_energy)]
    corrected_rows = []
    round_rows = []
    all_locality = []
    for round_id in range(1, 9):
        batch = [row for row in campaign if int(row["round"]) == round_id]
        if len(batch) != 10:
            raise ValueError(f"Expected ten campaign candidates in round {round_id}")
        similarities = [
            float(DataStructs.TanimotoSimilarity(row["fp"], incumbent["fp"])) for row in batch
        ]
        seed_similarities = [
            float(DataStructs.TanimotoSimilarity(row["fp"], seed["fp"])) for row in batch
        ]
        all_locality.extend(similarities)
        for row, similarity, seed_similarity in zip(batch, similarities, seed_similarities):
            corrected_rows.append(
                (
                    round_id,
                    row["id"],
                    row["energy"],
                    incumbent["id"],
                    incumbent_energy,
                    similarity,
                    seed_similarity,
                )
            )
        round_winner = max(batch, key=lambda row: float(row["energy"]))
        round_best = float(round_winner["energy"])
        if round_best > incumbent_energy:
            incumbent = round_winner
            incumbent_energy = round_best
        endpoint = max(int(row["id"]) for row in batch) + 1
        best_events.append((endpoint, incumbent_energy))
        round_rows.append(
            (
                round_id,
                endpoint,
                round_best,
                incumbent_energy,
                float(np.median(similarities)),
                float(np.median(seed_similarities)),
                float(np.mean(np.asarray(similarities) >= 0.4)),
            )
        )

    write_csv(
        out_data / "campaign_corrected_evaluations.csv",
        (
            "round",
            "candidate_id",
            "energy_ev",
            "pre_round_incumbent_id",
            "pre_round_incumbent_energy_ev",
            "tanimoto_to_pre_round_incumbent",
            "tanimoto_to_seed",
        ),
        corrected_rows,
    )
    write_csv(
        out_data / "campaign_corrected_rounds.csv",
        (
            "round",
            "cumulative_stored_dft_evaluations",
            "round_best_energy_ev",
            "best_so_far_after_round_ev",
            "median_tanimoto_to_pre_round_incumbent",
            "median_tanimoto_to_seed",
            "fraction_tanimoto_ge_0.4",
        ),
        round_rows,
    )
    write_csv(
        out_data / "campaign_source_audit.csv",
        ("source", "stored_rows", "finite_energies", "seed_rows", "candidate_rows", "failed_or_missing_energies"),
        [(str(source.relative_to(workspace)), len(campaign), len(campaign), 1, len(campaign) - 1, 0)],
    )

    apply_style(7.2)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.0, 2.35))

    floor, ceiling = -4.5, 2.9
    round_colors = plt.cm.Blues(np.linspace(0.40, 0.90, 8))
    calls = np.arange(1, len(campaign) + 1)
    energies = np.asarray([float(row["energy"]) for row in campaign])
    for round_id in range(1, 9):
        mask = np.asarray([int(row["round"]) == round_id for row in campaign])
        clipped = np.maximum(energies[mask], floor + 0.12)
        below = energies[mask] < floor
        ax_left.scatter(
            calls[mask][~below],
            clipped[~below],
            s=9,
            color=round_colors[round_id - 1],
            edgecolor=SURFACE,
            linewidth=0.3,
            zorder=3,
        )
        if np.any(below):
            ax_left.scatter(
                calls[mask][below],
                clipped[below],
                s=14,
                marker="v",
                color=round_colors[round_id - 1],
                edgecolor=SURFACE,
                linewidth=0.3,
                zorder=4,
            )
        if round_id > 1:
            ax_left.axvline(calls[mask][0] - 0.5, color=GRID_COLOR, linewidth=0.45, zorder=1)

    event_calls = np.asarray([call for call, _ in best_events])
    event_values = np.asarray([value for _, value in best_events])
    campaign_grid = np.arange(1, len(campaign) + 1)
    positions = np.searchsorted(event_calls, campaign_grid, side="right") - 1
    round_safe_best = event_values[positions]
    ax_left.step(campaign_grid, round_safe_best, where="post", color="#1050C0", linewidth=1.45, zorder=5)
    ax_left.scatter([1], [float(seed["energy"])], marker="*", s=35, color=INK, zorder=6)
    ax_left.annotate("FePc seed\n0.42 eV", (1, float(seed["energy"])), xytext=(6, -13),
                     textcoords="offset points", fontsize=6.1, color=INK)
    ax_left.annotate("2.59 eV", (81, 2.5912), xytext=(-3, 4), textcoords="offset points",
                     ha="right", va="bottom", fontsize=6.5, fontweight="bold", color="#1050C0")
    ax_left.annotate("−14.66 eV", (3, floor + 0.12), xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=5.8, color=MUTED)
    style_axis(ax_left)
    ax_left.set_xlim(0, 82)
    ax_left.set_ylim(floor, ceiling)
    ax_left.set_xlabel("Stored DFT evaluations")
    ax_left.set_ylabel("Computed energy (eV)")
    ax_left.set_title("(a) round-safe optimization trajectory", loc="left", fontweight="semibold", pad=4)

    round_ids = np.arange(1, 9)
    locality = np.asarray([float(row[4]) for row in round_rows])
    seed_similarity = np.asarray([float(row[5]) for row in round_rows])
    ax_right.fill_between(round_ids, seed_similarity, locality, color="#1050C0", alpha=0.09, linewidth=0)
    ax_right.plot(round_ids, locality, color="#1050C0", linewidth=1.45, marker="o", markersize=3.3,
                  markeredgecolor=SURFACE, markeredgewidth=0.4, label="pre-round incumbent")
    ax_right.plot(round_ids, seed_similarity, color=INK2, linewidth=1.05, linestyle=(0, (3.5, 1.8)),
                  marker="s", markersize=3.0, markeredgecolor=SURFACE, markeredgewidth=0.4,
                  label="FePc seed")
    style_axis(ax_right)
    ax_right.set_xlim(0.7, 8.3)
    ax_right.set_ylim(0.1, 0.78)
    ax_right.set_xticks(round_ids)
    ax_right.set_xlabel("Synchronous round")
    ax_right.set_ylabel("Median Tanimoto")
    ax_right.set_title("(b) local steps accumulate global drift", loc="left", fontweight="semibold", pad=4)
    ax_right.legend(loc="lower left", fontsize=6.6, handlelength=2.2)
    ax_right.text(
        0.98,
        0.05,
        f"campaign median = {np.median(all_locality):.2f};  {np.mean(np.asarray(all_locality) >= 0.4):.0%} ≥ 0.4",
        transform=ax_right.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
        color=INK2,
    )

    fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.19, wspace=0.28)
    save_figure(fig, out_figures / "fig3_campaign_corrected", dpi)
    plt.close(fig)


def write_readme(out_dir: Path) -> None:
    path = out_dir / "README.txt"
    text = (
        "Verified analysis generated by paper/verified_analysis.py\n"
        "\n"
        "Primary AUC definition:\n"
        "  - exactly five stored seeds (11, 22, 33, 44, 55) for each of eight tasks\n"
        "    and seven methods;\n"
        "  - 20 common initialization calls;\n"
        "  - incumbent updates only at a stored step/iteration batch endpoint, so row\n"
        "    order inside a synchronous batch cannot affect the result;\n"
        "  - last-observation-carried-forward through a fixed 1,000-call horizon; and\n"
        "  - trapezoidal area over calls 20--1,000 divided by 980.\n"
        "\n"
        "auc_rowwise is an explicitly labeled sensitivity metric.  It updates after\n"
        "each stored proposal row and is not used in figures, comparator selection, or\n"
        "rankings. Comparisons use paired resampling when controlled seeds are shared.\n"
        "REINVENT4 has repeated but unseeded runs and is therefore independently\n"
        "resampled. Strongest baselines are selected task-by-task by mean primary AUC.\n"
        "The comparison CSV filenames retain their historical paired_ prefix for\n"
        "compatibility; the comparison_design column is authoritative.\n"
        "Mean ranks first rank method means within each task and then average the eight\n"
        "task ranks.\n"
        "\n"
        "Locality definition:\n"
        "  Morgan radius-2, 2,048-bit Tanimoto similarity of every proposal to the\n"
        "  incumbent before its entire stored batch; the 20 initialization molecules\n"
        "  are excluded. The incumbent is updated only after the batch completes.\n"
        "\n"
        "Campaign source audit:\n"
        "  The committed campaign CSV has 81 stored rows and 81 finite energies: one\n"
        "  seed plus 80 candidates. Campaign locality and the displayed incumbent are\n"
        "  updated only after each ten-candidate round.\n"
    )
    # Keep ordinary output creation concentrated in this script; source-code edits are
    # made with apply_patch, while generated analysis artifacts are script outputs.
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_workspace = Path(__file__).resolve().parents[1]
    parser.add_argument("--raw-root", type=Path, default=default_workspace / "results")
    parser.add_argument("--figure-dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "verified_results")
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--rng-seed", type=int, default=20260821)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    figure_dir = args.figure_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)

    print(f"Reading raw runs from {raw_root}")
    results = collect(raw_root)
    print(f"Validated {len(results)} complete, unique-call trajectories")

    write_curve_table_and_plot(results, output_dir, figure_dir, args.bootstrap_reps, rng, args.dpi)
    _, comparison_summary = write_auc_tables(results, output_dir, args.bootstrap_reps, rng)
    plot_auc_deltas(comparison_summary, figure_dir, args.dpi)
    write_locality_tables_and_plot(results, output_dir, figure_dir, args.dpi)
    campaign_corrected_plot(default_workspace, output_dir, figure_dir, args.dpi)
    write_readme(output_dir)

    print(f"Wrote verified tables to {output_dir}")
    print(f"Wrote publication figures to {figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
