#!/usr/bin/env python3
"""Top-1 and top-10 AUC from the raw evaluation logs, on the paper's protocol.

Reuses ``verified_analysis``'s loader and batch-endpoint conventions verbatim,
so the top-1 column here reproduces Table 1 and the top-10 column is the PMO
metric (mean of the ten best molecules found so far) measured the same way.

For each run the incumbent set is held fixed while a synchronous batch is
evaluated and refreshed only at the batch endpoint; the resulting step curve is
integrated on the dense per-call grid from call 20 to 1,000 and divided by 980,
so both AUCs carry the units of the oracle score.  ``*_rowwise`` variants update
after every stored CSV row and are written only as a sensitivity check.

Outputs (paper/verified_results/topk/):
  topk_auc_by_seed.csv        one row per task x method x seed
  topk_auc_task_means.csv     task means and within-task ranks
  topk_auc_mean_ranks.csv     mean rank over the 30 oracle-seed cells, both metrics
  topk_auc_vs_best_baseline.csv  TRACE vs the strongest baseline per task
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from verified_analysis import (
    FIRST_CALL,
    GRID,
    HORIZON,
    METHOD_LABEL,
    METHODS,
    SEEDS,
    TASK_LABEL,
    TASKS,
    curve_from_events,
    load_rows,
    split_batches,
)

TOP_K = 10
# The six objectives of Table 1, in table order.
PAPER_TASKS = ("qed", "median_2", "osimertinib_mpo", "scaffold_hop", "deco_hop", "drd2")


def topk_mean(pool: Sequence[float], k: int) -> float:
    """Mean of the k best scores seen so far (all of them when fewer than k)."""
    best = sorted(pool, reverse=True)[:k]
    return float(np.mean(best))


def topk_curves(rows, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(batch-safe, rowwise) top-k mean curves on the fixed 20..1000 call grid."""
    init, batches = split_batches(rows)

    pool = [row.score for row in init]
    events = [(FIRST_CALL, topk_mean(pool, k))]
    for batch in batches:
        pool.extend(row.score for row in batch)
        events.append((batch[-1].call, topk_mean(pool, k)))
    batch_curve = curve_from_events(events)

    pool = [row.score for row in init]
    events = [(FIRST_CALL, topk_mean(pool, k))]
    for row in (r for r in rows if r.call > FIRST_CALL):
        pool.append(row.score)
        events.append((row.call, topk_mean(pool, k)))
    row_curve = curve_from_events(events)
    return batch_curve, row_curve


def normalized_auc(curve: np.ndarray) -> float:
    return float(np.trapezoid(curve, GRID) / float(HORIZON - FIRST_CALL))


def average_ranks_desc(values: dict[str, float]) -> dict[str, float]:
    """Competition-free average ranks, 1 = largest value, ties share the mean."""
    order = sorted(values, key=lambda m: -values[m])
    ranks: dict[str, float] = {}
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and np.isclose(values[order[j + 1]], values[order[i]]):
            j += 1
        rank = float(np.mean(np.arange(i + 1, j + 2)))
        for m in order[i : j + 1]:
            ranks[m] = rank
        i = j + 1
    return ranks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    workspace = Path(__file__).resolve().parents[1]
    parser.add_argument("--raw-root", type=Path, default=workspace / "results")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "verified_results" / "topk")
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--rng-seed", type=int, default=20260821)
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)

    methods = [m for m, *_ in METHODS]
    per_run: dict[tuple[str, str, int], dict[str, float]] = {}
    missing: list[str] = []

    for task, _ in TASKS:
        for method in methods:
            for seed in SEEDS:
                path = raw_root / task / method / f"seed_{seed}" / f"evaluations_seed_{seed}.csv"
                if not path.exists():
                    missing.append(str(path))
                    continue
                rows = load_rows(path)
                b1, r1 = topk_curves(rows, 1)
                b10, r10 = topk_curves(rows, TOP_K)
                per_run[(task, method, seed)] = {
                    "auc_top1_batch": normalized_auc(b1),
                    "auc_top1_rowwise": normalized_auc(r1),
                    "auc_top10_batch": normalized_auc(b10),
                    "auc_top10_rowwise": normalized_auc(r10),
                    "final_top1": float(b1[-1]),
                    "final_top10": float(b10[-1]),
                    "n_calls": float(len(rows)),
                }
    if missing:
        raise FileNotFoundError("Missing runs:\n" + "\n".join(missing))

    with (out_dir / "topk_auc_by_seed.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "task_label", "method_id", "method_label", "seed", "n_calls",
                    "auc_top1_batch_safe", "auc_top10_batch_safe",
                    "auc_top1_rowwise", "auc_top10_rowwise",
                    "final_top1", "final_top10"])
        for (task, method, seed), m in sorted(per_run.items()):
            w.writerow([task, TASK_LABEL[task], method, METHOD_LABEL[method], seed, int(m["n_calls"]),
                        f"{m['auc_top1_batch']:.6f}", f"{m['auc_top10_batch']:.6f}",
                        f"{m['auc_top1_rowwise']:.6f}", f"{m['auc_top10_rowwise']:.6f}",
                        f"{m['final_top1']:.6f}", f"{m['final_top10']:.6f}"])

    # Task means and within-task ranks, both metrics.
    task_mean: dict[tuple[str, str, str], float] = {}
    rows_out = []
    for task, _ in TASKS:
        for metric in ("auc_top1_batch", "auc_top10_batch"):
            means = {m: float(np.mean([per_run[(task, m, s)][metric] for s in SEEDS]))
                     for m in methods}
            ranks = average_ranks_desc(means)
            for m in methods:
                task_mean[(task, m, metric)] = means[m]
                rows_out.append((task, TASK_LABEL[task], m, METHOD_LABEL[m], metric,
                                 f"{means[m]:.6f}", f"{ranks[m]:.2f}"))
    with (out_dir / "topk_auc_task_means.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "task_label", "method_id", "method_label", "metric",
                    "mean_auc", "within_task_rank"])
        w.writerows(rows_out)

    # Mean rank over the 30 oracle-seed cells of the six headline tasks, as in Table 1.
    cell_ranks: dict[str, dict[str, list[float]]] = {
        metric: defaultdict(list) for metric in ("auc_top1_batch", "auc_top10_batch", "final_top1")
    }
    for metric in cell_ranks:
        for task in PAPER_TASKS:
            for seed in SEEDS:
                values = {m: per_run[(task, m, seed)][metric] for m in methods}
                for m, r in average_ranks_desc(values).items():
                    cell_ranks[metric][m].append(r)

    rank_rows = []
    for m in methods:
        row = {"method_id": m, "method_label": METHOD_LABEL[m]}
        for metric in ("auc_top1_batch", "auc_top10_batch", "final_top1"):
            v = np.asarray(cell_ranks[metric][m], dtype=float)
            boot = np.array([v[rng.integers(0, v.size, v.size)].mean()
                             for _ in range(args.bootstrap_reps)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            row[metric] = (float(v.mean()), float(lo), float(hi))
        rank_rows.append(row)
    rank_rows.sort(key=lambda r: r["auc_top10_batch"][0])

    with (out_dir / "topk_auc_mean_ranks.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method_id", "method_label",
                    "mean_rank_auc_top1", "ci_low_top1", "ci_high_top1",
                    "mean_rank_auc_top10", "ci_low_top10", "ci_high_top10",
                    "mean_rank_final_top1", "ci_low_final", "ci_high_final"])
        for r in rank_rows:
            w.writerow([r["method_id"], r["method_label"]]
                       + [f"{v:.4f}" for metric in ("auc_top1_batch", "auc_top10_batch", "final_top1")
                          for v in r[metric]])

    # TRACE vs the strongest baseline on each task, top-10 AUC.
    # Mirrors verified_analysis: comparator = highest baseline task mean, paired
    # over shared seeds except REINVENT4, whose repeated runs are unseeded.
    from verified_analysis import UNPAIRED_COMPARATORS

    reps = args.bootstrap_reps
    comp_rows = []
    for metric in ("auc_top1_batch", "auc_top10_batch"):
        for task in PAPER_TASKS:
            baselines = {m: task_mean[(task, m, metric)] for m in methods if m != "trace_turbomol"}
            comparator = max(baselines, key=baselines.get)
            trace = np.asarray([per_run[(task, "trace_turbomol", s)][metric] for s in SEEDS])
            base = np.asarray([per_run[(task, comparator, s)][metric] for s in SEEDS])
            deltas = trace - base
            paired = comparator not in UNPAIRED_COMPARATORS
            if paired:
                idx = rng.integers(0, len(SEEDS), size=(reps, len(SEEDS)))
                draws = deltas[idx].mean(axis=1)
                base_draws = base[idx].mean(axis=1)
            else:
                ti = rng.integers(0, len(SEEDS), size=(reps, len(SEEDS)))
                bi = rng.integers(0, len(SEEDS), size=(reps, len(SEEDS)))
                draws = trace[ti].mean(axis=1) - base[bi].mean(axis=1)
                base_draws = base[bi].mean(axis=1)
            lo, hi = np.percentile(draws, (2.5, 97.5))
            rel = 100.0 * draws / np.maximum(np.abs(base_draws), 1e-12)
            rlo, rhi = np.percentile(rel, (2.5, 97.5))
            comp_rows.append((
                metric, task, TASK_LABEL[task], comparator, METHOD_LABEL[comparator],
                "paired-seed" if paired else "independent-run",
                f"{trace.mean():.6f}", f"{base.mean():.6f}", f"{deltas.mean():.6f}",
                f"{lo:.6f}", f"{hi:.6f}",
                f"{100.0 * deltas.mean() / max(abs(base.mean()), 1e-12):.2f}",
                f"{rlo:.2f}", f"{rhi:.2f}",
                int(np.sum(deltas > 0)) if paired else "",
            ))
    with (out_dir / "topk_auc_vs_best_baseline.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "task", "task_label", "comparator", "comparator_label",
                    "comparison_design", "trace_mean", "comparator_mean", "mean_delta",
                    "ci_low", "ci_high", "relative_percent", "relative_ci_low",
                    "relative_ci_high", "trace_seed_wins"])
        w.writerows(comp_rows)

    print("\n### TRACE vs strongest baseline, top-10 AUC\n")
    print("Task | Comparator | TRACE | Baseline | Delta [95% CI] | Rel % | Seed wins")
    print(" | ".join("---" for _ in range(7)))
    for r in comp_rows:
        if r[0] != "auc_top10_batch":
            continue
        print(f"{r[2]} | {r[4]} | {r[6]} | {r[7]} | {float(r[8]):+.4f} "
              f"[{float(r[9]):+.4f}, {float(r[10]):+.4f}] | {r[11]}% | {r[14]}/5")

    # Console summary in Table 1 layout.
    for metric, title in (("auc_top1_batch", "Top-1 AUC"), ("auc_top10_batch", "Top-10 AUC")):
        print(f"\n### {title} (batch-safe, calls 20-1000, /980)\n")
        header = ["Method"] + [TASK_LABEL[t] for t in PAPER_TASKS] + ["Rank"]
        print(" | ".join(header))
        print(" | ".join("---" for _ in header))
        ordered = sorted(methods,
                         key=lambda m: float(np.mean(cell_ranks[metric][m])))
        for m in ordered:
            cells = [f"{task_mean[(t, m, metric)]:.3f}" for t in PAPER_TASKS]
            mr = float(np.mean(cell_ranks[metric][m]))
            print(" | ".join([METHOD_LABEL[m]] + cells + [f"{mr:.2f}"]))

    print(f"\nWrote 4 CSVs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
