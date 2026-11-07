#!/usr/bin/env python
"""Batch-size sweep figures: how each method holds up as batches get larger.

Every run spends the same 1020 oracle calls, so a larger batch buys parallelism
at the cost of sequential feedback rounds (200 iterations at batch 5 down to 25
at batch 40). Methods that depend on adapting to feedback should degrade; the
sweep measures how much.

Writes into figures/ablation/:
    batch_size_effect.{png,pdf,svg}   final score, sample efficiency, restarts
    batch_size_curves.{png,pdf,svg}   best-so-far vs oracle calls, per batch size
    *.csv                             the plotted numbers

    python analysis/plot_batch_size_sweep.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib.lines import Line2D

ORACLE = "scaffold_hop"
ORACLE_LABEL = "Scaffold hop"
BATCHES = [5, 10, 25, 40]

# Same palette slots as the main benchmark figure, so colours carry across the paper.
METHODS = [
    ("trace_turbomol", "TRACE", "#1050c0", (0, ()), "o"),
    ("graph_ga", "Graph-GA", "#ef5f38", (0, (4.0, 1.6)), "s"),
    ("smiles_ga", "SMILES-GA", "#20a0b0", (0, (1.2, 1.3)), "^"),
    ("gpbo_graph_ga", "GP-BO + Graph-GA", "#a00070", (0, (5.0, 1.4, 1.0, 1.4)), "D"),
    ("stoned_selfies", "STONED-SELFIES", "#b070ff", (0, (3.0, 1.2, 1.0, 1.2, 1.0, 1.2)), "v"),
]
EMPHASIS = "trace_turbomol"

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#1a1a1a", "#4d4d4d", "#767676", "#e4e4e4", "#b8b8b8", "#ffffff"


# --------------------------------------------------------------------------- #
# Style (matches results_new_plotting/plot_best_vs_iteration_grid.py)
# --------------------------------------------------------------------------- #


def apply_style(pt: float) -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "font.size": pt, "axes.labelsize": pt + 1, "axes.titlesize": pt + 1,
        "xtick.labelsize": pt - 0.5, "ytick.labelsize": pt - 0.5, "legend.fontsize": pt,
        "text.color": INK, "axes.titlecolor": INK, "axes.labelcolor": INK2,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.color": AXIS, "ytick.color": AXIS, "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "legend.frameon": False, "lines.solid_capstyle": "round",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def style_axis(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=2.6, pad=2.5)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))


def batch_axis(ax) -> None:
    """Batch size reads better on a log axis with the swept values as ticks."""
    ax.set_xscale("log")
    ax.set_xticks(BATCHES)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(BATCHES[0] * 0.82, BATCHES[-1] * 1.22)
    ax.set_xlabel("Batch size", fontsize=8 + 1, color=INK2)


def panel_label(ax, letter: str, pt: float) -> None:
    ax.text(-0.02, 1.14, f"({letter})", transform=ax.transAxes, fontsize=pt + 1.5,
            fontweight="bold", color=INK, ha="left", va="top")


def row_major_order(n_items: int, ncol: int) -> list:
    """Index order that makes matplotlib's column-major legend read left-to-right.

    matplotlib slices the handle list into `ncol` consecutive chunks of
    `ceil(n/ncol)` and fills each as a column, so entry `r*ncol + c` has to be
    handed over at position `c*nrow + r`. Ragged last columns are fine because
    the missing cells land at the end of both orders.
    """
    ncol = max(1, min(ncol, n_items))
    nrow = -(-n_items // ncol)
    return [r * ncol + c for c in range(ncol) for r in range(nrow) if r * ncol + c < n_items]


def bottom_legend(fig, y, ncol, pt, present):
    handles = [Line2D([0], [0], color=c, linestyle=d, linewidth=1.7 if m == EMPHASIS else 1.2,
                      marker=k, markersize=3.4, markeredgewidth=0.55, markeredgecolor=SURFACE)
               for m, _, c, d, k in present]
    order = row_major_order(len(present), ncol)
    legend = fig.legend([handles[i] for i in order], [present[i][1] for i in order],
                        loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
                        fontsize=pt, handlelength=2.6, handletextpad=0.6, columnspacing=2.0,
                        labelspacing=0.55)
    for text in legend.get_texts():
        text.set_color(INK)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

CALL_GRID = np.arange(20, 1021)


def load_curves(root: Path, batch: int, method: str):
    """Best-so-far per seed on the shared oracle-call grid: (n_seeds, n_calls)."""
    curves = []
    for path in sorted((root / f"batch_{batch}" / ORACLE / method).glob(
            "seed_*/trajectory_seed_*.csv")):
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
        if not rows:
            continue
        key = "iteration" if "iteration" in rows[0] else "generation"
        calls_key = "n_evaluated" if "n_evaluated" in rows[0] else "oracle_calls"
        rows.sort(key=lambda r: float(r[key]))
        calls = np.array([float(r[calls_key]) for r in rows])
        best = np.maximum.accumulate([float(r["best_so_far"]) for r in rows])
        curves.append(best[np.searchsorted(calls, CALL_GRID, side="right") - 1])
    return np.vstack(curves) if curves else None


def restarts_per_seed(root: Path, batch: int, method: str) -> np.ndarray:
    """Trust-region restart count per seed; seeds that never restart have no CSV."""
    seed_dirs = sorted((root / f"batch_{batch}" / ORACLE / method).glob("seed_*"))
    counts = []
    for seed_dir in seed_dirs:
        files = list(seed_dir.glob("restarts_seed_*.csv"))
        counts.append(sum(sum(1 for _ in f.open()) - 1 for f in files))
    return np.array(counts, dtype=float)


def boot_ci(runs: np.ndarray, reps: int, rng):
    """95% percentile bootstrap CI for the mean, resampling whole runs.

    Works for a (n_seeds,) vector and for a (n_seeds, n_points) curve stack.
    """
    idx = rng.integers(0, runs.shape[0], size=(reps, runs.shape[0]))
    draws = runs[idx].mean(axis=1)
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


# --------------------------------------------------------------------------- #
# Figure 1: what batch size costs
# --------------------------------------------------------------------------- #


def figure_effect(data, restarts, reps, rng, pt=8.0, width=7.2):
    apply_style(pt)
    height = 3.62
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(1, 3, left=0.068, right=0.995, top=1 - 0.34 / height,
                            bottom=1.36 / height, wspace=0.34)
    rows_out = []

    # (a) final score, (b) AUC over the run -- same shape, different statistic.
    for index, (letter, title, ylabel, reduce_fn) in enumerate([
        ("a", "Final score", f"Final {ORACLE_LABEL.lower()} score", lambda c: c[:, -1]),
        ("b", "Sample efficiency", "AUC of best-so-far", lambda c: c.mean(axis=1)),
    ]):
        ax = fig.add_subplot(grid[0, index])
        style_axis(ax)
        batch_axis(ax)
        for method, label, colour, dash, marker in METHODS:
            xs, means, los, his = [], [], [], []
            for batch in BATCHES:
                curves = data.get((batch, method))
                if curves is None:
                    continue
                values = reduce_fn(curves)
                low, high = boot_ci(values, reps, rng)
                xs.append(batch)
                means.append(values.mean())
                los.append(low)
                his.append(high)
                rows_out.append((title, label, batch, values.mean(), low, high))
            if not xs:
                continue
            emphasis = method == EMPHASIS
            ax.fill_between(xs, los, his, color=colour, alpha=0.13, linewidth=0)
            ax.plot(xs, means, color=colour, linestyle=dash, linewidth=1.7 if emphasis else 1.2,
                    marker=marker, markersize=3.6, markeredgewidth=0.55,
                    markeredgecolor=SURFACE, zorder=6 if emphasis else 4)
        ax.set_ylabel(ylabel, fontsize=pt + 1, color=INK2)
        ax.set_title(title, fontsize=pt + 1, color=INK, pad=5, fontweight="semibold")
        panel_label(ax, letter, pt)

    # (c) the mechanism: fewer iterations means fewer trust-region restarts.
    ax = fig.add_subplot(grid[0, 2])
    style_axis(ax)
    batch_axis(ax)
    colour = dict((m, c) for m, _, c, _, _ in METHODS)[EMPHASIS]
    xs, means = [], []
    for batch in BATCHES:
        counts = restarts.get(batch)
        if counts is None or not counts.size:
            continue
        jitter = np.linspace(-0.03, 0.03, counts.size)
        ax.plot(batch * (1 + jitter), counts, linestyle="none", marker="o", markersize=3.0,
                color=colour, alpha=0.45, markeredgewidth=0)
        xs.append(batch)
        means.append(counts.mean())
        rows_out.append(("Restarts", "TRACE", batch, counts.mean(), counts.min(), counts.max()))
    ax.plot(xs, means, color=colour, linewidth=1.7, marker="o", markersize=3.6,
            markeredgewidth=0.55, markeredgecolor=SURFACE)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Trust-region restarts per run", fontsize=pt + 1, color=INK2)
    ax.set_title("Restarts (TRACE only)", fontsize=pt + 1, color=INK, pad=5, fontweight="semibold")
    panel_label(ax, "c", pt)

    present = [m for m in METHODS if any((b, m[0]) in data for b in BATCHES)]
    bottom_legend(fig, 0.46 / height, 3, pt, present)
    n_seeds = next(iter(data.values())).shape[0]
    fig.text(0.5, 0.08 / height,
             f"{ORACLE_LABEL}, {n_seeds} seeds, fixed budget of {CALL_GRID[-1]} oracle calls at "
             "every batch size. Bands are 95% bootstrap CIs.",
             ha="center", va="bottom", fontsize=pt - 1, color=MUTED)
    return fig, rows_out


# --------------------------------------------------------------------------- #
# Figure 2: the curves behind it
# --------------------------------------------------------------------------- #


def figure_curves(data, reps, rng, pt=8.0, width=7.2):
    apply_style(pt)
    height = 2.98
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(1, len(BATCHES), left=0.068, right=0.995,
                            top=1 - 0.34 / height, bottom=1.02 / height, wspace=0.26)
    rows_out = []
    share = None

    for index, batch in enumerate(BATCHES):
        ax = fig.add_subplot(grid[0, index], sharey=share)
        share = share or ax
        style_axis(ax)
        for method, label, colour, dash, marker in METHODS:
            curves = data.get((batch, method))
            if curves is None:
                continue
            mean = curves.mean(axis=0)
            low, high = boot_ci(curves, reps, rng)
            emphasis = method == EMPHASIS
            ax.fill_between(CALL_GRID, low, high, color=colour, alpha=0.11, linewidth=0)
            ax.plot(CALL_GRID, mean, color=colour, linestyle=dash,
                    linewidth=1.6 if emphasis else 1.1, marker=marker,
                    markevery=(0, len(CALL_GRID) // 4), markersize=3.0,
                    markeredgewidth=0.5, markeredgecolor=SURFACE,
                    zorder=6 if emphasis else 4)
            rows_out += [("curve", label, batch, int(c), m, a, b)
                         for c, m, a, b in zip(CALL_GRID[::50], mean[::50],
                                               low[::50], high[::50])]
        ax.set_xlim(CALL_GRID[0], CALL_GRID[-1])
        ax.xaxis.set_major_locator(mticker.MultipleLocator(500))
        ax.set_title(f"Batch {batch}", fontsize=pt + 1, color=INK, pad=5, fontweight="semibold")
        panel_label(ax, chr(ord("a") + index), pt)
        if index:
            ax.tick_params(labelleft=False)

    fig.supxlabel("Oracle calls", fontsize=pt + 1.5, color=INK2, y=0.68 / height)
    fig.supylabel(f"{ORACLE_LABEL} score", fontsize=pt + 1.5, color=INK2, x=0.004,
                  y=(1 - 0.34 / height + 1.02 / height) / 2)
    present = [m for m in METHODS if any((b, m[0]) in data for b in BATCHES)]
    bottom_legend(fig, 0.10 / height, 3, pt, present)
    return fig, rows_out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def save(fig, out_dir: Path, stem: str, formats, dpi: int):
    paths = [out_dir / f"{stem}.{fmt}" for fmt in formats]
    for path in paths:
        fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return paths


def write_csv(path: Path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", default=str(repo / "ablations" / "batch_size"))
    parser.add_argument("--out-dir", default=str(repo / "figures" / "ablation"))
    parser.add_argument("--formats", default="png,pdf,svg")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    data = {(b, m): load_curves(root, b, m) for b in BATCHES for m, *_ in METHODS}
    data = {k: v for k, v in data.items() if v is not None}
    if not data:
        print(f"No trajectories found under {root}")
        return 1
    restarts = {b: restarts_per_seed(root, b, EMPHASIS) for b in BATCHES}

    rng = np.random.default_rng(args.rng_seed)
    written = []

    fig, rows = figure_effect(data, restarts, args.reps, rng)
    written += save(fig, out_dir, "batch_size_effect", formats, args.dpi)
    written.append(write_csv(out_dir / "batch_size_effect.csv",
                             ["panel", "method", "batch_size", "mean", "ci_low", "ci_high"], rows))

    fig, rows = figure_curves(data, args.reps, rng)
    written += save(fig, out_dir, "batch_size_curves", formats, args.dpi)
    written.append(write_csv(out_dir / "batch_size_curves.csv",
                             ["panel", "method", "batch_size", "oracle_calls",
                              "mean", "ci_low", "ci_high"], rows))

    print(f"{ORACLE}: batch sizes {BATCHES}, {len(METHODS)} methods, "
          f"{next(iter(data.values())).shape[0]} seeds, {args.reps} bootstrap resamples")
    for path in written:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
