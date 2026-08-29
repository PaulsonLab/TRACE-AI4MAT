"""Plot TRACE optimization curves for every oracle.

The conventions here match the figures in the paper (results_new_plotting/), so a
curve produced from this package is directly comparable with a published panel:

  * Curves come from `trajectory_seed_*.csv`, i.e. the incumbent at each batch
    endpoint, running-maxed. A batch is proposed jointly, so no member of it gets
    credit until the batch is finished. This is *not* the same as the row-wise
    best-so-far in evaluations_seed_*.csv, which steps up mid-batch and reads
    higher -- on DRD2 the two differ by about 0.03 in AUC.
  * The band is a 95% percentile bootstrap confidence interval for the mean,
    resampling whole runs (seeds), never individual iterations.
  * TRACE blue (#1050c0), heavy stroke, sparse white-edged markers, one shared
    axis label per grid; PDF + SVG + PNG for every figure.

Outputs
-------
    trace_best_vs_calls_main.{pdf,svg,png}   2x3, the paper's six headline oracles
    trace_best_vs_calls_all.{pdf,svg,png}    2x4, all eight bundled oracles
    per_oracle/<oracle>.{pdf,svg,png}        one oracle per file
    trace_curves.csv                         mean, CI and per-seed values
    trace_summary.csv                        final best and normalised AUC

Usage
-----
    python plot_trace_results.py
    python plot_trace_results.py --results results_repro --out figures_repro
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_DIR = Path(__file__).resolve().parent

# Oracle ids and display names, in the paper's order.
ORACLES: List[Tuple[str, str]] = [
    ("penalized_logp", "Penalized logP"),
    ("qed", "QED"),
    ("median_2", "Median molecule 2"),
    ("osimertinib_mpo", "Osimertinib MPO"),
    ("scaffold_hop", "Scaffold hop"),
    ("deco_hop", "Deco hop"),
    ("drd2", "DRD2"),
    ("jnk3", "JNK3"),
]
LABEL = dict(ORACLES)

# The six reported in the paper's headline figure and table. penalized_logp is a
# diagnostic (it rewards molecular size) and jnk3 was run but not reported.
MAIN_ORACLES = ["qed", "median_2", "osimertinib_mpo", "scaffold_hop", "deco_hop", "drd2"]

TRACE_BLUE = "#1050c0"
INK, MUTED, GRID = "#1a1a1a", "#767676", "#e4e4e4"
SEEDS = [11, 22, 33, 44, 55]

FS_LABEL, FS_TICK, FS_TITLE, FS_LEG = 24, 24, 27, 18


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.facecolor": "white", "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": FS_LABEL, "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE, "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEG,
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "#b8b8b8",
        "axes.linewidth": 1.6, "xtick.major.width": 1.6, "ytick.major.width": 1.6,
    })


def style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, linewidth=1.2)
    ax.set_axisbelow(True)
    ax.tick_params(pad=6)


def load_curves(results_root: Path, oracle: str) -> Optional[Dict]:
    """Batch-endpoint incumbent per seed, with the true cumulative call count.

    `n_evaluated` is the exact number of unique oracle calls spent once that
    batch has been scored, so the x axis is the real budget rather than an
    iteration index rescaled by the nominal batch size.
    """
    d = results_root / oracle / "trace_turbomol"
    curves, calls, seeds = [], None, []
    for path in sorted(d.glob("seed_*/trajectory_seed_*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        rows.sort(key=lambda r: int(r["iteration"]))
        curves.append(np.maximum.accumulate([float(r["best_so_far"]) for r in rows]))
        calls = np.asarray([int(r["n_evaluated"]) for r in rows], dtype=float)
        seeds.append(int(rows[0]["seed"]))
    if not curves:
        return None

    n = min(len(c) for c in curves)
    return {
        "oracle": oracle,
        "seeds": seeds,
        "calls": calls[:n],
        "runs": np.vstack([c[:n] for c in curves]),
    }


def bootstrap_ci(runs: np.ndarray, reps: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """95% percentile CI for the mean, resampling whole runs."""
    idx = rng.integers(0, runs.shape[0], size=(reps, runs.shape[0]))
    draws = runs[idx].mean(axis=1)
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def draw_panel(ax, data: Dict, reps: int, rng: np.random.Generator) -> None:
    x, runs = data["calls"], data["runs"]
    lo, hi = bootstrap_ci(runs, reps, rng)
    ax.fill_between(x, lo, hi, color=TRACE_BLUE, alpha=0.20, linewidth=0, zorder=8)
    ax.plot(x, runs.mean(0), color=TRACE_BLUE, linewidth=5.0, zorder=10,
            marker="o", markevery=(0, 5), markersize=12,
            markeredgecolor="white", markeredgewidth=2.0)
    ax.set_title(LABEL.get(data["oracle"], data["oracle"]), pad=14)
    style_axes(ax)


def figure_legend(fig, n_seeds: int, **kw) -> None:
    handles = [
        Line2D([], [], color=TRACE_BLUE, lw=5.0, marker="o", markersize=12,
               markeredgecolor="white", markeredgewidth=2.0, label="TRACE (mean)"),
        Patch(facecolor=TRACE_BLUE, alpha=0.20,
              label=f"95% bootstrap CI over {n_seeds} seeds"),
    ]
    fig.legend(handles=handles, frameon=False, **kw)


def save(fig, out: Path, name: str, dpi: int) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=dpi)
    plt.close(fig)
    print(f"[fig] {name}.{{pdf,svg,png}}")


def plot_grid(collected: List[Dict], out: Path, name: str, ncols: int,
              reps: int, rng: np.random.Generator, dpi: int) -> None:
    nrows = int(np.ceil(len(collected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.3 * ncols, 6.25 * nrows),
                             squeeze=False,
                             gridspec_kw={"hspace": 0.32, "wspace": 0.24,
                                          "left": 0.105, "right": 0.985,
                                          "top": 0.950, "bottom": 0.150})
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, data in zip(axes.ravel(), collected):
        ax.set_visible(True)
        draw_panel(ax, data, reps, rng)
        ax.tick_params(labelsize=FS_TICK + 6)
        ax.set_title(LABEL.get(data["oracle"], data["oracle"]), pad=14, fontsize=FS_TITLE + 4)

    fig.supxlabel("Number of Oracle Calls", fontsize=FS_LABEL + 10, y=0.048)
    fig.supylabel("Best Score Found", fontsize=FS_LABEL + 10)
    figure_legend(fig, max(len(d["seeds"]) for d in collected),
                  loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.055),
                  columnspacing=1.8, handlelength=2.6, fontsize=FS_LEG + 10)
    save(fig, out, name, dpi)


def plot_per_oracle(collected: List[Dict], out: Path, reps: int,
                    rng: np.random.Generator, dpi: int) -> None:
    sub = out / "per_oracle"
    sub.mkdir(parents=True, exist_ok=True)
    for data in collected:
        fig, ax = plt.subplots(figsize=(9.5, 7.0))
        draw_panel(ax, data, reps, rng)
        ax.set_xlabel("Number of Oracle Calls")
        ax.set_ylabel("Best Score Found")
        figure_legend(fig, len(data["seeds"]), loc="lower right",
                      bbox_to_anchor=(0.985, 0.055), fontsize=FS_LEG)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(sub / f"{data['oracle']}.{ext}", dpi=dpi)
        plt.close(fig)
    print(f"[fig] per_oracle/<oracle>.{{pdf,svg,png}}  ({len(collected)} oracles)")


def top1_auc(calls: np.ndarray, curve: np.ndarray) -> float:
    """Area under the best-so-far curve against unique oracle calls, per call.

    The incumbent is a step function that only rises at a batch endpoint, so it
    is held flat across each batch and integrated on the dense per-call grid.
    Trapezoid-integrating the 21 endpoints directly would interpolate across the
    steps and inflate the area -- by 0.023 on DRD2, enough to change a ranking.
    Values keep the units of the oracle score.
    """
    grid = np.arange(int(calls[0]), int(calls[-1]) + 1)
    held = curve[np.searchsorted(calls, grid, side="right") - 1]
    return float(np.trapezoid(held, grid) / (grid[-1] - grid[0]))


def write_tables(collected: List[Dict], out: Path, reps: int, rng: np.random.Generator) -> None:
    with open(out / "trace_curves.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["oracle", "iteration", "oracle_calls", "n_seeds", "mean", "ci_low", "ci_high"]
                   + [f"seed_{s}" for s in SEEDS])
        for d in collected:
            lo, hi = bootstrap_ci(d["runs"], reps, rng)
            mean = d["runs"].mean(0)
            by_seed = dict(zip(d["seeds"], d["runs"]))
            for i, call in enumerate(d["calls"]):
                w.writerow([d["oracle"], i, int(call), len(d["seeds"]),
                            f"{mean[i]:.6f}", f"{lo[i]:.6f}", f"{hi[i]:.6f}"]
                           + [f"{by_seed[s][i]:.6f}" if s in by_seed else "" for s in SEEDS])

    with open(out / "trace_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["oracle", "n_seeds", "oracle_calls", "final_best_mean", "final_best_std",
                    "final_best_min", "final_best_max", "top1_auc"])
        for d in collected:
            finals = d["runs"][:, -1]
            x = d["calls"]
            auc = float(np.mean([top1_auc(x, r) for r in d["runs"]]))
            w.writerow([d["oracle"], len(finals), int(x[-1]),
                        f"{finals.mean():.6f}", f"{finals.std():.6f}",
                        f"{finals.min():.6f}", f"{finals.max():.6f}", f"{auc:.6f}"])
    print("[csv] trace_curves.csv, trace_summary.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot TRACE best-so-far curves, matching the paper's figure conventions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    p.add_argument("--oracles", nargs="+", default=["all"],
                   help="'all', 'main' (the paper's six), or explicit oracle names.")
    p.add_argument("--bootstrap-reps", type=int, default=20000)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the bootstrap.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()

    results_root = Path(args.results)
    if not results_root.is_absolute():
        results_root = REPO_DIR / results_root
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_DIR / out
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    all_ids = [o for o, _ in ORACLES]
    if "all" in args.oracles:
        wanted = all_ids
    elif "main" in args.oracles:
        wanted = MAIN_ORACLES
    else:
        wanted = list(args.oracles)

    collected = []
    for oracle in wanted:
        d = load_curves(results_root, oracle)
        if d is None:
            print(f"[warn] no TRACE results for {oracle} under {results_root}")
            continue
        collected.append(d)
    if not collected:
        raise SystemExit(f"No TRACE results found under {results_root}")

    main_set = [d for d in collected if d["oracle"] in MAIN_ORACLES]
    if len(main_set) == len(MAIN_ORACLES):
        plot_grid(main_set, out, "trace_best_vs_calls_main", 3, args.bootstrap_reps, rng, args.dpi)
    if len(collected) > len(main_set):
        plot_grid(collected, out, "trace_best_vs_calls_all", 4, args.bootstrap_reps, rng, args.dpi)

    plot_per_oracle(collected, out, args.bootstrap_reps, rng, args.dpi)
    write_tables(collected, out, args.bootstrap_reps, rng)
    print(f"\nDone. Figures and tables in {out}")


if __name__ == "__main__":
    main()
