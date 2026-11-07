#!/usr/bin/env python
"""Generator x trust-region ablation: the full 2x2, one oracle at a time.

    generator / search        with trust region        without trust region
    TRACE generator           trace_turbomol           gpbo_trace_gen
    Graph-GA generator        trace_tr_graph_ga        gpbo_graph_ga

Each arm is read from ablations/proposal_region/<oracle>/<arm>/. For scaffold_hop the
gpbo_graph_ga cell was produced by the batch-size sweep rather than re-run here, so a
missing arm falls back to the sweep slice at the matching batch size; that run shares
the same seeds, initial molecules, and call budget.

Encoding: one colour per arm, with dash + marker as a redundant generator cue
(solid/circle = TRACE, dashed/square = Graph-GA). Panel (c) states the
trust-region effect directly: the paired per-seed difference (with TR minus
without TR), one column per generator, drawn in that generator's with-TR colour.

Writes figures/ablation/generator_tr_ablation_<oracle>.{png,pdf,svg} plus the plotted numbers
and a 2x2 summary table.

    python analysis/plot_generator_tr_ablation.py --oracle scaffold_hop
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib.lines import Line2D

ORACLE_LABELS = {
    "scaffold_hop": "Scaffold hop",
    "median_2": "Median 2",
    "osimertinib_mpo": "Osimertinib MPO",
    "penalized_logp": "Penalized logP",
    "qed": "QED",
    "drd2": "DRD2",
}

# One hue per arm (validated as a 4-slot categorical palette: all pairs clear the
# CVD floor), with dash + marker kept as a redundant generator cue.
TRACE_TR, GA_TR = "#1050c0", "#0f8a6a"
TRACE_NOTR, GA_NOTR = "#a00070", "#b25c00"
SOLID, DASH = (0, ()), (0, (5.0, 1.4, 1.0, 1.4))

# (folder, label, generator, has_tr, colour, dash, marker)
ARMS = [
    ("trace_turbomol", "TRACE gen. + trust region", "TRACE", True, TRACE_TR, SOLID, "o"),
    ("trace_tr_graph_ga", "Graph-GA gen. + trust region", "Graph-GA", True, GA_TR, DASH, "s"),
    ("gpbo_trace_gen", "TRACE gen., no trust region", "TRACE", False, TRACE_NOTR, SOLID, "o"),
    ("gpbo_graph_ga", "Graph-GA gen., no trust region", "Graph-GA", False, GA_NOTR, DASH, "s"),
]
GENERATORS = ["TRACE", "Graph-GA"]

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#1a1a1a", "#4d4d4d", "#767676", "#e4e4e4", "#b8b8b8", "#ffffff"


# --------------------------------------------------------------------------- #
# Style (matches plot_trust_region_ablation.py)
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
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))


def panel_label(ax, letter: str, pt: float) -> None:
    ax.text(-0.02, 1.14, f"({letter})", transform=ax.transAxes, fontsize=pt + 1.5,
            fontweight="bold", color=INK, ha="left", va="top")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def read_trajectory(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    calls_key = "n_evaluated" if "n_evaluated" in rows[0] else "oracle_calls"
    rows.sort(key=lambda r: float(r["iteration"]))
    return {
        "seed": int(float(rows[0]["seed"])),
        "calls": np.array([float(r[calls_key]) for r in rows]),
        "best": np.maximum.accumulate([float(r["best_so_far"]) for r in rows]),
    }


def load_arm(arm_dir: Path) -> list:
    runs = [read_trajectory(p) for p in sorted(arm_dir.glob("seed_*/trajectory_seed_*.csv"))]
    return sorted(runs, key=lambda d: d["seed"])          # seed order, so pairing is by seed


def on_grid(calls: np.ndarray, best: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return best[np.searchsorted(calls, grid, side="right") - 1]


def boot_ci(runs: np.ndarray, reps: int, rng) -> tuple:
    idx = rng.integers(0, runs.shape[0], size=(reps, runs.shape[0]))
    draws = runs[idx].mean(axis=1)
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def boot_ci_1d(values: np.ndarray, reps: int, rng) -> tuple:
    idx = rng.integers(0, values.size, size=(reps, values.size))
    draws = values[idx].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def sign_test_p(wins: int, n: int) -> float:
    from math import comb
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #


def build_figure(arms_data, oracle_label, reps, rng, pt=8.0, width=7.2):
    apply_style(pt)
    height = 3.48
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(1, 3, left=0.062, right=0.995, top=1 - 0.34 / height,
                            bottom=1.26 / height, wspace=0.36)
    rows_out = []

    # (a) best-so-far vs oracle calls ---------------------------------------- #
    ax = fig.add_subplot(grid[0, 0])
    style_axis(ax)
    lo_call = max(int(d[0]["calls"][0]) for d in arms_data.values())
    hi_call = min(int(d[0]["calls"][-1]) for d in arms_data.values())
    calls = np.arange(lo_call, hi_call + 1)
    for folder, label, _gen, _tr, colour, dash, marker in ARMS:
        runs = np.vstack([on_grid(d["calls"], d["best"], calls) for d in arms_data[folder]])
        mean = runs.mean(axis=0)
        low, high = boot_ci(runs, reps, rng)
        ax.fill_between(calls, low, high, color=colour, alpha=0.10, linewidth=0)
        ax.plot(calls, mean, color=colour, linestyle=dash, linewidth=1.7,
                marker=marker, markevery=(0, len(calls) // 5), markersize=3.2,
                markeredgewidth=0.55, markeredgecolor=SURFACE)
        rows_out += [("best_vs_calls", label, int(c), m, a, b)
                     for c, m, a, b in zip(calls, mean, low, high)]
    ax.set_xlim(lo_call, hi_call)
    ax.set_xlabel("Oracle calls", fontsize=pt + 1, color=INK2)
    ax.set_ylabel(f"{oracle_label} score", fontsize=pt + 1, color=INK2)
    ax.set_title("Best score so far", fontsize=pt + 1, color=INK, pad=5, fontweight="semibold")
    panel_label(ax, "a", pt)

    # (b) paired per-seed finals, grouped by generator ------------------------ #
    ax = fig.add_subplot(grid[0, 1])
    style_axis(ax)
    finals = {folder: np.array([d["best"][-1] for d in arms_data[folder]]) for folder, *_ in ARMS}
    positions, deltas = {}, {}
    for g_idx, generator in enumerate(GENERATORS):
        tr_arm = next(a for a in ARMS if a[2] == generator and a[3])
        no_arm = next(a for a in ARMS if a[2] == generator and not a[3])
        x_tr, x_no = 3 * g_idx, 3 * g_idx + 1
        positions[generator] = (x_tr, x_no)
        left, right = finals[tr_arm[0]], finals[no_arm[0]]
        deltas[generator] = left - right                      # paired by seed
        for a_val, b_val in zip(left, right):
            ax.plot([x_tr, x_no], [a_val, b_val], color=MUTED, linewidth=0.8, alpha=0.6, zorder=2)
        for x_pos, arm, values in ((x_tr, tr_arm, left), (x_no, no_arm, right)):
            ax.plot(np.full(values.size, x_pos), values, linestyle="none", marker=arm[6],
                    markersize=4.8, color=arm[4], markeredgewidth=0.7,
                    markeredgecolor=SURFACE, zorder=4)
            ax.plot([x_pos - 0.28, x_pos + 0.28], [values.mean()] * 2,
                    color=arm[4], linewidth=1.8, zorder=5)
            rows_out += [("final_by_seed", arm[1], int(d["seed"]), v, "", "")
                         for d, v in zip(arms_data[arm[0]], values)]
    ax.set_xlim(-0.8, 4.8)
    ax.set_xticks([p for pair in positions.values() for p in pair])
    ax.set_xticklabels(["TR", "no TR", "TR", "no TR"])
    for g_idx, generator in enumerate(GENERATORS):
        ax.text(3 * g_idx + 0.5, -0.16, f"{generator} gen.", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=pt - 0.5, color=INK2)
    ax.set_ylabel(f"Final {oracle_label.lower()} score", fontsize=pt + 1, color=INK2)
    ax.set_title("Paired by seed", fontsize=pt + 1, color=INK, pad=5, fontweight="semibold")
    panel_label(ax, "b", pt)

    # (c) trust-region gain within each generator ---------------------------- #
    ax = fig.add_subplot(grid[0, 2])
    style_axis(ax)
    ax.axhline(0.0, color=AXIS, linewidth=0.8, zorder=1)
    for g_idx, generator in enumerate(GENERATORS):
        arm = next(a for a in ARMS if a[2] == generator and a[3])
        colour = arm[4]                                   # that generator's with-TR hue
        values = deltas[generator]
        low, high = boot_ci_1d(values, reps, rng)
        ax.plot([g_idx, g_idx], [low, high], color=colour, linewidth=1.4, zorder=3)
        ax.plot(np.full(values.size, g_idx) + rng.uniform(-0.07, 0.07, values.size), values,
                linestyle="none", marker=arm[6], markersize=4.4, color=colour, alpha=0.55,
                markeredgewidth=0.6, markeredgecolor=SURFACE, zorder=4)
        ax.plot([g_idx - 0.22, g_idx + 0.22], [values.mean()] * 2, color=colour,
                linewidth=1.9, zorder=5)
        wins = int((values > 0).sum())
        # Pinned to the top of the panel: anchoring to the CI runs into the title
        # whenever the interval reaches the top of the axes.
        ax.text(g_idx, 0.99, f"{wins}/{values.size}\n$p$={sign_test_p(wins, values.size):.3f}",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=pt - 1.5, color=MUTED, linespacing=1.3)
        rows_out.append(("trust_region_gain_mean", f"{generator} generator", g_idx,
                         float(values.mean()), low, high))
        rows_out += [("trust_region_gain_by_seed", f"{generator} generator", int(d["seed"]), v, "", "")
                     for d, v in zip(arms_data[arm[0]], values)]
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{g} gen." for g in GENERATORS])
    ax.set_ylabel("Final score gain from trust region", fontsize=pt + 1, color=INK2)
    ax.set_title("Trust-region gain", fontsize=pt + 1, color=INK, pad=5, fontweight="semibold")
    panel_label(ax, "c", pt)

    # Legend + the point of the figure, stated once.
    handles = [Line2D([0], [0], color=c, linestyle=d, linewidth=1.7, marker=k,
                      markersize=3.4, markeredgewidth=0.55, markeredgecolor=SURFACE)
               for _f, _l, _g, _t, c, d, k in ARMS]
    legend = fig.legend(handles, [a[1] for a in ARMS], loc="lower center",
                        bbox_to_anchor=(0.5, 0.42 / height), ncol=2, frameon=False,
                        fontsize=pt, handlelength=2.6, handletextpad=0.6, columnspacing=2.4)
    for text in legend.get_texts():
        text.set_color(INK)
    n_seeds = len(arms_data[ARMS[0][0]])
    fig.text(0.5, 0.13 / height,
             f"All four arms share the same {n_seeds} seeds, the same starting molecules and the "
             f"same {hi_call}-call budget; only the generator and the trust region change. "
             "Bands and bars are 95% bootstrap CIs.",
             ha="center", va="bottom", fontsize=pt - 1, color=MUTED)
    return fig, rows_out, finals, deltas


def resolve_arm_dir(folder: str, oracle: str, root: Path, sweep_root: Path) -> Path:
    """Prefer the arm re-run under this benchmark; fall back to the sweep slice."""
    local = root / oracle / folder
    if any(local.glob("seed_*/trajectory_seed_*.csv")):
        return local
    return sweep_root / oracle / folder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oracle", default="scaffold_hop")
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", default=str(repo / "ablations" / "proposal_region"),
                        help="directory holding <oracle>/<arm>/seed_*/")
    parser.add_argument("--sweep-root",
                        default=str(repo / "ablations" / "batch_size" / "batch_10"),
                        help="batch-size-sweep slice to fall back to for missing arms")
    parser.add_argument("--out-dir", default=str(repo / "figures" / "ablation"))
    parser.add_argument("--formats", default="png,pdf,svg")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args()

    oracle = args.oracle
    oracle_label = ORACLE_LABELS.get(oracle, oracle.replace("_", " ").capitalize())
    root = Path(args.root).resolve()
    sweep_root = Path(args.sweep_root).resolve()

    arm_dirs = {folder: resolve_arm_dir(folder, oracle, root, sweep_root) for folder, *_ in ARMS}
    arms_data = {folder: load_arm(path) for folder, path in arm_dirs.items()}
    missing = [f for f, d in arms_data.items() if not d]
    if missing:
        for folder in missing:
            print(f"No trajectories for {folder} under {arm_dirs[folder]}")
        return 1

    seed_sets = {tuple(d["seed"] for d in runs) for runs in arms_data.values()}
    if len(seed_sets) != 1:
        print(f"Arms do not share seeds: {seed_sets}")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)
    fig, rows_out, finals, deltas = build_figure(arms_data, oracle_label, args.reps, rng)

    stem = f"generator_tr_ablation_{oracle}"
    written = []
    for fmt in (f.strip() for f in args.formats.split(",") if f.strip()):
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=args.dpi)
        written.append(path)
    plt.close(fig)

    csv_path = out_dir / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "arm", "x", "value", "ci_low", "ci_high"])
        writer.writerows(rows_out)
    written.append(csv_path)

    # 2x2 summary table.
    table_path = out_dir / f"{stem}_summary.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["oracle", "generator", "trust_region", "arm", "n_seeds", "mean_final_best",
                         "std_final_best", "sem_final_best", "mean_gain_from_trust_region"])
        for folder, label, generator, has_tr, *_ in ARMS:
            values = finals[folder]
            writer.writerow([
                oracle, generator, "yes" if has_tr else "no", folder, values.size,
                f"{values.mean():.6f}", f"{values.std(ddof=1):.6f}",
                f"{values.std(ddof=1) / np.sqrt(values.size):.6f}",
                f"{deltas[generator].mean():.6f}" if has_tr else "",
            ])
    written.append(table_path)

    print(f"[{oracle}]")
    print(f"{'arm':<34s} {'mean final':>11s} {'sd':>8s}")
    for folder, label, *_ in ARMS:
        values = finals[folder]
        print(f"{label:<34s} {values.mean():11.4f} {values.std(ddof=1):8.4f}")
    for generator in GENERATORS:
        values = deltas[generator]
        wins = int((values > 0).sum())
        print(f"trust-region gain, {generator:<9s} generator: {values.mean():+.4f} "
              f"(wins {wins}/{values.size}, sign test p={sign_test_p(wins, values.size):.3f})")
    for path in written:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
