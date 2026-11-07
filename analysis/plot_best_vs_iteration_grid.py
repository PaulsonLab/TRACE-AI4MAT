#!/usr/bin/env python
"""Journal-style benchmark figures with bootstrap confidence intervals.

Reads   <root>/<oracle>/<method>/seed_*/trajectory_seed_*.csv
Writes  figures/best_vs_iteration_grid.{png,pdf,svg}   mean best-so-far vs iteration, one panel per oracle
        figures/performance_profile.{png,pdf,svg}      score distribution over all (oracle, seed) runs
        figures/*.csv                                  the plotted numbers

Statistics follow Agarwal et al., "Deep RL at the Edge of the Statistical
Precipice" (NeurIPS 2021): percentile bootstrap 95% confidence intervals
instead of SD/SEM bands, and a performance profile alongside the curves. Bootstrap
resampling is over whole runs (seeds), never over iterations, so the
within-run correlation of a best-so-far curve is preserved. For the profile
the resampling is stratified by oracle.

Run with the benchmark environment interpreter:
    ~/anaconda3/envs/trace_benchmark/bin/python plot_best_vs_iteration_grid.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# What gets plotted
# --------------------------------------------------------------------------- #

ORACLES = [
    ("penalized_logp", "Penalized logP"),
    ("qed", "QED"),
    ("median_2", "Median molecule 2"),
    ("osimertinib_mpo", "Osimertinib MPO"),
    ("scaffold_hop", "Scaffold hop"),
    ("deco_hop", "Deco hop"),
    ("drd2", "DRD2"),
    ("jnk3", "JNK3"),
]

# (id, label, colour, dash, marker). The palette clears the dataviz all-pairs
# gates on white (worst-pair CVD dE 9.1, normal-vision dE 16.4, all >= 3:1
# contrast); dashes and markers are secondary encoding so identity survives
# colourblindness and greyscale printing.
METHODS = [
    ("trace_turbomol", "TRACE", "#1050c0", (0, ()), "o"),
    ("graph_ga", "Graph-GA", "#ef5f38", (0, (4.0, 1.6)), "s"),
    ("smiles_ga", "SMILES-GA", "#20a0b0", (0, (1.2, 1.3)), "^"),
    ("gpbo_graph_ga", "GP-BO + Graph-GA", "#a00070", (0, (5.0, 1.4, 1.0, 1.4)), "D"),
    ("stoned_selfies", "STONED-SELFIES", "#b070ff", (0, (3.0, 1.2, 1.0, 1.2, 1.0, 1.2)), "v"),
    ("reinvent4_external", "REINVENT4", "#606000", (0, (6.5, 1.5, 1.5, 1.5)), "P"),
    # Frozen SELFIES-VAE latent-space BO, budget-scaled trust region (the arm whose
    # failure_tolerance = ceil(dim/bsz) = 6 actually adapts within 20 iterations).
    # #8510f9 clears every dataviz gate including the 3:1 contrast floor that the
    # previous mint (#47d18c, 1.9:1) failed; its tightest pairs are CVD dE 9.1 vs
    # TRACE blue and normal-vision dE 16.4 vs STONED's purple. No green passes --
    # that family is boxed in by teal and olive.
    ("lolbo_frozen_tr_scaled", "LOL-BO", "#8510f9", (0, (2.4, 1.2)), "X"),
]
# The no-trust-region ablation shares LOL-BO's hue (same method, TR vs no TR) and
# is appended only when --include-global is passed, so the headline figures stay
# at 7 series unless you ask for the ablation.
GLOBAL_ARM = ("lolbo_frozen_global", "LOL-BO (no TR)", "#8510f9", (0, (1.0, 1.4)), "P")

MAIN_ORACLES = ["qed", "median_2", "osimertinib_mpo", "scaffold_hop", "deco_hop", "drd2"]
ORACLE_SETS = {
    "main": MAIN_ORACLES,
    "all": [o for o, _ in ORACLES],
    "penalized_logp": ["penalized_logp"],
    "jnk3": ["jnk3"],
}

EMPHASIS = "trace_turbomol"          # drawn thicker and on top
LOW_CONTRAST: set[str] = set()   # all series now clear 3:1 vs the surface

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#1a1a1a", "#4d4d4d", "#767676", "#e4e4e4", "#b8b8b8", "#ffffff"

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_runs(root: Path, oracle: str, method: str) -> Optional[np.ndarray]:
    """Best-so-far curves for one (oracle, method) as a (n_seeds, n_iters) array."""
    curves = []
    for path in sorted((root / oracle / method).glob("seed_*/trajectory_seed_*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        key = "iteration" if "iteration" in rows[0] else "generation"
        rows.sort(key=lambda r: float(r[key]))
        curves.append(np.maximum.accumulate([float(r["best_so_far"]) for r in rows]))

    if not curves:
        return None
    length = min(len(c) for c in curves)          # trim to the shortest run
    return np.vstack([c[:length] for c in curves])


def load_all(root: Path, keep: Optional[List[str]] = None) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], np.ndarray]]:
    runs, oracles = {}, []
    for oracle, label in ORACLES:
        if keep is not None and oracle not in keep:
            continue
        found = {m: load_runs(root, oracle, m) for m, *_ in METHODS}
        found = {m: a for m, a in found.items() if a is not None}
        if found:
            oracles.append((oracle, label))
            runs.update({(oracle, m): a for m, a in found.items()})
    return oracles, runs


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def boot_ci(runs: np.ndarray, reps: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """95% percentile bootstrap CI for the mean of a (n_seeds, n_iters) array.

    Whole runs are resampled, so every iteration in a replicate comes from the
    same set of seeds.
    """
    idx = rng.integers(0, runs.shape[0], size=(reps, runs.shape[0]))
    draws = runs[idx].mean(axis=1)                        # (reps, n_iters)
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def normalised_finals(oracles, runs) -> Dict[str, np.ndarray]:
    """Final scores as the fraction of attainable headroom captured, per method.

    Every method starts from the identical molecule within a given (oracle,
    seed), so each run is scored as

        (final - start_of_that_seed) / (best_any_method_on_that_oracle - start)

    which removes the shared starting-point variance and puts all six oracles
    on one 0-1 scale. Returns one (n_oracles, n_seeds) array per method.
    """
    out = {m: [] for m, *_ in METHODS}
    for oracle, _ in oracles:
        present = [(m, runs[(oracle, m)]) for m, *_ in METHODS if (oracle, m) in runs]
        start = present[0][1][:, 0]                                  # shared per seed
        ceiling = max(float(a[:, -1].max()) for _, a in present)
        headroom = np.maximum(ceiling - start, 1e-12)
        for method, a in present:
            out[method].append((a[:, -1] - start) / headroom)
    return {m: np.vstack(rows) for m, rows in out.items() if rows}


def profile(scores: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Fraction of runs scoring at or above each threshold."""
    return (scores.ravel()[:, None] >= taus[None, :]).mean(axis=0)


def profile_ci(scores: np.ndarray, taus: np.ndarray, reps: int, rng: np.random.Generator):
    """95% CI for a performance profile, bootstrap stratified by oracle.

    `scores` is (n_oracles, n_seeds); seeds are resampled within each oracle.
    """
    n_oracles, n_seeds = scores.shape
    rows = np.arange(n_oracles)[:, None, None]
    idx = rng.integers(0, n_seeds, size=(reps, n_oracles, n_seeds))
    draws = np.stack([profile(scores[rows[:, 0], i], taus) for i in idx])
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


# --------------------------------------------------------------------------- #
# Style
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
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",   # embed TrueType
    })


def style_axis(ax, integer_x: bool = True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=2.6, pad=2.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=integer_x, nbins=5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))


def draw_series(ax, x, centre, lo, hi, style) -> None:
    method, _, colour, dash, marker = style
    emphasis = method == EMPHASIS
    z = 6 if emphasis else 4
    width = 1.7 if emphasis else (1.6 if method in LOW_CONTRAST else 1.2)
    ax.fill_between(x, lo, hi, color=colour, alpha=0.13, linewidth=0, zorder=z - 2)
    ax.plot(x, centre, color=colour, linestyle=dash, linewidth=width,
            marker=marker, markevery=(0, max(1, len(x) // 5)), markersize=3.2,
            markeredgewidth=0.55, markeredgecolor=SURFACE, zorder=z, clip_on=False)


def bottom_legend(fig, y, ncol, pt, present):
    """Shared legend; entries reordered so a column-major legend reads left-to-right."""
    handles = [
        Line2D([0], [0], color=c, linestyle=d,
               linewidth=1.7 if m == EMPHASIS else (1.6 if m in LOW_CONTRAST else 1.2),
               marker=k, markersize=3.4, markeredgewidth=0.55, markeredgecolor=SURFACE)
        for m, _, c, d, k in present
    ]
    labels = [label for _, label, *_ in present]
    order = (range(len(present)) if len(present) % ncol
             else [i for c in range(ncol) for i in range(c, len(present), ncol)])
    legend = fig.legend([handles[i] for i in order], [labels[i] for i in order],
                        loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol,
                        frameon=False, fontsize=pt, handlelength=2.6,
                        handletextpad=0.6, columnspacing=2.0, labelspacing=0.55)
    for text in legend.get_texts():
        text.set_color(INK)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def figure_curves(oracles, runs, reps, rng, relative=False, pt=8.0, ncols=4, width=9.4):
    """Grid of mean best-so-far vs iteration, one panel per oracle.

    With `relative`, each run is plotted as its improvement over its own
    starting molecule. Every method sees the identical start within a given
    (oracle, seed), so that shared variance is pure nuisance: subtracting it
    removes the wide, uninformative interval at iteration 0 that otherwise sets
    the y-scale on the oracles whose starting scores vary a lot.
    """
    apply_style(pt)
    nrows = math.ceil(len(oracles) / ncols)
    # Below the panels, in inches: tick labels, "Iteration", legend, footnote.
    height = 2.05 * nrows + 1.30
    fig = plt.figure(figsize=(width, height))
    top, bottom = 1 - 0.30 / height, 1.30 / height
    grid = fig.add_gridspec(nrows, ncols, left=0.075, right=0.995, top=top,
                            bottom=bottom, wspace=0.30, hspace=0.42)

    table, n_iters = [], 0
    for index, (oracle, label) in enumerate(oracles):
        ax = fig.add_subplot(grid[index // ncols, index % ncols])
        style_axis(ax)
        for style in METHODS:
            data = runs.get((oracle, style[0]))
            if data is None:
                continue
            if relative:
                data = data - data[:, :1]          # improvement over this seed's start
            x = np.arange(data.shape[1])
            n_iters = max(n_iters, data.shape[1])
            centre = data.mean(axis=0)
            lo, hi = boot_ci(data, reps, rng)
            draw_series(ax, x, centre, lo, hi, style)
            table += [(oracle, style[0], style[1], int(i), data.shape[0], c, a, b)
                      for i, c, a, b in zip(x, centre, lo, hi)]
        ax.set_xlim(0, n_iters - 1)
        ax.set_title(label, fontsize=pt + 1, color=INK, pad=5.0, fontweight="semibold")
        ax.text(-0.02, 1.13, f"({chr(ord('a') + index)})", transform=ax.transAxes,
                fontsize=pt + 1.5, fontweight="bold", color=INK, ha="left", va="top")

    for index in range(len(oracles), nrows * ncols):
        fig.add_subplot(grid[index // ncols, index % ncols]).set_axis_off()

    fig.supxlabel("Iteration", fontsize=pt + 1.5, color=INK2, y=0.94 / height)
    fig.supylabel("Improvement over start" if relative else "Best score so far",
                  fontsize=pt + 1.5, color=INK2, x=0.006, y=(top + bottom) / 2)
    n_seeds = next(iter(runs.values())).shape[0]
    present = [s for s in METHODS if any((o, s[0]) in runs for o, _ in oracles)]
    bottom_legend(fig, 0.26 / height, 3, pt, present)
    # fig.text(0.5, 0.04 / height, f"Mean with 95% bootstrap CI ({reps} resamples over {n_seeds} seeds)",
    #          ha="center", va="bottom", fontsize=pt - 1, color=MUTED)
    return fig, table


def figure_profile(oracles, runs, reps, rng, pt=8.0, width=5.0):
    """Score distribution: fraction of runs above each normalised-score threshold."""
    apply_style(pt)
    finals = normalised_finals(oracles, runs)
    taus = np.linspace(0.0, 1.0, 201)

    height = 3.95
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0.115, 1.20 / height, 0.870, 1 - 1.44 / height])
    style_axis(ax, integer_x=False)

    table = []
    for style in METHODS:
        scores = finals.get(style[0])
        if scores is None:
            continue
        centre = profile(scores, taus)
        lo, hi = profile_ci(scores, taus, reps, rng)
        draw_series(ax, taus, centre, lo, hi, style)
        table += [(style[0], style[1], t, c, a, b) for t, c, a, b in zip(taus, centre, lo, hi)]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"Normalised score  $\tau$", fontsize=pt + 1, color=INK2, labelpad=3)
    ax.set_ylabel(r"Fraction of runs with score $\geq \tau$", fontsize=pt + 1, color=INK2)

    n_oracles, n_seeds = next(iter(finals.values())).shape
    bottom_legend(fig, 0.30 / height, 3, pt, [s for s in METHODS if s[0] in finals])
    fig.text(0.5, 0.04 / height,
             f"{n_oracles * n_seeds} runs per method ({n_oracles} oracles x {n_seeds} seeds); "
             f"95% stratified bootstrap CI ({reps} resamples)",
             ha="center", va="bottom", fontsize=pt - 1, color=MUTED)
    return fig, table


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def save(fig, out_dir: Path, stem: str, formats, dpi: int) -> List[Path]:
    paths = [out_dir / f"{stem}.{fmt}" for fmt in formats]
    for path in paths:
        fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return paths


def write_csv(path: Path, header, rows) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None, help="Results root (default: this script's folder).")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--formats", default="png,pdf,svg")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--reps", type=int, default=10000, help="Bootstrap resamples.")
    parser.add_argument("--rng-seed", type=int, default=0, help="Bootstrap RNG seed (reproducibility).")
    parser.add_argument("--oracles", default="main",
                        help="'main' (the six headline oracles), 'all', or a comma-separated list. "
                             "penalized_logp and jnk3 are plotted separately by default.")
    parser.add_argument("--name", default=None, help="Output basename (default derived from --oracles).")
    parser.add_argument("--include-global", action="store_true",
                        help="add the no-trust-region latent-BO ablation as an extra series.")
    parser.add_argument("--relative", action="store_true",
                        help="Plot improvement over each run's starting molecule "
                             "(drops the shared starting-point variance).")
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    if args.include_global and GLOBAL_ARM[0] not in {m for m, *_ in METHODS}:
        METHODS.append(GLOBAL_ARM)

    keep = ORACLE_SETS.get(args.oracles) or [o.strip() for o in args.oracles.split(",") if o.strip()]
    unknown = [o for o in keep if o not in {name for name, _ in ORACLES}]
    if unknown:
        raise SystemExit(f"Unknown oracle(s): {unknown}")
    basename = args.name or ("best_vs_iteration_grid" if len(keep) > 1 else f"best_vs_iteration_{keep[0]}")

    oracles, runs = load_all(root, keep)
    if not runs:
        print(f"No trajectory CSVs found under {root} for {keep}")
        return 1

    rng = np.random.default_rng(args.rng_seed)
    written: List[Path] = []

    # one panel stands alone; six tile 3-wide as before
    ncols = 3 if len(oracles) > 2 else len(oracles)
    width = 7.2 if len(oracles) > 2 else 3.1 * len(oracles) + 0.6
    fig, curve_rows = figure_curves(oracles, runs, args.reps, rng, relative=args.relative,
                                    ncols=ncols, width=width)
    written += save(fig, out_dir, basename, formats, args.dpi)
    written.append(write_csv(
        out_dir / f"{basename}.csv",
        ["oracle", "method_id", "method_label", "iteration", "n_seeds", "mean", "ci_low", "ci_high"],
        curve_rows,
    ))

    if len(oracles) < 2:  # a profile over one oracle is meaningless
        print(f"{len(oracles)} oracle x {len(METHODS)} methods -> {basename}")
        for path in written:
            print(f"  wrote {path}")
        return 0

    fig, profile_rows = figure_profile(oracles, runs, args.reps, rng)
    written += save(fig, out_dir, "performance_profile", formats, args.dpi)
    written.append(write_csv(
        out_dir / "performance_profile.csv",
        ["method_id", "method_label", "tau", "fraction_of_runs", "ci_low", "ci_high"],
        profile_rows,
    ))

    print(f"{len(oracles)} oracles x {len(METHODS)} methods, "
          f"{next(iter(runs.values())).shape[0]} seeds, {args.reps} bootstrap resamples "
          f"(rng seed {args.rng_seed})")
    for path in written:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
