"""Regenerate every data figure in the paper from the data in this repository.

Figure families (select with --figures):

  benchmark  figures/paper/    Fig. 2  low_budget_benchmark_trajectories
                               Fig. 10 best_vs_calls_split_penalized_logp
                               plus JNK3 and benchmark_curves.csv
  ablation   figures/ablation/ Fig. 3  proposal mechanism x adaptive region
                               Fig. 8  region controller diagnostics
  batch      figures/ablation/ Fig. 9  batch size sensitivity on scaffold hopping
  fepc       figures/fepc/     Fig. 4  campaign trajectory and chemical space map
                               Fig. 5  locality relative to the evolving incumbent
                               Fig. 6  final transition lineage
                               Fig. 7  round-wise molecule gallery

Figure 1 is a hand-drawn schematic and has no data to regenerate.

The benchmark family is computed here through verified_analysis.py, the same
module behind topk_auc_analysis.py, so the figures and Tables 1-2 cannot drift
apart. collect() requires the complete 8 x 7 x 5 run set and raises naming any
missing evaluation log, so this doubles as a completeness check on the published
results. The other families delegate to the scripts that originally produced
them, run against the staged copies of their data.

Usage:
    python analysis/reproduce_figures.py                    # benchmark only
    python analysis/reproduce_figures.py --figures all      # every figure
    python analysis/reproduce_figures.py --figures fepc
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verified_analysis import (  # noqa: E402
    FIRST_CALL,
    GRID,
    HORIZON,
    METHOD_LABEL,
    METHOD_STYLE,
    METHODS,
    SEEDS,
    TASK_LABEL,
    apply_style,
    bootstrap_curve_ci,
    collect,
    save_figure,
    style_axis,
    write_csv,
)

# Paper Figure 2. Penalized logP and JNK3 are reported separately.
MAIN_TASKS = ("qed", "median_2", "osimertinib_mpo", "scaffold_hop", "deco_hop", "drd2")

SPLIT = 70          # call count at which the x-axis breaks
INIT_EDGE = 14      # left edge, so the 20-molecule initialization is visible
BAND_ALPHA = 0.16
INIT_COLOR = "#ECECEC"


def _draw(ax_left, ax_right, task, results, reps, rng, rows_out):
    """Draw one objective across the split-axis pair."""
    for method, *_ in METHODS:
        curves = np.stack([results[(task, method, s)].batch_curve for s in SEEDS])
        mean = curves.mean(axis=0)
        lo, hi = bootstrap_curve_ci(curves, reps, rng)
        color, dash, _marker = METHOD_STYLE[method]
        for ax in (ax_left, ax_right):
            ax.fill_between(GRID, lo, hi, color=color, alpha=BAND_ALPHA, linewidth=0)
            ax.plot(GRID, mean, color=color, linestyle=dash, linewidth=1.25,
                    solid_capstyle="round")
        for n, m, l, h in zip(GRID, mean, lo, hi):
            if n in (20, 60, 250, 500, 750, 1000):
                rows_out.append((task, METHOD_LABEL[method], int(n),
                                 f"{m:.6f}", f"{l:.6f}", f"{h:.6f}"))

    ax_left.axvspan(INIT_EDGE, FIRST_CALL, color=INIT_COLOR, linewidth=0, zorder=0)
    ax_left.set_xlim(INIT_EDGE, SPLIT)
    ax_right.set_xlim(SPLIT, HORIZON)
    # style_axis installs its own tick locator, so it must run before we pin ticks
    for ax in (ax_left, ax_right):
        style_axis(ax)
    ax_left.set_xticks([20, 60])
    ax_right.set_xticks([250, 500, 750, 1000])
    ax_right.tick_params(labelleft=False, left=False)
    # break markers
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=5, linestyle="none",
              color="#B5B5B5", mec="#B5B5B5", mew=0.9, clip_on=False)
    ax_left.plot([1, 1], [0, 1], transform=ax_left.transAxes, **kw)
    ax_right.plot([0, 0], [0, 1], transform=ax_right.transAxes, **kw)
    ax_left.set_title(TASK_LABEL[task], loc="left", pad=4)


def _legend(fig, ncol, y=-0.015):
    handles = [plt.Line2D([], [], color=METHOD_STYLE[m][0], linestyle=METHOD_STYLE[m][1],
                          linewidth=1.4, label=METHOD_LABEL[m]) for m, *_ in METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, y), columnspacing=1.6, handlelength=2.4)


def figure_grid(tasks, results, reps, rng, out_dir, stem, dpi, rows_out):
    """Multi-panel grid: 3 columns, one split-axis pair per objective."""
    ncols = 3
    nrows = (len(tasks) + ncols - 1) // ncols
    fig = plt.figure(figsize=(8.4, 2.5 * nrows + 0.9))
    # Spacer columns separate the objective groups; wspace alone would widen the
    # split-axis break by the same amount and destroy the broken-axis effect.
    gs = GridSpec(nrows, ncols * 3 - 1, figure=fig,
                  width_ratios=([1, 2.6, 0.75] * ncols)[:-1],
                  wspace=0.04, hspace=0.55)
    for i, task in enumerate(tasks):
        r, c = divmod(i, ncols)
        axl = fig.add_subplot(gs[r, 3 * c])
        axr = fig.add_subplot(gs[r, 3 * c + 1], sharey=axl)
        _draw(axl, axr, task, results, reps, rng, rows_out)
        if c == 0:
            axl.set_ylabel("Best score found")
    fig.supxlabel("Cumulative oracle calls", y=0.105, fontsize=7.6)
    _legend(fig, ncol=4)
    fig.subplots_adjust(bottom=0.185, top=0.955, left=0.075, right=0.99)
    save_figure(fig, out_dir / stem, dpi)
    plt.close(fig)
    print(f"  wrote {stem}")


def figure_single(task, results, reps, rng, out_dir, stem, dpi, rows_out):
    fig = plt.figure(figsize=(6.0, 3.2))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 2.6], wspace=0.05)
    axl = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[0, 1], sharey=axl)
    _draw(axl, axr, task, results, reps, rng, rows_out)
    axl.set_ylabel("Best score found")
    fig.supxlabel("Cumulative oracle calls", y=0.175, fontsize=7.6)
    _legend(fig, ncol=4, y=0.005)
    fig.subplots_adjust(bottom=0.33, top=0.93, left=0.105, right=0.99)
    save_figure(fig, out_dir / stem, dpi)
    plt.close(fig)
    print(f"  wrote {stem}")


# Paper Figure 3: the crossed design, as stored on disk.
ABLATION_ARMS = (
    ("trace_turbomol",    "TRACE gen + region",    "#1050C0", (0, ())),
    ("gpbo_trace_gen",    "TRACE gen, no region",  "#1050C0", (0, (3.0, 1.4))),
    ("trace_tr_graph_ga", "Graph-GA gen + region", "#E65F39", (0, ())),
    ("gpbo_graph_ga",     "Graph-GA gen, no region", "#E65F39", (0, (3.0, 1.4))),
)
ABLATION_TASKS = (("osimertinib_mpo", "Osimertinib MPO"), ("scaffold_hop", "Scaffold hop"))
ABLATION_HORIZON = 1020   # 20 initialization + 100 iterations x q=10


def _batch_safe_curve(path: Path, horizon: int) -> np.ndarray:
    """Best-so-far on a 1-call grid, updated only at completed batch endpoints.

    Matches the convention in Appendix B.2: the incumbent is refreshed when a
    whole batch has been observed and held until the next batch endpoint. Read
    directly rather than via verified_analysis.load_rows, which pins the horizon
    at the 1,000-call benchmark protocol; the ablation runs to 1,020.
    """
    import csv
    by_iter: dict[int, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        # TRACE arms log "iteration"; the GP-BO arms log "step".
        step_col = next((c for c in ("iteration", "step") if c in (reader.fieldnames or [])), None)
        if step_col is None:
            raise ValueError(f"{path}: no 'iteration' or 'step' column")
        for r in reader:
            it = 0 if r.get("phase") == "init" else int(r[step_col])
            by_iter.setdefault(it, []).append(float(r["score"]))
    if not by_iter:
        raise ValueError(f"no rows in {path}")
    n, best = 0, -np.inf
    events = []
    for it in sorted(by_iter):
        n += len(by_iter[it])
        best = max(best, max(by_iter[it]))
        events.append((n, best))
    grid = np.arange(FIRST_CALL, horizon + 1)
    curve = np.full(grid.shape, events[0][1], dtype=float)
    for call, value in events:
        curve[grid >= call] = value
    return curve


def figure_region_ablation(repo, reps, rng, out_dir, dpi):
    """Paper Figure 3: proposal mechanism x adaptive region control."""
    root = repo / "ablations" / "proposal_region"
    grid = np.arange(FIRST_CALL, ABLATION_HORIZON + 1)
    fig = plt.figure(figsize=(6.6, 3.0))
    gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 2.6, 0.75, 1, 2.6], wspace=0.04)
    for t_idx, (task, label) in enumerate(ABLATION_TASKS):
        axl = fig.add_subplot(gs[0, 3 * t_idx])
        axr = fig.add_subplot(gs[0, 3 * t_idx + 1], sharey=axl)
        for arm, _lab, color, dash in ABLATION_ARMS:
            paths = sorted((root / task / arm).glob("seed_*/evaluations_seed_*.csv"))
            if not paths:
                raise FileNotFoundError(f"no runs for {task}/{arm} under {root}")
            curves = np.stack([_batch_safe_curve(p, ABLATION_HORIZON) for p in paths])
            lo, hi = bootstrap_curve_ci(curves, reps, rng)
            for ax in (axl, axr):
                ax.fill_between(grid, lo, hi, color=color, alpha=BAND_ALPHA, linewidth=0)
                ax.plot(grid, curves.mean(axis=0), color=color, linestyle=dash, linewidth=1.25)
        for ax in (axl, axr):
            style_axis(ax)
        axl.axvspan(INIT_EDGE, FIRST_CALL, color=INIT_COLOR, linewidth=0, zorder=0)
        axl.set_xlim(INIT_EDGE, SPLIT); axr.set_xlim(SPLIT, ABLATION_HORIZON)
        axl.set_xticks([20, 60]); axr.set_xticks([250, 500, 750, 1000])
        axr.tick_params(labelleft=False, left=False)
        axl.spines["right"].set_visible(False); axr.spines["left"].set_visible(False)
        axl.set_title(label, loc="left", pad=4)
        if t_idx == 0:
            axl.set_ylabel("Best score found")
    handles = [plt.Line2D([], [], color=c, linestyle=d, linewidth=1.4, label=l)
               for _a, l, c, d in ABLATION_ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.6, handlelength=2.4)
    fig.supxlabel("Cumulative oracle calls", y=0.195, fontsize=7.6)
    fig.subplots_adjust(bottom=0.35, top=0.92, left=0.095, right=0.99)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_dir / "proposal_region_ablation_trajectories", dpi)
    plt.close(fig)
    print("  wrote proposal_region_ablation_trajectories")


def _run(script: Path, *argv: str) -> None:
    """Delegate one figure family to the script that originally produced it."""
    cmd = [sys.executable, str(script), *argv]
    print(f"\n--- {script.name} {' '.join(argv[:2])} ---")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    tail = (proc.stdout or proc.stderr).strip().splitlines()
    for line in tail[-6:]:
        print("  " + line)
    if proc.returncode != 0:
        raise SystemExit(f"{script.name} failed (exit {proc.returncode}):\n{proc.stderr}")


def other_families(which, here, repo, dpi) -> None:
    fig_abl = repo / "figures" / "ablation"
    if which in ("all", "ablation"):
        apply_style()
        figure_region_ablation(repo, 4000, np.random.default_rng(0), fig_abl, dpi)
        for oracle in ("osimertinib_mpo", "scaffold_hop"):
            _run(here / "plot_generator_tr_ablation.py", "--oracle", oracle,
                 "--root", str(repo / "ablations" / "proposal_region"),
                 "--sweep-root", str(repo / "ablations" / "batch_size"),
                 "--out-dir", str(fig_abl), "--dpi", str(dpi))
    if which in ("all", "batch"):
        _run(here / "plot_batch_size_sweep.py",
             "--root", str(repo / "ablations" / "batch_size"),
             "--out-dir", str(fig_abl), "--dpi", str(dpi))
    if which in ("all", "fepc"):
        _run(here / "plot_fepc_case_study.py",
             "--campaign", str(repo / "fepc_campaign" / "campaign_evaluations.csv"),
             "--out-dir", str(repo / "figures" / "fepc"), "--dpi", str(dpi))


def main() -> int:
    here = Path(__file__).resolve().parent
    repo = here.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", choices=("benchmark", "ablation", "batch", "fepc", "all"),
                    default="benchmark", help="which figure family to regenerate")
    ap.add_argument("--raw-root", default=str(repo / "results"),
                    help="directory holding <task>/<method>/seed_<n>/evaluations_seed_<n>.csv")
    ap.add_argument("--out-dir", default=str(repo / "figures" / "paper"))
    ap.add_argument("--tasks", choices=("main", "all"), default="main",
                    help="'main' = paper Figure 2 plus penalized logP; 'all' adds JNK3")
    ap.add_argument("--reps", type=int, default=4000, help="bootstrap resamples")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if args.figures != "benchmark":
        other_families(args.figures, here, repo, args.dpi)
        if args.figures != "all":
            return 0

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading runs from {raw_root} ...")
    results = collect(raw_root)   # raises unless all 8 x 7 x 5 logs are present
    print(f"  {len(results)} runs loaded and validated "
          f"(calls {FIRST_CALL}-{HORIZON}, {len(SEEDS)} seeds)")

    apply_style()
    rng = np.random.default_rng(args.rng_seed)
    rows: list[tuple] = []

    figure_grid(MAIN_TASKS, results, args.reps, rng, out_dir,
                "low_budget_benchmark_trajectories", args.dpi, rows)
    figure_single("penalized_logp", results, args.reps, rng, out_dir,
                  "best_vs_calls_split_penalized_logp", args.dpi, rows)
    if args.tasks == "all":
        figure_single("jnk3", results, args.reps, rng, out_dir,
                      "best_vs_calls_split_jnk3", args.dpi, rows)

    write_csv(out_dir / "benchmark_curves.csv",
              ("task", "method", "oracle_calls", "mean_best", "ci_low", "ci_high"),
              rows)
    print(f"  wrote benchmark_curves.csv ({len(rows)} rows)")
    print(f"\nFigures written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
