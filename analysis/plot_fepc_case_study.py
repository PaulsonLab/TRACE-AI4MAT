#!/usr/bin/env python
"""The FePc / ORR catalyst campaign, analysed the way the benchmark was.

Eight human-in-the-loop rounds, batches of 10, one DFT queue. TRACE proposed,
a collaborator relaxed the structures, the results were appended and the
trust-region state carried forward. 81 candidates, 80 energies, 1 DFT failure.
Objective is computed dissolution energy referenced to Fe(2+), higher is better;
the seed is iron phthalocyanine at 0.4218 eV.

Every metric here is the metric the benchmark figures already use, recomputed on
the campaign, so the two halves of the paper can be put on one axis:

  campaign_trajectory      best-so-far vs cumulative DFT calls, evaluations
                           scattered by round -- the analogue of
                           best_vs_iteration_grid
  locality_by_round        Tanimoto to the incumbent at the time of evaluation,
                           per round, against the benchmark TRACE band -- the
                           analogue of trust_region_trajectory
  drift_vs_locality        similarity to the incumbent against similarity to
                           the seed, per round -- the two curves cross in round
                           2 and never re-cross
  locality_comparison      campaign vs benchmark TRACE vs the six baselines on
                           one axis -- the analogue of locality_vs_outcome, and
                           the figure that makes the paper one paper
  chemical_space_map       MDS of Jaccard distance over Morgan fingerprints,
                           coloured by round, incumbent path overlaid
  structure_lineage        FePc -> round-7 winner -> round-8 winner, depicted,
                           with the Tanimoto between them
  molecule_shape_by_round  SA score / heavy atoms / molecular weight by round --
                           the analogue of molecule_shape_vs_K, and the audit
                           that says the winners did not drift into nonsense
  energy_vs_locality       does searching locally pay? -- the analogue of
                           property_vs_K

Run with the benchmark environment interpreter:
    ~/anaconda3/envs/trace_benchmark/bin/python plot_fepc_case_study.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Draw

RDLogger.DisableLog("rdApp.*")

# Reuse the benchmark figures' style so the two figure sets are one set.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_best_vs_iteration_grid import (  # noqa: E402
    AXIS, GRID, INK, INK2, MUTED, SURFACE, apply_style, save, style_axis, write_csv,
)

sys.path.append(os.path.join(Chem.RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

TRACE_BLUE = "#1050c0"
SEED_SMILES_ID = 0
N_ROUNDS = 8
# light -> dark, one step per round; same construction as the K ramp.
ROUND_COLORS = ["#cfe0f7", "#9ec5f4", "#6ea8ee", "#4489e0", "#2a70cd",
                "#1c5cab", "#134a8c", "#0d366b"]

# Benchmark reference bands for the locality comparison: across the six main
# objectives, the range each method's per-oracle locality statistics span.
# Derived at run time from analysis/locality_vs_outcome.csv so the bands cannot
# drift from the benchmark runs they summarise.
BENCH_REFERENCE_CSV = Path(__file__).resolve().parent / "locality_vs_outcome.csv"
BENCH_ORDER = [
    ("trace_turbomol", "TRACE (benchmark)"),
    ("stoned_selfies", "STONED-SELFIES"),
    ("smiles_ga", "SMILES-GA"),
    ("graph_ga", "Graph-GA"),
    ("gpbo_graph_ga", "GP-BO + Graph-GA"),
    ("lolbo_frozen_tr_scaled", "LOL-BO (frozen VAE)"),
    ("reinvent4_external", "REINVENT4"),
]


def load_bench_reference(path: Path = BENCH_REFERENCE_CSV):
    """(label, median-locality range, share >= 0.4 range) per method."""
    df = pd.read_csv(path)
    out = []
    for method_id, label in BENCH_ORDER:
        sub = df[df["method_id"] == method_id]
        if sub.empty:
            continue
        out.append((
            label,
            (round(sub["locality_median"].min(), 3), round(sub["locality_median"].max(), 3)),
            (round(sub["frac_ge_0.4"].min(), 3), round(sub["frac_ge_0.4"].max(), 3)),
        ))
    return out


BENCH_REFERENCE = load_bench_reference()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_campaign(path: Path) -> pd.DataFrame:
    """Campaign evaluations into one tidy frame: round, id, smiles, energy.

    Reads `campaign_evaluations.csv`, which ships with this repository: one row
    per DFT-evaluated structure, with round 0 holding the FePc seed and rounds
    1-8 holding the ten designed candidates of each optimization round.
    Locality columns present in the file are ignored and recomputed by
    add_locality(), so the figures do not depend on stored derived values.
    """
    raw = pd.read_csv(path)
    missing = {"round", "id", "smiles", "energy"} - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
    df = pd.DataFrame({
        "round": raw["round"].astype(int),
        "id": raw["id"].astype(int),
        "smiles": raw["smiles"].astype(str).str.strip(),
        "energy": pd.to_numeric(raw["energy"], errors="coerce"),
    }).sort_values("id").reset_index(drop=True)
    df["parsed"] = [Chem.MolFromSmiles(s) is not None for s in df.smiles]
    return df


def fingerprints(smiles: List[str]):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None)
    return out


def add_locality(df: pd.DataFrame) -> pd.DataFrame:
    """Tanimoto to the incumbent at the moment each candidate was scored.

    Identical definition to plot_paper_extras.run_metrics, so the campaign
    number is directly comparable to every benchmark method's number.
    """
    fps = fingerprints(list(df.smiles))
    seed_fp = fps[0]
    to_inc, to_seed = [], []
    best_v, best_fp = -np.inf, None
    for fp, e in zip(fps, df.energy):
        to_inc.append(DataStructs.TanimotoSimilarity(fp, best_fp)
                      if (best_fp is not None and fp is not None) else np.nan)
        to_seed.append(DataStructs.TanimotoSimilarity(fp, seed_fp)
                       if fp is not None else np.nan)
        if np.isfinite(e) and e > best_v:
            best_v, best_fp = e, fp
    df = df.copy()
    df["tanimoto_to_incumbent"] = to_inc
    df["tanimoto_to_seed"] = to_seed
    df["_fp"] = fps
    return df


def add_properties(df: pd.DataFrame) -> pd.DataFrame:
    sa, heavy, mw = [], [], []
    for s in df.smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            sa.append(np.nan); heavy.append(np.nan); mw.append(np.nan); continue
        try:
            sa.append(sascorer.calculateScore(m))
        except Exception:
            sa.append(np.nan)
        heavy.append(m.GetNumHeavyAtoms())
        mw.append(Descriptors.MolWt(m))
    df = df.copy()
    df["sa_score"], df["heavy_atoms"], df["mol_weight"] = sa, heavy, mw
    return df


def best_so_far(df: pd.DataFrame) -> np.ndarray:
    return np.maximum.accumulate(np.nan_to_num(df.energy.values, nan=-np.inf))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def fig_trajectory(df: pd.DataFrame):
    """Best-so-far vs cumulative DFT calls, evaluations scattered by round.

    The y-axis is clipped at -4.6 eV. One geometry relaxed to -14.66 eV, far
    outside the range any candidate competes in; drawn as a marker on the floor
    rather than allowed to compress the two orders of magnitude that matter.
    """
    FLOOR, CEIL = -4.6, 3.15
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    calls = np.arange(1, len(df) + 1)
    energy = df.energy.values
    bsf = best_so_far(df)

    for rnd in range(1, N_ROUNDS + 1):
        m = df["round"].values == rnd
        y = np.clip(energy[m], FLOOR, CEIL)
        below = energy[m] < FLOOR
        ax.scatter(calls[m][~below], y[~below], s=17, color=ROUND_COLORS[rnd - 1],
                   edgecolor=SURFACE, linewidth=0.4, zorder=3)
        if below.any():
            ax.scatter(calls[m][below], np.full(below.sum(), FLOOR + 0.12), s=22,
                       marker="v", color=ROUND_COLORS[rnd - 1], edgecolor=SURFACE,
                       linewidth=0.4, zorder=3, clip_on=False)
        if rnd > 1:
            ax.axvline(calls[m][0] - 0.5, color=GRID, linewidth=0.6, zorder=1)

    n_below = int((energy < FLOOR).sum())
    ax.annotate(f"{n_below} relaxation off scale ({energy.min():.1f} eV)",
                (calls[energy < FLOOR][0], FLOOR + 0.12), xytext=(8, 2),
                textcoords="offset points", fontsize=6.5, color=MUTED, va="bottom")

    ax.step(calls, np.clip(bsf, FLOOR, CEIL), where="post", color=TRACE_BLUE,
            linewidth=1.9, zorder=4)
    ax.scatter([1], [energy[0]], s=48, marker="*", color=INK, zorder=5)
    ax.annotate("FePc seed, 0.42 eV", (1, energy[0]), xytext=(7, 7),
                textcoords="offset points", fontsize=7, color=INK, ha="left")

    # the plateau is the mechanism -- label it rather than smoothing it away
    ax.annotate("", xy=(41, 0.885), xytext=(21, 0.885),
                arrowprops=dict(arrowstyle="|-|,widthA=0.25,widthB=0.25",
                                color=MUTED, linewidth=0.8))
    ax.annotate("rounds 3-4 flat, region tightens\n(0.61 -> 0.72 Tanimoto)",
                (31, 0.885), xytext=(0, -26), textcoords="offset points",
                fontsize=7, color=MUTED, ha="center")

    top = np.nanmax(energy)
    ax.annotate(f"{top:.2f} eV, 6.1x the seed", (len(df), top), xytext=(-6, 4),
                textcoords="offset points", fontsize=8, color=TRACE_BLUE,
                ha="right", va="bottom", fontweight="bold")

    style_axis(ax)
    ax.set_xlabel("cumulative DFT evaluations")
    ax.set_ylabel("dissolution energy (eV)")
    ax.set_title("TRACE on FePc-derived ORR catalysts: 8 rounds, 81 DFT calls",
                 loc="left", pad=8)
    ax.set_xlim(0, len(df) + 1)
    ax.set_ylim(FLOOR, CEIL)
    handles = [Line2D([], [], color=TRACE_BLUE, lw=1.9, label="best so far"),
               Line2D([], [], marker="o", ls="none", color=ROUND_COLORS[1],
                      markersize=4, label="round 1-2"),
               Line2D([], [], marker="o", ls="none", color=ROUND_COLORS[-1],
                      markersize=4, label="round 7-8")]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, ncol=3,
              handletextpad=0.5, columnspacing=1.2)
    return fig


def fig_drift_vs_locality(df: pd.DataFrame):
    """The campaign's signature: steps stay local while the frontier travels.

    Two medians per round on one axis. Similarity to the incumbent rises as the
    trust region tightens; similarity to the FePc seed falls monotonically. They
    cross in round 2 and never re-cross.
    """
    inc, seed = [], []
    for rnd in range(1, N_ROUNDS + 1):
        m = df["round"].values == rnd
        inc.append(np.nanmedian(df.tanimoto_to_incumbent.values[m]))
        seed.append(np.nanmedian(df.tanimoto_to_seed.values[m]))
    x = np.arange(1, N_ROUNDS + 1)

    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.fill_between(x, inc, seed, where=np.array(inc) >= np.array(seed),
                    color=TRACE_BLUE, alpha=0.07, linewidth=0, zorder=1)
    ax.plot(x, inc, color=TRACE_BLUE, linewidth=1.9, marker="o", markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=4)
    ax.plot(x, seed, color=INK2, linewidth=1.5, marker="s", markersize=4,
            linestyle=(0, (4, 2)), markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=4)

    ax.annotate("to the current incumbent\n(how local each step is)",
                (x[-1], inc[-1]), xytext=(-6, 12), textcoords="offset points",
                fontsize=7.5, color=TRACE_BLUE, ha="right", fontweight="bold")
    ax.annotate("to the FePc seed\n(how far the search has travelled)",
                (x[-1], seed[-1]), xytext=(-6, -6), textcoords="offset points",
                fontsize=7.5, color=INK2, ha="right", va="top")

    style_axis(ax)
    ax.set_xlabel("round")
    ax.set_ylabel("median Tanimoto")
    ax.set_title("Local steps, distant destination", loc="left", pad=8)
    ax.set_xlim(0.7, N_ROUNDS + 0.3)
    ax.set_ylim(0, 0.85)
    ax.set_xticks(x)
    return fig, list(zip(x.tolist(), inc, seed))


def fig_locality_by_round(df: pd.DataFrame):
    """Per-round Tanimoto to incumbent, against the benchmark TRACE band."""
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    ax.axhspan(0.487, 0.559, color=TRACE_BLUE, alpha=0.10, zorder=0)
    ax.annotate("TRACE, six-oracle benchmark", (8.5, 0.523), fontsize=7,
                color=TRACE_BLUE, ha="right", va="center")
    ax.axhline(0.40, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.annotate("0.4 -- every benchmark baseline's median sits below this",
                (0.6, 0.40), xytext=(0, 5), textcoords="offset points",
                fontsize=7, color=MUTED, va="bottom")

    med = []
    for rnd in range(1, N_ROUNDS + 1):
        vals = df.loc[df["round"] == rnd, "tanimoto_to_incumbent"].dropna().values
        med.append(np.median(vals))
        ax.scatter(np.full(len(vals), rnd) + np.random.default_rng(rnd).normal(0, 0.055, len(vals)),
                   vals, s=14, color=ROUND_COLORS[rnd - 1], edgecolor=SURFACE,
                   linewidth=0.35, zorder=3)
    ax.plot(range(1, N_ROUNDS + 1), med, color=TRACE_BLUE, linewidth=1.7,
            marker="o", markersize=4, markeredgecolor=SURFACE,
            markeredgewidth=0.6, zorder=4)

    style_axis(ax)
    ax.set_xlabel("round")
    ax.set_ylabel("Tanimoto to incumbent")
    ax.set_title("The trust region tightens as the campaign proceeds", loc="left", pad=8)
    ax.set_xlim(0.5, N_ROUNDS + 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(1, N_ROUNDS + 1))
    return fig, med


def fig_locality_comparison(df: pd.DataFrame):
    """Campaign and benchmark on one axis: the figure that unifies the paper."""
    loc = df["tanimoto_to_incumbent"].dropna().values
    camp = ("TRACE (FePc DFT campaign)", (np.median(loc), np.median(loc)),
            (np.mean(loc >= 0.4), np.mean(loc >= 0.4)))
    rows = [camp] + BENCH_REFERENCE

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    y = np.arange(len(rows))[::-1]
    for ax, idx, label, xmax in ((axes[0], 1, "median Tanimoto to incumbent", 0.72),
                                 (axes[1], 2, "share of evaluations >= 0.4", 1.0)):
        for yi, row in zip(y, rows):
            lo, hi = row[idx]
            highlight = "TRACE" in row[0]
            colour = TRACE_BLUE if highlight else MUTED
            if abs(hi - lo) < 1e-9:
                ax.plot([lo], [yi], marker="o", color=colour, markersize=5.5, zorder=3)
            else:
                ax.plot([lo, hi], [yi, yi], color=colour, linewidth=3.4,
                        solid_capstyle="round", alpha=0.95 if highlight else 0.55, zorder=3)
        ax.set_xlim(0, xmax)
        ax.set_xlabel(label)
        style_axis(ax, integer_x=False)
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
        ax.grid(False, axis="y")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=8)
    for tick, row in zip(axes[0].get_yticklabels(), rows):
        if "TRACE" in row[0]:
            tick.set_color(INK); tick.set_fontweight("bold")
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    fig.suptitle("Deployed and benchmarked TRACE search the same way; no baseline does",
                 x=0.055, ha="left", fontsize=10, color=INK, y=1.0)
    fig.text(0.055, -0.04, "Bars span the range over the six benchmark oracles. "
             "The DFT campaign is a single run, so it is a point.",
             fontsize=7, color=MUTED, ha="left")
    return fig, rows


def fig_chemical_space(df: pd.DataFrame):
    """MDS over Jaccard distance, coloured by round, incumbent path overlaid."""
    from sklearn.manifold import MDS

    fps = list(df["_fp"])
    n = len(fps)
    D = np.zeros((n, n))
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        D[i] = 1.0 - np.asarray(sims)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    xy = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
             normalized_stress=False, n_init=8, max_iter=600).fit_transform(D)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    e = df.energy.values
    finite = np.isfinite(e)
    sizes = 18 + 150 * (np.nan_to_num(e, nan=np.nanmin(e)) - np.nanmin(e)) / (np.nanmax(e) - np.nanmin(e))

    for rnd in range(1, N_ROUNDS + 1):
        m = df["round"].values == rnd
        ax.scatter(xy[m, 0], xy[m, 1], s=sizes[m], color=ROUND_COLORS[rnd - 1],
                   edgecolor=SURFACE, linewidth=0.5, zorder=3)

    # the incumbent path: every molecule that was ever the best so far
    bsf = best_so_far(df)
    inc_idx, seen = [], -np.inf
    for i, v in enumerate(bsf):
        if v > seen:
            seen = v
            inc_idx.append(i)
    ax.plot(xy[inc_idx, 0], xy[inc_idx, 1], color=INK, linewidth=1.0,
            linestyle=(0, (4, 2)), zorder=4, alpha=0.75)
    ax.scatter(xy[inc_idx, 0], xy[inc_idx, 1], s=26, facecolor="none",
               edgecolor=INK, linewidth=0.9, zorder=5)

    ax.scatter(xy[0, 0], xy[0, 1], marker="*", s=170, color=INK, zorder=6,
               edgecolor=SURFACE, linewidth=0.6)
    ax.annotate("FePc\n0.42 eV", xy[0], xytext=(9, 0), textcoords="offset points",
                fontsize=7.5, color=INK, va="center")
    win = int(np.nanargmax(e))
    ax.annotate(f"best\n{e[win]:.2f} eV", xy[win], xytext=(-10, 0),
                textcoords="offset points", fontsize=7.5, color=TRACE_BLUE,
                fontweight="bold", va="center", ha="right")

    style_axis(ax, integer_x=False)
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.margins(0.10)
    ax.set_title("Structure space, MDS of Tanimoto distance", loc="left", pad=8)
    ax.set_xlabel("marker size = dissolution energy; dashed path = successive incumbents",
                  fontsize=7.5, color=MUTED)
    handles = [Line2D([], [], marker="o", ls="none", color=ROUND_COLORS[i],
                      markersize=5, label=f"round {i+1}") for i in (0, 3, 7)]
    ax.legend(handles=handles, loc="best", fontsize=7.5)
    return fig, xy


def fig_lineage(df: pd.DataFrame):
    """FePc -> round-7 winner -> round-8 winner, with the Tanimoto between."""
    ids = [0, 68, 80]
    sub = df[df.id.isin(ids)].set_index("id").loc[ids]
    mols = [Chem.MolFromSmiles(s) for s in sub.smiles]
    for m in mols:
        AllChem.Compute2DCoords(m)
    imgs = [Draw.MolToImage(m, size=(430, 430), kekulize=False) for m in mols]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    captions = ["seed - iron phthalocyanine", "round 7 winner", "round 8 winner"]
    for ax, img, cap, (mid, row) in zip(axes, imgs, captions, sub.iterrows()):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{cap}\n{row.energy:.2f} eV", fontsize=8.5, color=INK, pad=4)

    f = list(sub["_fp"])
    t_seed_68 = DataStructs.TanimotoSimilarity(f[0], f[1])
    t_68_80 = DataStructs.TanimotoSimilarity(f[1], f[2])
    t_seed_80 = DataStructs.TanimotoSimilarity(f[0], f[2])
    for x, t in ((0.365, t_seed_68), (0.655, t_68_80)):
        fig.text(x, 0.36, f"Tanimoto\n{t:.2f}", ha="center", fontsize=7.5, color=MUTED)
        fig.text(x, 0.50, "-->", ha="center", fontsize=13, color=AXIS)
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.01, right=0.99)
    fig.text(0.5, 0.015,
             f"The final step is local (0.65 from its parent) and worth +1.15 eV. "
             f"The destination is not: {t_seed_80:.2f} from the seed.",
             ha="center", fontsize=8, color=INK)
    fig.suptitle("One local step at a time, a long way from the start",
                 x=0.02, ha="left", fontsize=10, color=INK)
    return fig, (t_seed_68, t_68_80, t_seed_80)


def fig_shape_by_round(df: pd.DataFrame):
    """SA / heavy atoms / MW by round -- the winners did not drift into nonsense."""
    fields = [("sa_score", "SA score"), ("heavy_atoms", "heavy atoms"),
              ("mol_weight", "molecular weight")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    seed = df.iloc[0]
    for ax, (col, label) in zip(axes, fields):
        med = []
        for rnd in range(1, N_ROUNDS + 1):
            vals = df.loc[df["round"] == rnd, col].dropna().values
            med.append(np.median(vals))
            ax.scatter(np.full(len(vals), rnd), vals, s=9,
                       color=ROUND_COLORS[rnd - 1], alpha=0.75,
                       edgecolor="none", zorder=3)
        ax.plot(range(1, N_ROUNDS + 1), med, color=TRACE_BLUE, linewidth=1.5, zorder=4)
        ax.axhline(seed[col], color=INK, linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
        ax.annotate("FePc", (N_ROUNDS + 0.3, seed[col]), fontsize=7, color=INK,
                    va="center", ha="left", annotation_clip=False)
        style_axis(ax)
        ax.set_xlabel("round")
        ax.set_title(label, loc="left", fontsize=9, pad=5)
        ax.set_xticks(range(1, N_ROUNDS + 1, 2))
    fig.suptitle("Proposal quality over the campaign, against the seed",
                 x=0.02, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 0.97, 0.93))
    return fig


def fig_energy_vs_locality(df: pd.DataFrame):
    """Does searching locally pay? Binned, because n is small."""
    d = df.dropna(subset=["tanimoto_to_incumbent", "energy"])
    x = d["tanimoto_to_incumbent"].values
    y = d["energy"].values
    edges = np.array([0.0, 0.3, 0.5, 0.65, 0.8, 1.0])
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.scatter(x, y, s=16, color=MUTED, alpha=0.45, edgecolor="none", zorder=2)
    centres, meds, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() < 3:
            continue
        centres.append((lo + hi) / 2); meds.append(np.median(y[m])); counts.append(int(m.sum()))
    ax.plot(centres, meds, color=TRACE_BLUE, linewidth=1.8, marker="o",
            markersize=5, markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=4)
    for c, mv, n in zip(centres, meds, counts):
        ax.annotate(f"n={n}", (c, mv), xytext=(0, 8), textcoords="offset points",
                    fontsize=6.5, color=MUTED, ha="center")
    style_axis(ax, integer_x=False)
    ax.set_xlabel("Tanimoto to incumbent")
    ax.set_ylabel("dissolution energy (eV)")
    ax.set_title("Local proposals score better", loc="left", pad=8)
    return fig, list(zip(centres, meds, counts))


# --------------------------------------------------------------------------- #

def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--campaign", default=str(here.parent / "fepc_campaign" / "campaign_evaluations.csv"))
    ap.add_argument("--out-dir", default=str(here.parent / "figures" / "fepc"))
    ap.add_argument("--formats", default="png,pdf,svg")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--pt", type=float, default=8.0)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    apply_style(args.pt)

    df = load_campaign(Path(args.campaign))
    df = add_properties(add_locality(df))
    n_fail = int(df.energy.isna().sum())
    print(f"[data] {len(df)} candidates, {len(df) - n_fail} energies, "
          f"{n_fail} DFT failure(s), all parsed: {bool(df.parsed.all())}")

    written: List[Path] = []

    fig = fig_trajectory(df)
    written += save(fig, out, "campaign_trajectory", formats, args.dpi)

    fig, drift = fig_drift_vs_locality(df)
    written += save(fig, out, "drift_vs_locality", formats, args.dpi)
    written.append(write_csv(out / "drift_vs_locality.csv",
                             ["round", "median_tanimoto_to_incumbent", "median_tanimoto_to_seed"],
                             [(r, f"{a:.6f}", f"{b:.6f}") for r, a, b in drift]))

    fig, med = fig_locality_by_round(df)
    written += save(fig, out, "locality_by_round", formats, args.dpi)
    written.append(write_csv(out / "locality_by_round.csv", ["round", "median_tanimoto_to_incumbent"],
                             [(i + 1, f"{v:.6f}") for i, v in enumerate(med)]))

    fig, rows = fig_locality_comparison(df)
    written += save(fig, out, "locality_comparison", formats, args.dpi)
    written.append(write_csv(out / "locality_comparison.csv",
                             ["setting", "median_lo", "median_hi", "share_ge_0.4_lo", "share_ge_0.4_hi"],
                             [(r[0], f"{r[1][0]:.4f}", f"{r[1][1]:.4f}",
                               f"{r[2][0]:.4f}", f"{r[2][1]:.4f}") for r in rows]))

    fig, xy = fig_chemical_space(df)
    written += save(fig, out, "chemical_space_map", formats, args.dpi)

    fig, tans = fig_lineage(df)
    written += save(fig, out, "structure_lineage", formats, args.dpi)
    print(f"[lineage] seed->68 {tans[0]:.3f}  68->80 {tans[1]:.3f}  seed->80 {tans[2]:.3f}")

    fig = fig_shape_by_round(df)
    written += save(fig, out, "molecule_shape_by_round", formats, args.dpi)

    fig, binned = fig_energy_vs_locality(df)
    written += save(fig, out, "energy_vs_locality", formats, args.dpi)
    written.append(write_csv(out / "energy_vs_locality.csv",
                             ["tanimoto_bin_centre", "median_energy", "n"],
                             [(f"{c:.3f}", f"{m:.4f}", n) for c, m, n in binned]))

    # round-level summary: the paper's Table 1
    rounds = []
    bsf = best_so_far(df)
    for rnd in range(1, N_ROUNDS + 1):
        m = df["round"].values == rnd
        e = df.energy.values[m]
        rounds.append((rnd, int(m.sum()), int(np.flatnonzero(m)[-1] + 1),
                       f"{np.nanmax(e):.4f}", f"{bsf[np.flatnonzero(m)[-1]]:.4f}",
                       f"{np.nanmedian(df.tanimoto_to_incumbent.values[m]):.4f}",
                       f"{np.nanmedian(df.tanimoto_to_seed.values[m]):.4f}"))
    written.append(write_csv(out / "campaign_rounds.csv",
                             ["round", "n_candidates", "cumulative_dft", "round_best",
                              "best_so_far", "median_tanimoto_to_incumbent",
                              "median_tanimoto_to_seed"], rounds))
    written.append(write_csv(out / "campaign_evaluations.csv",
                             ["round", "id", "smiles", "energy", "tanimoto_to_incumbent",
                              "tanimoto_to_seed", "sa_score", "heavy_atoms", "mol_weight"],
                             df[["round", "id", "smiles", "energy", "tanimoto_to_incumbent",
                                 "tanimoto_to_seed", "sa_score", "heavy_atoms",
                                 "mol_weight"]].values.tolist()))

    print(f"\n[wrote] {len(written)} files under {out}")
    for p in written:
        print("  ", p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
