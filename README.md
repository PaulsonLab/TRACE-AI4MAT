# TRACE — Trust Region Adaptive Chemical Exploration

Code, data and analysis for

> **TRACE: Adaptive Trust-Region Bayesian Optimization for Open-Ended Molecular Design**
> Madhav R. Muthyala, Mahsa Askari, Farshud Sorourifar, Evan Zhou, Joe Gauthier, Joel A. Paulson
> AI4Mat workshop, NeurIPS 2026.

TRACE is a batch Bayesian-optimization method for molecular design that works
directly in discrete chemical space: no latent space, no pretrained generative
model. At each iteration it grows a candidate pool around the current best
molecule with a chemistry-aware perturbation generator, restricts that pool to a
**trust region** defined jointly by a Tanimoto similarity floor and a cardinality
budget, fits a Tanimoto-kernel GP over fragprints to everything scored so far, and
picks the next batch by q-Log Expected Improvement. The trust region expands after
sustained improvement and contracts after sustained failure.

This repository holds the method, the oracles, **every run behind every number and
every figure in the paper** — TRACE and all six comparators — and the scripts that
turn those runs back into the paper's tables and figures.

---

## Quick start

```bash
pip install -r requirements.txt

python verify_package.py                       # confirm the oracles are intact
python analysis/topk_auc_analysis.py           # rebuild Tables 1 and 2
python analysis/reproduce_figures.py --figures all   # rebuild every data figure
python run_trace.py --oracle qed --out my_run  # run TRACE on one oracle
```

`verify_package.py` is the integrity check. It re-scores 2,000 molecules sampled
from the bundled runs with the bundled oracle code and asserts every score matches
what was recorded. If it passes, the oracle code in this repository is
demonstrably the code that scored the results in this repository.

---

## Reproducing the paper

Everything reported in the paper is derived from one primary record:
`<...>/seed_<s>/evaluations_seed_<s>.csv`, one row per unique oracle call. The
tables and the figures read that file and nothing else, so they cannot drift apart.

### Tables 1 and 2 — top-1 and top-10 AUC

```bash
python analysis/topk_auc_analysis.py
```

Reads `results/`, prints both tables as Markdown and writes four CSVs to
`analysis/verified_results/topk/`. It labels the latent-space comparator
"Frozen-VAE BO"; the paper calls the same runs "LOL-BO (frozen VAE)". The
expected output is the paper's: TRACE mean
rank **1.73** on top-1 and **1.27** on top-10 across the six main objectives. Every
per-task AUC matches the paper at the reported three decimals, and the top-10
ranks match exactly.

### Figures

```bash
python analysis/reproduce_figures.py --figures all --tasks all
```

| Paper figure | Output | Produced by |
| --- | --- | --- |
| Fig. 2 | `figures/paper/low_budget_benchmark_trajectories` | `reproduce_figures.py --figures benchmark` |
| Fig. 3 | `figures/ablation/proposal_region_ablation_trajectories` | `reproduce_figures.py --figures ablation` |
| Fig. 4–7 | `figures/fepc/*` | `analysis/plot_fepc_case_study.py` |
| Fig. 8 | `figures/ablation/generator_tr_ablation_<oracle>` | `analysis/plot_generator_tr_ablation.py` |
| Fig. 9 | `figures/ablation/batch_size_{effect,curves}` | `analysis/plot_batch_size_sweep.py` |
| Fig. 10 | `figures/paper/best_vs_calls_split_penalized_logp` | `reproduce_figures.py --figures benchmark` |
| — | `figures/paper/best_vs_calls_split_jnk3` | `--tasks all` (appendix objective) |


---

## Layout

```text
run_trace.py                  Single entry point: runs TRACE on any/all oracles
trace_optimizer.py            TRACE controller, fragprint Tanimoto GP, qLogEI
generator.py                  Chemistry-aware candidate generator
verify_package.py             Oracle integrity self-check
plot_trace_results.py         TRACE-only figures

benchmark_suite/              The eight objectives
    oracles.py                All eight scoring oracles
    common.py                 Constraints, SA scoring, fingerprints, penalized logP
    rf_oracle.py              Portable numpy re-implementation of the JNK3 forest
oracle/                       Bundled oracle model files
    drd2_current.pkl.gz       PMO/TDC DRD2 SVC (gzipped scikit-learn pickle)
    jnk3_forest.npz           TDC JNK3 random forest, exported to plain numpy

initial_data/<oracle>/        Starting molecules: seed_{11,22,33,44,55}.csv + manifest

results/<oracle>/<method>/    Main benchmark: 8 oracles x 7 methods x 5 seeds
ablations/
    proposal_region/          Fig. 3 and 8: {TRACE, Graph-GA} gen x {region, no region}
    batch_size/               Fig. 9: batch in {5, 10, 25, 40} on scaffold hopping
fepc_campaign/                Fig. 4-7: the eight-round FePc DFT campaign

analysis/                     Everything that turns runs into paper numbers
    verified_analysis.py      Loader, batch-safe curves, bootstrap CIs, permutation tests
    topk_auc_analysis.py      Tables 1 and 2
    verified_results/topk/    Those tables, as written by the run bundled here
    reproduce_figures.py      Driver: regenerates every data figure
    plot_generator_tr_ablation.py, plot_batch_size_sweep.py,
    plot_fepc_case_study.py, plot_best_vs_iteration_grid.py

plogp_benchmark_suite/        Our implementations of the four in-process baselines
    run_baseline.py           Runs one of them on one objective, paper protocol
baselines_external/           Configs and wrappers for the two external baselines
    reinvent4/                Suite config, scoring plugin, runner, evaluator
    lolbo/                    Frozen-VAE wrappers, trust region, surrogate, runner

figures/                      Regenerated output of the analysis scripts
LICENSE                       MIT, for the code in this repository
```

---

## Method

### Generator — [generator.py](generator.py)

`HybridPerturbator` builds a library of chemically valid neighbours around one or
more seed molecules. It applies 14 graph-edit operators — `add_fragment`,
`apply_reaction`, `fuse_ring`, `bridge_ring`, `internal_bridge`, `spiro_ring`,
`mutate_atom`, `open_ring`, `delete_fragment`, `delete_ring`, `shrink_ring`,
`peel_ring`, `micro_mutate`, `micro_halogen` — chosen by a softmax bandit that
reweights each operator by its running mean reward, so operators that have been
producing accepted molecules get sampled more often.

Every generated molecule is sanitized against the run's `MoleculeConstraints`
(max SA score, max SMILES length, max heavy atoms), so the generator cannot emit
anything the benchmark would reject. `trace_optimizer.py` wraps it as
`SAConstrainedHybridPerturbator`, which additionally enforces the SA ceiling
inside `_sanitize` rather than as a post-filter.

Both chemistry profiles run in every iteration, feeding different parts of the
candidate set — they are not alternatives:

- `strict` + `mode="balanced"` — the local pool, `N_target = max(50, K)`,
  regenerated around the current center every iteration.
- `relaxed` + `mode="global"` — the global pool, 1000 molecules. Built once at
  startup, seeded by *all* valid initial molecules (so the generator's multi-seed
  path is active there), and rebuilt only on restart, seeded by the new center.
  It is **not** refreshed between iterations unless
  `--refresh-global-pool-each-iter true` is passed, which no bundled result used.
- `relaxed` + `mode="global"` again — a fallback pool of `max(10·q, 100)`, built
  only when the trust region cannot fill a batch.

The generator's chemistry envelope is re-derived from the current center every
iteration, not just at restarts. 

### Optimizer — [trace_optimizer.py](trace_optimizer.py)

**Trust region.** State is a pair `(τ, K)`: a Tanimoto similarity floor `τ` and a
cardinality budget `K`. A candidate is admitted iff `T(x, center) ≥ τ`; the
admitted set is then sorted by similarity to the center and truncated to the top
`K`. Here `T` is plain RDKit Tanimoto on the **binary radius-2, 2048-bit Morgan
fingerprint** from `get_fp` in [generator.py](generator.py) — deliberately *not*
the representation the GP uses. The same fingerprint drives the similarity sort,
the diversity pick and the restart-diversity check, and every constant below is
calibrated on it.

τ and `K` are *coupled* — `τ` is a monotone decreasing linear function of `K`:

```
ratio = clip((K - K_min) / (K_max - K_min), 0, 1)
τ     = s_min_ceil - ratio · (s_min_ceil - s_min_floor)
```

so a large budget implies a permissive similarity floor and a small budget a
strict one. Defaults: `K_init = 200`, `K_min = 20`, `K_max = 1000`,
`s_min_floor = 0.10`, `s_min_ceil = 0.50`.

**Adaptation.** A TuRBO-style ladder. A batch counts as a success if its best
score beats the incumbent by more than `eps_improve · max(1, |best|)`
(`eps_improve = 1e-3`). After 5 consecutive successes `K` doubles; after 3
consecutive failures `K` halves. If a halving would put `K` below the restart
floor `max(K_min, batch_size)`, the run restarts instead: `K` resets to `K_init`
and the center jumps elsewhere. The new center is the highest-scoring molecule
that is Tanimoto-dissimilar (below `restart_diversity_threshold = 0.70`) from
every center already used — preferring previously productive iteration centers,
falling back to the full evaluated history, and finally to the global best. Every
restart is logged with its reason and its new center.

**Surrogate and acquisition.** A `SingleTaskGP` with a `ScaleKernel(TanimotoKernel)`
over **fragprints** (radius-3 Morgan 2048 bits concatenated with 85 RDKit `fr_*`
fragment counts), fit by exact marginal likelihood on all scored molecules, with
targets standardized per fit. The batch is chosen by `qLogExpectedImprovement`
(256-sample `IIDNormalSampler`) under `optimize_acqf_discrete` over the admitted
trust-region candidates.

### Oracles — [benchmark_suite/oracles.py](benchmark_suite/oracles.py)

| Oracle | Type |
| --- | --- |
| `penalized_logp` | RDKit property |
| `qed` | RDKit property |
| `median_2` | GuacaMol dual-target similarity (tadalafil + sildenafil) |
| `osimertinib_mpo` | GuacaMol multi-property objective |
| `scaffold_hop` | GuacaMol pharmacophore + SMARTS |
| `deco_hop` | GuacaMol pharmacophore + SMARTS |
| `drd2` | PMO/TDC activity classifier (bundled SVC) |
| `jnk3` | TDC activity classifier (bundled random forest) |


---

## Benchmark protocol

All seven methods share the same objective, the same constraints, the same five
seeds, the same unique-oracle-call budget and, where a method accepts starting
molecules, the same initial design.

| Setting | Value |
| --- | --- |
| Oracle-call budget | 1,000 unique molecules per seed |
| Initial design | 20 molecules, scored (counts against the budget) |
| Adaptive batches | 20 iterations × 49 molecules = 980 |
| Seeds | 11, 22, 33, 44, 55 |
| Initial molecules | `initial_data/<oracle>/`, sampled from ZINC-250k |

### Running the comparators

```bash
python -m plogp_benchmark_suite.run_baseline --method graph_ga --oracle qed
```

`--method` takes `graph_ga`, `smiles_ga`, `stoned_selfies` or `gpbo_graph_ga`,
and the `results/<oracle>/` subfolder of the same name holds what that method
produced. The remaining two comparators, `reinvent4_external` and
`lolbo_frozen_tr_scaled`, need their own environments and are covered below.

`run_baseline.py` supplies each baseline with the objective from
`benchmark_suite/oracles.py` and the starting molecules from
`initial_data/<oracle>/` — the same two inputs TRACE gets — and defaults to the
protocol above. The four baselines themselves are libraries; import
`run_*_benchmark` directly to vary anything the runner does not expose.

REINVENT4 and LOL-BO need their own environments and their own pretrained weights,
which are not redistributed here. `baselines_external/<method>/` holds the exact
configuration, the runner and the evaluator — enough to repeat the run, not enough
to skip installing the upstream project.

- `reinvent4/oracle_suite.json` records the per-oracle constraints and the budget
  rule. `reinvent_plugins/components/comp_trace_oracles.py` is the scoring
  component that exposes these objectives to REINVENT4;
  `run_reinvent_suite.py` drives the suite and `evaluate_reinvent_output.py`
  re-scores the generated SMILES. Inside the REINVENT4 environment these import a
  `trace_reinvent` package — a copy of the objective definitions carried into that
  environment, since it is a separate Python install. That copy is not shipped
  here to avoid a second, drift-prone oracle definition in one repository. It is
  not a different objective: scoring 150 molecules per task under both
  implementations gives a maximum absolute difference of 0 on all eight.
- `lolbo/` holds the frozen-VAE wrappers, the latent trust region, the deep-kernel
  surrogate and the runner, each documenting what it takes unmodified from
  upstream and what it re-implements (mostly removing hard-coded `.cuda()` calls).
  `trace_benchmark_env.yml` pins the environment the LOL-BO runs used. The
  GuacaMol-pretrained VAE weights are upstream's and are not redistributed.


---

## Data

### `results/<oracle>/<method>/`

```text
benchmark_summary.json          Mean/std final best across seeds, config echo
benchmark_mean_std.csv/.png     Aggregate curve
best_so_far_matrix.npy          (n_seeds × n_iters+1) best-so-far matrix
final_scores_by_seed.csv        Final best per seed
seed_<s>/
    evaluations_seed_<s>.csv    Every scored molecule, in call order  <- main record
    trajectory_seed_<s>.csv     Per-iteration state: best, K, τ, status, center
    summary_seed_<s>.json       Best molecule, best score, SA, restarts
    turbo_state_seed_<s>.json   Final controller state (TRACE only)
    checkpoint_seed_<s>.json    Full run state (TRACE only)
    restarts_seed_<s>.csv       One row per restart (TRACE only; absent if none)
```

`evaluations_seed_<s>.csv` is the primary record — one row per unique oracle call,
with `oracle_call`, `smiles`, `score`, `best_so_far`, `sa`, `heavy_atoms`, plus the
controller's `K`, `s_min` and `center_smiles` at that point.

`trajectory_seed_<s>.csv` is what to read to see the trust region working: `K`,
`s_min`, `status` (`stable` / `expanded` / `shrunk` / `restart`) and `action`
(`success` / `failure`) record every controller decision.

### `ablations/`

`proposal_region/<oracle>/<arm>/` is the crossed design behind Figures 3 and 8 —
{TRACE proposer, Graph-GA proposer} × {region on, off} at q=10 for 100 iterations
(1,020 cumulative calls, so these runs exceed the 1,000-call horizon the main
loader enforces). The paper reports `osimertinib_mpo` and `scaffold_hop`; the
`median_2` arms were run the same way and are included here as well.
Arm folders map to paper labels as `trace_turbomol` = TRACE gen + region,
`gpbo_trace_gen` = TRACE gen no region, `trace_tr_graph_ga` = Graph-GA gen +
region, `gpbo_graph_ga` = Graph-GA gen no region.

`batch_size/batch_<q>/scaffold_hop/<method>/` is the Figure 9 sweep: the same 1,020
calls spent at q ∈ {5, 10, 25, 40}, so a larger batch buys parallelism at the cost
of sequential feedback rounds.

### `fepc_campaign/`

The eight-round iron-phthalocyanine ORR campaign: 81 rows — the FePc seed at round
0 (0.4218 eV) plus eight rounds of ten designed candidates, every one with a
returned energy. The best, at round 8, reaches 2.5912 eV.
`campaign_evaluations.csv` is the per-molecule record with round, identifier,
SMILES and computed dissolution energy;
`campaign_rounds.csv` and the three `*locality*` files are the round-level
aggregates the figures draw. `analysis/plot_fepc_case_study.py` regenerates all of
them from `campaign_evaluations.csv` alone.

---

## Running TRACE

```bash
python run_trace.py --oracle all --out results_repro
python run_trace.py --oracle qed --out results_repro
python run_trace.py --oracle median_2 drd2 jnk3 --out results_repro
```

Smoke test (minutes, not a paper result):

```bash
python run_trace.py --oracle qed --n-iters 2 --batch-size 5 --seeds 11 \
    --max-oracle-calls 30 --global-pool-size 60 --out smoke_test
```

Runs are CPU-only; no GPU is needed. `--skip-existing true` resumes a partially
finished sweep at oracle granularity.

`initial_data/<oracle>/seed_<s>.csv` holds the 20 starting molecules for that
oracle and seed, with `manifest.json` recording exactly how they were drawn from
ZINC-250k: pool size, score quantile window, minimum pairwise Tanimoto distance,
and the constraints in force. `run_trace.py` validates the manifest against the
requested oracle and seeds before starting, so initial data from one oracle cannot
silently be used for another.

---

## Environment

See `requirements.txt` for pinned versions. Verified on Python 3.14 (Linux, CPU)
and Python 3.14.2 (Windows 11). 

---

## License and attribution

The code in this repository is MIT licensed (`LICENSE`). That covers our code
only, not the third-party material bundled with it, which stays under its own
terms: the DRD2 and JNK3 oracle models (PMO / Therapeutics Data Commons), the
GuacaMol objective definitions, the starting molecules drawn from ZINC-250k, the
fragprints representation and Tanimoto kernel introduced by GAUCHE, RDKit's
contributed SA scorer, and the Graph-GA and GP-BO comparators, which are adapted
from the MIT-licensed [`wenhao-gao/mol_opt`](https://github.com/wenhao-gao/mol_opt).
Consult those projects and the papers cited throughout this README for their
licence terms before redistributing, and confirm in particular that the upstream
terms permit redistribution of the two bundled model files in `oracle/`.
REINVENT4 and LOL-BO are not redistributed here at all — install them from
upstream under their own licences. When reporting scores on these objectives,
cite the original sources rather than this repository.

## Citing

```bibtex
@inproceedings{muthyala2026trace,
  title     = {{TRACE}: Adaptive Trust-Region {B}ayesian Optimization for
               Open-Ended Molecular Design},
  author    = {Muthyala, Madhav R. and Askari, Mahsa and Sorourifar, Farshud and
               Zhou, Evan and Gauthier, Joe and Paulson, Joel A.},
  booktitle = {AI4Mat Workshop, NeurIPS},
  year      = {2026},
}
```
