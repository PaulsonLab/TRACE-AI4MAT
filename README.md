# TRACE — code, oracles, and results

Submission package for the AI4Mat workshop paper.

TRACE is a batch Bayesian-optimization method for molecular design that works
directly in discrete chemical space: no latent space, no pretrained generative
model. At each iteration it grows a candidate pool around the current best
molecule with a chemistry-aware perturbation generator, restricts that pool to a
**trust region** defined jointly by a Tanimoto similarity floor and a cardinality
budget, fits a Tanimoto-kernel GP over fragprints to everything scored so far, and
picks the next batch by q-Log Expected Improvement. The trust region expands after
sustained improvement and contracts after sustained failure.

This package contains everything needed to run TRACE and to plot every TRACE
result in the paper.

---

## Quick start

```bash
pip install -r requirements.txt

python verify_package.py                       # confirm the package is intact
python plot_trace_results.py                   # plot the bundled results
python run_trace.py --oracle qed --out my_run  # reproduce one oracle
```

`verify_package.py` is the check that matters. It re-scores 2,000 molecules
sampled from the bundled runs with the bundled oracle code and asserts every
score matches what was recorded. If it passes, the **oracle** code in this folder
is demonstrably the code that scored the results in this folder. It says nothing
about the surrogate, which has since moved to fragprints + qLogEI (see the note
above) — `run_trace.py` no longer reproduces the bundled curves.

---

## Layout

```text
run_trace.py                  Single entry point: runs TRACE on any/all oracles
plot_trace_results.py         Figures and tables from run outputs
verify_package.py             Integrity + reproduction self-check

generator.py                  Chemistry-aware candidate generator
trace_optimizer.py            TRACE controller, fragprint Tanimoto GP, qLogEI
benchmark_suite/
    oracles.py                All eight scoring oracles
    common.py                 Constraints, SA scoring, fingerprints, penalized logP
    rf_oracle.py              Portable numpy re-implementation of the JNK3 forest

oracle/                       Bundled oracle model files
    drd2_current.pkl.gz       PMO/TDC DRD2 SVC (gzipped scikit-learn pickle)
    jnk3_forest.npz           TDC JNK3 random forest, exported to plain numpy

initial_data/<oracle>/        Starting molecules: seed_{11,22,33,44,55}.csv + manifest
results/<oracle>/
                              The TRACE results reported in the paper
figures/                      Output of plot_trace_results.py
LICENSE                       MIT, for the code in this repository
```

---

## The three components

### 1. Generator — [generator.py](generator.py)

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

### 2. Optimizer — [trace_optimizer.py](trace_optimizer.py)

**Trust region.** State is a pair `(τ, K)`: a Tanimoto similarity floor `τ` and a
cardinality budget `K`. A candidate is admitted iff `T(x, center) ≥ τ`; the
admitted set is then sorted by similarity to the center and truncated to the top
`K`. Here `T` is plain RDKit Tanimoto on the **binary radius-2, 2048-bit Morgan
fingerprint** from `get_fp` in [generator.py](generator.py) — deliberately *not*
the representation the GP uses (see *Surrogate and acquisition* below). The same
fingerprint drives the similarity sort, the diversity pick and the
restart-diversity check, and every constant below is calibrated on it.

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

There is a second, separate restart trigger that the ladder description above
does not cover, and in practice it is the only one that fires: if the trust
region cannot supply a full batch, TRACE first retries with a relaxed-profile
global pool, and restarts if it still comes up short. All 27 restarts across the
bundled runs came from that path — at `batch_size = 49` the ladder floor of
`max(20, 49)` needs nine consecutive failed batches out of twenty, which never
happened. At smaller batch sizes the ladder path does fire.

**Surrogate and acquisition.** A `SingleTaskGP` with a `ScaleKernel(TanimotoKernel)`
over **fragprints**, fit by exact marginal likelihood on all scored molecules, with
targets standardized per fit. The batch is chosen by `qLogExpectedImprovement`
(256-sample `IIDNormalSampler`) under `optimize_acqf_discrete` over the admitted
trust-region candidates.



### 3. Oracles — [benchmark_suite/oracles.py](benchmark_suite/oracles.py)

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

## How TRACE was run

Every TRACE result in `results/` came from this configuration:

| Setting | Value |
| --- | --- |
| Oracle-call budget | 1,000 unique molecules per seed |
| Initial design | 20 molecules, scored (counts against the budget) |
| Adaptive batches | 20 iterations × 49 molecules = 980 |
| Seeds | 11, 22, 33, 44, 55 |
| Global pool size | 1,000 candidates per iteration |
| Pool refresh | not refreshed each iteration (`--refresh-global-pool-each-iter false`) |
| Initial molecules | `initial_data/<oracle>/`, sampled from ZINC-250k |

The budget identity `20 + 20 × 49 = 1000` is asserted at startup by
`--max-oracle-calls`, so a mis-specified run fails immediately instead of
silently producing a curve on the wrong budget.

Oracle calls are counted in **unique** molecules. Re-proposing a molecule already
in history is free, which is what makes the x-axis comparable across methods with
different duplicate rates.

Reproduce the full set:

```bash
python run_trace.py --oracle all --out results_repro
python plot_trace_results.py --results results_repro --out figures_repro
```

One oracle, or a subset:

```bash
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

### Initial molecules

`initial_data/<oracle>/seed_<s>.csv` holds the 20 starting molecules for that
oracle and seed, with `manifest.json` recording exactly how they were drawn from
ZINC-250k: pool size, score quantile window, minimum pairwise Tanimoto distance,
and the constraints in force. `run_trace.py` validates the manifest against the
requested oracle and seeds before starting, so initial data from one oracle
cannot silently be used for another.

---

## Results

`results/<oracle>/` holds, for each of the 8 oracles:

```text
benchmark_summary.json          Mean/std final best across seeds, config echo
benchmark_mean_std.csv/.png     Aggregate curve
best_so_far_matrix.npy          (n_seeds × n_iters+1) best-so-far matrix
final_scores_by_seed.csv        Final best per seed
seed_<s>/
    evaluations_seed_<s>.csv    Every scored molecule, in call order  <- main record
    trajectory_seed_<s>.csv     Per-iteration state: best, K, τ, status, center
    summary_seed_<s>.json       Best molecule, best score, SA, restarts
    turbo_state_seed_<s>.json   Final controller state
    checkpoint_seed_<s>.json    Full run state
```

`evaluations_seed_<s>.csv` is the primary record — one row per unique oracle
call, with `oracle_call`, `smiles`, `score`, `best_so_far`, `sa`, `heavy_atoms`,
plus the controller's `K`, `s_min` and `center_smiles` at that point. Every
figure and table in this package is derived from it.

`trajectory_seed_<s>.csv` is what to read to see the trust region working: the
`K`, `s_min`, `status` (`stable` / `expanded` / `shrunk` / `restart`) and
`action` (`success` / `failure`) columns record every controller decision.

## Plotting

```bash
python plot_trace_results.py                                  # bundled results
python plot_trace_results.py --results results_repro --out figures_repro
python plot_trace_results.py --oracles main                   # the paper's six
python plot_trace_results.py --oracles qed drd2 jnk3          # any subset
```

Produces, in `figures/`:

```text
trace_best_vs_calls_main.{pdf,svg,png}   2x3 grid, the six oracles the paper reports
trace_best_vs_calls_all.{pdf,svg,png}    2x4 grid, all eight bundled oracles
per_oracle/<oracle>.{pdf,svg,png}        one oracle per file
trace_curves.csv                         mean, 95% CI and per-seed value at every point
trace_summary.csv                        final best and top-1 AUC per oracle
```


## Environment

Verified on Python 3.14 (Linux, CPU) and Python 3.14.2 (Windows 11); see
`requirements.txt` for pinned versions.

## Citing

If you use this code, please cite the TRACE paper. A BibTeX entry will be added here once the paper has a venue record.
