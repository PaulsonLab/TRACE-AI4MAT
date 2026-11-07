import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem, DataStructs

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from plogp_benchmark_suite.common import (
    MAX_SA_SCORE,
    CachedMoleculeOracle,
    MoleculeConstraints,
    PenalizedLogPOracle,
    default_initial_smiles_by_seed,
    finish_run,
    get_fp,
    parse_required_atomic_nums,
    sanitize_smiles_for_constraints,
    set_all_seeds,
    str_to_bool,
    write_json,
    write_rows_csv,
)
from plogp_benchmark_suite.graph_ga_ops import (
    propose_graph_ga_offspring,
    trim_population,
)


def select_diverse_unscored(candidates: List[str], target_size: int) -> List[str]:
    """Pick a diverse subset without using oracle scores."""
    unique = list(dict.fromkeys(candidates))
    if len(unique) <= int(target_size):
        return unique

    fps = {}
    for smiles in unique:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fps[smiles] = get_fp(mol)

    available = [smiles for smiles in unique if smiles in fps]
    if len(available) <= int(target_size):
        return available

    selected = [available[0]]
    remaining = available[1:]
    while len(selected) < int(target_size) and remaining:
        best_smiles = None
        best_distance = -1.0
        for smiles in remaining:
            fp = fps[smiles]
            min_distance = min(1.0 - DataStructs.TanimotoSimilarity(fp, fps[picked]) for picked in selected)
            if min_distance > best_distance:
                best_distance = min_distance
                best_smiles = smiles
        if best_smiles is None:
            break
        selected.append(best_smiles)
        remaining.remove(best_smiles)
    return selected


def run_graph_ga_seed(
    initial_smiles: List[str],
    seed: int,
    out_dir: Path,
    n_iters: int,
    batch_size: int,
    constraints: MoleculeConstraints,
    population_size: int = 120,
    mutation_rate: float = 0.067,
    max_attempt_multiplier: int = 100,
    candidate_pool_size: Optional[int] = None,
    oracle_spec=None,
) -> Dict:
    set_all_seeds(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_population = sanitize_smiles_for_constraints(initial_smiles, constraints)
    if not init_population:
        raise ValueError(f"No valid initial population for seed {seed}.")

    max_oracle_calls = len(init_population) + int(n_iters) * int(batch_size)
    if oracle_spec is None:
        oracle = PenalizedLogPOracle(max_oracle_calls=max_oracle_calls, constraints=constraints)
    else:
        oracle = CachedMoleculeOracle(
            max_oracle_calls=max_oracle_calls,
            constraints=constraints,
            score_mol=oracle_spec.score_mol,
            score_smiles=oracle_spec.score_smiles,
            oracle_name=oracle_spec.name,
            reward_label=oracle_spec.reward_label,
            metadata=oracle_spec.metadata,
        )

    population = init_population[:]
    if not population:
        raise ValueError(f"No valid initial population for seed {seed}.")

    init_scores = oracle.score_many(population, phase="init", step=0)
    population, population_scores = trim_population(
        population,
        score_lookup=lambda smiles: oracle.cache.get(smiles),
        population_size=population_size,
    )

    trajectory = [
        {
            "seed": int(seed),
            "generation": 0,
            "oracle_calls": int(len(oracle)),
            "n_batch": int(len([s for s in init_scores if s is not None])),
            "best_so_far": float(oracle.best()["score"]),
            "best_smiles": oracle.best()["smiles"],
            "population_size": int(len(population)),
            "target_batch": int(len([s for s in init_scores if s is not None])),
            "n_generated": int(len(population)),
        }
    ]

    generation = 0
    underfilled_batches = 0
    while not oracle.finished and generation < n_iters:
        generation += 1
        target = min(batch_size, oracle.remaining())
        pool_target = max(target, int(candidate_pool_size or target))
        offspring = propose_graph_ga_offspring(
            population_smiles=population,
            population_scores=population_scores,
            constraints=constraints,
            offspring_size=pool_target,
            mutation_rate=mutation_rate,
            max_attempt_multiplier=max_attempt_multiplier,
            exclude_smiles=oracle.cache.keys(),
        )

        if not offspring:
            # If crossover stalls, keep the run honest: no fake oracle calls.
            break

        batch = select_diverse_unscored(offspring, target_size=target)
        calls_before = len(oracle)
        oracle.score_many(batch, phase="iter", step=generation)
        n_new_calls = len(oracle) - calls_before
        if n_new_calls < target:
            underfilled_batches += 1

        candidate_pool = population + batch
        population, population_scores = trim_population(
            candidate_pool,
            score_lookup=lambda smiles: oracle.cache.get(smiles),
            population_size=population_size,
        )

        trajectory.append(
            {
                "seed": int(seed),
                "generation": int(generation),
                "oracle_calls": int(len(oracle)),
                "n_batch": int(n_new_calls),
                "best_so_far": float(oracle.best()["score"]),
                "best_smiles": oracle.best()["smiles"],
                "population_size": int(len(population)),
                "target_batch": int(target),
                "n_generated": int(len(batch)),
                "n_candidate_pool": int(len(offspring)),
            }
        )

    return finish_run(
        out_dir=out_dir,
        method="graph_ga",
        seed=seed,
        oracle=oracle,
        trajectory=trajectory,
        extra_summary={
            "n_iters_requested": int(n_iters),
            "batch_size": int(batch_size),
            "population_size": int(population_size),
            "mutation_rate": float(mutation_rate),
            "max_attempt_multiplier": int(max_attempt_multiplier),
            "candidate_pool_size": int(candidate_pool_size or batch_size),
            "candidate_pool_selection": "diverse_unscored",
            "underfilled_batches": int(underfilled_batches),
            "generations_completed": int(generation),
        },
    )


def run_graph_ga_benchmark(
    out_root: str = "benchmark_graph_ga_plogp",
    seeds: Optional[List[int]] = None,
    n_iters: int = 50,
    batch_size: int = 5,
    max_sa_score: float = MAX_SA_SCORE,
    max_smiles_length: Optional[int] = None,
    max_heavy_atoms: Optional[int] = None,
    required_atomic_nums: Optional[List[int]] = None,
    population_size: int = 120,
    mutation_rate: float = 0.067,
    max_attempt_multiplier: int = 100,
    candidate_pool_size: Optional[int] = None,
    initial_smiles_by_seed: Optional[Dict[int, List[str]]] = None,
    oracle_spec=None,
) -> Dict:
    if seeds is None:
        seeds = [11, 22, 33, 44, 55]

    constraints = MoleculeConstraints(
        max_sa_score=max_sa_score,
        max_smiles_length=max_smiles_length,
        max_heavy_atoms=max_heavy_atoms,
        required_atomic_nums=required_atomic_nums,
    )
    initials = initial_smiles_by_seed or default_initial_smiles_by_seed()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for seed in seeds:
        summary = run_graph_ga_seed(
            initial_smiles=initials[seed],
            seed=seed,
            out_dir=out_root / f"seed_{seed}",
            n_iters=n_iters,
            batch_size=batch_size,
            constraints=constraints,
            population_size=population_size,
            mutation_rate=mutation_rate,
            max_attempt_multiplier=max_attempt_multiplier,
            candidate_pool_size=candidate_pool_size,
            oracle_spec=oracle_spec,
        )
        summaries.append(summary)

    final_rows = [
        {
            "seed": int(s["seed"]),
            "best_score": float(s["best_score"]),
            "best_smiles": s["best_smiles"],
            "n_oracle_calls": int(s["n_oracle_calls"]),
        }
        for s in summaries
    ]
    write_rows_csv(out_root / "final_scores_by_seed.csv", final_rows)

    best_scores = [float(row["best_score"]) for row in final_rows]
    overall = {
        "method": "graph_ga",
        "oracle": oracle_spec.name if oracle_spec is not None else "penalized_logp",
        "reward_label": oracle_spec.reward_label if oracle_spec is not None else "penalized logP",
        "seeds": [int(s) for s in seeds],
        "n_iters": int(n_iters),
        "batch_size": int(batch_size),
        "population_size": int(population_size),
        "mutation_rate": float(mutation_rate),
        "max_attempt_multiplier": int(max_attempt_multiplier),
        "candidate_pool_size": int(candidate_pool_size or batch_size),
        "candidate_pool_selection": "diverse_unscored",
        "mean_final_best": float(np.mean(best_scores)),
        "std_final_best": float(np.std(best_scores)),
        "best_score_overall": float(np.max(best_scores)),
        "out_root": str(out_root),
    }
    write_json(out_root / "benchmark_summary.json", overall)
    return overall


def parse_args():
    parser = argparse.ArgumentParser(description="Run a focused Graph-GA pLogP benchmark.")
    parser.add_argument("--out", default="benchmark_graph_ga_plogp")
    parser.add_argument("--n-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=120)
    parser.add_argument("--mutation-rate", type=float, default=0.067)
    parser.add_argument("--max-attempt-multiplier", type=int, default=100)
    parser.add_argument("--candidate-pool-size", type=int, default=None)
    parser.add_argument("--max-sa", type=float, default=MAX_SA_SCORE)
    parser.add_argument("--max-smiles-length", type=int, default=None)
    parser.add_argument("--max-heavy-atoms", type=int, default=None)
    parser.add_argument("--require-fe", type=str_to_bool, default=False, metavar="true/false")
    parser.add_argument("--require-elements", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=[11, 22, 33, 44, 55])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_graph_ga_benchmark(
        out_root=args.out,
        seeds=args.seeds,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        max_sa_score=args.max_sa,
        max_smiles_length=args.max_smiles_length,
        max_heavy_atoms=args.max_heavy_atoms,
        required_atomic_nums=parse_required_atomic_nums(args.require_elements, args.require_fe),
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        max_attempt_multiplier=args.max_attempt_multiplier,
        candidate_pool_size=args.candidate_pool_size,
    )
