import argparse
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, DataStructs

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from plogp_benchmark_suite.common import (
    MAX_SA_SCORE,
    CachedMoleculeOracle,
    MoleculeConstraints,
    PenalizedLogPOracle,
    canonicalize_smiles,
    default_initial_smiles_by_seed,
    finish_run,
    get_fp,
    molecule_satisfies_constraints,
    parse_required_atomic_nums,
    sanitize_smiles_for_constraints,
    set_all_seeds,
    str_to_bool,
    write_json,
    write_rows_csv,
)
from plogp_benchmark_suite.graph_ga_ops import trim_population


TOKEN_RE = re.compile(
    r"(\[[^\[\]]+\]|Br|Cl|Si|Se|Na|Li|Mg|Ca|Al|[BCNOFPSIbcnops]|"
    r"\%\d{2}|\d|=|#|-|/|\\|\.|\(|\)|\+|-)"
)

BASE_ATOM_TOKENS = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Cl",
    "Br",
    "c",
    "n",
    "o",
    "s",
]

BASE_EDIT_TOKENS = ["C", "N", "O", "F", "Cl", "Br", "=", "#", "(", ")"]


def tokenize_smiles(smiles: str) -> List[str]:
    tokens = TOKEN_RE.findall(smiles)
    return tokens if tokens else list(smiles)


def detokenize_smiles(tokens: Sequence[str]) -> str:
    return "".join(tokens)


def randomized_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        return Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True)
    except TypeError:
        return Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)


def population_alphabet(population: Sequence[str]) -> List[str]:
    tokens = set(BASE_EDIT_TOKENS)
    for smiles in population:
        tokens.update(tokenize_smiles(smiles))
    return sorted(tokens)


def rank_parent_pool(
    population: Sequence[str],
    score_lookup,
    parent_pool_size: int,
) -> Tuple[List[str], np.ndarray]:
    pairs = []
    for smiles in population:
        score = score_lookup(smiles)
        if score is not None:
            pairs.append((float(score), smiles))
    pairs.sort(reverse=True, key=lambda item: item[0])
    pairs = pairs[: max(1, int(parent_pool_size))]
    parents = [smiles for _, smiles in pairs]
    if not parents:
        return [], np.asarray([], dtype=float)

    ranks = np.arange(1, len(parents) + 1, dtype=float)
    weights = 1.0 / ranks
    weights = weights / weights.sum()
    return parents, weights


def select_diverse_unscored(candidates: List[str], target_size: int) -> List[str]:
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


def crossover_tokens(parent_a: str, parent_b: str) -> List[str]:
    tokens_a = tokenize_smiles(randomized_smiles(parent_a))
    tokens_b = tokenize_smiles(randomized_smiles(parent_b))
    if len(tokens_a) < 2:
        return tokens_b[:]
    if len(tokens_b) < 2:
        return tokens_a[:]

    cut_a = random.randint(1, len(tokens_a) - 1)
    cut_b = random.randint(1, len(tokens_b) - 1)
    if random.random() < 0.5:
        return tokens_a[:cut_a] + tokens_b[cut_b:]
    return tokens_b[:cut_b] + tokens_a[cut_a:]


def mutate_tokens(
    tokens: Sequence[str],
    alphabet: Sequence[str],
    mutation_rate: float,
    min_mutations: int,
    max_mutations: int,
    max_token_delta: int,
) -> List[str]:
    mutated = list(tokens)
    if not mutated:
        mutated = [random.choice(BASE_ATOM_TOKENS)]

    n_edits = random.randint(max(1, int(min_mutations)), max(1, int(max_mutations)))
    target_cap = max(1, len(tokens) + int(max_token_delta))
    target_floor = max(1, len(tokens) - int(max_token_delta))

    for _ in range(n_edits):
        if random.random() > float(mutation_rate):
            continue

        choices = ["replace", "insert", "delete", "atom_substitute"]
        if len(mutated) <= target_floor:
            choices = ["replace", "insert", "atom_substitute"]
        if len(mutated) >= target_cap:
            choices = ["replace", "delete", "atom_substitute"]

        op = random.choice(choices)
        if op == "insert":
            pos = random.randint(0, len(mutated))
            token = random.choice(BASE_ATOM_TOKENS if random.random() < 0.7 else list(alphabet))
            mutated.insert(pos, token)
        elif op == "delete" and mutated:
            pos = random.randint(0, len(mutated) - 1)
            mutated.pop(pos)
        elif op == "atom_substitute" and mutated:
            atom_positions = [
                i
                for i, token in enumerate(mutated)
                if token in BASE_ATOM_TOKENS or token.startswith("[")
            ]
            if atom_positions:
                pos = random.choice(atom_positions)
                mutated[pos] = random.choice(BASE_ATOM_TOKENS)
        elif mutated:
            pos = random.randint(0, len(mutated) - 1)
            token = random.choice(BASE_ATOM_TOKENS if random.random() < 0.7 else list(alphabet))
            mutated[pos] = token

    return mutated[:target_cap]


def child_from_parents(
    parent_a: str,
    parent_b: str,
    alphabet: Sequence[str],
    constraints: MoleculeConstraints,
    mutation_rate: float,
    min_mutations: int,
    max_mutations: int,
    max_token_delta: int,
) -> Optional[str]:
    if random.random() < 0.8:
        tokens = crossover_tokens(parent_a, parent_b)
    else:
        tokens = tokenize_smiles(randomized_smiles(parent_a))

    tokens = mutate_tokens(
        tokens,
        alphabet=alphabet,
        mutation_rate=mutation_rate,
        min_mutations=min_mutations,
        max_mutations=max_mutations,
        max_token_delta=max_token_delta,
    )
    candidate = detokenize_smiles(tokens)
    canonical = canonicalize_smiles(candidate)
    if canonical is None:
        return None
    mol = Chem.MolFromSmiles(canonical)
    if not molecule_satisfies_constraints(mol, constraints):
        return None
    return canonical


def propose_smiles_ga_offspring(
    population_smiles: Sequence[str],
    score_lookup,
    constraints: MoleculeConstraints,
    target_size: int,
    exclude_smiles: Iterable[str],
    parent_pool_size: int = 40,
    mutation_rate: float = 0.7,
    min_mutations: int = 1,
    max_mutations: int = 5,
    max_token_delta: int = 8,
    max_attempt_multiplier: int = 300,
) -> List[str]:
    parents, weights = rank_parent_pool(population_smiles, score_lookup, parent_pool_size)
    if not parents:
        return []

    alphabet = population_alphabet(parents)
    excluded = set(exclude_smiles)
    accepted: List[str] = []
    accepted_set = set()
    attempts = max(int(target_size) * int(max_attempt_multiplier), int(target_size))

    for _ in range(attempts):
        if len(accepted) >= int(target_size):
            break
        parent_a = str(np.random.choice(parents, p=weights))
        parent_b = str(np.random.choice(parents, p=weights))
        child = child_from_parents(
            parent_a,
            parent_b,
            alphabet=alphabet,
            constraints=constraints,
            mutation_rate=mutation_rate,
            min_mutations=min_mutations,
            max_mutations=max_mutations,
            max_token_delta=max_token_delta,
        )
        if child is None:
            continue
        if child in excluded or child in accepted_set:
            continue
        accepted_set.add(child)
        accepted.append(child)

    return accepted


def make_oracle(max_oracle_calls: int, constraints: MoleculeConstraints, oracle_spec=None) -> CachedMoleculeOracle:
    if oracle_spec is None:
        return PenalizedLogPOracle(max_oracle_calls=max_oracle_calls, constraints=constraints)
    return CachedMoleculeOracle(
        max_oracle_calls=max_oracle_calls,
        constraints=constraints,
        score_mol=oracle_spec.score_mol,
        score_smiles=oracle_spec.score_smiles,
        oracle_name=oracle_spec.name,
        reward_label=oracle_spec.reward_label,
        metadata=oracle_spec.metadata,
    )


def run_smiles_ga_seed(
    initial_smiles: List[str],
    seed: int,
    out_dir: Path,
    n_iters: int,
    batch_size: int,
    constraints: MoleculeConstraints,
    population_size: int = 120,
    parent_pool_size: int = 40,
    mutation_rate: float = 0.7,
    min_mutations: int = 1,
    max_mutations: int = 5,
    max_token_delta: int = 8,
    max_attempt_multiplier: int = 300,
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
    oracle = make_oracle(max_oracle_calls=max_oracle_calls, constraints=constraints, oracle_spec=oracle_spec)

    init_scores = oracle.score_many(init_population, phase="init", step=0)
    population, population_scores = trim_population(
        init_population,
        score_lookup=lambda smiles: oracle.cache.get(smiles),
        population_size=population_size,
    )
    if not population:
        raise ValueError(f"No valid scored initial population for seed {seed}.")

    trajectory = [
        {
            "seed": int(seed),
            "generation": 0,
            "oracle_calls": int(len(oracle)),
            "n_batch": int(len([score for score in init_scores if score is not None])),
            "best_so_far": float(oracle.best()["score"]),
            "best_smiles": oracle.best()["smiles"],
            "population_size": int(len(population)),
            "target_batch": int(len([score for score in init_scores if score is not None])),
            "n_generated": int(len(population)),
        }
    ]

    generation = 0
    underfilled_batches = 0
    while not oracle.finished and generation < int(n_iters):
        generation += 1
        target = min(int(batch_size), oracle.remaining())
        pool_target = max(target, int(candidate_pool_size or target))
        candidate_pool = propose_smiles_ga_offspring(
            population_smiles=population,
            score_lookup=lambda smiles: oracle.cache.get(smiles),
            constraints=constraints,
            target_size=pool_target,
            exclude_smiles=oracle.cache.keys(),
            parent_pool_size=parent_pool_size,
            mutation_rate=mutation_rate,
            min_mutations=min_mutations,
            max_mutations=max_mutations,
            max_token_delta=max_token_delta,
            max_attempt_multiplier=max_attempt_multiplier,
        )

        if not candidate_pool:
            break

        batch = select_diverse_unscored(candidate_pool, target_size=target)
        calls_before = len(oracle)
        oracle.score_many(batch, phase="iter", step=generation)
        n_new_calls = len(oracle) - calls_before
        if n_new_calls < target:
            underfilled_batches += 1

        evaluated_pool = population + batch
        population, population_scores = trim_population(
            evaluated_pool,
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
                "n_candidate_pool": int(len(candidate_pool)),
            }
        )

        if n_new_calls <= 0:
            break

    return finish_run(
        out_dir=out_dir,
        method="smiles_ga",
        seed=seed,
        oracle=oracle,
        trajectory=trajectory,
        extra_summary={
            "n_iters_requested": int(n_iters),
            "batch_size": int(batch_size),
            "population_size": int(population_size),
            "parent_pool_size": int(parent_pool_size),
            "mutation_rate": float(mutation_rate),
            "min_mutations": int(min_mutations),
            "max_mutations": int(max_mutations),
            "max_token_delta": int(max_token_delta),
            "max_attempt_multiplier": int(max_attempt_multiplier),
            "candidate_pool_size": int(candidate_pool_size or batch_size),
            "candidate_pool_selection": "diverse_unscored",
            "underfilled_batches": int(underfilled_batches),
            "generations_completed": int(generation),
        },
    )


def run_smiles_ga_benchmark(
    out_root: str = "benchmark_smiles_ga_plogp",
    seeds: Optional[List[int]] = None,
    n_iters: int = 50,
    batch_size: int = 5,
    max_sa_score: float = MAX_SA_SCORE,
    max_smiles_length: Optional[int] = None,
    max_heavy_atoms: Optional[int] = None,
    required_atomic_nums: Optional[List[int]] = None,
    population_size: int = 120,
    parent_pool_size: int = 40,
    mutation_rate: float = 0.7,
    min_mutations: int = 1,
    max_mutations: int = 5,
    max_token_delta: int = 8,
    max_attempt_multiplier: int = 300,
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
        summary = run_smiles_ga_seed(
            initial_smiles=initials[int(seed)],
            seed=int(seed),
            out_dir=out_root / f"seed_{seed}",
            n_iters=n_iters,
            batch_size=batch_size,
            constraints=constraints,
            population_size=population_size,
            parent_pool_size=parent_pool_size,
            mutation_rate=mutation_rate,
            min_mutations=min_mutations,
            max_mutations=max_mutations,
            max_token_delta=max_token_delta,
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
        "method": "smiles_ga",
        "oracle": oracle_spec.name if oracle_spec is not None else "penalized_logp",
        "reward_label": oracle_spec.reward_label if oracle_spec is not None else "penalized logP",
        "seeds": [int(s) for s in seeds],
        "n_iters": int(n_iters),
        "batch_size": int(batch_size),
        "population_size": int(population_size),
        "parent_pool_size": int(parent_pool_size),
        "mutation_rate": float(mutation_rate),
        "min_mutations": int(min_mutations),
        "max_mutations": int(max_mutations),
        "max_token_delta": int(max_token_delta),
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
    parser = argparse.ArgumentParser(description="Run a SMILES-GA molecular optimization benchmark.")
    parser.add_argument("--out", default="benchmark_smiles_ga_plogp")
    parser.add_argument("--n-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=120)
    parser.add_argument("--parent-pool-size", type=int, default=40)
    parser.add_argument("--mutation-rate", type=float, default=0.7)
    parser.add_argument("--min-mutations", type=int, default=1)
    parser.add_argument("--max-mutations", type=int, default=5)
    parser.add_argument("--max-token-delta", type=int, default=8)
    parser.add_argument("--max-attempt-multiplier", type=int, default=300)
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
    run_smiles_ga_benchmark(
        out_root=args.out,
        seeds=args.seeds,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        max_sa_score=args.max_sa,
        max_smiles_length=args.max_smiles_length,
        max_heavy_atoms=args.max_heavy_atoms,
        required_atomic_nums=parse_required_atomic_nums(args.require_elements, args.require_fe),
        population_size=args.population_size,
        parent_pool_size=args.parent_pool_size,
        mutation_rate=args.mutation_rate,
        min_mutations=args.min_mutations,
        max_mutations=args.max_mutations,
        max_token_delta=args.max_token_delta,
        max_attempt_multiplier=args.max_attempt_multiplier,
        candidate_pool_size=args.candidate_pool_size,
    )
