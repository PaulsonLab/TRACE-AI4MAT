"""Literature-faithful STONED-SELFIES baseline.

This is a *separate* implementation from ``stoned_selfies_baseline.py``. It exists
so we can compare the canonical STONED optimizer (as published) against the
GA-style variant currently wired into the suite.

Canonical algorithm (identical in both reference implementations):
  * Original: aspuru-guzik-group/stoned-selfies -> GA_rediscover.py
  * Benchmark baseline: wenhao-gao/mol_opt -> main/stoned/run.py

Reference loop (generation_size = 500 in both):
    population = [best_molecule]                       # seeded from one start
    each generation:
        score population
        best = argmax(fitness)                         # single global champion
        new_population = [mutate_selfie(best) for _ in range(gen_size - 1)]
        new_population.append(best)                    # strict elitism

Key properties that define STONED and that THIS file restores:
  1. Strict elitism: every offspring is a mutant of the *current global best*
     only (not a rank-weighted pool of parents).
  2. Exactly ONE token edit per offspring (insert / replace / delete), not 1-5.
  3. The whole generation = single-token mutants of that one champion.
  4. Alphabet = selfies.get_semantic_robust_alphabet() (same as the references
     and as the existing baseline).

Apples-to-apples fairness (NO advantage over the other methods): this run is
locked to the SAME harness configuration as graph_ga / smiles_ga / gpbo:
  * same oracle + same hard call budget (n_initial + n_iters * batch_size),
  * same per-generation evaluation budget (batch_size offspring per generation),
  * same MoleculeConstraints (max SA / heavy atoms / SMILES length), enforced as
    a hard reject on every child exactly like the other methods,
  * same initial data (the champion is seeded from the best-scoring molecule of
    the shared initial set, instead of from the references' random ZINC draw).

The ONLY forced deviation from the references is generation_size: 500 -> batch_size
(=49 here), because the shared 1000-call budget cannot afford 500/generation.
This is the apples-to-apples knob; everything else is the published algorithm.

Two design points worth confirming before we wire this in (see DISCUSSION_NOTES
at the bottom):
  (A) the length ceiling for mutations (`max_molecules_len`), and
  (B) whether seeding the champion from shared initial data (vs. a random draw)
      is acceptable for the comparison.
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    import selfies as sf
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "STONED-SELFIES benchmark support requires the `selfies` package. "
        "Install it in the TRACE environment with: python -m pip install selfies"
    ) from exc

from plogp_benchmark_suite.common import (
    MAX_SA_SCORE,
    CachedMoleculeOracle,
    MoleculeConstraints,
    PenalizedLogPOracle,
    canonicalize_smiles,
    default_initial_smiles_by_seed,
    finish_run,
    molecule_satisfies_constraints,
    parse_required_atomic_nums,
    sanitize_smiles_for_constraints,
    set_all_seeds,
    str_to_bool,
    write_json,
    write_rows_csv,
)


def selfies_tokens(selfies: str) -> List[str]:
    return list(sf.split_selfies(selfies))


def semantic_alphabet() -> List[str]:
    """Same alphabet the original STONED and the PMO baseline use."""
    alphabet = list(sf.get_semantic_robust_alphabet())
    alphabet.sort()
    return alphabet


def mutate_once(
    tokens: Sequence[str],
    alphabet: Sequence[str],
    max_tokens: int,
) -> List[str]:
    """Apply EXACTLY ONE STONED-style SELFIES edit (insert / replace / delete).

    This mirrors the reference ``mutate_selfie``: a single random operation on a
    single random position. Over-length insertions are reverted (no edit), which
    the retry loop in ``generate_best_mutants`` simply resamples.
    """
    mutated = list(tokens)
    if not mutated:
        op = "insert"
    else:
        op = random.choice(["insert", "replace", "delete"])

    if op == "insert":
        if len(mutated) >= int(max_tokens):
            return mutated  # at ceiling -> no growth this attempt
        pos = random.randint(0, len(mutated))
        mutated.insert(pos, random.choice(alphabet))
    elif op == "replace":
        pos = random.randint(0, len(mutated) - 1)
        mutated[pos] = random.choice(alphabet)
    else:  # delete
        pos = random.randint(0, len(mutated) - 1)
        mutated.pop(pos)

    return mutated


def generate_best_mutants(
    best_smiles: str,
    alphabet: Sequence[str],
    constraints: MoleculeConstraints,
    target_size: int,
    exclude_smiles,
    max_molecules_len: int,
    max_attempt_multiplier: int = 100,
) -> List[str]:
    """Generate `target_size` unique single-token mutants of the champion.

    Strict elitism: the only parent is `best_smiles`. Every child is decoded,
    canonicalized, and hard-filtered through the shared MoleculeConstraints,
    exactly like the candidates the other methods submit.
    """
    best_mol = Chem.MolFromSmiles(best_smiles)
    if best_mol is None:
        return []
    canonical_parent = Chem.MolToSmiles(best_mol, isomericSmiles=True)
    try:
        base_tokens = selfies_tokens(sf.encoder(canonical_parent))
    except Exception:
        return []

    excluded = set(exclude_smiles)
    accepted: List[str] = []
    accepted_set = set()
    attempts = max(int(target_size) * int(max_attempt_multiplier), int(target_size))

    for _ in range(attempts):
        if len(accepted) >= int(target_size):
            break
        child_tokens = mutate_once(base_tokens, alphabet, max_tokens=max_molecules_len)
        try:
            child_smiles = sf.decoder("".join(child_tokens))
        except Exception:
            continue
        canonical_child = canonicalize_smiles(child_smiles)
        if canonical_child is None:
            continue
        child_mol = Chem.MolFromSmiles(canonical_child)
        if not molecule_satisfies_constraints(child_mol, constraints):
            continue
        if canonical_child in excluded or canonical_child in accepted_set:
            continue
        accepted_set.add(canonical_child)
        accepted.append(canonical_child)

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


def _default_length_ceiling(constraints: MoleculeConstraints) -> int:
    """Length cap (in SELFIES tokens) for mutations.

    The references recompute this each generation as max length in the current
    population (which, seeded from a 500-molecule random ZINC draw, gives a large
    fixed ceiling). For an apples-to-apples run we instead give STONED the same
    size headroom the shared constraints already allow every method, and let
    MoleculeConstraints be the real gate. See DISCUSSION_NOTES (A).
    """
    if constraints.max_smiles_length:
        return int(constraints.max_smiles_length)
    if constraints.max_heavy_atoms:
        return int(constraints.max_heavy_atoms) * 2
    return 100


def run_stoned_literature_seed(
    initial_smiles: List[str],
    seed: int,
    out_dir: Path,
    n_iters: int,
    batch_size: int,
    constraints: MoleculeConstraints,
    max_molecules_len: Optional[int] = None,
    max_attempt_multiplier: int = 100,
    method_name: str = "stoned_literature",
    oracle_spec=None,
) -> Dict:
    set_all_seeds(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_population = sanitize_smiles_for_constraints(initial_smiles, constraints)
    if not init_population:
        raise ValueError(f"No valid initial population for seed {seed}.")

    # Same hard budget formula the other methods use.
    max_oracle_calls = len(init_population) + int(n_iters) * int(batch_size)
    oracle = make_oracle(max_oracle_calls=max_oracle_calls, constraints=constraints, oracle_spec=oracle_spec)

    init_scores = oracle.score_many(init_population, phase="init", step=0)
    if oracle.best()["smiles"] is None:
        raise ValueError(f"No valid scored initial population for seed {seed}.")

    alphabet = semantic_alphabet()
    length_ceiling = int(max_molecules_len) if max_molecules_len else _default_length_ceiling(constraints)

    trajectory = [
        {
            "seed": int(seed),
            "generation": 0,
            "oracle_calls": int(len(oracle)),
            "n_batch": int(len([s for s in init_scores if s is not None])),
            "best_so_far": float(oracle.best()["score"]),
            "best_smiles": oracle.best()["smiles"],
            "target_batch": int(len([s for s in init_scores if s is not None])),
            "n_generated": int(len([s for s in init_scores if s is not None])),
        }
    ]

    generation = 0
    underfilled_batches = 0
    while not oracle.finished and generation < int(n_iters):
        generation += 1
        target = min(int(batch_size), oracle.remaining())

        # Strict elitism: the sole parent is the current global champion.
        champion = oracle.best()["smiles"]
        offspring = generate_best_mutants(
            best_smiles=champion,
            alphabet=alphabet,
            constraints=constraints,
            target_size=target,
            exclude_smiles=oracle.cache.keys(),
            max_molecules_len=length_ceiling,
            max_attempt_multiplier=max_attempt_multiplier,
        )

        if not offspring:
            break

        calls_before = len(oracle)
        oracle.score_many(offspring, phase="iter", step=generation)
        n_new_calls = len(oracle) - calls_before
        if n_new_calls < target:
            underfilled_batches += 1

        trajectory.append(
            {
                "seed": int(seed),
                "generation": int(generation),
                "oracle_calls": int(len(oracle)),
                "n_batch": int(n_new_calls),
                "best_so_far": float(oracle.best()["score"]),
                "best_smiles": oracle.best()["smiles"],
                "champion_parent": champion,
                "target_batch": int(target),
                "n_generated": int(len(offspring)),
            }
        )

        if n_new_calls <= 0:
            break

    return finish_run(
        out_dir=out_dir,
        method=method_name,
        seed=seed,
        oracle=oracle,
        trajectory=trajectory,
        extra_summary={
            "variant": "literature_faithful_elitist",
            "n_iters_requested": int(n_iters),
            "batch_size": int(batch_size),
            "selection": "strict_elitism_single_champion",
            "mutations_per_child": 1,
            "generation_size": int(batch_size),
            "length_ceiling_tokens": int(length_ceiling),
            "max_attempt_multiplier": int(max_attempt_multiplier),
            "underfilled_batches": int(underfilled_batches),
            "generations_completed": int(generation),
        },
    )


def run_stoned_literature_benchmark(
    out_root: str = "benchmark_stoned_literature_plogp",
    seeds: Optional[List[int]] = None,
    n_iters: int = 20,
    batch_size: int = 49,
    max_sa_score: float = MAX_SA_SCORE,
    max_smiles_length: Optional[int] = None,
    max_heavy_atoms: Optional[int] = None,
    required_atomic_nums: Optional[List[int]] = None,
    max_molecules_len: Optional[int] = None,
    max_attempt_multiplier: int = 100,
    method_name: str = "stoned_literature",
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
        summary = run_stoned_literature_seed(
            initial_smiles=initials[int(seed)],
            seed=int(seed),
            out_dir=out_root / f"seed_{seed}",
            n_iters=n_iters,
            batch_size=batch_size,
            constraints=constraints,
            max_molecules_len=max_molecules_len,
            max_attempt_multiplier=max_attempt_multiplier,
            method_name=method_name,
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
        "method": method_name,
        "variant": "literature_faithful_elitist",
        "oracle": oracle_spec.name if oracle_spec is not None else "penalized_logp",
        "reward_label": oracle_spec.reward_label if oracle_spec is not None else "penalized logP",
        "seeds": [int(s) for s in seeds],
        "n_iters": int(n_iters),
        "batch_size": int(batch_size),
        "selection": "strict_elitism_single_champion",
        "mutations_per_child": 1,
        "max_attempt_multiplier": int(max_attempt_multiplier),
        "mean_final_best": float(np.mean(best_scores)),
        "std_final_best": float(np.std(best_scores)),
        "best_score_overall": float(np.max(best_scores)),
        "out_root": str(out_root),
    }
    write_json(out_root / "benchmark_summary.json", overall)
    return overall


def parse_args():
    parser = argparse.ArgumentParser(description="Run a literature-faithful STONED-SELFIES benchmark.")
    parser.add_argument("--out", default="benchmark_stoned_literature_plogp")
    parser.add_argument("--n-iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=49)
    parser.add_argument("--max-molecules-len", type=int, default=None)
    parser.add_argument("--max-attempt-multiplier", type=int, default=100)
    parser.add_argument("--max-sa", type=float, default=MAX_SA_SCORE)
    parser.add_argument("--max-smiles-length", type=int, default=None)
    parser.add_argument("--max-heavy-atoms", type=int, default=None)
    parser.add_argument("--require-fe", type=str_to_bool, default=False, metavar="true/false")
    parser.add_argument("--require-elements", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=[11, 22, 33, 44, 55])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stoned_literature_benchmark(
        out_root=args.out,
        seeds=args.seeds,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        max_sa_score=args.max_sa,
        max_smiles_length=args.max_smiles_length,
        max_heavy_atoms=args.max_heavy_atoms,
        required_atomic_nums=parse_required_atomic_nums(args.require_elements, args.require_fe),
        max_molecules_len=args.max_molecules_len,
        max_attempt_multiplier=args.max_attempt_multiplier,
    )


# ---------------------------------------------------------------------------
# DISCUSSION_NOTES (read before wiring into run_all_oracle_benchmarks.py)
# ---------------------------------------------------------------------------
# (A) Length ceiling `max_molecules_len`:
#     References recompute it per generation as max length in the population,
#     which - seeded from 500 random ZINC molecules - is effectively a large
#     fixed ceiling. Here we derive it from the shared constraints so STONED has
#     the same size headroom as every other method, and MoleculeConstraints is
#     the binding gate. Alternative: track max length of the current generation
#     (literal-faithful) - but that can plateau STONED below the shared size cap
#     and would make the comparison *less* fair, not more.
#
# (B) Champion seeding:
#     References seed from a random draw; we seed the champion from the best of
#     the SAME initial set the other methods get. This is the apples-to-apples
#     choice (identical starting information). Flagging in case you'd rather seed
#     STONED from a random ZINC draw to match the paper exactly.
#
# (C) Forced deviation: generation_size 500 -> batch_size (49), dictated by the
#     shared 1000-call budget. This is the one place the published number can't
#     be matched without giving STONED more evaluations than the other methods.
# ---------------------------------------------------------------------------
