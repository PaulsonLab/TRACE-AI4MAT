from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from trace_reinvent.oracles import CachedMoleculeOracle, MoleculeConstraints, get_oracle_spec
from trace_reinvent.results import finish_seed_run

SMILES_COLUMNS = [
    "smiles",
    "SMILES",
    "Smiles",
    "molecule",
    "Molecule",
    "sampled_smiles",
    "Sampled_SMILES",
    "generated_smiles",
    "Generated_SMILES",
]


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def infer_column(fieldnames: List[str], preferred: Optional[str], candidates: List[str]) -> str:
    if preferred:
        if preferred not in fieldnames:
            raise ValueError(f"Column '{preferred}' not found. Available columns: {fieldnames}")
        return preferred
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    lower = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise ValueError(f"Could not infer SMILES column. Available columns: {fieldnames}")


def read_smiles_rows(path: Path, smiles_column: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = read_csv_dicts(path)
    if not rows:
        return []
    column = infer_column(list(rows[0].keys()), smiles_column, SMILES_COLUMNS)
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        smiles = row.get(column)
        if smiles:
            out.append({"source_index": idx, "smiles": smiles, "source_row": row})
    return out


def read_initial_smiles(initial_data: Optional[Path], seed: int, n_initial: Optional[int]) -> List[str]:
    if initial_data is None:
        return []
    path = Path(initial_data)
    if path.is_dir():
        path = path / f"seed_{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Initial-data CSV not found: {path}")
    rows = read_smiles_rows(path)
    smiles = [row["smiles"] for row in rows]
    if n_initial is not None and int(n_initial) > 0:
        smiles = smiles[: int(n_initial)]
    return smiles


def evaluate_reinvent_csv(
    csv_path: Path,
    out_root: Path,
    oracle_name: str,
    seed: int,
    max_oracle_calls: int,
    constraints: MoleculeConstraints,
    n_iters: int,
    batch_size: int,
    initial_data: Optional[Path] = None,
    n_initial: Optional[int] = None,
    smiles_column: Optional[str] = None,
) -> Dict[str, Any]:
    spec = get_oracle_spec(oracle_name)
    oracle = CachedMoleculeOracle(max_oracle_calls=max_oracle_calls, constraints=constraints, spec=spec)
    generated_samples: List[Dict[str, Any]] = []

    initial_smiles = read_initial_smiles(initial_data, seed, n_initial)
    for rank, smiles in enumerate(initial_smiles, start=1):
        status = oracle.score_one(smiles, phase="initial", iteration=0, batch_rank=rank, source_index=rank)
        generated_samples.append(
            {
                "source": "initial_data",
                "source_index": rank,
                "iteration": 0,
                "batch_rank": rank,
                "input_smiles": smiles,
                "canonical_smiles": status.get("canonical_smiles"),
                "accepted": bool(status.get("accepted")),
                "reject_reason": status.get("reason"),
                "score": status.get("score"),
                "best_so_far_after": status.get("best_so_far"),
                "oracle_calls_after": status.get("oracle_calls"),
            }
        )
    best_by_iteration: Dict[int, Dict[str, Any]] = {
        0: {
            "oracle_calls": int(len(oracle)),
            "best_so_far": float(oracle.best()["score"]),
            "best_smiles": oracle.best()["smiles"],
        }
    }

    rows = read_smiles_rows(csv_path, smiles_column=smiles_column)
    for row in rows:
        source_index = int(row["source_index"])
        iteration = int((source_index - 1) // int(batch_size)) + 1
        batch_rank = int((source_index - 1) % int(batch_size)) + 1
        if iteration > int(n_iters):
            break
        status = oracle.score_one(
            row["smiles"],
            phase="reinvent4",
            iteration=iteration,
            batch_rank=batch_rank,
            source_index=source_index,
        )
        generated_samples.append(
            {
                "source": str(csv_path),
                "source_index": source_index,
                "iteration": iteration,
                "batch_rank": batch_rank,
                "input_smiles": row["smiles"],
                "canonical_smiles": status.get("canonical_smiles"),
                "accepted": bool(status.get("accepted")),
                "reject_reason": status.get("reason"),
                "score": status.get("score"),
                "best_so_far_after": status.get("best_so_far"),
                "oracle_calls_after": status.get("oracle_calls"),
            }
        )
        best_by_iteration[iteration] = {
            "oracle_calls": int(len(oracle)),
            "best_so_far": float(oracle.best()["score"]),
            "best_smiles": oracle.best()["smiles"],
        }
        if oracle.finished:
            break

    trajectory: List[Dict[str, Any]] = []
    last = best_by_iteration[0]
    for iteration in range(0, int(n_iters) + 1):
        if iteration in best_by_iteration:
            last = best_by_iteration[iteration]
        trajectory.append(
            {
                "iteration": iteration,
                "oracle_calls": int(last["oracle_calls"]),
                "best_so_far": float(last["best_so_far"]),
                "best_smiles": last["best_smiles"],
            }
        )

    out_dir = Path(out_root) / f"seed_{seed}"
    return finish_seed_run(
        out_dir=out_dir,
        method="reinvent4_external",
        seed=seed,
        oracle=oracle,
        trajectory=trajectory,
        generated_samples=generated_samples,
        extra_summary={
            "source_csv": str(csv_path),
            "initial_data": str(initial_data) if initial_data else None,
            "n_initial_requested": n_initial,
            "n_initial_loaded": int(len(initial_smiles)),
            "n_iters": int(n_iters),
            "batch_size": int(batch_size),
            "fixed_step_policy": (
                "REINVENT is evaluated for exactly n_iters fixed batches. Invalid, duplicate, "
                "or constraint-failing molecules are recorded but do not consume unique oracle budget."
            ),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate REINVENT4 CSV output with TRACE oracle accounting.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-root", required=True, help="Method root, e.g. benchmark_reinvent4_1k/qed/reinvent4_external")
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=49)
    parser.add_argument("--max-oracle-calls", type=int, default=1000)
    parser.add_argument("--initial-data", default=None)
    parser.add_argument("--n-initial", type=int, default=20)
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--max-sa", type=float, default=6.5)
    parser.add_argument("--max-smiles-length", type=int, default=100)
    parser.add_argument("--max-heavy-atoms", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    constraints = MoleculeConstraints(
        max_sa_score=args.max_sa,
        max_smiles_length=args.max_smiles_length,
        max_heavy_atoms=args.max_heavy_atoms,
    )
    evaluate_reinvent_csv(
        csv_path=Path(args.csv),
        out_root=Path(args.out_root),
        oracle_name=args.oracle,
        seed=args.seed,
        max_oracle_calls=args.max_oracle_calls,
        constraints=constraints,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        initial_data=Path(args.initial_data) if args.initial_data else None,
        n_initial=args.n_initial,
        smiles_column=args.smiles_column,
    )


if __name__ == "__main__":
    main()
