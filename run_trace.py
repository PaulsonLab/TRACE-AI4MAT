"""Single entry point for the TRACE molecular-optimization benchmark.

This script wires together the three components of the method and runs them
under a fixed oracle-call budget:

    generator.py         chemistry-aware candidate generator (HybridPerturbator)
    trace_optimizer.py   the TRACE controller + fragprint GP surrogate + qLogEI
    benchmark_suite/oracles.py   the eight scoring oracles

Everything the benchmark needs -- initial molecules, oracle models, constraints --
is bundled in this folder, so a run is reproducible from a clean checkout.

Examples
--------
Run the paper's 1,000-call protocol on one oracle::

    python run_trace.py --oracle qed --out results_repro

Note: the surrogate has changed since the bundled results/ were generated (it now
uses fragprints + qLogExpectedImprovement, previously binary ECFP + qEI), so a run
measures the current configuration rather than reproducing the bundled curves. The
oracles, budget, initial data and trust-region controller are unchanged.

Reproduce all eight oracles::

    python run_trace.py --oracle all --out results_repro

Fast smoke test (a few minutes, not a paper result)::

    python run_trace.py --oracle qed --n-iters 2 --batch-size 5 \
        --seeds 11 --max-oracle-calls 30 --out smoke_test
"""

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rdkit import Chem

from benchmark_suite.common import (
    MoleculeConstraints,
    sanitize_smiles_for_constraints,
)
from benchmark_suite.oracles import get_oracle_spec
from trace_optimizer import run_benchmark as run_trace_benchmark


REPO_DIR = Path(__file__).resolve().parent

# The eight oracles reported in the paper, in table order.
ORACLE_ORDER = [
    "penalized_logp",
    "qed",
    "median_2",
    "osimertinib_mpo",
    "scaffold_hop",
    "deco_hop",
    "drd2",
    "jnk3",
]

# Chemistry constraints per oracle. These are part of the benchmark definition:
# every method in the paper is held to the same filter, so changing them here
# makes a run non-comparable with the bundled results. penalized_logP is the one
# exception to the shared 6.5/100/50 setting -- its optimum is a long aliphatic
# chain, so it needs the looser size and SA envelope to be optimizable at all.
ORACLE_CONSTRAINTS: Dict[str, Dict[str, Any]] = {
    "penalized_logp": {"max_sa": 7.0, "max_smiles_length": 120, "max_heavy_atoms": 60},
    "qed": {"max_sa": 6.5, "max_smiles_length": 100, "max_heavy_atoms": 50},
    "median_2": {"max_sa": 6.5, "max_smiles_length": 100, "max_heavy_atoms": 50},
    "osimertinib_mpo": {"max_sa": 6.5, "max_smiles_length": 100, "max_heavy_atoms": 50},
    "scaffold_hop": {"max_sa": 6.5, "max_smiles_length": 100, "max_heavy_atoms": 50},
    "deco_hop": {"max_sa": 6.5, "max_smiles_length": 100, "max_heavy_atoms": 50},
    "drd2": {"max_sa": 6.5, "max_smiles_length": 100, "max_heavy_atoms": 50},
    "jnk3": {"max_sa": 6.5, "max_smiles_length": 100, "max_heavy_atoms": 50},
}


def portable(path: Path) -> str:
    """Path relative to the package when it lives inside it, else as given.

    Run records get committed alongside results, so anything the package itself
    computes is stored relative to the package root: absolute paths do not
    resolve on anyone else's machine and leak the operator's home directory.
    """
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def read_initial_smiles_by_seed(initial_dir: Path, seeds: List[int]) -> Dict[int, List[str]]:
    """Load the versioned starting molecules, one CSV per seed."""
    initial_dir = Path(initial_dir)
    if not initial_dir.exists():
        raise FileNotFoundError(f"Initial-data directory not found: {initial_dir}")

    out: Dict[int, List[str]] = {}
    for seed in seeds:
        path = initial_dir / f"seed_{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing initial-data file for seed {seed}: {path}")
        smiles = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                value = row.get("canonical_smiles") or row.get("smiles")
                if value:
                    smiles.append(value)
        if not smiles:
            raise ValueError(f"No initial molecules found in {path}")
        out[int(seed)] = list(dict.fromkeys(smiles))
    return out


def validate_initial_manifest(initial_dir: Path, oracle_name: str, seeds: List[int]) -> Optional[Dict[str, Any]]:
    """Fail loudly if the initial data was built for a different oracle or seed set."""
    manifest_path = Path(initial_dir) / "manifest.json"
    if not manifest_path.exists():
        return None

    manifest = read_json(manifest_path)
    manifest_oracle = str(manifest.get("oracle_name", ""))
    if manifest_oracle and manifest_oracle != oracle_name:
        raise ValueError(
            f"Initial-data oracle mismatch: manifest has '{manifest_oracle}', "
            f"but --oracle resolved to '{oracle_name}'."
        )

    manifest_seeds = {int(s) for s in manifest.get("seeds", [])}
    missing = sorted(set(int(s) for s in seeds) - manifest_seeds) if manifest_seeds else []
    if missing:
        raise ValueError(f"Initial data for {oracle_name} does not cover seed(s): {missing}")
    return manifest


def trace_score_mol_factory(oracle_spec):
    """Adapt an OracleSpec to the score_mol(mol) -> float callback TRACE expects."""

    def score_mol(mol: Chem.Mol) -> float:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return float(oracle_spec.score_smiles_value(smiles))

    return score_mol


def run_one_oracle(oracle_name: str, args: argparse.Namespace) -> Path:
    oracle_spec = get_oracle_spec(oracle_name)
    limits = ORACLE_CONSTRAINTS[oracle_name]

    initial_dir = REPO_DIR / args.initial_root / oracle_name
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = REPO_DIR / out_root
    run_dir = out_root / oracle_name / "trace_turbomol"

    if args.skip_existing and (run_dir / "benchmark_summary.json").exists():
        print(f"[skip] TRACE {oracle_name} already complete: {run_dir}")
        return run_dir

    constraints = MoleculeConstraints(
        max_sa_score=limits["max_sa"],
        max_smiles_length=limits["max_smiles_length"],
        max_heavy_atoms=limits["max_heavy_atoms"],
        required_atomic_nums=None,
    )

    initial_manifest = validate_initial_manifest(initial_dir, oracle_spec.name, args.seeds)
    initials_raw = read_initial_smiles_by_seed(initial_dir, args.seeds)
    initials = {
        int(seed): sanitize_smiles_for_constraints(smiles, constraints)
        for seed, smiles in initials_raw.items()
    }
    for seed, smiles in initials.items():
        if not smiles:
            raise ValueError(f"No initial molecules for seed {seed} satisfy the {oracle_name} constraints.")

    # Oracle-call accounting: the initial molecules are scored too, so the total
    # budget is n_initial + n_iters * batch_size. The check below is what keeps
    # every method in the paper on an identical 1,000-call budget.
    expected_budget_by_seed = {
        int(seed): len(initials[int(seed)]) + int(args.n_iters) * int(args.batch_size)
        for seed in args.seeds
    }
    if args.max_oracle_calls is not None:
        mismatches = {
            seed: budget
            for seed, budget in expected_budget_by_seed.items()
            if int(budget) != int(args.max_oracle_calls)
        }
        if mismatches:
            raise ValueError(
                "--max-oracle-calls does not match n_initial + n_iters * batch_size. "
                f"requested={args.max_oracle_calls}, computed_by_seed={mismatches}. "
                "For 20 initial molecules and a 1000-call budget use --n-iters 20 --batch-size 49."
            )

    print(f"[run] TRACE {oracle_spec.display_name} -> {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        run_dir / "run_manifest.json",
        {
            "benchmark": "trace_single_method",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            # argv verbatim: this echoes whatever was typed, including any
            # absolute --out path the caller chose.
            "command": sys.argv[:],
            "python": platform.python_version(),
            "platform": platform.platform(),
            "oracle": oracle_spec.name,
            "reward_label": oracle_spec.reward_label,
            "oracle_metadata": oracle_spec.metadata,
            "seeds": [int(s) for s in args.seeds],
            "n_iters": int(args.n_iters),
            "batch_size": int(args.batch_size),
            "max_oracle_calls": args.max_oracle_calls,
            "expected_budget_by_seed": expected_budget_by_seed,
            "constraints": {
                "max_sa_score": limits["max_sa"],
                "max_smiles_length": limits["max_smiles_length"],
                "max_heavy_atoms": limits["max_heavy_atoms"],
            },
            "trace_global_pool_size": int(args.global_pool_size),
            "trace_refresh_global_pool_each_iter": bool(args.refresh_global_pool_each_iter),
            "initial_data_dir": portable(initial_dir),
            "out_dir": portable(run_dir),
            "initial_manifest": initial_manifest,
        },
    )

    run_trace_benchmark(
        initial_smiles_by_seed=initials,
        seeds=args.seeds,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        out_root=str(run_dir),
        global_pool_size=args.global_pool_size,
        refresh_global_pool_each_iter=args.refresh_global_pool_each_iter,
        max_sa_score=limits["max_sa"],
        max_smiles_length=limits["max_smiles_length"],
        max_heavy_atoms=limits["max_heavy_atoms"],
        required_atomic_nums=None,
        score_mol=trace_score_mol_factory(oracle_spec),
        objective_name=oracle_spec.name,
        reward_label=oracle_spec.reward_label,
    )
    return run_dir


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TRACE on the bundled molecular-optimization oracles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--oracle",
        nargs="+",
        default=["all"],
        help=f"'all' or any of: {', '.join(ORACLE_ORDER)}",
    )
    parser.add_argument("--out", default="results_repro", help="Output root directory.")
    parser.add_argument("--initial-root", default="initial_data")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--n-iters", type=int, default=20, help="Adaptive batches after the initial design.")
    parser.add_argument("--batch-size", type=int, default=49, help="Molecules scored per batch.")
    parser.add_argument(
        "--max-oracle-calls",
        type=int,
        default=1000,
        help="Asserted total budget per seed. Pass 0 to disable the check.",
    )
    parser.add_argument(
        "--global-pool-size",
        type=int,
        default=1000,
        help="Size of the global exploration pool TRACE samples outside the trust region.",
    )
    parser.add_argument(
        "--refresh-global-pool-each-iter", type=str_to_bool, default=False, metavar="true/false"
    )
    parser.add_argument("--skip-existing", type=str_to_bool, default=False, metavar="true/false")
    args = parser.parse_args()
    if args.max_oracle_calls is not None and int(args.max_oracle_calls) <= 0:
        args.max_oracle_calls = None
    return args


def normalize_oracles(raw: List[str]) -> List[str]:
    selected: List[str] = []
    for item in raw:
        key = item.strip().lower().replace("-", "_")
        if key == "all":
            return ORACLE_ORDER[:]
        if key not in ORACLE_CONSTRAINTS:
            raise ValueError(f"Unknown oracle '{item}'. Use 'all' or one of: {', '.join(ORACLE_ORDER)}")
        if key not in selected:
            selected.append(key)
    return selected


def main() -> None:
    args = parse_args()
    oracles = normalize_oracles(args.oracle)
    print(f"TRACE run: oracles={oracles} seeds={args.seeds} budget={args.max_oracle_calls}")
    for oracle_name in oracles:
        run_dir = run_one_oracle(oracle_name, args)
        print(f"[done] {oracle_name} -> {run_dir}")
    print(f"\nAll runs complete. Plot with:\n  python plot_trace_results.py --results {args.out}")


if __name__ == "__main__":
    main()
