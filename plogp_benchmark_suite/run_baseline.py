"""Run one in-process baseline on one objective, on the paper's protocol.

The four baselines in this package are libraries: each exposes a
``run_*_benchmark`` function that takes an ``oracle_spec`` and a dict of starting
molecules. This is the glue that supplies both from the repository -- the same
objective definitions in ``benchmark_suite/oracles.py`` that scored every bundled
result, and the same starting molecules in ``initial_data/<oracle>/``.

Defaults reproduce the main benchmark: 20 initial molecules, 20 iterations of 49,
1,000 unique oracle calls, seeds 11/22/33/44/55. The per-oracle SA, length and
heavy-atom constraints are the ones the bundled runs used.

    python -m plogp_benchmark_suite.run_baseline --method graph_ga --oracle qed
    python -m plogp_benchmark_suite.run_baseline --method smiles_ga --oracle drd2 \
        --out my_runs --seeds 11

Methods: graph_ga, smiles_ga, stoned_selfies, gpbo_graph_ga.
For TRACE itself use ``run_trace.py`` at the repository root.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmark_suite.oracles import get_oracle_spec

from plogp_benchmark_suite.gpbo_graph_ga_baseline import run_gpbo_benchmark
from plogp_benchmark_suite.graph_ga_baseline import run_graph_ga_benchmark
from plogp_benchmark_suite.smiles_ga_baseline import run_smiles_ga_benchmark
from plogp_benchmark_suite.stoned_selfies_baseline import run_stoned_selfies_benchmark

REPO = Path(__file__).resolve().parents[1]

# The constraints each objective was run under. penalized_logp is looser because
# its reward is unbounded in molecule size; the rest share one envelope.
ORACLE_CONSTRAINTS: Dict[str, Dict[str, float]] = {
    "penalized_logp": {"max_sa": 7.0, "max_smiles_length": 120, "max_heavy_atoms": 60},
    "qed": {"max_sa": 6.5, "max_smiles_length": 100, "max_heavy_atoms": 50},
    "median_2": {"max_sa": 6.5, "max_smiles_length": 100, "max_heavy_atoms": 50},
    "osimertinib_mpo": {"max_sa": 6.5, "max_smiles_length": 100, "max_heavy_atoms": 50},
    "scaffold_hop": {"max_sa": 6.5, "max_smiles_length": 100, "max_heavy_atoms": 50},
    "deco_hop": {"max_sa": 6.5, "max_smiles_length": 100, "max_heavy_atoms": 50},
    "drd2": {"max_sa": 6.5, "max_smiles_length": 100, "max_heavy_atoms": 50},
    "jnk3": {"max_sa": 6.5, "max_smiles_length": 100, "max_heavy_atoms": 50},
}

RUNNERS = {
    "graph_ga": run_graph_ga_benchmark,
    "smiles_ga": run_smiles_ga_benchmark,
    "stoned_selfies": run_stoned_selfies_benchmark,
    "gpbo_graph_ga": run_gpbo_benchmark,
}


def load_initial_molecules(oracle: str, seeds: List[int]) -> Dict[int, List[str]]:
    """The same starting molecules TRACE was given, per seed."""
    out: Dict[int, List[str]] = {}
    for seed in seeds:
        path = REPO / "initial_data" / oracle / f"seed_{seed}.csv"
        if not path.exists():
            raise SystemExit(f"No initial data for {oracle} seed {seed}: {path}")
        with open(path, newline="") as fh:
            out[seed] = [row["smiles"] for row in csv.DictReader(fh)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--oracle", required=True, choices=sorted(ORACLE_CONSTRAINTS))
    ap.add_argument("--out", default=None,
                    help="output root (default: <method>_<oracle>_repro)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[11, 22, 33, 44, 55])
    ap.add_argument("--n-iters", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=49)
    args = ap.parse_args()

    constraints = ORACLE_CONSTRAINTS[args.oracle]
    initial = load_initial_molecules(args.oracle, args.seeds)
    out_root = args.out or f"{args.method}_{args.oracle}_repro"

    n_initial = len(next(iter(initial.values())))
    print(f"{args.method} on {args.oracle}: {len(args.seeds)} seeds x "
          f"({n_initial} initial + {args.n_iters} x {args.batch_size}) = "
          f"{n_initial + args.n_iters * args.batch_size} oracle calls per seed")

    result = RUNNERS[args.method](
        out_root=out_root,
        seeds=args.seeds,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        max_sa_score=constraints["max_sa"],
        max_smiles_length=int(constraints["max_smiles_length"]),
        max_heavy_atoms=int(constraints["max_heavy_atoms"]),
        initial_smiles_by_seed=initial,
        oracle_spec=get_oracle_spec(args.oracle),
    )
    print(f"mean final best: {result.get('mean_final_best')}")
    print(f"wrote {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
