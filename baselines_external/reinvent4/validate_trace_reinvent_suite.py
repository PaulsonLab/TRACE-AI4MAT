from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import List

from evaluate_reinvent_output import evaluate_reinvent_csv
from run_reinvent_suite import ORACLE_CONFIGS, normalize_oracles
from trace_reinvent.oracles import MoleculeConstraints, get_oracle_spec
from trace_reinvent.results import aggregate_oracle, aggregate_suite

TOY_SMILES = [
    "CCO",
    "c1ccccc1",
    "not_a_smiles",
    "CCO",
    "CCN(CC)CC",
    "O=C(O)c1ccccc1",
    "CC(C)Oc1ccccc1",
    "C1CCCCC1",
]


def write_toy_csv(path: Path, smiles: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles"])
        writer.writeheader()
        for smi in smiles:
            writer.writerow({"smiles": smi})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TRACE REINVENT evaluator without running REINVENT4.")
    parser.add_argument("--out-root", default="validation_output")
    parser.add_argument("--oracles", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--n-iters", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-initial", type=int, default=3)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-drd2-if-unavailable", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_root = Path(__file__).resolve().parent
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = suite_root / out_root
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    oracles = normalize_oracles(args.oracles)
    completed = []
    for oracle in oracles:
        try:
            get_oracle_spec(oracle)
        except Exception as exc:
            if oracle == "drd2" and args.skip_drd2_if_unavailable:
                print(f"[skip] drd2 unavailable: {exc}")
                continue
            raise
        cfg = ORACLE_CONFIGS[oracle]
        toy_csv = out_root / "_toy_reinvent_csvs" / f"{oracle}_seed_{args.seed}.csv"
        write_toy_csv(toy_csv, TOY_SMILES)
        method_root = out_root / oracle / "reinvent4_external"
        constraints = MoleculeConstraints(
            max_sa_score=float(cfg["max_sa"]),
            max_smiles_length=int(cfg["max_smiles_length"]),
            max_heavy_atoms=int(cfg["max_heavy_atoms"]),
        )
        evaluate_reinvent_csv(
            csv_path=toy_csv,
            out_root=method_root,
            oracle_name=oracle,
            seed=int(args.seed),
            max_oracle_calls=int(args.n_initial + args.n_iters * args.batch_size),
            constraints=constraints,
            n_iters=int(args.n_iters),
            batch_size=int(args.batch_size),
            initial_data=suite_root / "initial_data" / oracle,
            n_initial=int(args.n_initial),
        )
        spec = get_oracle_spec(oracle)
        aggregate_oracle(
            oracle_out=out_root / oracle,
            method="reinvent4_external",
            expected_budget_by_seed={int(args.seed): int(args.n_initial + args.n_iters * args.batch_size)},
            n_iters=int(args.n_iters),
            objective_name=spec.display_name,
            reward_label=spec.reward_label,
        )
        completed.append(oracle)
        print(f"[ok] {oracle}")
    aggregate_suite(out_root, completed)
    print(f"[validation complete] {out_root}")


if __name__ == "__main__":
    main()
