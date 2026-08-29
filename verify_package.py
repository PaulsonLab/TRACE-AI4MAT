"""Self-check for the TRACE submission package.

Confirms three things, in order:

  1. Every dependency imports and every bundled file is present.
  2. All eight oracles load (including the two model-backed ones).
  3. The bundled oracle code reproduces the bundled result scores exactly --
     i.e. the numbers in results/ really came from the code in this folder.
  4. Every aggregate file agrees with the per-seed records it summarises.

Run it before trusting anything else here::

    python verify_package.py
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

REPO_DIR = Path(__file__).resolve().parent

ORACLES = [
    "penalized_logp",
    "qed",
    "median_2",
    "osimertinib_mpo",
    "scaffold_hop",
    "deco_hop",
    "drd2",
    "jnk3",
]
SEEDS = [11, 22, 33, 44, 55]

REQUIRED_FILES = [
    "run_trace.py",
    "plot_trace_results.py",
    "generator.py",
    "trace_optimizer.py",
    "benchmark_suite/oracles.py",
    "benchmark_suite/common.py",
    "benchmark_suite/rf_oracle.py",
    "oracle/drd2_current.pkl.gz",
    "oracle/jnk3_forest.npz",
]

TOLERANCE = 1e-6

_failures: List[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        _failures.append(label)
    return ok


def check_imports() -> None:
    print("\n[1/4] Environment")
    import platform

    print(f"  python {platform.python_version()} on {platform.system()}")
    for module, attr in [
        ("rdkit", "__version__"),
        ("numpy", "__version__"),
        ("torch", "__version__"),
        ("botorch", "__version__"),
        ("gpytorch", "__version__"),
        ("sklearn", "__version__"),
        ("matplotlib", "__version__"),
    ]:
        try:
            mod = __import__(module)
            check(f"import {module}", True, getattr(mod, attr, "?"))
        except Exception as exc:  # noqa: BLE001 - report, don't crash the whole check
            check(f"import {module}", False, str(exc))

    for rel in REQUIRED_FILES:
        check(f"file {rel}", (REPO_DIR / rel).exists())

    for oracle in ORACLES:
        d = REPO_DIR / "initial_data" / oracle
        missing = [s for s in SEEDS if not (d / f"seed_{s}.csv").exists()]
        check(f"initial data {oracle}", not missing, "" if not missing else f"missing seeds {missing}")


def check_oracles() -> None:
    print("\n[2/4] Oracles")
    from benchmark_suite.oracles import FINGERPRINT_MODE, get_oracle_spec

    print(f"  ECFP/FCFP fingerprint mode: {FINGERPRINT_MODE}")
    probe = "Cc1cc(C)c2sc(N(Cc3ccco3)C(=O)c3cccc(S(=O)(=O)C(C)C)c3)nc2c1"
    for oracle in ORACLES:
        try:
            spec = get_oracle_spec(oracle)
            value = spec.score_smiles_value(probe)
            check(f"oracle {oracle}", value == value, f"probe score {value:.6f}")
        except Exception as exc:  # noqa: BLE001
            check(f"oracle {oracle}", False, str(exc))


def check_reproduction(n_samples: int, results_root: Path) -> None:
    print(f"\n[3/4] Reproducing recorded scores ({n_samples} molecules per oracle-seed)")
    from benchmark_suite.oracles import get_oracle_spec

    rng = random.Random(0)
    for oracle in ORACLES:
        spec = get_oracle_spec(oracle)
        checked = 0
        mismatched: List[Tuple[int, str, float, float]] = []
        worst = 0.0
        for seed in SEEDS:
            path = results_root / oracle / "trace_turbomol" / f"seed_{seed}" / f"evaluations_seed_{seed}.csv"
            if not path.exists():
                continue
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            sample = rows if len(rows) <= n_samples else rng.sample(rows, n_samples)
            for row in sample:
                recorded = float(row["score"])
                recomputed = spec.score_smiles_value(row["smiles"])
                delta = abs(recorded - recomputed)
                worst = max(worst, delta)
                checked += 1
                if delta > TOLERANCE:
                    mismatched.append((seed, row["smiles"], recorded, recomputed))

        if checked == 0:
            check(f"reproduce {oracle}", False, "no result files found")
            continue
        ok = not mismatched
        detail = f"{checked} molecules, max |delta| = {worst:.2e}"
        if not ok:
            detail = f"{len(mismatched)}/{checked} mismatched, max |delta| = {worst:.2e}"
        check(f"reproduce {oracle}", ok, detail)
        if mismatched:
            seed, smiles, rec, got = mismatched[0]
            print(f"          e.g. seed {seed}: recorded {rec:.6f} vs recomputed {got:.6f}  {smiles}")


def check_aggregates(results_root: Path) -> None:
    """Aggregates must agree with the per-seed records they summarise.

    Added after a real defect: one oracle's aggregate files had been written from
    a 3-seed run and never regenerated when the remaining seeds were added, so
    benchmark_mean_std.csv silently described 3 of 5 seeds while the seed
    directories held all 5.
    """
    print("\n[4/4] Aggregate consistency")
    import numpy as np

    for oracle in ORACLES:
        d = results_root / oracle / "trace_turbomol"
        if not d.is_dir():
            check(f"aggregates {oracle}", False, "missing")
            continue
        try:
            seed_dirs = sorted(int(x.name.split("_")[1]) for x in d.glob("seed_*") if x.is_dir())
            matrix = np.load(d / "best_so_far_matrix.npy")
            summary = json.loads((d / "benchmark_summary.json").read_text())
            with open(d / "final_scores_by_seed.csv", newline="", encoding="utf-8") as f:
                final_seeds = sorted(int(r["seed"]) for r in csv.DictReader(f))
            with open(d / "benchmark_mean_std.csv", newline="", encoding="utf-8") as f:
                curve_cols = sorted(
                    int(c.split("_")[1]) for c in next(csv.reader(f)) if c.startswith("seed_")
                )

            per_seed_best = []
            for s in seed_dirs:
                per_seed_best.append(
                    float(json.loads((d / f"seed_{s}" / f"summary_seed_{s}.json").read_text())["best_score"])
                )
            truth = float(np.mean(per_seed_best))

            problems = []
            if matrix.shape[0] != len(seed_dirs):
                problems.append(f"matrix has {matrix.shape[0]} rows for {len(seed_dirs)} seeds")
            if final_seeds != seed_dirs:
                problems.append(f"final_scores covers {final_seeds}, dirs are {seed_dirs}")
            if curve_cols != seed_dirs:
                problems.append(f"mean_std covers {curve_cols}, dirs are {seed_dirs}")
            if abs(float(np.mean(matrix[:, -1])) - truth) > 1e-9:
                problems.append(f"matrix mean {np.mean(matrix[:, -1]):.6f} != per-seed {truth:.6f}")
            if abs(float(summary["mean_final_best"]) - truth) > 1e-9:
                problems.append(f"summary mean {summary['mean_final_best']:.6f} != per-seed {truth:.6f}")

            check(f"aggregates {oracle}", not problems,
                  f"{len(seed_dirs)} seeds, mean {truth:.6f}" if not problems else "; ".join(problems))
        except Exception as exc:  # noqa: BLE001
            check(f"aggregates {oracle}", False, str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the TRACE submission package.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--n-samples", type=int, default=50, help="Molecules sampled per oracle-seed.")
    args = parser.parse_args()

    results_root = Path(args.results)
    if not results_root.is_absolute():
        results_root = REPO_DIR / results_root

    print("TRACE submission package verification")
    print(f"package: {REPO_DIR}")

    check_imports()
    check_oracles()
    check_reproduction(args.n_samples, results_root)
    check_aggregates(results_root)

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {', '.join(_failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
