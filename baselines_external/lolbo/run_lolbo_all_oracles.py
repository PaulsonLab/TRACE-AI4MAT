"""Run the frozen-VAE LOL-BO arms across the six TRACE oracles.

Mirrors run_all_oracle_benchmarks.py: same oracle set, same per-oracle chemistry
constraints, same versioned initial molecules, same seeds, same
`n_initial + n_iters * batch_size` budget. Constraints are not re-typed here --
they are read straight out of run_all_oracle_benchmarks.py's ORACLE_CONFIGS
literal, so the two can never drift apart.

Output layout matches every other method, so the existing plotting scripts pick
the arms up with one new row in their METHODS list:

    <out-root>/<oracle>/<method>/seed_<seed>/{trajectory,evaluations,summary,latent_diagnostics}_seed_<seed>.*

Arms (see README.md):
    literal        upstream's hard-coded failure_tolerance = 32
    budget_scaled  upstream's own commented-out formula, ceil(max(4/bsz, dim/bsz))

Example (six oracles, five seeds, both arms, 1,000 calls each):

    python run_lolbo_all_oracles.py --out-root ../benchmark_lolbo --device cuda
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
for _path in (str(REPO_ROOT), str(PACKAGE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

from benchmark_oracles import read_initial_smiles_by_seed, validate_initial_manifest  # noqa: E402
from plogp_benchmark_suite.common import MoleculeConstraints  # noqa: E402
from plogp_benchmark_suite.oracles import get_oracle_spec  # noqa: E402

LEGACY_ORACLE_MODULE = "trace_reinvent.oracles"


def legacy_get_oracle_spec():
    """The pre-2026-06-20 oracle definitions, preserved in the REINVENT4 suite.

    plogp_benchmark_suite/oracles.py changed on 2026-06-20, switching the GuacaMol
    ECFP/FCFP similarity from hashed bit vectors to unhashed count fingerprints.
    That reorders molecules (~Spearman 0.78), so median_2 and osimertinib_mpo
    scored before and after that date are different objectives. Every stored
    baseline in results_new_plotting predates the change, and
    reinvent4_oracle_suite/trace_reinvent/oracles.py (2026-06-14) still carries the
    old definitions -- verified to reproduce the stored scores exactly, 150/150 on
    all six oracles. This flag exists so the LOL-BO arms can be run against the
    same objective those baselines were optimised on.
    """
    legacy_root = REPO_ROOT / "reinvent4_oracle_suite"
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))
    from trace_reinvent.oracles import get_oracle_spec as spec_fn

    return spec_fn

from frozen_vae import DEFAULT_STATE_DICT  # noqa: E402
from lolbo_runner import TR_TOLERANCE_MODES, resolve_device, run_lolbo_benchmark  # noqa: E402

ARM_METHOD_NAMES = {
    "literal": "lolbo_frozen",
    "budget_scaled": "lolbo_frozen_tr_scaled",
    "global": "lolbo_frozen_global",   # no-trust-region ablation
}


def load_oracle_configs() -> Dict[str, Dict[str, Any]]:
    """Read ORACLE_CONFIGS out of run_all_oracle_benchmarks.py without importing it."""
    source_path = REPO_ROOT / "run_all_oracle_benchmarks.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
        if target == "ORACLE_CONFIGS":
            return ast.literal_eval(node.value)
    raise RuntimeError(f"ORACLE_CONFIGS not found in {source_path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def git_revision(path: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(path), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def build_manifest(args, oracles: List[str], arms: List[str], configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    upstream_commit = ""
    commit_file = PACKAGE_DIR / "vendor" / "COMMIT.txt"
    if commit_file.exists():
        upstream_commit = commit_file.read_text().strip().split("=")[-1]
    return {
        "benchmark": "trace_lolbo_frozen_vae_baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv[:],
        "repo_dir": str(REPO_ROOT),
        "git_revision": git_revision(REPO_ROOT),
        "upstream": {
            "source": "https://github.com/nataliemaus/lolbo",
            "license": "MIT",
            "commit": upstream_commit,
            "vae_state_dict": str(args.vae_state_dict),
            "vae_state_dict_sha256": sha256(Path(args.vae_state_dict)),
        },
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "torch": package_version("torch"),
            "botorch": package_version("botorch"),
            "gpytorch": package_version("gpytorch"),
            "rdkit": package_version("rdkit"),
            "selfies": package_version("selfies"),
            "numpy": package_version("numpy"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "fairness_contract": [
            "Same six oracle implementations and reward transforms as every other TRACE method.",
            "Same per-oracle chemistry constraints, read from run_all_oracle_benchmarks.py ORACLE_CONFIGS.",
            "Same versioned initial molecules per seed; initial molecules count against the budget.",
            "Per-seed oracle budget is n_valid_initial_molecules + n_iters * batch_size.",
            "Duplicate/cached molecules do not consume oracle budget.",
            "Primary comparison is best-so-far versus unique oracle calls.",
            "Trust region, surrogate and acquisition are LOL-BO's (vendored); the VAE is frozen.",
            "Decoded batches are refilled up to --max-fill-rounds so each iteration spends the full batch, "
            "matching the GP-BO baseline's resample policy.",
        ],
        "known_deviations_from_upstream_lolbo": [
            "10,000 labelled initialisation points -> the suite's 20 shared initial molecules.",
            "bsz=10 over 70k+ calls -> bsz=49 over 1,000 calls (shared PMO-1K grid).",
            "guacamol.standard_benchmarks scoring -> TRACE's local oracle specs (same task definitions).",
            "No chemistry constraints -> the suite's shared SA / heavy-atom / SMILES-length filter.",
            "End-to-end VAE fine-tuning and recenter() are never invoked (frozen VAE).",
            "Stereochemistry is stripped before encoding: the checkpoint vocabulary has no stereo tokens.",
        ],
        "arms": {arm: ARM_METHOD_NAMES[arm] for arm in arms},
        "oracle_source": LEGACY_ORACLE_MODULE if getattr(args, "legacy_oracle", False) else "plogp_benchmark_suite.oracles",
        "oracles": oracles,
        "oracle_configs": {name: configs[name] for name in oracles},
        "seeds": [int(seed) for seed in args.seeds],
        "n_iters": int(args.n_iters),
        "batch_size": int(args.batch_size),
        "expected_budget_per_seed": int(args.n_initial + args.n_iters * args.batch_size),
        "surrogate": {
            "model": "GPModelDKL (SVGP + deep kernel, hidden dims (256, 256))",
            "learning_rate": float(args.learning_rate),
            "init_n_epochs": int(args.init_n_epochs),
            "num_update_epochs": int(args.num_update_epochs),
            "acquisition": "Thompson sampling (MaxPosteriorSampling, replacement=False)",
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen-VAE LOL-BO across the six TRACE oracles.")
    parser.add_argument("--out-root", default=str(REPO_ROOT / "benchmark_lolbo"))
    parser.add_argument("--oracles", nargs="*", default=["all"])
    parser.add_argument("--arms", nargs="*", choices=list(TR_TOLERANCE_MODES), default=list(TR_TOLERANCE_MODES))
    parser.add_argument("--seeds", type=int, nargs="*", default=[11, 22, 33, 44, 55])
    parser.add_argument("--n-iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=49)
    parser.add_argument("--n-initial", type=int, default=20, help="expected initial molecules, for the budget check")
    parser.add_argument("--initial-data-root", default=str(REPO_ROOT / "initial_data"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vae-state-dict", default=str(DEFAULT_STATE_DICT))
    parser.add_argument("--max-string-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--init-n-epochs", type=int, default=20)
    parser.add_argument("--num-update-epochs", type=int, default=2)
    parser.add_argument("--n-candidates", type=int, default=None)
    parser.add_argument("--max-fill-rounds", type=int, default=12)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--legacy-oracle",
        action="store_true",
        help="score with the pre-2026-06-20 oracle definitions from reinvent4_oracle_suite "
        "(needed to compare against the stored June baselines on median_2 and osimertinib_mpo)",
    )
    parser.add_argument(
        "--mirror-into",
        default=None,
        help="also copy each finished <oracle>/<method> directory into this results root "
        "(e.g. ../results_new_plotting) so the plotting scripts see it next to the other methods",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configs = load_oracle_configs()

    requested = args.oracles
    if len(requested) == 1 and requested[0] == "all":
        oracles = list(configs.keys())
    else:
        unknown = [name for name in requested if name not in configs]
        if unknown:
            raise SystemExit(f"Unknown oracles: {unknown}. Known: {list(configs)}")
        oracles = requested

    spec_factory = legacy_get_oracle_spec() if args.legacy_oracle else get_oracle_spec
    oracle_source = LEGACY_ORACLE_MODULE if args.legacy_oracle else "plogp_benchmark_suite.oracles"
    if args.legacy_oracle:
        print(f"[oracle] using LEGACY definitions from {oracle_source}", flush=True)

    device = resolve_device(args.device)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(exist_ok=True)

    manifest = build_manifest(args, oracles, args.arms, configs)
    with open(out_root / "run_manifest_lolbo.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[manifest] {out_root / 'run_manifest_lolbo.json'}  device={device}", flush=True)

    failures: List[str] = []
    for oracle_name in oracles:
        cfg = configs[oracle_name]
        oracle_spec = spec_factory(cfg["oracle_arg"])
        constraints = MoleculeConstraints(
            max_sa_score=float(cfg["max_sa"]),
            max_smiles_length=cfg["max_smiles_length"],
            max_heavy_atoms=cfg["max_heavy_atoms"],
            required_atomic_nums=None,
        )
        initial_dir = Path(args.initial_data_root) / cfg["initial_dir"]
        validate_initial_manifest(initial_dir, oracle_spec, constraints, list(args.seeds))
        initials = read_initial_smiles_by_seed(initial_dir, list(args.seeds))

        for arm in args.arms:
            method = ARM_METHOD_NAMES[arm]
            method_root = out_root / oracle_name / method
            if args.skip_existing and (method_root / "benchmark_summary.json").exists():
                print(f"[skip] {oracle_name}/{method} already done", flush=True)
                continue

            print(f"[run] {oracle_name}/{method} seeds={args.seeds}", flush=True)
            try:
                overall = run_lolbo_benchmark(
                    initial_smiles_by_seed=initials,
                    oracle_spec=oracle_spec,
                    out_root=method_root,
                    seeds=args.seeds,
                    n_iters=args.n_iters,
                    batch_size=args.batch_size,
                    constraints=constraints,
                    device=device,
                    method=method,
                    tr_tolerances=arm,
                    vae_state_dict=Path(args.vae_state_dict),
                    max_string_length=args.max_string_length,
                    learning_rate=args.learning_rate,
                    init_n_epochs=args.init_n_epochs,
                    num_update_epochs=args.num_update_epochs,
                    n_candidates=args.n_candidates,
                    max_fill_rounds=args.max_fill_rounds,
                    oracle_source=oracle_source,
                )
                print(
                    f"[done] {oracle_name}/{method} mean_final_best="
                    f"{overall['mean_final_best']:.4f} +/- {overall['std_final_best']:.4f} "
                    f"decode_yield={overall['mean_decode_yield']:.3f}",
                    flush=True,
                )
                if args.mirror_into:
                    destination = Path(args.mirror_into) / oracle_name / method
                    if destination.exists():
                        shutil.rmtree(destination)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(method_root, destination)
                    print(f"[mirror] {destination}", flush=True)
            except Exception:
                failures.append(f"{oracle_name}/{method}")
                log_path = out_root / "logs" / f"{oracle_name}_{method}_traceback.log"
                log_path.write_text(traceback.format_exc())
                print(f"[FAIL] {oracle_name}/{method} -> {log_path}", flush=True)

    if failures:
        print(f"\n{len(failures)} run(s) failed: {', '.join(failures)}", flush=True)
        raise SystemExit(1)
    print("\nAll runs finished.", flush=True)


if __name__ == "__main__":
    main()
