"""Install gate for the frozen-VAE LOL-BO baseline. Run this before any real run.

Checks, in order:

  1. checkpoint present and SHA256 matches PROVENANCE.md
  2. torch / CUDA / package versions
  3. VAE loads with strict=True and is frozen
  4. vocabulary coverage of this benchmark's initial molecules, with and without
     stereochemistry stripping (the checkpoint has no stereo tokens)
  5. teacher-forced reconstruction accuracy on initial molecules and ZINC
  6. encode -> decode round trip: validity, and how often the molecule survives
  7. prior samples: decode validity and how many pass the shared constraints
  8. shared harness reachable: oracle specs score, initial data loads

Exit code 0 means the package is safe to run; 1 means a hard gate failed.
Numbers 5-7 are reported, not gated: they are properties of the frozen latent
space and belong in the paper, not in a pass/fail test.

    python verify_lolbo.py --device cuda
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
for _path in (str(REPO_ROOT), str(PACKAGE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from frozen_vae import DEFAULT_STATE_DICT, FrozenSelfiesVAE, flatten_smiles  # noqa: E402

ORACLES = ["penalized_logp", "qed", "median_2", "osimertinib_mpo", "scaffold_hop", "drd2"]
SEEDS = [11, 22, 33, 44, 55]

failures: List[str] = []


def check(condition: bool, message: str) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {message}")
    if not condition:
        failures.append(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initial_smiles() -> List[str]:
    out: List[str] = []
    for oracle in ORACLES:
        for seed in SEEDS:
            path = REPO_ROOT / "initial_data" / oracle / f"seed_{seed}.csv"
            if not path.exists():
                continue
            with open(path, newline="") as handle:
                for row in csv.DictReader(handle):
                    value = row.get("canonical_smiles") or row.get("smiles")
                    if value:
                        out.append(value)
    return out


def zinc_sample(n: int, seed: int = 0) -> List[str]:
    path = REPO_ROOT / "zinc250k.csv"
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    column = "smiles" if rows and "smiles" in rows[0] else list(rows[0])[0]
    random.Random(seed).shuffle(rows)
    return [row[column].strip() for row in rows[:n]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-zinc", type=int, default=500)
    parser.add_argument("--n-prior", type=int, default=200)
    parser.add_argument("--state-dict", default=str(DEFAULT_STATE_DICT))
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n[1] checkpoint")
    ckpt = Path(args.state_dict)
    check(ckpt.exists(), f"checkpoint exists: {ckpt}")
    if not ckpt.exists():
        return 1
    digest = sha256(ckpt)
    provenance = (PACKAGE_DIR / "PROVENANCE.md").read_text() if (PACKAGE_DIR / "PROVENANCE.md").exists() else ""
    recorded = re.search(r"([0-9a-f]{64})\s+\./checkpoints/SELFIES-VAE-state-dict\.pt", provenance)
    check(bool(recorded), "PROVENANCE.md records a checkpoint SHA256")
    if recorded:
        check(digest == recorded.group(1), f"checkpoint SHA256 matches PROVENANCE.md ({digest[:12]}...)")

    print("\n[2] environment")
    print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  device={device}")
    if torch.cuda.is_available():
        print(f"  gpu: {torch.cuda.get_device_name(0)}")
    for name in ("botorch", "gpytorch", "rdkit", "selfies", "numpy"):
        try:
            from importlib.metadata import version

            print(f"  {name} {version(name)}")
        except Exception:
            print(f"  {name} MISSING")
    check(not (device.startswith("cuda") and not torch.cuda.is_available()), "requested device is available")

    print("\n[3] VAE loads")
    vae = FrozenSelfiesVAE(state_dict_path=ckpt, device=device)
    n_params = sum(p.numel() for p in vae.vae.parameters())
    n_trainable = sum(p.numel() for p in vae.vae.parameters() if p.requires_grad)
    print(f"  parameters: {n_params:,}  latent dim: {vae.dim}  vocab: {len(vae.dataobj.vocab)} tokens")
    check(n_trainable == 0, "VAE parameters are frozen (requires_grad=False)")
    check(not vae.vae.training, "VAE is in eval mode")

    print("\n[4] vocabulary coverage of the benchmark's initial molecules")
    smiles = initial_smiles()
    check(len(smiles) > 0, f"initial data readable ({len(smiles)} molecules)")
    strict = FrozenSelfiesVAE(state_dict_path=ckpt, device=device, strip_stereo=False)
    n_strict = len(strict.encode(smiles).encoded_smiles)
    result = vae.encode(smiles)
    n_stripped = len(result.encoded_smiles)
    print(f"  as canonicalised by TRACE (isomeric): {n_strict}/{len(smiles)} = {100 * n_strict / len(smiles):.1f}%")
    print(f"  stereochemistry stripped:             {n_stripped}/{len(smiles)} = {100 * n_stripped / len(smiles):.1f}%")
    print(f"  molecules whose stereo had to be removed: {result.n_stereo_stripped}")
    check(n_stripped >= 0.99 * len(smiles), "at least 99% of initial molecules encode after stereo stripping")

    print("\n[5] teacher-forced reconstruction (reported, not gated)")
    stats = vae.reconstruction_stats(smiles)
    print(f"  initial data: token acc {stats['recon_token_acc']:.3f}  exact string acc {stats['recon_string_acc']:.3f}")
    zinc = zinc_sample(args.n_zinc)
    if zinc:
        zstats = vae.reconstruction_stats(zinc)
        print(f"  zinc250k n={zstats['n_encoded']}: token acc {zstats['recon_token_acc']:.3f}  "
              f"exact string acc {zstats['recon_string_acc']:.3f}")

    print("\n[6] encode -> decode round trip (reported, not gated)")
    probe = smiles[:200]
    encoded = vae.encode(probe)
    decoded = vae.decode(encoded.z)
    n_valid = sum(1 for s in decoded if s and Chem.MolFromSmiles(s) is not None)
    identical = 0
    for source, target in zip(encoded.encoded_smiles, decoded):
        if not target:
            continue
        flat_source, flat_target = flatten_smiles(source), flatten_smiles(target)
        if flat_source is not None and flat_source == flat_target:
            identical += 1
    n = max(len(decoded), 1)
    print(f"  decoded valid:            {n_valid}/{n} = {100 * n_valid / n:.1f}%")
    print(f"  recovers the input molecule: {identical}/{n} = {100 * identical / n:.1f}%")

    print("\n[7] prior samples (reported, not gated)")
    from plogp_benchmark_suite.common import MoleculeConstraints, molecule_satisfies_constraints

    constraints = MoleculeConstraints(max_sa_score=6.5, max_smiles_length=100, max_heavy_atoms=50)
    prior_z = vae.sample_prior_z(args.n_prior)
    prior_smiles = vae.decode(prior_z)
    valid = [s for s in prior_smiles if s and Chem.MolFromSmiles(s) is not None]
    unique = set(Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True) for s in valid)
    in_constraints = [s for s in valid if molecule_satisfies_constraints(Chem.MolFromSmiles(s), constraints)]
    print(f"  valid:            {len(valid)}/{args.n_prior} = {100 * len(valid) / args.n_prior:.1f}%")
    print(f"  unique:           {len(unique)}/{args.n_prior} = {100 * len(unique) / args.n_prior:.1f}%")
    print(f"  pass constraints: {len(in_constraints)}/{args.n_prior} = "
          f"{100 * len(in_constraints) / args.n_prior:.1f}%  (SA<=6.5, <=50 heavy atoms, <=100 chars)")

    print("\n[8] shared harness")
    try:
        from plogp_benchmark_suite.oracles import get_oracle_spec

        for oracle in ORACLES:
            spec = get_oracle_spec(oracle)
            score = spec.score_smiles_value("CC(=O)Oc1ccccc1C(=O)O")
            check(score == score, f"oracle {oracle} scores aspirin ({score:.4f})")
    except Exception as exc:  # pragma: no cover
        check(False, f"oracle specs import and score: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 62)
    if failures:
        print(f"{len(failures)} gate(s) FAILED:")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("All gates passed. Safe to run run_lolbo_all_oracles.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
