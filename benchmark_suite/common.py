import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen

try:
    from rdkit.Chem import RDConfig

    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
except ImportError:
    sascorer = None

if sascorer is None:
    raise ImportError("RDKit SA_Score / sascorer is required for this benchmark.")


FP_RADIUS = 2
FP_NBITS = 2048
MAX_SA_SCORE = 7.0

RDLogger.DisableLog("rdApp.warning")


@dataclass
class MoleculeConstraints:
    max_sa_score: float = MAX_SA_SCORE
    max_smiles_length: Optional[int] = None
    max_heavy_atoms: Optional[int] = None
    required_atomic_nums: Optional[List[int]] = None


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def get_sa_score(mol: Chem.Mol) -> float:
    if mol is None:
        return float("inf")
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return float("inf")


def largest_ring_size(mol: Chem.Mol) -> int:
    rings = mol.GetRingInfo().AtomRings()
    return max((len(ring) for ring in rings), default=0)


def penalized_logp_mol(mol: Chem.Mol) -> float:
    if mol is None:
        return -100.0
    try:
        log_p = float(Crippen.MolLogP(mol))
        sa = get_sa_score(mol)
        if not math.isfinite(sa):
            return -100.0
        cycle_penalty = max(largest_ring_size(mol) - 6, 0)
        return float(log_p - sa - cycle_penalty)
    except Exception:
        return -100.0


def penalized_logp_smiles(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return penalized_logp_mol(mol)


def mol_has_required_elements(mol: Chem.Mol, required_atomic_nums: Optional[List[int]]) -> bool:
    if not required_atomic_nums:
        return True
    present = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
    return all(int(z) in present for z in required_atomic_nums)


def molecule_satisfies_constraints(mol: Chem.Mol, constraints: MoleculeConstraints) -> bool:
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False

    sa = get_sa_score(mol)
    if not math.isfinite(sa) or sa > constraints.max_sa_score:
        return False
    if constraints.max_heavy_atoms is not None and mol.GetNumHeavyAtoms() > constraints.max_heavy_atoms:
        return False
    if constraints.max_smiles_length is not None:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        if len(smiles) > constraints.max_smiles_length:
            return False
    if not mol_has_required_elements(mol, constraints.required_atomic_nums):
        return False
    return True


def smiles_satisfies_constraints(smiles: str, constraints: MoleculeConstraints) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return molecule_satisfies_constraints(mol, constraints)


def sanitize_smiles_for_constraints(
    smiles_list: Iterable[str],
    constraints: MoleculeConstraints,
) -> List[str]:
    out = []
    seen = set()
    for smiles in smiles_list:
        canonical = canonicalize_smiles(smiles)
        if canonical is None or canonical in seen:
            continue
        if smiles_satisfies_constraints(canonical, constraints):
            seen.add(canonical)
            out.append(canonical)
    return out


def get_fp(mol: Chem.Mol, n_bits: int = FP_NBITS):
    return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, n_bits)


def smiles_to_fp_array(smiles: str, n_bits: int = FP_NBITS) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = get_fp(mol, n_bits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class CachedMoleculeOracle:
    """
    Unique-molecule oracle with a hard call budget.

    Cached repeats do not consume budget, matching the sample-efficiency
    convention used by PMO/mol_opt.
    """

    def __init__(
        self,
        max_oracle_calls: int,
        constraints: MoleculeConstraints,
        score_mol: Optional[Callable[[Chem.Mol], float]] = None,
        score_smiles: Optional[Callable[[str], float]] = None,
        oracle_name: str = "custom",
        reward_label: str = "score",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.max_oracle_calls = int(max_oracle_calls)
        self.constraints = constraints
        self.score_mol = score_mol
        self.score_smiles = score_smiles
        self.oracle_name = str(oracle_name)
        self.reward_label = str(reward_label)
        self.metadata = metadata or {}
        self.cache: Dict[str, float] = {}
        self.rows: List[Dict[str, Any]] = []

        if self.score_mol is None and self.score_smiles is None:
            raise ValueError("CachedMoleculeOracle requires score_mol or score_smiles.")

    def __len__(self) -> int:
        return len(self.cache)

    @property
    def finished(self) -> bool:
        return len(self.cache) >= self.max_oracle_calls

    def remaining(self) -> int:
        return max(0, self.max_oracle_calls - len(self.cache))

    def score_one(self, smiles: str, phase: str = "eval", step: int = 0) -> Optional[float]:
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            return None

        if canonical in self.cache:
            return self.cache[canonical]
        if self.finished:
            return None

        mol = Chem.MolFromSmiles(canonical)
        if not molecule_satisfies_constraints(mol, self.constraints):
            return None

        try:
            if self.score_smiles is not None:
                score = float(self.score_smiles(canonical))
            else:
                score = float(self.score_mol(mol))  # type: ignore[misc]
        except Exception:
            return None

        if not math.isfinite(score):
            return None

        self.cache[canonical] = score
        best_so_far = max(self.cache.values())
        self.rows.append(
            {
                "oracle_call": int(len(self.cache)),
                "oracle": self.oracle_name,
                "reward_label": self.reward_label,
                "phase": phase,
                "step": int(step),
                "smiles": canonical,
                "score": float(score),
                "best_so_far": float(best_so_far),
                "sa": float(get_sa_score(mol)),
                "heavy_atoms": int(mol.GetNumHeavyAtoms()),
                "smiles_length": int(len(canonical)),
            }
        )
        return score

    def score_many(self, smiles_list: Iterable[str], phase: str = "eval", step: int = 0) -> List[Optional[float]]:
        scores = []
        for smiles in smiles_list:
            scores.append(self.score_one(smiles, phase=phase, step=step))
            if self.finished:
                break
        return scores

    def best(self) -> Dict[str, Any]:
        if not self.cache:
            return {"smiles": None, "score": -float("inf")}
        smiles, score = max(self.cache.items(), key=lambda item: item[1])
        return {"smiles": smiles, "score": float(score)}


class PenalizedLogPOracle(CachedMoleculeOracle):
    def __init__(self, max_oracle_calls: int, constraints: MoleculeConstraints):
        super().__init__(
            max_oracle_calls=max_oracle_calls,
            constraints=constraints,
            score_mol=penalized_logp_mol,
            oracle_name="penalized_logp",
            reward_label="penalized logP",
            metadata={
                "definition": "RDKit MolLogP - SA score - max(largest_ring_size - 6, 0)",
            },
        )


def default_initial_smiles_by_seed() -> Dict[int, List[str]]:
    return {
        11: [
            "CC1(C)CC2CC3C24C2=C(C=C2)C314",
            "COC(=O)C1=CC2=C1CCC2",
            "CC(C)c1ccccc1-c1ccccc1",
        ],
        22: [
            "O=C1Oc2cccc1c2",
            "CC(C)c1ccccc1-c1ccc(N)cc1",
            "COC(=O)c1c2cc3c-2c1C3",
        ],
        33: [
            "CC1(C)CC2CC3C24C2=C(C=C2)C314",
            "O=C1Oc2cccc1c2",
            "COC(=O)C1=CC2=C1CCC2",
        ],
        44: [
            "CC(C)c1ccccc1-c1ccccc1",
            "CC(C)c1ccccc1-c1ccc(N)cc1",
            "COC(=O)c1c2cc3c-2c1C3",
        ],
        55: [
            "COC(=O)C1=CC2=C1CCC2",
            "O=C1Oc2cccc1c2",
            "CC1(C)CC2CC3C24C2=C(C=C2)C314",
        ],
    }


def write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)


def finish_run(
    out_dir: Path,
    method: str,
    seed: int,
    oracle: CachedMoleculeOracle,
    trajectory: List[Dict[str, Any]],
    extra_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(out_dir / f"evaluations_seed_{seed}.csv", oracle.rows)
    write_rows_csv(out_dir / f"trajectory_seed_{seed}.csv", trajectory)

    best = oracle.best()
    summary = {
        "method": method,
        "seed": int(seed),
        "oracle": getattr(oracle, "oracle_name", None),
        "reward_label": getattr(oracle, "reward_label", "score"),
        "oracle_metadata": getattr(oracle, "metadata", {}),
        "best_smiles": best["smiles"],
        "best_score": float(best["score"]),
        "n_oracle_calls": int(len(oracle)),
        "max_oracle_calls": int(oracle.max_oracle_calls),
        "constraints": asdict(oracle.constraints),
        "evaluations_csv": str(out_dir / f"evaluations_seed_{seed}.csv"),
        "trajectory_csv": str(out_dir / f"trajectory_seed_{seed}.csv"),
    }
    if extra_summary:
        summary.update(extra_summary)
    write_json(out_dir / f"summary_seed_{seed}.json", summary)
    return summary


def parse_required_atomic_nums(values: Optional[List[str]], require_fe: bool = False) -> Optional[List[int]]:
    atomic_nums = []
    if require_fe:
        atomic_nums.append(26)

    periodic_table = Chem.GetPeriodicTable()
    for raw in values or []:
        token = str(raw).strip()
        if not token:
            continue
        if token.isdigit():
            atomic_nums.append(int(token))
            continue
        symbol = token[0].upper() + token[1:].lower()
        atomic_num = int(periodic_table.GetAtomicNumber(symbol))
        if atomic_num <= 0:
            raise ValueError(f"Unknown element: {raw}")
        atomic_nums.append(atomic_num)

    atomic_nums = list(dict.fromkeys(atomic_nums))
    return atomic_nums or None


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected true/false, got: {value}")
