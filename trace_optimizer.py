import argparse
import csv
import json
import math
import os
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.normal import IIDNormalSampler
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors

try:
    from generator import HybridPerturbator, derive_config_from_seed, get_fp
except ImportError:
    print("Error: generator.py not found. Run this file from the TRACE code folder.")
    sys.exit(1)


warnings.filterwarnings("ignore")

MAX_SA_SCORE = 7.0

try:
    from rdkit.Chem import RDConfig

    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
except ImportError:
    sascorer = None

if sascorer is None:
    raise ImportError(
        "RDKit SA_Score / sascorer is required because this benchmark enforces "
        "a hard synthetic-accessibility constraint."
    )


# ============================================================
# Objective and chemistry filters
# ============================================================

@dataclass
class MoleculeConstraints:
    max_sa_score: float = MAX_SA_SCORE
    max_smiles_length: Optional[int] = None
    max_heavy_atoms: Optional[int] = None
    required_atomic_nums: Optional[List[int]] = None


def get_sa_score(mol: Chem.Mol) -> float:
    if mol is None:
        return float("inf")
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return float("inf")


def is_sa_feasible_mol(mol: Chem.Mol, max_sa_score: float = MAX_SA_SCORE) -> bool:
    sa = get_sa_score(mol)
    return bool(np.isfinite(sa) and sa <= float(max_sa_score))


def mol_has_required_elements(mol: Chem.Mol, required_atomic_nums: Optional[List[int]]) -> bool:
    if not required_atomic_nums:
        return True
    present = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
    return all(int(z) in present for z in required_atomic_nums)


def molecule_satisfies_constraints(mol: Chem.Mol, constraints: MoleculeConstraints) -> bool:
    if mol is None:
        return False
    if not is_sa_feasible_mol(mol, constraints.max_sa_score):
        return False
    if constraints.max_heavy_atoms is not None and mol.GetNumHeavyAtoms() > constraints.max_heavy_atoms:
        return False
    if constraints.max_smiles_length is not None:
        smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        if len(smi) > constraints.max_smiles_length:
            return False
    if not mol_has_required_elements(mol, constraints.required_atomic_nums):
        return False
    return True


def penalized_logp(mol: Chem.Mol) -> float:
    """
    Standard penalized logP objective:

        MolLogP - synthetic_accessibility - large_cycle_penalty

    The large-cycle penalty is max(largest_ring_size - 6, 0), following the
    common molecular-optimization benchmark convention.
    """
    if mol is None:
        return -100.0

    try:
        log_p = float(Crippen.MolLogP(mol))
        sa = get_sa_score(mol)
        if not np.isfinite(sa):
            return -100.0

        rings = mol.GetRingInfo().AtomRings()
        largest_ring = max((len(ring) for ring in rings), default=0)
        cycle_penalty = max(largest_ring - 6, 0)
        return float(log_p - sa - cycle_penalty)
    except Exception:
        return -100.0


class SAConstrainedHybridPerturbator(HybridPerturbator):
    def __init__(self, config=None, probs=None, max_sa_score: float = MAX_SA_SCORE):
        super().__init__(config=config, probs=probs)
        self.max_sa_score = float(max_sa_score)
        try:
            self.config.max_sa_score = float(max_sa_score)
        except Exception:
            pass

    def _sanitize(self, mol: Chem.Mol):
        mol = super()._sanitize(mol)
        if mol is None:
            return None
        if not is_sa_feasible_mol(mol, self.max_sa_score):
            return None
        return mol


def build_generator(
    seed_mol: Chem.Mol,
    chem_profile: str,
    constraints: MoleculeConstraints,
) -> SAConstrainedHybridPerturbator:
    cfg = derive_config_from_seed(seed_mol, chem_profile=chem_profile)
    cfg.max_sa_score = float(constraints.max_sa_score)
    if constraints.max_smiles_length is not None:
        cfg.max_smiles_length = int(constraints.max_smiles_length)
    if constraints.max_heavy_atoms is not None:
        cfg.max_heavy_atoms = min(int(cfg.max_heavy_atoms), int(constraints.max_heavy_atoms))
    if constraints.required_atomic_nums:
        cfg.require_element_atomic_nums = [int(z) for z in constraints.required_atomic_nums]
    return SAConstrainedHybridPerturbator(config=cfg, max_sa_score=constraints.max_sa_score)


# ============================================================
# GP model and acquisition
# ============================================================

# ---------------------------------------------------------------------------
# Fragprints: the GP's molecular representation (GAUCHE, Griffiths et al. 2023).
#
# A 2048-bit radius-3 Morgan bit vector concatenated with the 85 RDKit
# fragment counts (the `fr_*` entries of Descriptors.descList).
#
# This is the *surrogate's* representation ONLY. It is deliberately NOT the
# fingerprint the trust region uses: admission (`T(x, center) >= tau`), the
# similarity sort, `diverse_pick` and the restart-diversity check all go
# through `get_fp` from generator.py -- a binary radius-2 ECFP with plain
# RDKit Tanimoto -- and every trust-region constant (`s_min_floor = 0.10`,
# `s_min_ceil = 0.50`, `restart_diversity_threshold = 0.70`) is calibrated on
# that. Redefining `get_fp` would change the algorithm, not the surrogate.
#
# GAUCHE's own featurisation slices `Descriptors.descList[115:]` to reach the
# fragment block. That index was pinned to a 2021-era RDKit; on RDKit 2026.03
# it returns 102 descriptors of which 17 are not fragment counts, including
# MolLogP and MolMR -- continuous, possibly negative, and on the
# `penalized_logp` oracle a direct leak of the objective into the features.
# Negative entries also break the Tanimoto kernel's non-negativity. Selecting
# by the `fr_` name prefix gives GAUCHE's intended 85 on any RDKit version.
# ---------------------------------------------------------------------------

FRAGPRINT_RADIUS = 3
FRAGPRINT_NBITS = 2048
_FRAGMENT_FNS = [(name, fn) for name, fn in Descriptors.descList if name.startswith("fr_")]
FRAGPRINT_DIM = FRAGPRINT_NBITS + len(_FRAGMENT_FNS)

_fragprint_cache: Dict[str, np.ndarray] = {}


def fragprint(smiles: str) -> np.ndarray:
    """2048-bit radius-3 Morgan bits ++ 85 RDKit fragment counts, cached by SMILES."""
    vec = _fragprint_cache.get(smiles)
    if vec is not None:
        return vec

    vec = np.zeros(FRAGPRINT_DIM, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        bits = np.zeros((FRAGPRINT_NBITS,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(mol, FRAGPRINT_RADIUS, FRAGPRINT_NBITS),
            bits,
        )
        vec[:FRAGPRINT_NBITS] = bits
        for j, (_, fn) in enumerate(_FRAGMENT_FNS):
            try:
                vec[FRAGPRINT_NBITS + j] = float(fn(mol))
            except Exception:
                pass
    _fragprint_cache[smiles] = vec
    return vec


def smiles_to_tensor(smiles_list) -> torch.Tensor:
    return torch.from_numpy(np.stack([fragprint(s) for s in smiles_list])).float()


class TanimotoKernel(Kernel):
    """MinMax/Tanimoto kernel, k(x, x') = <x,x'> / (||x||^2 + ||x'||^2 - <x,x'>).

    The norms are squared L2, which is the form proven positive semi-definite
    for non-negative real vectors and the form GAUCHE uses. An L1 (`x.sum()`)
    norm coincides with this on 0/1 vectors, so it was equivalent while the GP
    ran on binary ECFP -- it is not equivalent on the fragment counts, which
    are non-negative integers, so the squared norm is required here.
    """

    has_lengthscale = False

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return torch.ones(x1.shape[:-1], device=x1.device, dtype=x1.dtype)

        if x1.dim() == 2:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 2:
            x2 = x2.unsqueeze(0)

        inter = torch.matmul(x1, x2.transpose(-1, -2))
        x1_norm = (x1 * x1).sum(dim=-1).unsqueeze(-1)
        x2_norm = (x2 * x2).sum(dim=-1).unsqueeze(-2)
        union = torch.clamp(x1_norm + x2_norm - inter, min=1e-6)
        return (inter / union).clamp_min(0.0).squeeze(0)


class TanimotoGP(SingleTaskGP):
    def __init__(self, train_X, train_Y):
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-4))
        covar_module = ScaleKernel(TanimotoKernel())
        super().__init__(train_X, train_Y, covar_module=covar_module, likelihood=likelihood)


def fit_gp_model(train_smiles, y_vals):
    if len(train_smiles) < 2:
        return None, torch.tensor(0.0), torch.tensor(1.0)

    train_X = smiles_to_tensor(train_smiles)
    train_Y = torch.tensor(y_vals, dtype=torch.float32).unsqueeze(-1)

    y_mean = train_Y.mean()
    y_std = train_Y.std()
    if float(y_std.item()) < 1e-6:
        y_std = torch.tensor(1.0, dtype=torch.float32)

    model = TanimotoGP(train_X, (train_Y - y_mean) / y_std)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    try:
        fit_gpytorch_mll(mll)
        model.eval()
        model.likelihood.eval()
    except Exception:
        pass

    return model, y_mean, y_std


def acquire_qlogei(model, candidates, q=1, y_best_norm=0.0):
    """Batch selection by qLogExpectedImprovement over the admitted candidates.

    qLogEI optimises a smoothed log-EI rather than EI itself. The two have the
    same maximiser in exact arithmetic; the log form does not underflow to a
    flat zero once every candidate sits well below the incumbent, which is the
    regime a shrinking trust region walks into on the harder oracles.
    """
    if not candidates:
        return []

    q = max(1, min(int(q), len(candidates)))
    X_cand = smiles_to_tensor(candidates)
    sampler = IIDNormalSampler(sample_shape=torch.Size([256]))
    acq_func = qLogExpectedImprovement(model=model, best_f=y_best_norm, sampler=sampler)

    try:
        selected_X, _ = optimize_acqf_discrete(
            acq_function=acq_func,
            q=q,
            choices=X_cand,
            max_batch_size=2048,
            unique=True,
        )
    except Exception:
        return []

    selected = []
    for row in selected_X:
        matches = (X_cand == row).all(dim=1).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            selected.append(candidates[matches[0].item()])

    return list(dict.fromkeys(selected))


# ============================================================
# TuRBO-style state with batch-aware restart floor
# ============================================================

@dataclass
class TurboState:
    K: int = 200
    K_init: int = 200
    K_min: int = 20
    K_max: int = 1000
    tau_succ: int = 5
    tau_fail: int = 3
    expand_factor: float = 2.0
    shrink_factor: float = 0.5
    eps_improve: float = 1e-3
    s_min_floor: float = 0.10
    s_min_ceil: float = 0.50
    best_value: float = -float("inf")
    success_counter: int = 0
    failure_counter: int = 0

    # ---- region-ablation controls -------------------------------------
    # Defaults reproduce the coupled TRACE controller exactly (arm A).
    #   region_enabled=False -> no tau filter and no top-K truncation (arm E)
    #   k_adapts=False       -> K frozen at K_init (arms C, E)
    #   tau_mode="coupled"   -> tau = map(K)                (arms A, E)
    #   tau_mode="fixed"     -> tau = tau_fixed_value       (arm B)
    #   tau_mode="independent" -> tau = map(K_tau), K_tau on its own ladder
    #                             with its own counters      (arms C, D)
    region_enabled: bool = True
    k_adapts: bool = True
    tau_mode: str = "coupled"
    tau_fixed_value: float = 0.30
    K_tau: int = 200
    K_tau_init: int = 200
    simctrl_succ: int = 5
    simctrl_fail: int = 3
    simctrl_success_counter: int = 0
    simctrl_failure_counter: int = 0

    def restart_floor(self, batch_size: int) -> int:
        return max(int(self.K_min), int(batch_size))

    def _map_k_to_s(self, k: float) -> float:
        """Eq. 5: monotone decreasing linear map from budget to similarity floor."""
        ratio = (float(k) - self.K_min) / max(self.K_max - self.K_min, 1)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        s_min = self.s_min_ceil - ratio * (self.s_min_ceil - self.s_min_floor)
        return float(np.clip(s_min, self.s_min_floor, self.s_min_ceil))

    def get_s_min(self) -> float:
        if not self.region_enabled:
            return 0.0
        if self.tau_mode == "fixed":
            return float(np.clip(self.tau_fixed_value, 0.0, 1.0))
        if self.tau_mode == "independent":
            return self._map_k_to_s(self.K_tau)
        return self._map_k_to_s(self.K)

    def effective_K(self, n_available: int) -> int:
        """Cardinality budget actually applied to the admitted pool."""
        if not self.region_enabled:
            return int(n_available)
        return int(self.K)

    def reset_after_restart(self):
        self.K = int(self.K_init)
        self.K_tau = int(self.K_tau_init)
        self.success_counter = 0
        self.failure_counter = 0
        self.simctrl_success_counter = 0
        self.simctrl_failure_counter = 0

    def update(self, batch_max: float, batch_size: int) -> Tuple[str, str, str]:
        threshold = self.best_value + self.eps_improve * max(1.0, abs(self.best_value))

        if batch_max > threshold:
            self.success_counter += 1
            self.failure_counter = 0
            self.simctrl_success_counter += 1
            self.simctrl_failure_counter = 0
            action = "success"
        else:
            self.success_counter = 0
            self.failure_counter += 1
            self.simctrl_success_counter = 0
            self.simctrl_failure_counter += 1
            action = "failure"

        self.best_value = max(self.best_value, batch_max)
        status = "stable"
        reason = "none"

        # ---- independent tau ladder (arms C and D only) ------------------
        # Same success/failure signal as the K ladder, but its own counters
        # and its own thresholds, so the two can desynchronise. This is what
        # makes internally incoherent (tau, K) states reachable.
        sim_ladder_exhausted = False
        if self.region_enabled and self.tau_mode == "independent":
            if self.simctrl_success_counter >= self.simctrl_succ:
                self.K_tau = min(self.K_max, int(self.K_tau * self.expand_factor))
                self.simctrl_success_counter = 0
            elif self.simctrl_failure_counter >= self.simctrl_fail:
                new_K_tau = int(self.K_tau * self.shrink_factor)
                if new_K_tau < self.K_min:
                    sim_ladder_exhausted = True
                else:
                    self.K_tau = new_K_tau
                    self.simctrl_failure_counter = 0

        if not self.k_adapts:
            # K is frozen, so the K ladder cannot own the restart trigger.
            # When the tau ladder is the only adaptive component (arm C) it
            # inherits the restart role, mirroring the coupled rule.
            if sim_ladder_exhausted:
                self.reset_after_restart()
                return action, "restart", "tau_ladder_below_floor_restart"
            return action, status, reason

        if self.success_counter >= self.tau_succ:
            self.K = min(self.K_max, int(self.K * self.expand_factor))
            self.success_counter = 0
            status = "expanded"
            reason = "success_threshold"

        elif self.failure_counter >= self.tau_fail:
            new_K = int(self.K * self.shrink_factor)
            if new_K < self.restart_floor(batch_size):
                self.reset_after_restart()
                status = "restart"
                reason = "next_shrink_below_batch_restart_floor"
            else:
                self.K = new_K
                self.failure_counter = 0
                status = "shrunk"
                reason = "failure_threshold"

        return action, status, reason


# ============================================================
# Utilities
# ============================================================

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def write_json(path: Path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]):
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


def append_row_csv(path: Path, row: Dict[str, Any], fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(row.keys())

    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def diverse_pick(candidates, fp_cache, n):
    if not candidates:
        return []
    if len(candidates) <= n:
        return candidates[:]

    picked = [candidates[0]]
    remaining = candidates[1:]

    while len(picked) < n and remaining:
        best_smi = None
        best_min_dist = -1.0
        for smi in remaining:
            fp = fp_cache[smi]
            min_dist = min(
                1.0 - DataStructs.TanimotoSimilarity(fp, fp_cache[p])
                for p in picked
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_smi = smi
        if best_smi is None:
            break
        picked.append(best_smi)
        remaining.remove(best_smi)

    return picked


def top_off_batch(selected, candidates, fp_cache, target_size):
    selected = list(dict.fromkeys(selected))
    if len(selected) >= target_size:
        return selected[:target_size]

    used = set(selected)
    remaining = [s for s in candidates if s not in used]
    selected.extend(diverse_pick(remaining, fp_cache, target_size - len(selected)))
    return selected[:target_size]


def save_state(state: TurboState, path: Path):
    write_json(path, asdict(state))


def save_checkpoint(
    path: Path,
    seed: int,
    n_completed_batches: int,
    state: TurboState,
    active_center_smi: str,
    history_smi: List[str],
    history_y: List[float],
    trajectory: List[Dict[str, Any]],
    restart_events: List[Dict[str, Any]],
):
    best_idx = int(np.argmax(history_y))
    write_json(
        path,
        {
            "seed": int(seed),
            "n_completed_batches": int(n_completed_batches),
            "state": asdict(state),
            "active_center_smiles": active_center_smi,
            "best_smiles": history_smi[best_idx],
            "best_score": float(history_y[best_idx]),
            "n_evaluated": int(len(history_smi)),
            "n_restarts": int(len(restart_events)),
            "history_smiles": history_smi,
            "history_scores": [float(v) for v in history_y],
            "trajectory": trajectory,
            "restart_events": restart_events,
        },
    )


# ============================================================
# Candidate pools and restart centers
# ============================================================

def build_global_pool(
    center_mol: Chem.Mol,
    seed_mols: List[Chem.Mol],
    global_pool_size: int,
    constraints: MoleculeConstraints,
    label: str,
) -> List[str]:
    try:
        generator = build_generator(center_mol, "relaxed", constraints=constraints)
        smiles = generator.generate_library_auto(
            start_mols=seed_mols,
            N_target=global_pool_size,
            mode="global",
            verbose=False,
        )
        print(f"[Pool {label}] built {len(smiles)} candidates from {Chem.MolToSmiles(center_mol)[:60]}")
        return list(dict.fromkeys(smiles))
    except Exception as exc:
        print(f"[WARN] global pool build failed for {label}: {exc}")
        return []


def build_tr_candidates(
    center_smi: str,
    state: TurboState,
    history_set: set,
    global_pool: List[str],
    local_pool: List[str],
    fp_cache: Dict[str, Any],
    get_fp_c,
    constraints: MoleculeConstraints,
) -> Tuple[List[str], int, int]:
    center_fp = get_fp_c(center_smi)
    s_min = state.get_s_min()
    candidates = list(dict.fromkeys(global_pool + local_pool))

    tr_candidates = []
    for smi in candidates:
        if smi in history_set:
            continue
        mol = Chem.MolFromSmiles(smi)
        if not molecule_satisfies_constraints(mol, constraints):
            continue
        try:
            fp = get_fp_c(smi)
        except Exception:
            continue
        if DataStructs.TanimotoSimilarity(center_fp, fp) >= s_min:
            tr_candidates.append(smi)

    tr_candidates.sort(
        key=lambda smi: DataStructs.TanimotoSimilarity(center_fp, get_fp_c(smi)),
        reverse=True,
    )
    # n_clearing_tau = |{x in P \ D : T(x, c) >= tau}| BEFORE the top-K cut.
    # Comparing it with K tells us which of the two constraints actually binds.
    n_clearing_tau = len(tr_candidates)
    admitted = tr_candidates[: state.effective_K(n_clearing_tau)]
    return admitted, len(candidates), n_clearing_tau


def collect_center_candidates(center_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_smi: Dict[str, Dict[str, Any]] = {}

    for rec in center_records:
        smi = rec["center_smiles"]
        score = float(rec["center_score"])
        existing = by_smi.get(smi)
        if existing is None:
            by_smi[smi] = {
                "smiles": smi,
                "score": score,
                "times_used": 1,
                "best_batch_improvement": float(rec.get("batch_improvement", 0.0)),
                "max_tr_candidates": int(rec.get("n_tr_candidates", 0)),
                "last_iteration": int(rec.get("iteration", 0)),
            }
        else:
            existing["score"] = max(existing["score"], score)
            existing["times_used"] += 1
            existing["best_batch_improvement"] = max(
                existing["best_batch_improvement"],
                float(rec.get("batch_improvement", 0.0)),
            )
            existing["max_tr_candidates"] = max(
                existing["max_tr_candidates"],
                int(rec.get("n_tr_candidates", 0)),
            )
            existing["last_iteration"] = max(
                existing["last_iteration"],
                int(rec.get("iteration", 0)),
            )

    return sorted(
        by_smi.values(),
        key=lambda item: (
            item["score"],
            item["best_batch_improvement"],
            item["max_tr_candidates"],
            item["last_iteration"],
        ),
        reverse=True,
    )


def is_diverse_from_used(
    smi: str,
    used_restart_centers: set,
    fp_cache: Dict[str, Any],
    get_fp_c,
    threshold: float,
) -> bool:
    if smi in used_restart_centers:
        return False

    fp = fp_cache.get(smi)
    if fp is None:
        fp = get_fp_c(smi)

    for used in used_restart_centers:
        fp_used = fp_cache.get(used)
        if fp_used is None:
            try:
                fp_used = get_fp_c(used)
            except Exception:
                continue
        if DataStructs.TanimotoSimilarity(fp, fp_used) > threshold:
            return False

    return True


def select_restart_center(
    center_records: List[Dict[str, Any]],
    history_smi: List[str],
    history_y: List[float],
    fp_cache: Dict[str, Any],
    get_fp_c,
    used_restart_centers: set,
    diversity_threshold: float,
    excluded_centers: Optional[set] = None,
) -> Tuple[str, str]:
    """
    Prefer observed iteration centers, sorted by score and productivity. If all
    centers have already been tried or are too similar, fall back to the full
    evaluated history sorted by objective value.
    """
    excluded_centers = excluded_centers or set()

    for item in collect_center_candidates(center_records):
        smi = item["smiles"]
        if smi in excluded_centers:
            continue
        try:
            if is_diverse_from_used(
                smi,
                used_restart_centers,
                fp_cache,
                get_fp_c,
                diversity_threshold,
            ):
                return smi, "center_records"
        except Exception:
            continue

    for smi, _ in sorted(zip(history_smi, history_y), key=lambda pair: pair[1], reverse=True):
        if smi in excluded_centers:
            continue
        try:
            if is_diverse_from_used(
                smi,
                used_restart_centers,
                fp_cache,
                get_fp_c,
                diversity_threshold,
            ):
                return smi, "history_fallback"
        except Exception:
            continue

    return history_smi[int(np.argmax(history_y))], "global_best_fallback"


def do_restart(
    reason: str,
    attempted_iteration: int,
    state: TurboState,
    center_records: List[Dict[str, Any]],
    history_smi: List[str],
    history_y: List[float],
    fp_cache: Dict[str, Any],
    get_fp_c,
    used_restart_centers: set,
    restart_events: List[Dict[str, Any]],
    global_pool_size: int,
    constraints: MoleculeConstraints,
    diversity_threshold: float,
    score_mol: Callable[[Chem.Mol], float] = penalized_logp,
    excluded_centers: Optional[set] = None,
) -> Tuple[str, List[str]]:
    state.reset_after_restart()

    restart_center_smi, source = select_restart_center(
        center_records=center_records,
        history_smi=history_smi,
        history_y=history_y,
        fp_cache=fp_cache,
        get_fp_c=get_fp_c,
        used_restart_centers=used_restart_centers,
        diversity_threshold=diversity_threshold,
        excluded_centers=excluded_centers,
    )
    used_restart_centers.add(restart_center_smi)

    restart_center_mol = Chem.MolFromSmiles(restart_center_smi)
    restart_score = float(dict(zip(history_smi, history_y)).get(restart_center_smi, score_mol(restart_center_mol)))
    restart_number = len(restart_events) + 1

    print("\n" + "*" * 80)
    print(f"[RESTART {restart_number}] reason={reason}")
    print(f"[RESTART {restart_number}] source={source}")
    print(f"[RESTART {restart_number}] center_score={restart_score:.4f}")
    print(f"[RESTART {restart_number}] center={restart_center_smi[:80]}")
    print("*" * 80 + "\n")

    global_pool = build_global_pool(
        center_mol=restart_center_mol,
        seed_mols=[restart_center_mol],
        global_pool_size=global_pool_size,
        constraints=constraints,
        label=f"restart_{restart_number}",
    )

    restart_events.append(
        {
            "restart_number": int(restart_number),
            "attempted_iteration": int(attempted_iteration),
            "reason": reason,
            "source": source,
            "center_smiles": restart_center_smi,
            "center_score": restart_score,
            "K_after_restart": int(state.K),
            "s_min_after_restart": float(state.get_s_min()),
            "global_pool_size": int(len(global_pool)),
        }
    )

    return restart_center_smi, global_pool


# ============================================================
# Single-seed and multi-seed runs
# ============================================================

def run_trace_optimization(
    initial_smiles_list: List[str],
    n_iters: int = 50,
    batch_size: int = 5,
    seed: int = 0,
    run_dir: str = "trace_run/seed_0",
    global_pool_size: int = 1000,
    refresh_global_pool_each_iter: bool = False,
    max_sa_score: float = MAX_SA_SCORE,
    max_smiles_length: Optional[int] = None,
    max_heavy_atoms: Optional[int] = None,
    required_atomic_nums: Optional[List[int]] = None,
    restart_diversity_threshold: float = 0.70,
    max_restarts_per_batch: int = 3,
    score_mol: Optional[Callable[[Chem.Mol], float]] = None,
    objective_name: str = "penalized_logp",
    reward_label: str = "penalized logP",
    state_overrides: Optional[Dict[str, Any]] = None,
    arm_label: str = "coupled",
) -> Dict[str, Any]:
    if not initial_smiles_list:
        raise ValueError("Must provide at least one initial SMILES string.")
    score_mol = score_mol or penalized_logp

    set_all_seeds(seed)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    state_file = run_dir / f"turbo_state_seed_{seed}.json"
    checkpoint_file = run_dir / f"checkpoint_seed_{seed}.json"
    eval_csv = run_dir / f"evaluations_seed_{seed}.csv"
    trajectory_csv = run_dir / f"trajectory_seed_{seed}.csv"
    restarts_csv = run_dir / f"restarts_seed_{seed}.csv"
    summary_json = run_dir / f"summary_seed_{seed}.json"

    for path in [state_file, checkpoint_file, eval_csv, trajectory_csv, restarts_csv, summary_json]:
        if path.exists():
            path.unlink()

    state = TurboState()
    if state_overrides:
        valid = set(state.__dataclass_fields__)
        unknown = sorted(set(state_overrides) - valid)
        if unknown:
            raise ValueError(f"Unknown TurboState override(s): {unknown}")
        for key, value in state_overrides.items():
            setattr(state, key, value)
        # K_init / K_sim_init anchor restarts; keep the live values consistent
        # with them unless the caller set the live values explicitly.
        if "K_init" in state_overrides and "K" not in state_overrides:
            state.K = int(state.K_init)
        if "K_tau_init" in state_overrides and "K_tau" not in state_overrides:
            state.K_tau = int(state.K_tau_init)
    constraints = MoleculeConstraints(
        max_sa_score=float(max_sa_score),
        max_smiles_length=max_smiles_length,
        max_heavy_atoms=max_heavy_atoms,
        required_atomic_nums=required_atomic_nums,
    )
    history_smi: List[str] = []
    history_y: List[float] = []
    history_set = set()
    fp_cache: Dict[str, Any] = {}
    trajectory: List[Dict[str, Any]] = []
    restart_events: List[Dict[str, Any]] = []
    center_records: List[Dict[str, Any]] = []
    used_restart_centers = set()

    def get_fp_c(smi):
        if smi not in fp_cache:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES encountered: {smi}")
            fp_cache[smi] = get_fp(mol)
        return fp_cache[smi]

    print("\n" + "=" * 90)
    print(f"[RUN] seed={seed} objective={objective_name} iters={n_iters} batch={batch_size}")
    print(f"[RUN] constraints={asdict(constraints)} output={run_dir}")
    print(f"[RUN] restart floor = max(K_min, batch_size)")
    print("=" * 90)

    init_unique = list(dict.fromkeys(s for s in initial_smiles_list if isinstance(s, str) and s.strip()))

    for init_rank, smi in enumerate(init_unique, start=1):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"[Init] skipping invalid SMILES: {smi}")
            continue

        if not molecule_satisfies_constraints(mol, constraints):
            print(f"[Init] rejected by constraints: SA={get_sa_score(mol):.3f} {smi}")
            continue

        sa = get_sa_score(mol)
        score = float(score_mol(mol))
        history_smi.append(smi)
        history_y.append(score)
        history_set.add(smi)
        get_fp_c(smi)

        append_row_csv(
            eval_csv,
            {
                "seed": int(seed),
                "oracle": objective_name,
                "reward_label": reward_label,
                "oracle_call": int(len(history_smi)),
                "phase": "init",
                "iteration": 0,
                "batch_rank": int(init_rank),
                "smiles": smi,
                "sa": float(sa),
                "heavy_atoms": int(mol.GetNumHeavyAtoms()),
                "smiles_length": int(len(smi)),
                "score": float(score),
                "best_so_far": float(max(history_y)),
                "K": int(state.K),
                "s_min": float(state.get_s_min()),
                "n_restarts": 0,
                "center_smiles": "",
            },
        )

    if not history_smi:
        raise ValueError(f"No valid initial molecules satisfy constraints: {asdict(constraints)}.")

    best_idx = int(np.argmax(history_y))
    active_center_smi = history_smi[best_idx]
    state.best_value = float(history_y[best_idx])

    init_seed_mols = []
    for smi in history_smi:
        mol = Chem.MolFromSmiles(smi)
        if molecule_satisfies_constraints(mol, constraints):
            init_seed_mols.append(mol)

    global_pool = build_global_pool(
        center_mol=Chem.MolFromSmiles(active_center_smi),
        seed_mols=init_seed_mols,
        global_pool_size=global_pool_size,
        constraints=constraints,
        label="init",
    )

    trajectory.append(
        {
            "seed": int(seed),
            "iteration": 0,
            "n_evaluated": int(len(history_smi)),
            "n_batch": int(len(history_smi)),
            "batch_best": float(state.best_value),
            "best_so_far": float(state.best_value),
            "K": int(state.K),
            "s_min": float(state.get_s_min()),
            "status": "init",
            "action": "init",
            "restart_reason": "none",
            "n_restarts": 0,
            "center_smiles": active_center_smi,
            "best_smiles": active_center_smi,
        }
    )

    save_state(state, state_file)
    save_checkpoint(
        checkpoint_file,
        seed,
        0,
        state,
        active_center_smi,
        history_smi,
        history_y,
        trajectory,
        restart_events,
    )

    completed_batches = 0
    restart_attempts_for_batch = 0

    while completed_batches < n_iters:
        attempted_iteration = completed_batches + 1

        if state.K < state.restart_floor(batch_size):
            active_center_smi, global_pool = do_restart(
                reason="K_below_batch_restart_floor",
                attempted_iteration=attempted_iteration,
                state=state,
                center_records=center_records,
                history_smi=history_smi,
                history_y=history_y,
                fp_cache=fp_cache,
                get_fp_c=get_fp_c,
                used_restart_centers=used_restart_centers,
                restart_events=restart_events,
                global_pool_size=global_pool_size,
                constraints=constraints,
                diversity_threshold=restart_diversity_threshold,
                score_mol=score_mol,
                excluded_centers={active_center_smi},
            )
            restart_attempts_for_batch += 1
            write_rows_csv(restarts_csv, restart_events)
            continue

        center_mol = Chem.MolFromSmiles(active_center_smi)
        center_score = float(dict(zip(history_smi, history_y)).get(active_center_smi, score_mol(center_mol)))

        print(
            f"\n[Iter {attempted_iteration:02d}] center_score={center_score:.4f} "
            f"best={state.best_value:.4f} K={state.K} s_min={state.get_s_min():.3f} "
            f"restarts={len(restart_events)}"
        )

        if refresh_global_pool_each_iter:
            global_pool = build_global_pool(
                center_mol=center_mol,
                seed_mols=[center_mol],
                global_pool_size=global_pool_size,
                constraints=constraints,
                label=f"iter_{attempted_iteration}",
            )

        try:
            local_generator = build_generator(center_mol, "strict", constraints=constraints)
            local_pool = local_generator.generate_library_auto(
                start_mol=center_mol,
                N_target=max(50, int(state.K)),
                mode="balanced",
                verbose=False,
            )
        except Exception as exc:
            print(f"[WARN] local generation failed: {exc}")
            local_pool = []

        tr_candidates, n_raw_candidates, n_clearing_tau = build_tr_candidates(
            center_smi=active_center_smi,
            state=state,
            history_set=history_set,
            global_pool=global_pool,
            local_pool=local_pool,
            fp_cache=fp_cache,
            get_fp_c=get_fp_c,
            constraints=constraints,
        )

        if len(tr_candidates) < batch_size:
            print(
                f"[Iter {attempted_iteration:02d}] trust region underfilled "
                f"({len(tr_candidates)} < batch_size={batch_size}); trying fallback generation."
            )
            try:
                fallback_generator = build_generator(center_mol, "relaxed", constraints=constraints)
                fallback_pool = fallback_generator.generate_library_auto(
                    start_mol=center_mol,
                    N_target=max(batch_size * 10, 100),
                    mode="global",
                    verbose=False,
                )
            except Exception:
                fallback_pool = []

            tr_candidates, n_raw_candidates, n_clearing_tau = build_tr_candidates(
                center_smi=active_center_smi,
                state=state,
                history_set=history_set,
                global_pool=global_pool,
                local_pool=local_pool + fallback_pool,
                fp_cache=fp_cache,
                get_fp_c=get_fp_c,
                constraints=constraints,
            )

        if len(tr_candidates) < batch_size:
            center_records.append(
                {
                    "seed": int(seed),
                    "iteration": int(attempted_iteration),
                    "center_smiles": active_center_smi,
                    "center_score": float(center_score),
                    "n_raw_candidates": int(n_raw_candidates),
                    "n_tr_candidates": int(len(tr_candidates)),
                    "batch_improvement": 0.0,
                    "status": "candidate_starved",
                }
            )

            active_center_smi, global_pool = do_restart(
                reason=f"trust_region_underfilled_{len(tr_candidates)}_lt_batch_{batch_size}",
                attempted_iteration=attempted_iteration,
                state=state,
                center_records=center_records,
                history_smi=history_smi,
                history_y=history_y,
                fp_cache=fp_cache,
                get_fp_c=get_fp_c,
                used_restart_centers=used_restart_centers,
                restart_events=restart_events,
                global_pool_size=global_pool_size,
                constraints=constraints,
                diversity_threshold=restart_diversity_threshold,
                score_mol=score_mol,
                excluded_centers={active_center_smi},
            )
            restart_attempts_for_batch += 1
            write_rows_csv(restarts_csv, restart_events)
            save_state(state, state_file)
            save_checkpoint(
                checkpoint_file,
                seed,
                completed_batches,
                state,
                active_center_smi,
                history_smi,
                history_y,
                trajectory,
                restart_events,
            )

            if restart_attempts_for_batch > max_restarts_per_batch:
                raise RuntimeError(
                    "Could not form a full batch after repeated restarts. "
                    "Increase global_pool_size, relax chemistry constraints, or lower batch_size."
                )
            continue

        restart_attempts_for_batch = 0
        q = int(batch_size)

        # Region state as it stood when this batch's pool was admitted, i.e.
        # BEFORE state.update() below. These are the numbers the binding
        # diagnostic must use.
        K_at_admission = int(state.effective_K(n_clearing_tau))
        s_min_at_admission = float(state.get_s_min())
        tau_binds_now = int(n_clearing_tau < K_at_admission)

        if len(history_y) < 5:
            print("[Acq] diversity pick during warm start")
            batch = diverse_pick(tr_candidates, fp_cache, q)
        else:
            print(f"[Acq] GP + qLogEI over {len(tr_candidates)} trust-region candidates")
            model, y_mean, y_std = fit_gp_model(history_smi, history_y)
            if model is None:
                batch = diverse_pick(tr_candidates, fp_cache, q)
            else:
                best_f_norm = (state.best_value - y_mean) / y_std
                batch = acquire_qlogei(model, tr_candidates, q=q, y_best_norm=best_f_norm)
                batch = top_off_batch(batch, tr_candidates, fp_cache, q)

        batch = list(dict.fromkeys(batch))[:q]
        if len(batch) != batch_size:
            raise RuntimeError(
                f"Internal error: selected batch has size {len(batch)}, expected {batch_size}."
            )

        best_before = float(state.best_value)
        batch_y = []

        for batch_rank, smi in enumerate(batch, start=1):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            if not molecule_satisfies_constraints(mol, constraints):
                continue

            sa = get_sa_score(mol)
            score = float(score_mol(mol))
            history_smi.append(smi)
            history_y.append(score)
            history_set.add(smi)
            get_fp_c(smi)
            batch_y.append(score)

            append_row_csv(
                eval_csv,
                {
                    "seed": int(seed),
                "oracle": objective_name,
                "reward_label": reward_label,
                "oracle_call": int(len(history_smi)),
                "phase": "iter",
                "iteration": int(attempted_iteration),
                "batch_rank": int(batch_rank),
                    "smiles": smi,
                    "sa": float(sa),
                    "heavy_atoms": int(mol.GetNumHeavyAtoms()),
                    "smiles_length": int(len(smi)),
                    "score": float(score),
                    "best_so_far": float(max(history_y)),
                    "K": int(state.K),
                    "s_min": float(state.get_s_min()),
                    "n_restarts": int(len(restart_events)),
                    "center_smiles": active_center_smi,
                },
            )

            print(f"  > rank={batch_rank} SA={sa:.3f} {reward_label}={score:.4f} {smi[:80]}")

        if len(batch_y) != batch_size:
            raise RuntimeError(
                f"Only {len(batch_y)} molecules passed final checks from a selected batch of {batch_size}."
            )

        action, status, reason = state.update(max(batch_y), batch_size=batch_size)
        completed_batches += 1

        best_idx = int(np.argmax(history_y))
        best_smi = history_smi[best_idx]
        batch_best = float(max(batch_y))
        batch_improvement = max(0.0, batch_best - best_before)

        center_records.append(
            {
                "seed": int(seed),
                "iteration": int(attempted_iteration),
                "center_smiles": active_center_smi,
                "center_score": float(center_score),
                "n_raw_candidates": int(n_raw_candidates),
                "n_tr_candidates": int(len(tr_candidates)),
                "batch_improvement": float(batch_improvement),
                "status": status,
            }
        )

        if status == "restart":
            restart_reason = reason
        else:
            restart_reason = "none"

        trajectory.append(
            {
                "seed": int(seed),
                "iteration": int(attempted_iteration),
                "n_evaluated": int(len(history_smi)),
                "n_batch": int(len(batch_y)),
                "batch_best": float(batch_best),
                "best_so_far": float(state.best_value),
                "K": int(state.K),
                "K_tau": int(state.K_tau),
                "restart_floor": int(state.restart_floor(batch_size)),
                "s_min": float(state.get_s_min()),
                "K_at_admission": K_at_admission,
                "s_min_at_admission": s_min_at_admission,
                "n_raw_candidates": int(n_raw_candidates),
                "n_clearing_tau": int(n_clearing_tau),
                "n_admitted": int(len(tr_candidates)),
                "tau_binds": tau_binds_now,
                "status": status,
                "action": action,
                "restart_reason": restart_reason,
                "n_restarts": int(len(restart_events)),
                "center_smiles": active_center_smi,
                "best_smiles": best_smi,
            }
        )

        print(
            f"[Update] {action.upper()} {status.upper()} best={state.best_value:.4f} "
            f"K={state.K} floor={state.restart_floor(batch_size)}"
        )

        if status == "restart":
            active_center_smi, global_pool = do_restart(
                reason=reason,
                attempted_iteration=attempted_iteration + 1,
                state=state,
                center_records=center_records,
                history_smi=history_smi,
                history_y=history_y,
                fp_cache=fp_cache,
                get_fp_c=get_fp_c,
                used_restart_centers=used_restart_centers,
                restart_events=restart_events,
                global_pool_size=global_pool_size,
                constraints=constraints,
                diversity_threshold=restart_diversity_threshold,
                score_mol=score_mol,
                excluded_centers={active_center_smi},
            )
            trajectory[-1]["n_restarts"] = int(len(restart_events))
            write_rows_csv(restarts_csv, restart_events)
        else:
            active_center_smi = best_smi

        write_rows_csv(trajectory_csv, trajectory)
        save_state(state, state_file)
        save_checkpoint(
            checkpoint_file,
            seed,
            completed_batches,
            state,
            active_center_smi,
            history_smi,
            history_y,
            trajectory,
            restart_events,
        )

    final_best_idx = int(np.argmax(history_y))
    final_best_mol = Chem.MolFromSmiles(history_smi[final_best_idx])
    summary = {
        "method": "trace_turbomol",
        "arm": str(arm_label),
        "region_config": {
            "region_enabled": bool(state.region_enabled),
            "k_adapts": bool(state.k_adapts),
            "tau_mode": str(state.tau_mode),
            "tau_fixed_value": float(state.tau_fixed_value),
            "K_init": int(state.K_init),
            "K_min": int(state.K_min),
            "K_max": int(state.K_max),
            "tau_succ": int(state.tau_succ),
            "tau_fail": int(state.tau_fail),
            "simctrl_succ": int(state.simctrl_succ),
            "simctrl_fail": int(state.simctrl_fail),
            "s_min_floor": float(state.s_min_floor),
            "s_min_ceil": float(state.s_min_ceil),
        },
        "seed": int(seed),
        "objective": objective_name,
        "reward_label": reward_label,
        "max_sa_score": float(max_sa_score),
        "max_smiles_length": max_smiles_length,
        "max_heavy_atoms": max_heavy_atoms,
        "required_atomic_nums": required_atomic_nums,
        "n_iterations": int(n_iters),
        "batch_size": int(batch_size),
        "restart_floor": int(state.restart_floor(batch_size)),
        "best_smiles": history_smi[final_best_idx],
        "best_score": float(history_y[final_best_idx]),
        "best_sa": float(get_sa_score(final_best_mol)),
        "n_evaluated_total": int(len(history_smi)),
        "n_restarts": int(len(restart_events)),
        "run_dir": str(run_dir),
        "state_file": str(state_file),
        "checkpoint_file": str(checkpoint_file),
        "trajectory_csv": str(trajectory_csv),
        "evaluations_csv": str(eval_csv),
        "restarts_csv": str(restarts_csv),
    }
    write_json(summary_json, summary)

    print(
        f"\n[DONE] seed={seed} best_{reward_label}={summary['best_score']:.4f} "
        f"best_SA={summary['best_sa']:.3f} restarts={summary['n_restarts']}"
    )

    return {
        "seed": int(seed),
        "run_dir": str(run_dir),
        "best_score": float(summary["best_score"]),
        "best_smiles": summary["best_smiles"],
        "n_restarts": int(summary["n_restarts"]),
        "trajectory": trajectory,
        "summary": summary,
    }


def run_benchmark(
    initial_smiles_by_seed: Dict[int, List[str]],
    seeds: Optional[List[int]] = None,
    n_iters: int = 50,
    batch_size: int = 5,
    out_root: str = "trace_run",
    global_pool_size: int = 1000,
    refresh_global_pool_each_iter: bool = False,
    max_sa_score: float = MAX_SA_SCORE,
    max_smiles_length: Optional[int] = None,
    max_heavy_atoms: Optional[int] = None,
    required_atomic_nums: Optional[List[int]] = None,
    restart_diversity_threshold: float = 0.70,
    score_mol: Optional[Callable[[Chem.Mol], float]] = None,
    objective_name: str = "penalized_logp",
    reward_label: str = "penalized logP",
    state_overrides: Optional[Dict[str, Any]] = None,
    arm_label: str = "coupled",
) -> Dict[str, Any]:
    if seeds is None:
        seeds = [11, 22, 33, 44, 55]

    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)

    run_results = []
    best_matrix = []

    for seed in seeds:
        if seed not in initial_smiles_by_seed:
            raise ValueError(f"Missing initial SMILES list for seed {seed}.")

        result = run_trace_optimization(
            initial_smiles_list=initial_smiles_by_seed[seed],
            n_iters=n_iters,
            batch_size=batch_size,
            seed=seed,
            run_dir=str(out_root_path / f"seed_{seed}"),
            global_pool_size=global_pool_size,
            refresh_global_pool_each_iter=refresh_global_pool_each_iter,
            max_sa_score=max_sa_score,
            max_smiles_length=max_smiles_length,
            max_heavy_atoms=max_heavy_atoms,
            required_atomic_nums=required_atomic_nums,
            restart_diversity_threshold=restart_diversity_threshold,
            score_mol=score_mol,
            objective_name=objective_name,
            reward_label=reward_label,
            state_overrides=state_overrides,
            arm_label=arm_label,
        )
        run_results.append(result)

        best_curve = [float(row["best_so_far"]) for row in result["trajectory"] if row["iteration"] <= n_iters]
        if len(best_curve) < n_iters + 1:
            best_curve.extend([best_curve[-1]] * (n_iters + 1 - len(best_curve)))
        best_matrix.append(best_curve[: n_iters + 1])

    best_matrix = np.asarray(best_matrix, dtype=float)
    mean_curve = np.mean(best_matrix, axis=0)
    std_curve = np.std(best_matrix, axis=0)

    np.save(out_root_path / "best_so_far_matrix.npy", best_matrix)

    final_rows = [
        {
            "seed": int(result["seed"]),
            "best_score": float(result["best_score"]),
            "best_smiles": result["best_smiles"],
            "best_sa": float(result["summary"]["best_sa"]),
            "n_restarts": int(result["n_restarts"]),
            "run_dir": result["run_dir"],
        }
        for result in run_results
    ]
    write_rows_csv(out_root_path / "final_scores_by_seed.csv", final_rows)

    agg_rows = []
    for i in range(n_iters + 1):
        row = {
            "iteration": int(i),
            "mean_best_so_far": float(mean_curve[i]),
            "std_best_so_far": float(std_curve[i]),
        }
        for j, seed in enumerate(seeds):
            row[f"seed_{seed}"] = float(best_matrix[j, i])
        agg_rows.append(row)
    write_rows_csv(out_root_path / "benchmark_mean_std.csv", agg_rows)

    x = np.arange(n_iters + 1)
    plt.figure(figsize=(9, 6))
    for row in best_matrix:
        plt.plot(x, row, alpha=0.25)
    plt.plot(x, mean_curve, linewidth=2.5, label="Mean best-so-far")
    plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, label="+/- 1 std")
    plt.xlabel("Completed BO batch (0 = initial molecules)")
    plt.ylabel(f"Best {reward_label} so far")
    plt.title(
        f"TRACE/TuRBO-qLogEI {objective_name} benchmark | {len(seeds)} seeds | "
        f"{n_iters} batches | batch={batch_size} | SA<={max_sa_score:.1f}"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_root_path / "benchmark_mean_std.png", dpi=250)
    plt.close()

    summary = {
        "method": "trace_turbomol",
        "objective": objective_name,
        "reward_label": reward_label,
        "n_seeds": int(len(seeds)),
        "seeds": [int(s) for s in seeds],
        "n_iters": int(n_iters),
        "batch_size": int(batch_size),
        "max_sa_score": float(max_sa_score),
        "max_smiles_length": max_smiles_length,
        "max_heavy_atoms": max_heavy_atoms,
        "required_atomic_nums": required_atomic_nums,
        "restart_floor_rule": "max(K_min, batch_size)",
        "gp_features": f"fragprints(morgan_r{FRAGPRINT_RADIUS}_{FRAGPRINT_NBITS}bit + {len(_FRAGMENT_FNS)} rdkit fr_ counts)",
        "gp_feature_dim": int(FRAGPRINT_DIM),
        "gp_kernel": "ScaleKernel(TanimotoKernel[squared-L2 MinMax])",
        "acquisition": "qLogExpectedImprovement",
        "mean_final_best": float(mean_curve[-1]),
        "std_final_best": float(std_curve[-1]),
        "best_seed": int(seeds[int(np.argmax([r["best_score"] for r in run_results]))]),
        "best_score_overall": float(max(r["best_score"] for r in run_results)),
        "total_restarts": int(sum(r["n_restarts"] for r in run_results)),
        "out_root": str(out_root_path),
    }
    write_json(out_root_path / "benchmark_summary.json", summary)

    print("\n" + "=" * 90)
    print("[BENCHMARK DONE]")
    print(f"Mean final best : {summary['mean_final_best']:.4f} +/- {summary['std_final_best']:.4f}")
    print(f"Best overall    : {summary['best_score_overall']:.4f}")
    print(f"Total restarts  : {summary['total_restarts']}")
    print(f"Output          : {out_root_path}")
    print("=" * 90)

    return summary


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
            raise ValueError(f"Unknown element symbol or atomic number: {raw}")
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
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Clean TuRBO-qLogEI benchmark for penalized logP.")
    parser.add_argument("--out", default="trace_run", help="Output directory.")
    parser.add_argument("--n-iters", type=int, default=5, help="Number of full BO batches per seed.")
    parser.add_argument("--batch-size", type=int, default=5, help="Oracle evaluations per BO batch.")
    parser.add_argument("--global-pool-size", type=int, default=1000, help="Generator global pool size.")
    parser.add_argument("--max-sa", type=float, default=MAX_SA_SCORE, help="Hard maximum SA score.")
    parser.add_argument(
        "--max-smiles-length",
        type=int,
        default=80,
        help="Optional maximum canonical SMILES length. If omitted, uses generator.py defaults.",
    )
    parser.add_argument(
        "--max-heavy-atoms",
        type=int,
        default=None,
        help="Optional maximum heavy atom count. If omitted, uses generator.py derived defaults.",
    )
    parser.add_argument(
        "--require-fe",
        type=str_to_bool,
        default=False,
        metavar="true/false",
        help="Require every initial/generated/evaluated molecule to contain Fe.",
    )
    parser.add_argument(
        "--require-elements",
        nargs="*",
        default=None,
        help="Require these elements in every molecule. Accepts symbols or atomic numbers, e.g. Fe N or 26 7.",
    )
    parser.add_argument(
        "--refresh-global-pool-each-iter",
        type=str_to_bool,
        default=False,
        metavar="true/false",
        help="Rebuild the global pool from the active center every completed batch.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[11, 22, 33, 44, 55],
        help="Benchmark seeds to run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    required_atomic_nums = parse_required_atomic_nums(args.require_elements, require_fe=args.require_fe)
    run_benchmark(
        initial_smiles_by_seed=default_initial_smiles_by_seed(),
        seeds=args.seeds,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        out_root=args.out,
        global_pool_size=args.global_pool_size,
        refresh_global_pool_each_iter=args.refresh_global_pool_each_iter,
        max_sa_score=args.max_sa,
        max_smiles_length=args.max_smiles_length,
        max_heavy_atoms=args.max_heavy_atoms,
        required_atomic_nums=required_atomic_nums,
    )
