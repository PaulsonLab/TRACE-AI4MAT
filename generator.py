#############################################################################
# WORKING CODE FOR GENERATOR  (UPDATED: Fe-only filter + multi-seed support)
#############################################################################

import os
import sys
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdchem, AllChem, DataStructs, GraphDescriptors
from rdkit import RDLogger

try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
except ImportError:
    sascorer = None

# ============================================================
# 0. GLOBAL CONFIG
# ============================================================

RDLogger.DisableLog("rdApp.*")  # silence RDKit chatter

FP_RADIUS = 2
FP_NBITS = 2048

# Crude max valence table for mutate_atom (helps avoid nonsense)
MAX_VALENCE = {
    6: 4,   # C
    7: 3,   # N
    8: 2,   # O
    9: 1,   # F
    16: 6,  # S
    17: 1,  # Cl
    35: 1,  # Br
    26: 6,  # Fe (not perfect; used only for a crude mutate filter)
}


@dataclass
class HybridConfig:
    """
    Chemistry constraints derived from the seed molecule and a coarse chem_profile.
    """
    min_heavy_atoms: int = 5
    max_heavy_atoms: int = 150
    strict_sanitization: bool = True
    keep_largest_fragment: bool = True
    max_smiles_length: int = 200
    min_bertz: float = 100.0
    min_rings: int = 1

    # NEW: require certain elements to be present in every accepted molecule
    # Example: [26] means "must contain Fe"
    require_element_atomic_nums: Optional[List[int]] = None
    max_sa_score: Optional[float] = None


# ============================================================
# 1. AUTO-TUNING PRESETS
# ============================================================

def derive_config_from_seed(seed_mol: Chem.Mol, chem_profile: str = "default") -> HybridConfig:
    """
    Derive chemistry constraints from the seed molecule and a high-level profile.
    chem_profile ∈ {"strict", "default", "relaxed"}.
    """
    n0 = seed_mol.GetNumHeavyAtoms()
    rinfo = seed_mol.GetRingInfo()
    r0 = rinfo.NumRings()
    try:
        b0 = GraphDescriptors.BertzCT(seed_mol)
    except Exception:
        b0 = 200.0  # fallback

    MAX_HEAVY_CAP = 150
    MIN_SMILES_LENGTH = 150

    chem_profile = chem_profile.lower()
    if chem_profile == "strict":
        min_heavy = max(4, int(0.7 * n0))
        max_heavy = max(min_heavy + 1, int(1.5 * n0))
        min_rings = max(1, min(r0, 4))
        min_bertz = 0.5 * b0
    elif chem_profile == "relaxed":
        min_heavy = max(3, int(0.3 * n0))
        max_heavy = max(min_heavy + 1, int(2.5 * n0))
        min_rings = 1
        min_bertz = 0.2 * b0
    else:  # "default"
        min_heavy = max(4, int(0.5 * n0))
        max_heavy = max(min_heavy + 1, int(2.0 * n0))
        min_rings = max(1, min(r0, 3))
        min_bertz = 0.3 * b0

    max_heavy = min(max_heavy, MAX_HEAVY_CAP)

    return HybridConfig(
        min_heavy_atoms=min_heavy,
        max_heavy_atoms=max_heavy,
        strict_sanitization=True,
        keep_largest_fragment=True,
        max_smiles_length=MIN_SMILES_LENGTH,
        min_bertz=min_bertz,
        min_rings=min_rings,
        require_element_atomic_nums=None,  # set later if desired
    )


# Mode presets: "local", "balanced", "global"
MODE_PRESETS: Dict[str, Dict[str, float]] = {
    "local": {
        "inner_q": 0.40,
        "outer_q": 0.90,
        "base_reset_prob": 0.20,
        "max_reset_prob": 0.80,
        "w_shell": 1.0,
        "w_div_min": 0.3,
        "w_div_max": 0.6,
        "w_local_min": 0.8,
        "w_local_max": 1.5,
        "bandit_eta": 3.0,
        "novelty_quantile": 0.30,
        "random_walk_steps": 3,
    },
    "balanced": {
        "inner_q": 0.33,
        "outer_q": 0.66,
        "base_reset_prob": 0.15,
        "max_reset_prob": 0.70,
        "w_shell": 0.8,
        "w_div_min": 0.3,
        "w_div_max": 0.8,
        "w_local_min": 0.8,
        "w_local_max": 1.2,
        "bandit_eta": 3.0,
        "novelty_quantile": 0.25,
        "random_walk_steps": 4,
    },
    "global": {
        "inner_q": 0.25,
        "outer_q": 0.80,
        "base_reset_prob": 0.05,
        "max_reset_prob": 0.50,
        "w_shell": 0.6,
        "w_div_min": 0.4,
        "w_div_max": 1.0,
        "w_local_min": 0.6,
        "w_local_max": 1.0,
        "bandit_eta": 4.0,
        "novelty_quantile": 0.35,
        "random_walk_steps": 4,
    },
}


# ============================================================
# 2. KNOWLEDGE BASE
# ============================================================

GENERIC_ATOMS: List[int] = [6, 7, 8, 9, 16, 17, 35]  # C, N, O, F, S, Cl, Br

ADDITION_FRAGMENTS: List[str] = [
    "[*]C", "[*]CC", "[*]C(C)C", "[*]C(F)(F)F", "[*]OC", "[*]OCC",
    "[*]C(=O)N", "[*]C(=O)O", "[*]C(=O)OC",
    "[*]c1ccccc1", "[*]c1ccncc1",
    "[*]C#N", "[*]S(=O)(=O)C", "[*]N",
    "[*]F", "[*]Cl", "[*]Br", "[*]I",
    "[*]C1CC1", "[*]C1CCCCC1", "[*]C1=CC=CC=C1",
]

REACTION_PATTERNS: List[str] = [
    "[C:1](=[O:2])[OH]>>[C:1](=[O:2])NC",
    "[C:1](=[O:2])[OH]>>[C:1](=[O:2])OC",
    "[c:1][Cl,Br,I]>>[c:1]c2ccccc2",
    "[c:1][Cl,Br,I]>>[c:1]N",
    "[c:1][OH]>>[c:1]OC",
    "[#16X2:1]>>[#16:1](=O)(=O)",
    "[C:1]#[C:2]>>[C:1]c3cn[nH]c3",
]
COMPILED_REACTIONS = [AllChem.ReactionFromSmarts(x) for x in REACTION_PATTERNS]


# ============================================================
# 3. HELPERS
# ============================================================

def shortest_path_len(mol: Chem.Mol, i: int, j: int) -> int:
    if i == j:
        return 0
    try:
        path = Chem.GetShortestPath(mol, i, j)
        return (len(path) - 1) if path else 100
    except Exception:
        return 100


def add_chain_between(rw: Chem.RWMol, a_idx: int, b_idx: int, n_new: int) -> List[int]:
    if n_new == 0:
        if rw.GetBondBetweenAtoms(a_idx, b_idx) is None:
            rw.AddBond(a_idx, b_idx, rdchem.BondType.SINGLE)
        return []

    new_ids = []
    prev = a_idx
    for _ in range(n_new):
        nid = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(prev, nid, rdchem.BondType.SINGLE)
        prev = nid
        new_ids.append(nid)

    if rw.GetBondBetweenAtoms(prev, b_idx) is None:
        rw.AddBond(prev, b_idx, rdchem.BondType.SINGLE)
    return new_ids


def get_fp(m: Chem.Mol) -> DataStructs.ExplicitBitVect:
    return AllChem.GetMorganFingerprintAsBitVect(m, FP_RADIUS, FP_NBITS)


def mol_has_required_elements(mol: Chem.Mol, required_atomic_nums: Optional[List[int]]) -> bool:
    """
    Return True if mol contains all required atomic numbers at least once.
    Example: required_atomic_nums=[26] forces Fe to be present.
    """
    if not required_atomic_nums:
        return True
    present = set(a.GetAtomicNum() for a in mol.GetAtoms())
    return all(z in present for z in required_atomic_nums)


# ============================================================
# 4. KERNEL-BASED NOVELTY CONTROLLER
# ============================================================

class KernelNoveltyController:
    """
    Maintain a Cholesky factor L of (K + λI) over an anchor set of fingerprints.
    For a new fp, compute a variance-like gain Δ and map to [0,1] as novelty.
    """

    def __init__(self, lam: float = 1e-3, eps: float = 1e-9, max_points: Optional[int] = 100):
        self.lam = lam
        self.eps = eps
        self.max_points = max_points
        self.fps: List[DataStructs.ExplicitBitVect] = []
        self.L: Optional[np.ndarray] = None  # Cholesky of (K + λI)

    @staticmethod
    def tanimoto(fp1, fp2) -> float:
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def kernel(self, fp1, fp2) -> float:
        return float(self.tanimoto(fp1, fp2))

    def _compute_y_and_delta(self, fp: DataStructs.ExplicitBitVect) -> Tuple[Optional[np.ndarray], float]:
        k_xx = 1.0
        if not self.fps:
            delta = k_xx + self.lam
            return None, float(delta)

        n = len(self.fps)
        v = np.zeros((n,), dtype=float)
        for i, old_fp in enumerate(self.fps):
            v[i] = self.kernel(fp, old_fp)

        y = np.linalg.solve(self.L, v)
        delta = k_xx + self.lam - float(np.dot(y, y))
        delta = max(delta, self.eps)
        return y, delta

    def _delta_to_reward(self, delta: float) -> float:
        delta_clamped = max(self.lam, min(1.0 + self.lam, delta))
        raw = (delta_clamped - self.lam) / (1.0 - self.lam)
        return float(max(0.0, min(1.0, raw)))

    def add_point_and_get_reward(self, fp: DataStructs.ExplicitBitVect) -> float:
        if not self.fps:
            delta = 1.0 + self.lam
            reward = self._delta_to_reward(delta)
            self.L = np.array([[np.sqrt(delta)]], dtype=float)
            self.fps.append(fp)
            return reward

        y, delta = self._compute_y_and_delta(fp)
        reward = self._delta_to_reward(delta)

        if self.max_points is None or len(self.fps) < self.max_points:
            n = len(self.fps)
            L_new = np.zeros((n + 1, n + 1), dtype=float)
            L_new[:n, :n] = self.L
            L_new[n, :n] = y
            L_new[n, n] = np.sqrt(max(delta, self.eps))
            self.L = L_new
            self.fps.append(fp)

        return reward


# ============================================================
# 5. OPERATOR BANDIT
# ============================================================

class OperatorBandit:
    """
    Simple softmax bandit over operators.
    """

    def __init__(self, base_probs: Dict[str, float], eta: float = 3.0, min_prob: float = 1e-3):
        self.ops = list(base_probs.keys())
        self.base_probs = {op: float(p) for op, p in base_probs.items()}
        s = sum(self.base_probs.values())
        for k in self.base_probs:
            self.base_probs[k] /= s

        self.eta = eta
        self.min_prob = min_prob
        self.counts: Dict[str, int] = {op: 0 for op in self.ops}
        self.mean_rewards: Dict[str, float] = {op: 0.0 for op in self.ops}

    def update(self, op: str, reward: float) -> None:
        if op not in self.counts:
            return
        c = self.counts[op]
        m = self.mean_rewards[op]
        c_new = c + 1
        m_new = m + (reward - m) / c_new
        self.counts[op] = c_new
        self.mean_rewards[op] = m_new

    def get_probs(self) -> Dict[str, float]:
        scores = []
        for op in self.ops:
            base_p = self.base_probs[op]
            base_logit = np.log(base_p + 1e-12)
            m_r = self.mean_rewards[op]
            score = base_logit + self.eta * m_r
            scores.append(score)
        scores = np.array(scores, dtype=float)
        max_s = np.max(scores)
        exp_s = np.exp(scores - max_s)
        probs = exp_s / np.sum(exp_s)
        probs = np.maximum(probs, self.min_prob)
        probs /= np.sum(probs)
        return {op: float(p) for op, p in zip(self.ops, probs)}


# ============================================================
# 6. HYBRID PERTURBATOR
# ============================================================

class HybridPerturbator:
    def __init__(self, config: HybridConfig = None, probs: Dict[str, float] = None):
        self.config = config or HybridConfig()
        if self.config.max_sa_score is not None and sascorer is None:
            raise ImportError("SA_Score (sascorer) is required because max_sa_score is set, but it could not be imported.")

        default_probs = {
            "add_fragment": 0.05,
            "apply_reaction": 0.05,
            "fuse_ring": 0.10,
            "internal_bridge": 0.10,
            "spiro_ring": 0.10,
            "mutate_atom": 0.15,
            "open_ring": 0.02,
            "delete_fragment": 0.08,
            "delete_ring": 0.04,
            "shrink_ring": 0.04,
            "bridge_ring": 0.04,
            "peel_ring": 0.09,
            "micro_mutate": 0.09,
            "micro_halogen": 0.05,
        }
        self.probs = probs or default_probs
        total = sum(self.probs.values())
        for k in self.probs:
            self.probs[k] /= total
        self.base_probs = dict(self.probs)

    # ----------------- sanitization -----------------

    def _sanitize(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        if mol is None:
            return None
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)
            if self.config.strict_sanitization:
                Chem.SanitizeMol(mol)
            else:
                flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                Chem.SanitizeMol(mol, sanitizeOps=flags)

            frags = Chem.GetMolFrags(mol, asMols=True)
            if len(frags) > 1:
                if self.config.keep_largest_fragment:
                    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
                    mol.UpdatePropertyCache(strict=False)
                else:
                    return None

            n_heavy = mol.GetNumHeavyAtoms()
            if n_heavy < self.config.min_heavy_atoms:
                return None
            if n_heavy > self.config.max_heavy_atoms:
                return None

            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            if self.config.max_smiles_length is not None and len(smi) > self.config.max_smiles_length:
                return None

            # Enforce minimum ring count
            if mol.GetRingInfo().NumRings() < self.config.min_rings:
                return None

            bertz = GraphDescriptors.BertzCT(mol)
            if bertz < self.config.min_bertz:
                return None

            # NEW: enforce required elements (e.g., Fe)
            if not mol_has_required_elements(mol, self.config.require_element_atomic_nums):
                return None
            
            if self.config.max_sa_score is not None:
                try:
                    sa = float(sascorer.calculateScore(mol))
                except Exception:
                    return None
                if sa > self.config.max_sa_score:
                    return None

            return mol
        except Exception:
            return None

    # ----------------- operators -----------------

    def op_mutate_atom_generic(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        rw_mol = Chem.RWMol(mol)
        atom_indices = [a.GetIdx() for a in rw_mol.GetAtoms()]
        if not atom_indices:
            return None
        idx = random.choice(atom_indices)
        atom = rw_mol.GetAtomWithIdx(idx)
        current_num = atom.GetAtomicNum()
        candidates = [n for n in GENERIC_ATOMS if n != current_num]
        if not candidates:
            return None
        new_num = random.choice(candidates)

        atom.SetAtomicNum(new_num)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(False)

        deg = atom.GetTotalDegree()
        max_val = MAX_VALENCE.get(new_num, 4)
        if deg > max_val:
            return None

        return self._sanitize(rw_mol.GetMol())

    def op_micro_mutate_atom(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        fp_old = get_fp(mol)
        rw_mol = Chem.RWMol(mol)

        candidates: List[int] = []
        for atom in rw_mol.GetAtoms():
            if atom.IsInRing():
                continue
            if atom.GetTotalDegree() > 2:
                continue
            if atom.GetAtomicNum() not in GENERIC_ATOMS:
                continue
            candidates.append(atom.GetIdx())

        if not candidates:
            return None

        idx = random.choice(candidates)
        atom = rw_mol.GetAtomWithIdx(idx)
        current_num = atom.GetAtomicNum()

        choices = [z for z in GENERIC_ATOMS if z != current_num]
        if not choices:
            return None

        new_num = random.choice(choices)
        atom.SetAtomicNum(new_num)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(False)

        deg = atom.GetTotalDegree()
        max_val = MAX_VALENCE.get(new_num, 4)
        if deg > max_val:
            return None

        new_mol = self._sanitize(rw_mol.GetMol())
        if new_mol is None:
            return None

        fp_new = get_fp(new_mol)
        sim_local = DataStructs.TanimotoSimilarity(fp_old, fp_new)

        if sim_local < 0.6:
            return None
        return new_mol

    def op_micro_halogen_swap(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        fp_old = get_fp(mol)
        rw_mol = Chem.RWMol(mol)

        halogens = [a.GetIdx() for a in rw_mol.GetAtoms()
                    if a.GetAtomicNum() in (9, 17, 35)]
        if not halogens:
            return None

        idx = random.choice(halogens)
        atom = rw_mol.GetAtomWithIdx(idx)
        current_num = atom.GetAtomicNum()
        possible = [9, 17, 35]
        candidates = [z for z in possible if z != current_num]
        if not candidates:
            return None

        new_num = random.choice(candidates)
        atom.SetAtomicNum(new_num)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(False)

        new_mol = self._sanitize(rw_mol.GetMol())
        if new_mol is None:
            return None

        fp_new = get_fp(new_mol)
        sim_local = DataStructs.TanimotoSimilarity(fp_old, fp_new)
        if sim_local < 0.7:
            return None
        return new_mol

    def op_internal_bridge(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        rw_mol = Chem.RWMol(mol)
        atoms = list(rw_mol.GetAtoms())
        if len(atoms) < 5:
            return None
        candidates = []
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                if rw_mol.GetBondBetweenAtoms(i, j):
                    continue
                try:
                    path = Chem.GetShortestPath(rw_mol, i, j)
                    dist = len(path)
                    if 2 <= dist <= 5:
                        candidates.append((i, j))
                except Exception:
                    pass
        if not candidates:
            return None
        u, v = random.choice(candidates)
        try:
            if random.random() < 0.5:
                rw_mol.AddBond(u, v, rdchem.BondType.SINGLE)
            else:
                new_idx = rw_mol.AddAtom(Chem.Atom(6))
                rw_mol.AddBond(u, new_idx, rdchem.BondType.SINGLE)
                rw_mol.AddBond(new_idx, v, rdchem.BondType.SINGLE)
            return self._sanitize(rw_mol.GetMol())
        except Exception:
            return None

    def op_add_fragment(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        candidates = [a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() < 3]
        if not candidates:
            return None
        site_idx = random.choice(candidates)
        frag_smi = random.choice(ADDITION_FRAGMENTS)
        frag = Chem.MolFromSmiles(frag_smi)
        if not frag:
            return None
        try:
            combo = Chem.CombineMols(mol, frag)
            rw = Chem.RWMol(combo)
            dummies = [a for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
            if not dummies:
                return None
            dummy = dummies[0]
            dummy_idx = dummy.GetIdx()
            anchor = dummy.GetNeighbors()[0].GetIdx()
            rw.AddBond(site_idx, anchor, rdchem.BondType.SINGLE)
            rw.RemoveAtom(dummy_idx)
            return self._sanitize(rw.GetMol())
        except Exception:
            return None

    def op_apply_reaction(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        rxn = random.choice(COMPILED_REACTIONS)
        try:
            products = rxn.RunReactants((mol,))
            if not products:
                return None
            for p_tuple in products:
                for p in p_tuple:
                    sanitized = self._sanitize(p)
                    if sanitized:
                        return sanitized
            return None
        except Exception:
            return None

    def op_fuse_ring(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        bonds = [b for b in mol.GetBonds() if b.IsInRing()]
        if not bonds:
            return None
        target = random.choice(bonds)
        begin, end = target.GetBeginAtomIdx(), target.GetEndAtomIdx()
        ring_size = random.choice([5, 6])
        try:
            rw = Chem.RWMol(mol)
            add_chain_between(rw, begin, end, ring_size - 2)
            return self._sanitize(rw.GetMol())
        except Exception:
            return None

    def op_bridge_ring(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        ring_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
        if len(ring_atoms) < 4:
            return None
        random.shuffle(ring_atoms)
        a, b = ring_atoms[0], ring_atoms[1]
        dist = shortest_path_len(mol, a, b)
        if dist < 2 or dist > 4:
            return None
        try:
            rw = Chem.RWMol(mol)
            add_chain_between(rw, a, b, 1)
            return self._sanitize(rw.GetMol())
        except Exception:
            return None

    def op_spiro_ring(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        candidates = [a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() < 4]
        if not candidates:
            return None
        anchor_idx = random.choice(candidates)
        ring_size = random.choice([3, 4, 5])
        try:
            rw = Chem.RWMol(mol)
            prev = anchor_idx
            for _ in range(ring_size - 1):
                new_atom = rw.AddAtom(Chem.Atom(6))
                rw.AddBond(prev, new_atom, rdchem.BondType.SINGLE)
                prev = new_atom
            rw.AddBond(prev, anchor_idx, rdchem.BondType.SINGLE)
            return self._sanitize(rw.GetMol())
        except Exception:
            return None

    def op_open_ring(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        candidates = []
        for b in mol.GetBonds():
            if b.IsInRing() and b.GetBondType() == rdchem.BondType.SINGLE:
                candidates.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        if not candidates:
            return None
        b0, b1 = random.choice(candidates)
        try:
            rw = Chem.RWMol(mol)
            rw.RemoveBond(b0, b1)
            if len(Chem.GetMolFrags(rw, asMols=False)) > 1:
                return None
            return self._sanitize(rw.GetMol())
        except Exception:
            return None

    def op_delete_fragment(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        nonring_bonds = [b.GetIdx() for b in mol.GetBonds() if not b.IsInRing()]
        if not nonring_bonds:
            return None
        try:
            bid = random.choice(nonring_bonds)
            rw = Chem.RWMol(mol)
            b = rw.GetBondWithIdx(bid)
            a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            rw.RemoveBond(a1, a2)
            frags = Chem.GetMolFrags(rw.GetMol(), asMols=True)
            if len(frags) < 2:
                return None
            core = max(frags, key=lambda m: m.GetNumHeavyAtoms())
            return self._sanitize(core)
        except Exception:
            return None

    def op_delete_ring(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        ri = mol.GetRingInfo()
        atom_rings = list(ri.AtomRings())
        if not atom_rings:
            return None
        candidates = []
        for ring in atom_rings:
            if all(ri.NumAtomRings(a_idx) == 1 for a_idx in ring):
                candidates.append(ring)
        if not candidates:
            return None
        ring = random.choice(candidates)
        try:
            rw = Chem.RWMol(mol)
            for idx in sorted(ring, reverse=True):
                rw.RemoveAtom(idx)
            return self._sanitize(rw.GetMol())
        except Exception:
            return None

    def op_shrink_ring(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        ri = mol.GetRingInfo()
        atom_rings = list(ri.AtomRings())
        if not atom_rings:
            return None
        big_rings = [tuple(r) for r in atom_rings if len(r) >= 5]
        if not big_rings:
            return None
        ring = random.choice(big_rings)
        ring_set = set(ring)
        candidate_atoms = []
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            ring_neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in ring_set]
            if len(ring_neighbors) == 2:
                candidate_atoms.append((idx, ring_neighbors))
        if not candidate_atoms:
            return None
        try:
            idx, (n1, n2) = random.choice(candidate_atoms)
            rw = Chem.RWMol(mol)
            if rw.GetBondBetweenAtoms(n1, n2) is None:
                rw.AddBond(n1, n2, rdchem.BondType.SINGLE)
            rw.RemoveAtom(idx)
            return self._sanitize(rw.GetMol())
        except Exception:
            return None

    def op_peel_outer_ring(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        ri = mol.GetRingInfo()
        rings = list(ri.AtomRings())
        if not rings:
            return None
        candidates = []
        for ring in rings:
            exclusive_atoms = []
            for atom_idx in ring:
                if ri.NumAtomRings(atom_idx) == 1:
                    exclusive_atoms.append(atom_idx)
            if len(exclusive_atoms) > 0:
                candidates.append(exclusive_atoms)
        if not candidates:
            return None
        atoms_to_delete = random.choice(candidates)
        try:
            rw = Chem.RWMol(mol)
            for idx in sorted(atoms_to_delete, reverse=True):
                rw.RemoveAtom(idx)
            return self._sanitize(rw.GetMol())
        except Exception:
            return None

    # ----------------- dispatch -----------------

    def perturb_with_op(self, mol: Chem.Mol) -> Tuple[Optional[Chem.Mol], str]:
        r = random.random()
        cum = 0.0
        chosen_op = None
        for op, p in self.probs.items():
            cum += p
            if r <= cum:
                chosen_op = op
                break
        if chosen_op is None:
            chosen_op = "mutate_atom"

        op = chosen_op
        if op == "add_fragment":
            res = self.op_add_fragment(mol)
        elif op == "apply_reaction":
            res = self.op_apply_reaction(mol)
        elif op == "fuse_ring":
            res = self.op_fuse_ring(mol)
        elif op == "bridge_ring":
            res = self.op_bridge_ring(mol)
        elif op == "internal_bridge":
            res = self.op_internal_bridge(mol)
        elif op == "spiro_ring":
            res = self.op_spiro_ring(mol)
        elif op == "mutate_atom":
            res = self.op_mutate_atom_generic(mol)
        elif op == "open_ring":
            res = self.op_open_ring(mol)
        elif op == "delete_fragment":
            res = self.op_delete_fragment(mol)
        elif op == "delete_ring":
            res = self.op_delete_ring(mol)
        elif op == "shrink_ring":
            res = self.op_shrink_ring(mol)
        elif op == "peel_ring":
            res = self.op_peel_outer_ring(mol)
        elif op == "micro_mutate":
            res = self.op_micro_mutate_atom(mol)
        elif op == "micro_halogen":
            res = self.op_micro_halogen_swap(mol)
        else:
            res = self.op_mutate_atom_generic(mol)
        return res, op

    # ----------------- AUTO-TUNED GENERATOR -----------------

    def generate_library_auto(
        self,
        start_mol: Optional[Chem.Mol] = None,
        start_mols: Optional[List[Chem.Mol]] = None,
        N_target: int = 20000,
        mode: str = "balanced",
        verbose: bool = True,
    ) -> List[str]:
        """
        UPDATED:
          - You may pass a single seed via start_mol OR multiple seeds via start_mols.
          - Shells + locality are computed relative to the *currently active seed*.
        """
        if start_mols is None:
            if start_mol is None:
                raise ValueError("Provide start_mol or start_mols.")
            start_mols = [start_mol]

        # sanitize / validate seeds (also enforce required elements on seeds)
        seeds: List[Chem.Mol] = []
        for m in start_mols:
            mm = self._sanitize(m)
            if mm is not None:
                seeds.append(mm)
        if not seeds:
            raise ValueError("No valid seeds after sanitization / constraints.")

        mode_key = mode.lower()
        if mode_key not in MODE_PRESETS:
            if verbose:
                print(f"[WARN] Unknown mode '{mode}'. Falling back to 'balanced'.")
            mode_key = "balanced"
        mp = MODE_PRESETS[mode_key]

        seed_smiles = [Chem.MolToSmiles(m, isomericSmiles=True) for m in seeds]
        if verbose:
            print(f"[GENERIC AUTO] #seeds={len(seeds)} | mode={mode_key}, N_target={N_target}")
            for i, s in enumerate(seed_smiles):
                print(f"  seed[{i}]: {s}")

        seed_fps = [get_fp(m) for m in seeds]

        unique_smiles: Set[str] = set(seed_smiles)
        accepted_smiles: List[str] = []

        novelty_ctrl = KernelNoveltyController(lam=1e-3, max_points=100)

        bandit = OperatorBandit(self.base_probs, eta=mp["bandit_eta"], min_prob=1e-3)
        random_walk_steps = int(mp["random_walk_steps"])

        burn_in_fraction = 0.10
        burn_in = max(100, int(burn_in_fraction * N_target))
        reward_window = max(200, int(0.05 * N_target))

        auto_calibrate_novelty = True
        avg_reward_thresh: Optional[float] = None
        novelty_calibrated = False
        novelty_quantile = mp["novelty_quantile"]

        inner_q = mp["inner_q"]
        outer_q = mp["outer_q"]
        if not (0.0 < inner_q < outer_q < 1.0):
            raise ValueError("Require 0 < inner_q < outer_q < 1")

        target_inner_frac = inner_q
        target_outer_frac = 1.0 - outer_q
        target_mid_frac = max(0.0, outer_q - inner_q)

        base_reset_prob = mp["base_reset_prob"]
        max_reset_prob = mp["max_reset_prob"]
        w_shell = mp["w_shell"]

        w_div_min = mp["w_div_min"]
        w_div_max = mp["w_div_max"]
        w_local_min = mp["w_local_min"]
        w_local_max = mp["w_local_max"]
        exploration_anneal = 1.5
        exploitation_anneal = 1.5

        w_complexity = 0.2

        inner_thresh = 0.50
        outer_thresh = 0.80
        shell_calibrated = False

        max_failures = max(4 * N_target * random_walk_steps, 40000)

        n_accepted = 0
        failures = 0

        # start from a random seed
        active_seed_idx = random.randrange(len(seeds))
        curr_mol = seeds[active_seed_idx]
        curr_seed_mol = seeds[active_seed_idx]
        curr_seed_fp = seed_fps[active_seed_idx]

        recent_diversity_rewards: List[float] = []
        n_inner = 0
        n_outer = 0
        n_mid = 0

        init_diversity: List[float] = []
        init_dists: List[float] = []

        while n_accepted < N_target and failures < max_failures:
            phase = n_accepted / N_target if N_target > 0 else 0.0
            expl_factor = (1.0 - phase) ** exploration_anneal
            exploi_factor = phase ** exploitation_anneal
            w_div_t = w_div_min + (w_div_max - w_div_min) * expl_factor
            w_local_t = w_local_min + (w_local_max - w_local_min) * exploi_factor

            # INNER-FOCUSED EPISODES: reset to active seed more often if inner under target
            total_shell = max(1, n_inner + n_outer + n_mid)
            frac_inner = n_inner / total_shell
            inner_deficit = max(0.0, target_inner_frac - frac_inner)
            inner_focus_prob = min(0.7, 2.0 * inner_deficit)
            if random.random() < inner_focus_prob:
                curr_mol = curr_seed_mol

            temp_mol = curr_mol
            ops_used: List[Tuple[str, bool]] = []
            for _ in range(random_walk_steps):
                mod, op_used = self.perturb_with_op(temp_mol)
                success = mod is not None
                ops_used.append((op_used, success))
                if success:
                    temp_mol = mod

            candidate_mol = temp_mol
            if candidate_mol is None:
                failures += 1
                continue

            # Candidate must pass sanitization (already done by ops), but we keep this strict check:
            candidate_mol = self._sanitize(candidate_mol)
            if candidate_mol is None:
                failures += 1
                continue

            smi = Chem.MolToSmiles(candidate_mol, isomericSmiles=True)
            if smi in unique_smiles:
                failures += 1
                continue

            fp = get_fp(candidate_mol)
            r_div = novelty_ctrl.add_point_and_get_reward(fp)

            fp_curr = get_fp(curr_mol)
            sim_seed = DataStructs.TanimotoSimilarity(curr_seed_fp, fp)
            sim_curr = DataStructs.TanimotoSimilarity(fp_curr, fp)
            r_local = 0.5 * (sim_seed + sim_curr)

            try:
                bertz = GraphDescriptors.BertzCT(candidate_mol)
                r_complexity = min(1.0, max(0.0, bertz / 1000.0))
            except Exception:
                r_complexity = 0.0

            d_seed = 1.0 - sim_seed

            if n_accepted < burn_in:
                init_diversity.append(r_div)
                init_dists.append(d_seed)

            if (not shell_calibrated) and n_accepted >= burn_in and len(init_dists) >= 10:
                inner_thresh = float(np.quantile(init_dists, inner_q))
                outer_thresh = float(np.quantile(init_dists, outer_q))
                if outer_thresh <= inner_thresh:
                    outer_thresh = inner_thresh + 0.05
                shell_calibrated = True
                n_inner, n_outer, n_mid = 0, 0, 0
                if verbose:
                    print(
                        f"[AUTO] Shells calibrated: inner_thresh={inner_thresh:.3f}, "
                        f"outer_thresh={outer_thresh:.3f} "
                        f"(inner_q={inner_q:.2f}, outer_q={outer_q:.2f})"
                    )

            if (not novelty_calibrated) and auto_calibrate_novelty and n_accepted >= burn_in and len(init_diversity) >= 10:
                avg_reward_thresh = float(np.quantile(init_diversity, novelty_quantile))
                novelty_calibrated = True
                if verbose:
                    print(
                        f"[AUTO] Novelty threshold (quantile {novelty_quantile:.2f}): "
                        f"avg_reward_thresh={avg_reward_thresh:.3f}"
                    )

            if d_seed <= inner_thresh:
                ring_region = "inner"
            elif d_seed >= outer_thresh:
                ring_region = "outer"
            else:
                ring_region = "mid"

            total_shell_before = n_inner + n_outer + n_mid
            if total_shell_before > 0:
                f_in_before = n_inner / total_shell_before
                f_mid_before = n_mid / total_shell_before
                f_out_before = n_outer / total_shell_before
                err_before = ((f_in_before - target_inner_frac) ** 2 +
                              (f_mid_before - target_mid_frac) ** 2 +
                              (f_out_before - target_outer_frac) ** 2)

                n_in_after = n_inner + (1 if ring_region == "inner" else 0)
                n_mid_after = n_mid + (1 if ring_region == "mid" else 0)
                n_out_after = n_outer + (1 if ring_region == "outer" else 0)
                total_after = total_shell_before + 1
                f_in_after = n_in_after / total_after
                f_mid_after = n_mid_after / total_after
                f_out_after = n_out_after / total_after
                err_after = ((f_in_after - target_inner_frac) ** 2 +
                             (f_mid_after - target_mid_frac) ** 2 +
                             (f_out_after - target_outer_frac) ** 2)
                r_shell = err_before - err_after
            else:
                r_shell = 0.0

            total_reward = (
                w_div_t * r_div
                + w_local_t * r_local
                + w_complexity * r_complexity
                + w_shell * r_shell
            )
            total_reward = float(max(0.0, min(2.0, total_reward)))

            unique_smiles.add(smi)
            accepted_smiles.append(smi)
            n_accepted += 1

            if ring_region == "inner":
                n_inner += 1
            elif ring_region == "outer":
                n_outer += 1
            else:
                n_mid += 1

            successful_ops = [op_name for op_name, success in ops_used if success]
            if successful_ops:
                share = total_reward / len(successful_ops)
                for op_name, success in ops_used:
                    bandit.update(op_name, share if success else 0.0)
            else:
                for op_name, _ in ops_used:
                    bandit.update(op_name, 0.0)

            recent_diversity_rewards.append(r_div)
            if len(recent_diversity_rewards) > reward_window:
                recent_diversity_rewards.pop(0)

            total_shell = max(1, n_inner + n_outer + n_mid)
            frac_inner = n_inner / total_shell
            inner_deficit = max(0.0, target_inner_frac - frac_inner)
            reset_prob = base_reset_prob + 0.2 * inner_deficit
            reset_prob = min(max(reset_prob, 0.0), max_reset_prob)

            # occasionally switch to a different seed (important for multi-seed runs)
            if len(seeds) > 1 and random.random() < 0.05:
                active_seed_idx = random.randrange(len(seeds))
                curr_seed_mol = seeds[active_seed_idx]
                curr_seed_fp = seed_fps[active_seed_idx]

            if random.random() < reset_prob:
                curr_mol = curr_seed_mol
            else:
                curr_mol = candidate_mol

            prob_update_interval = max(50, int(0.05 * N_target))
            if (n_accepted % prob_update_interval) == 0:
                self.probs = bandit.get_probs()
                if verbose:
                    printable = {k: round(v, 2) for k, v in sorted(self.probs.items()) if v > 0.02}
                    print(
                        f"[GENERIC AUTO] n={n_accepted} "
                        f"inner={n_inner}, outer={n_outer}, mid={n_mid} "
                        f"w_div_t={w_div_t:.2f} w_local_t={w_local_t:.2f} "
                        f"seed_idx={active_seed_idx} "
                        f"Probs: {printable}"
                    )

            if avg_reward_thresh is not None and len(recent_diversity_rewards) == reward_window:
                avg_r = float(np.mean(recent_diversity_rewards))
                if avg_r < avg_reward_thresh:
                    if verbose:
                        print(f"[GENERIC AUTO] Saturation reached: Avg novelty {avg_r:.3f} < {avg_reward_thresh:.3f}")
                    break

        if verbose:
            print(
                f"[GENERIC AUTO] Done. Accepted {n_accepted} molecules "
                f"(inner={n_inner}, outer={n_outer}, mid={n_mid}, failures={failures})."
            )
        return accepted_smiles


# # ============================================================
# # 7. MAIN
# # ============================================================

# if __name__ == "__main__":
#     SEED = 42
#     random.seed(SEED)
#     np.random.seed(SEED)

#     # User-facing knobs:
#     N_TARGET = 1000
#     MODE = "balanced"         # "local", "balanced", "global"
#     CHEM_PROFILE = "default"  # "strict", "default", "relaxed"

#     # MULTI-SEED: put as many SMILES as you want here
#     base_smiles_list = [
#         "[Fe]1n2c3nc4nc(nc5n1c(nc6nc(nc2c7ccccc37)c8ccccc68)c9ccccc59)c%10ccccc4%10",
#         # "another_smiles_here_with_Fe",
#     ]

#     mols = []
#     for smi in base_smiles_list:
#         m = Chem.MolFromSmiles(smi)
#         if m is None:
#             print(f"[WARN] Could not parse seed: {smi}")
#             continue
#         mols.append(m)

#     if not mols:
#         print("Error: Could not parse any seed molecules.")
#         sys.exit(1)

#     # Derive constraints from the first seed (simple + consistent)
#     config = derive_config_from_seed(mols[0], chem_profile=CHEM_PROFILE)

#     # FE FILTER: require Fe in every accepted molecule
#     config.require_element_atomic_nums = [26]

#     engine = HybridPerturbator(config=config)

#     lib = engine.generate_library_auto(
#         start_mols=mols,
#         N_target=N_TARGET,
#         mode=MODE,
#         verbose=True,
#     )

#     with open("generic_chemical_space_Fe_only.smi", "w") as f:
#         for s in base_smiles_list:
#             f.write(f"{s}\tSeed\n")
#         for s in lib:
#             f.write(f"{s}\n")

#     print("Saved 'generic_chemical_space_Fe_only.smi'.")
