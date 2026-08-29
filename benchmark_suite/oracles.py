import gzip
import math
import os
import pickle
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, QED, rdMolDescriptors

try:
    from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
except Exception:
    Generate = None
    Gobbi_Pharm2D = None

from benchmark_suite.common import penalized_logp_mol


ScoreMol = Callable[[Chem.Mol], float]
ScoreSmiles = Callable[[str], float]


@dataclass
class OracleSpec:
    name: str
    display_name: str
    reward_label: str
    score_mol: Optional[ScoreMol] = None
    score_smiles: Optional[ScoreSmiles] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def score_smiles_value(self, smiles: str) -> float:
        canonical_mol = Chem.MolFromSmiles(smiles)
        if canonical_mol is None:
            return float("nan")
        if self.score_smiles is not None:
            return float(self.score_smiles(Chem.MolToSmiles(canonical_mol, isomericSmiles=True)))
        if self.score_mol is None:
            return float("nan")
        return float(self.score_mol(canonical_mol))


def gaussian(x: float, mu: float, sigma: float) -> float:
    return float(np.exp(-0.5 * ((float(x) - float(mu)) / float(sigma)) ** 2))


def min_gaussian(x: float, mu: float, sigma: float) -> float:
    return gaussian(max(float(x), float(mu)), mu, sigma)


def max_gaussian(x: float, mu: float, sigma: float) -> float:
    return gaussian(min(float(x), float(mu)), mu, sigma)


def clipped_score(x: float, upper_x: float, lower_x: float = 0.0, high_score: float = 1.0, low_score: float = 0.0) -> float:
    slope = (float(high_score) - float(low_score)) / (float(upper_x) - float(lower_x))
    intercept = float(high_score) - slope * float(upper_x)
    return float(np.clip(slope * float(x) + intercept, low_score, high_score))


def arithmetic_mean(values) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def geometric_mean(values) -> float:
    vals = [max(float(v), 0.0) for v in values]
    if not vals or any(not math.isfinite(v) for v in vals):
        return float("nan")
    if any(v <= 0.0 for v in vals):
        return 0.0
    return float(np.exp(np.mean(np.log(vals))))


def atom_count(mol: Chem.Mol, element: str) -> int:
    if element == "H":
        mol = Chem.AddHs(mol)
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == element)


# ---------------------------------------------------------------------------
# ECFP/FCFP variant used by the Tanimoto-based GuacaMol oracles.
#
# Two definitions are in circulation and they are NOT interchangeable:
#
#   "bitvector"  hashed 2048-bit Morgan fingerprint, binary Tanimoto.
#   "count"      unhashed count Morgan fingerprint, count-based Tanimoto.
#                This is what guacamol.utils.fingerprints does, so it is the
#                literal PMO/GuacaMol task definition.
#
# They reorder molecules relative to each other (~Spearman 0.78), so swapping
# one for the other changes the objective, not just its scale.
#
# The default here is "bitvector" because that is the definition under which
# every result in results/ was generated -- for median_2 and osimertinib_mpo,
# the two oracles this switch actually affects, and for every baseline they
# were compared against. Keeping it as the default is what makes the bundled
# curves reproducible from this code.
#
# Set TRACE_GUACAMOL_FP=count (or GUACAMOL_FINGERPRINT_MODE=count) to run the
# GuacaMol-faithful definition. Doing so changes the median_2 and
# osimertinib_mpo objectives, so results produced that way are not comparable
# with the bundled ones and must be regenerated for every method.
#
# Unaffected by this switch: penalized_logp, qed, scaffold_hop and deco_hop
# (pharmacophore / SMARTS, no ECFP), drd2 and jnk3 (each carries its own
# featurisation inside its model wrapper).
# ---------------------------------------------------------------------------

FINGERPRINT_MODES = ("bitvector", "count")
_DEFAULT_FINGERPRINT_MODE = "bitvector"
_FINGERPRINT_NBITS = 2048


def _resolve_fingerprint_mode() -> str:
    raw = os.environ.get("TRACE_GUACAMOL_FP") or os.environ.get("GUACAMOL_FINGERPRINT_MODE")
    if raw is None:
        return _DEFAULT_FINGERPRINT_MODE
    mode = str(raw).strip().lower()
    if mode not in FINGERPRINT_MODES:
        raise ValueError(
            f"Unknown fingerprint mode {raw!r}. Use one of: {', '.join(FINGERPRINT_MODES)}"
        )
    return mode


FINGERPRINT_MODE = _resolve_fingerprint_mode()

_MORGAN_ARGS = {
    "ECFP4": (2, False),
    "ECFP6": (3, False),
    "FCFP4": (2, True),
    "FCFP6": (3, True),
}


def fingerprint(mol: Chem.Mol, fp_type: str):
    fp_key = fp_type.upper()
    if fp_key in _MORGAN_ARGS:
        radius, use_features = _MORGAN_ARGS[fp_key]
        if FINGERPRINT_MODE == "count":
            return AllChem.GetMorganFingerprint(mol, radius, useFeatures=use_features)
        return AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=_FINGERPRINT_NBITS, useFeatures=use_features
        )
    if fp_key == "AP":
        return AllChem.GetAtomPairFingerprint(mol, maxLength=10)
    if fp_key == "PHCO" and Generate is not None and Gobbi_Pharm2D is not None:
        return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
    return Chem.RDKFingerprint(mol)


_TARGET_FP_CACHE: Dict[tuple, Any] = {}


def target_fingerprint(target_smiles: str, fp_type: str):
    key = (target_smiles, fp_type.upper(), FINGERPRINT_MODE)
    if key not in _TARGET_FP_CACHE:
        target_mol = Chem.MolFromSmiles(target_smiles)
        if target_mol is None:
            return None
        _TARGET_FP_CACHE[key] = fingerprint(target_mol, fp_type)
    return _TARGET_FP_CACHE[key]


def tanimoto_to_target(mol: Chem.Mol, target_smiles: str, fp_type: str) -> float:
    target_fp = target_fingerprint(target_smiles, fp_type)
    if target_fp is None:
        return float("nan")
    fp = fingerprint(mol, fp_type)
    return float(DataStructs.TanimotoSimilarity(fp, target_fp))


def smarts_score(mol: Chem.Mol, smarts: str, inverse: bool = False) -> float:
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return float("nan")
    has_match = bool(mol.GetSubstructMatches(pattern))
    if has_match:
        return 0.0 if inverse else 1.0
    return 1.0 if inverse else 0.0


def qed_score(mol: Chem.Mol) -> float:
    return float(QED.qed(mol))


OSIMERTINIB = "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34"


def osimertinib_mpo_score(mol: Chem.Mol) -> float:
    similar = clipped_score(tanimoto_to_target(mol, OSIMERTINIB, "FCFP4"), upper_x=0.8)
    not_too_similar = min_gaussian(tanimoto_to_target(mol, OSIMERTINIB, "ECFP6"), mu=0.85, sigma=0.1)
    tpsa_over_100 = max_gaussian(Descriptors.TPSA(mol), mu=100.0, sigma=10.0)
    logp_under_1 = min_gaussian(Crippen.MolLogP(mol), mu=1.0, sigma=1.0)
    return geometric_mean([similar, not_too_similar, tpsa_over_100, logp_under_1])


SCAFFOLD_HOP_TARGET = "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C"
SCAFFOLD_HOP_DECO = "[#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1"
SCAFFOLD_HOP_SCAFFOLD = "[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12"


def scaffold_hop_score(mol: Chem.Mol) -> float:
    pharmacophore_similarity = clipped_score(tanimoto_to_target(mol, SCAFFOLD_HOP_TARGET, "PHCO"), upper_x=0.75)
    keep_decoration = smarts_score(mol, SCAFFOLD_HOP_DECO, inverse=False)
    remove_scaffold = smarts_score(mol, SCAFFOLD_HOP_SCAFFOLD, inverse=True)
    return arithmetic_mean([pharmacophore_similarity, keep_decoration, remove_scaffold])


# guacamol.standard_benchmarks.decoration_hop(): keep the scaffold, replace the
# decorations. Same target molecule as scaffold hop, and the same two SMARTS, but
# their senses are swapped and the pharmacophore clip is 0.85 rather than 0.75.
DECO_HOP_TARGET = SCAFFOLD_HOP_TARGET
DECO_HOP_DECO_1 = "CS([#6])(=O)=O"
DECO_HOP_DECO_2 = "[#7]-c1ccc2ncsc2c1"
DECO_HOP_SCAFFOLD = SCAFFOLD_HOP_SCAFFOLD


def deco_hop_score(mol: Chem.Mol) -> float:
    pharmacophore_similarity = clipped_score(tanimoto_to_target(mol, DECO_HOP_TARGET, "PHCO"), upper_x=0.85)
    remove_decoration_1 = smarts_score(mol, DECO_HOP_DECO_1, inverse=True)
    remove_decoration_2 = smarts_score(mol, DECO_HOP_DECO_2, inverse=True)
    keep_scaffold = smarts_score(mol, DECO_HOP_SCAFFOLD, inverse=False)
    return arithmetic_mean(
        [pharmacophore_similarity, remove_decoration_1, remove_decoration_2, keep_scaffold]
    )


TADALAFIL = "O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C"
SILDENAFIL = "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"


def median_2_score(mol: Chem.Mol) -> float:
    return geometric_mean(
        [
            tanimoto_to_target(mol, TADALAFIL, "ECFP6"),
            tanimoto_to_target(mol, SILDENAFIL, "ECFP6"),
        ]
    )


class TDCOracle:
    def __init__(self, tdc_name: str):
        # PyTDC still imports rdkit.six on some oracle paths; newer RDKit
        # releases removed that compatibility module.
        if "rdkit.six" not in sys.modules:
            rdkit_six = types.ModuleType("rdkit.six")
            rdkit_six.iteritems = lambda mapping: iter(mapping.items())
            sys.modules["rdkit.six"] = rdkit_six
        try:
            from tdc import Oracle
        except ImportError as exc:
            raise ImportError(
                f"Oracle '{tdc_name}' requires PyTDC. Install PyTDC in the TRACE environment, "
                "then rerun the benchmark."
            ) from exc
        self.tdc_name = tdc_name
        self.oracle = Oracle(name=tdc_name)
        try:
            self.oracle.evaluator_func("CCO")
        except Exception as exc:
            raise RuntimeError(
                f"TDC oracle '{tdc_name}' loaded, but its raw evaluator failed. "
                "This usually means the bundled sklearn pickle is incompatible with "
                "the current scikit-learn/Python environment. Use a compatible TDC "
                "environment for this optional oracle, or use the stable local oracle set."
            ) from exc

    def __call__(self, smiles: str) -> float:
        value = self.oracle(smiles)
        if isinstance(value, (list, tuple)):
            value = value[0]
        return float(value)


class LocalRandomForestActivityOracle:
    """GSK3B / JNK3 from TDC's own model file, via the portable numpy export.

    TDC featurises these two with a *binary* ECFP4 bit vector
    (`GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)`), which is a different
    featurisation from the counts-based ECFP6 that DRD2 uses -- substituting one
    for the other silently defines a different oracle, so it is spelled out here.
    See benchmark_suite/rf_oracle.py for why the model is an .npz.
    """

    def __init__(self, target: str):
        from benchmark_suite.rf_oracle import NumpyForest

        forest_path = Path(__file__).resolve().parents[1] / "oracle" / f"{target}_forest.npz"
        if not forest_path.exists():
            raise FileNotFoundError(
                f"Missing exported forest: {forest_path}. Build it with:\n"
                f"  <legacy-python> -m benchmark_suite.rf_oracle export "
                f"oracle/{target}_current.pkl {forest_path}"
            )
        self.target = target
        self.forest = NumpyForest(forest_path)

    def __call__(self, smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return float("nan")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, features)
        return float(self.forest.predict_positive(features.reshape(1, -1))[0])


class LocalDRD2Oracle:
    """Compatibility wrapper for the bundled PMO/TDC DRD2 sklearn model."""

    # The model ships gzipped: the raw pickle is 34 MB and mostly compressible
    # padding, which shrinks to under 0.3 MB and keeps every file in this
    # repository below GitHub's 25 MB web-upload limit. The plain .pkl is still
    # accepted, so an existing uncompressed copy keeps working unchanged.
    MODEL_FILENAMES = ("drd2_current.pkl.gz", "drd2_current.pkl")

    def __init__(self):
        oracle_dir = Path(__file__).resolve().parents[1] / "oracle"
        model_path = next(
            (oracle_dir / name for name in self.MODEL_FILENAMES if (oracle_dir / name).exists()),
            None,
        )
        if model_path is None:
            raise FileNotFoundError(
                "Missing local DRD2 model. Expected one of "
                + ", ".join(str(oracle_dir / n) for n in self.MODEL_FILENAMES)
            )
        opener = gzip.open if model_path.suffix == ".gz" else open
        with opener(model_path, "rb") as f:
            self.model = pickle.load(f)

    @staticmethod
    def _fingerprints_from_mol(mol: Chem.Mol):
        fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
        arr = np.zeros((1, 2048), np.int32)
        for idx, value in fp.GetNonzeroElements().items():
            arr[0, int(idx) % 2048] += int(value)
        return arr

    def __call__(self, smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        probs = np.asarray(self.model.predict_proba(self._fingerprints_from_mol(mol)))
        if probs.ndim >= 2 and probs.shape[-1] > 1:
            return float(probs.reshape(-1, probs.shape[-1])[0, 1])
        return float(probs.reshape(-1)[-1])


def get_oracle_spec(name: str) -> OracleSpec:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"plogp", "penalized_logp", "penalized_logp_rdkit"}:
        return OracleSpec(
            name="penalized_logp",
            display_name="Penalized logP",
            reward_label="penalized logP",
            score_mol=penalized_logp_mol,
            metadata={
                "family": "rdkit_property",
                "definition": "RDKit MolLogP - SA score - max(largest_ring_size - 6, 0)",
            },
        )
    if key == "qed":
        return OracleSpec(
            name="qed",
            display_name="QED",
            reward_label="QED",
            score_mol=qed_score,
            metadata={
                "family": "rdkit_property",
                "range": [0.0, 1.0],
                "source": "RDKit/TDC/GuacaMol",
            },
        )
    if key in {"osimertinib", "osimertinib_mpo"}:
        return OracleSpec(
            name="osimertinib_mpo",
            display_name="Osimertinib MPO",
            reward_label="Osimertinib MPO",
            score_mol=osimertinib_mpo_score,
            metadata={
                "family": "guacamol_mpo",
                "range": [0.0, 1.0],
                "target_smiles": OSIMERTINIB,
                "source": "GuacaMol standard_benchmarks hard_osimertinib",
            },
        )
    if key in {"scaffold_hop", "scaffold"}:
        return OracleSpec(
            name="scaffold_hop",
            display_name="Scaffold Hop",
            reward_label="Scaffold Hop score",
            score_mol=scaffold_hop_score,
            metadata={
                "family": "guacamol_hop",
                "range": [0.0, 1.0],
                "target_smiles": SCAFFOLD_HOP_TARGET,
                "source": "GuacaMol standard_benchmarks scaffold_hop",
            },
        )
    if key in {"median_2", "median2", "median_tadalafil_sildenafil"}:
        return OracleSpec(
            name="median_2",
            display_name="Median molecules 2",
            reward_label="Median-2 score",
            score_mol=median_2_score,
            metadata={
                "family": "guacamol_similarity",
                "range": [0.0, 1.0],
                "target_smiles": [TADALAFIL, SILDENAFIL],
                "source": "GuacaMol standard_benchmarks median_tadalafil_sildenafil",
            },
        )
    if key in {"deco_hop", "deco", "decoration_hop"}:
        return OracleSpec(
            name="deco_hop",
            display_name="Deco Hop",
            reward_label="Deco Hop score",
            score_mol=deco_hop_score,
            metadata={
                "family": "guacamol_structural",
                "range": [0.0, 1.0],
                "source": "GuacaMol/TDC/PMO decoration_hop",
                "definition": "mean(clipped PHCO Tanimoto @0.85, no CS(=O)(=O)C, no [#7]-benzothiazole, scaffold present)",
            },
        )

    if key in {"jnk3", "jnk3_activity"}:
        jnk3 = LocalRandomForestActivityOracle("jnk3")
        return OracleSpec(
            name="jnk3",
            display_name="JNK3",
            reward_label="JNK3 activity",
            score_smiles=jnk3,
            metadata={
                "family": "activity_classifier",
                "range": [0.0, 1.0],
                "source": "TDC/PMO JNK3 random forest (jnk3_current.pkl), numpy export",
                "featurisation": "binary ECFP4, GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)",
            },
        )

    if key in {"drd2", "drd2_activity"}:
        drd2 = LocalDRD2Oracle()
        return OracleSpec(
            name="drd2",
            display_name="DRD2",
            reward_label="DRD2 activity probability",
            score_smiles=drd2,
            metadata={
                "family": "tdc_activity_classifier",
                "range": [0.0, 1.0],
                "tdc_name": "DRD2",
                "source": "TDC/PMO DRD2 oracle, local drd2_current.pkl.gz compatibility wrapper",
            },
        )
    raise ValueError(
        "Unknown oracle '{name}'. Available local oracles: qed, osimertinib_mpo, "
        "scaffold_hop, deco_hop, median_2, drd2, jnk3, penalized_logp.".format(name=name)
    )


def recommended_oracles() -> Dict[str, str]:
    return {
        "penalized_logp": "legacy RDKit pLogP control; useful but hackable",
        "qed": "bounded RDKit drug-likeness sanity check",
        "median_2": "GuacaMol dual-target similarity task",
        "drd2": "PMO/TDC dopamine receptor D2 activity classifier",
        "osimertinib_mpo": "GuacaMol multi-property lead-like objective",
        "scaffold_hop": "GuacaMol structural hop/motif-control objective",
    }
