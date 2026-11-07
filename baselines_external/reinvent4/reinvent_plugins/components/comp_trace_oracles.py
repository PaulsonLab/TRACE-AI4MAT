"""
REINVENT4 scoring component for the six TRACE benchmark oracles.

Keep this file inside reinvent4_oracle_suite/reinvent_plugins/components and set:
    TRACE_REINVENT_SUITE_ROOT=/path/to/reinvent4_oracle_suite
    PYTHONPATH=/path/to/reinvent4_oracle_suite:$PYTHONPATH

The run_reinvent_suite.py wrapper sets those variables automatically.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from pydantic.dataclasses import dataclass
from rdkit import Chem

_ENV_ROOT = os.environ.get("TRACE_REINVENT_SUITE_ROOT")
if _ENV_ROOT:
    _ROOT = Path(_ENV_ROOT).expanduser().resolve()
else:
    _ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reinvent_plugins.components.add_tag import add_tag
from reinvent_plugins.components.component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from trace_reinvent.oracles import MoleculeConstraints, constraint_rejection_reason, get_oracle_spec


def first(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


@add_tag("__parameters")
@dataclass
class Parameters:
    oracle_name: Any
    reward_low: Optional[Any] = None
    reward_high: Optional[Any] = None
    max_sa_score: Optional[Any] = None
    max_smiles_length: Optional[Any] = None
    max_heavy_atoms: Optional[Any] = None
    drd2_model_path: Optional[Any] = None


@add_tag("__component")
class TraceOracleReward:
    def __init__(self, params: Parameters):
        self.oracle_name = str(first(params.oracle_name)).strip()
        self.reward_low = first(params.reward_low, None)
        self.reward_high = first(params.reward_high, None)
        if self.reward_low is not None:
            self.reward_low = float(self.reward_low)
        if self.reward_high is not None:
            self.reward_high = float(self.reward_high)
        if self.reward_low is not None and self.reward_high is not None and self.reward_high <= self.reward_low:
            raise ValueError("reward_high must be greater than reward_low")

        self.constraints = MoleculeConstraints(
            max_sa_score=float(first(params.max_sa_score, 6.5)),
            max_smiles_length=None
            if first(params.max_smiles_length, None) in (None, "")
            else int(first(params.max_smiles_length)),
            max_heavy_atoms=None
            if first(params.max_heavy_atoms, None) in (None, "")
            else int(first(params.max_heavy_atoms)),
        )
        raw_model_path = first(params.drd2_model_path, None)
        self.spec = get_oracle_spec(self.oracle_name, Path(raw_model_path) if raw_model_path else None)

    def _reward_from_raw(self, raw_score: float) -> float:
        if self.reward_low is None or self.reward_high is None:
            return float(np.clip(raw_score, 0.0, 1.0))
        return float(np.clip((raw_score - self.reward_low) / (self.reward_high - self.reward_low), 0.0, 1.0))

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> ComponentResults:
        rewards = []
        raw_scores = []
        accepted = []
        reasons = []
        canonical_smiles = []

        for mol in mols:
            canonical = None
            if mol is not None:
                try:
                    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
                except Exception:
                    canonical = None
            canonical_smiles.append(canonical)

            reason = constraint_rejection_reason(mol, self.constraints)
            if reason is not None or canonical is None:
                rewards.append(0.0)
                raw_scores.append(np.nan)
                accepted.append(False)
                reasons.append(reason or "invalid_smiles")
                continue

            try:
                if self.spec.score_smiles is not None:
                    raw = float(self.spec.score_smiles(canonical))
                elif self.spec.score_mol is not None:
                    raw = float(self.spec.score_mol(mol))
                else:
                    raw = float("nan")
            except Exception:
                raw = float("nan")

            if not np.isfinite(raw):
                rewards.append(0.0)
                raw_scores.append(np.nan)
                accepted.append(False)
                reasons.append("scoring_failed")
                continue

            rewards.append(self._reward_from_raw(raw))
            raw_scores.append(raw)
            accepted.append(True)
            reasons.append("accepted")

        return ComponentResults(
            [np.asarray(rewards, dtype=float)],
            metadata={
                "trace_oracle": [self.oracle_name] * len(rewards),
                "raw_trace_score": raw_scores,
                "trace_oracle_accepted": accepted,
                "trace_oracle_reason": reasons,
                "canonical_smiles": canonical_smiles,
            },
        )
