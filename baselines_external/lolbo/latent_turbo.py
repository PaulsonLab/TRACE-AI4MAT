"""LOL-BO's trust region, device-agnostic.

`TurboState` and `update_state` are imported unmodified from the vendored
`lolbo/utils/bo_utils/turbo.py`, so the trust-region *logic* here is upstream's:

    length 0.8, length_min 0.5**7, length_max 1.6
    success_tolerance 10   -> expand (length *= 2) after 10 consecutive improvements
    failure_tolerance 32   -> shrink (length /= 2) after 32 consecutive failures
    restart when length < length_min
    "improvement" means max(Y_next) > best_value + 1e-3 * |best_value|

`generate_batch` is re-implemented because upstream hard-codes `.cuda()` on eight
lines, which makes CPU smoke tests impossible. Everything else is line-for-line
the same: candidate count min(5000, max(2000, 200*d)), a scrambled Sobol draw
inside the box `x_center +/- 8 * length / 2`, a perturbation mask that touches
each coordinate with probability min(20/d, 1) (forcing at least one coordinate
per candidate), and batch selection by Thompson sampling with
`MaxPosteriorSampling(replacement=False)`. Only the `acqf="ts"` path is kept --
LOL-BO's molecule runs use Thompson sampling, and its "ei" branch falls back to
"ts" on any exception anyway.

Note on tolerances at a 1,000-call budget: upstream hard-codes
failure_tolerance=32 (its dim-scaled formula is commented out in turbo.py). With
20 iterations of 49, 32 consecutive failures can never occur, so the literal
trust region cannot shrink or restart inside this budget. That is why
run_lolbo_all_oracles.py also runs a `budget_scaled` arm using upstream's own
commented-out formula, ceil(max(4/bsz, dim/bsz)).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import torch
from botorch.generation import MaxPosteriorSampling
from torch.quasirandom import SobolEngine

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
UPSTREAM_TURBO = VENDOR_DIR / "lolbo" / "utils" / "bo_utils" / "turbo.py"


def _load_upstream_symbols(source_path: Path, names):
    """Execute exactly the named top-level definitions from an upstream file.

    `import lolbo.utils.bo_utils.turbo` cannot be used directly: that module does
    `from .approximate_gp import *`, and approximate_gp.py imports
    `gpytorch.lazy.LazyTensor`, an API removed in gpytorch >= 1.11 (this repo runs
    1.15). Extracting the two definitions we need as AST nodes keeps them
    byte-identical to vendor/lolbo/utils/bo_utils/turbo.py -- they cannot drift --
    without importing the parts of that module we never call.
    """
    import ast
    from dataclasses import dataclass

    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    wanted = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in set(names)
    }
    missing = set(names) - set(wanted)
    if missing:
        raise ImportError(f"{source_path} no longer defines: {sorted(missing)}")

    module = ast.Module(body=[wanted[name] for name in names], type_ignores=[])
    namespace = {"math": math, "torch": torch, "dataclass": dataclass}
    exec(compile(module, filename=str(source_path), mode="exec"), namespace)  # noqa: S102
    return [namespace[name] for name in names]


TurboState, update_state = _load_upstream_symbols(UPSTREAM_TURBO, ["TurboState", "update_state"])

__all__ = ["TurboState", "update_state", "generate_batch", "generate_batch_global",
           "budget_scaled_failure_tolerance"]

# upstream turbo.py, commented out in TurboState.__post_init__
def budget_scaled_failure_tolerance(dim: int, batch_size: int) -> int:
    return math.ceil(max([4.0 / batch_size, float(dim) / batch_size]))


def generate_batch(
    state: TurboState,
    model,
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int,
    device: torch.device,
    n_candidates: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
):
    """Thompson-sample `batch_size` latent points inside the trust region."""
    assert torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    X = X.to(device=device, dtype=dtype)
    Y = Y.to(device=device, dtype=dtype)

    x_center = X[Y.argmax(), :].clone()
    weights = torch.ones_like(x_center) * 8  # less than 4 stdevs on either side max
    tr_lb = x_center - weights * state.length / 2.0
    tr_ub = x_center + weights * state.length / 2.0

    dim = X.shape[-1]
    sobol = SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(device=device, dtype=dtype)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
    X_next = thompson_sampling(X_cand, num_samples=batch_size)

    return X_next, x_center


def generate_batch_global(
    model,
    dim: int,
    batch_size: int,
    device: torch.device,
    n_candidates: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
):
    """No-trust-region ablation: Thompson-sample from the whole latent prior.

    Identical to `generate_batch` in every respect except where the candidates
    come from. Instead of a box around the incumbent, the 5,000 candidates are
    drawn from N(0, I) -- the InfoTransformerVAE's own prior, i.e. the region the
    decoder was trained to decode. Surrogate, acquisition (Thompson sampling with
    `MaxPosteriorSampling(replacement=False)`) and batch size are unchanged, so a
    difference in outcome is attributable to candidate locality alone.
    """
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * dim))
    X_cand = torch.randn(n_candidates, dim, device=device, dtype=dtype)
    thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
    return thompson_sampling(X_cand, num_samples=batch_size)
