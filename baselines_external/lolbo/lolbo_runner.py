"""LOL-BO with a frozen SELFIES VAE, run under the TRACE fairness contract.

One optimisation run = one (oracle, seed) pair, producing exactly the artifacts
every other TRACE method produces (`finish_run` from plogp_benchmark_suite.common),
plus a latent-space diagnostics CSV.

Algorithm (all of it upstream LOL-BO except where README.md says otherwise):

    encode the initial molecules            -> train_z (256-d posterior samples)
    fit GPModelDKL on (train_z, train_y)    -> init_n_epochs = 20
    TurboState(dim=256, batch_size=bsz)
    repeat until the oracle budget is spent:
        Thompson-sample bsz latents in the trust region  (latent_turbo.generate_batch)
        decode them, score the new ones                  (frozen VAE + shared oracle)
        update_state(...) on the realised batch          (vendored, unmodified)
        train the surrogate 2 epochs on the new batch
        restart the trust region if length < length_min

The VAE is frozen, so LOL-BO's end-to-end path (`update_models_end_to_end` +
`recenter`, both vendored under vendor/lolbo/) is never invoked. That makes this
arm LS-BO/TuRBO in LOL-BO's pretrained latent space, using LOL-BO's own trust
region, surrogate and acquisition. README.md states this in the results-claim
language the paper should use.

Batch filling: a decoded batch loses molecules to invalid SMILES, duplicates and
the shared chemistry constraints. Like the GP-BO baseline
(plogp_benchmark_suite/gpbo_graph_ga_baseline.py), the run re-samples the
shortfall (up to --max-fill-rounds times) so each iteration really spends
batch_size oracle calls, keeping the per-iteration x-axis aligned with every
other method. Every rejected decode is counted in the diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from rdkit import Chem, DataStructs

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
for _path in (str(REPO_ROOT), str(PACKAGE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from plogp_benchmark_suite.common import (  # noqa: E402  (shared with every TRACE method)
    CachedMoleculeOracle,
    MoleculeConstraints,
    canonicalize_smiles,
    finish_run,
    get_fp,
    molecule_satisfies_constraints,
    sanitize_smiles_for_constraints,
    set_all_seeds,
    write_json,
    write_rows_csv,
)

from frozen_vae import DEFAULT_STATE_DICT, LATENT_DIM, FrozenSelfiesVAE  # noqa: E402
from latent_turbo import (  # noqa: E402
    TurboState,
    budget_scaled_failure_tolerance,
    generate_batch,
    generate_batch_global,
    update_state,
)
from surrogate import initialize_surrogate, posterior_mean_std, update_surrogate  # noqa: E402

LITERAL_FAILURE_TOLERANCE = 32  # hard-coded in vendor/lolbo/utils/bo_utils/turbo.py
TR_TOLERANCE_MODES = ("literal", "budget_scaled", "global")
# "global" is the no-trust-region ablation: same surrogate and acquisition,
# candidates drawn from the whole latent prior instead of a box.


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device asked for CUDA but torch.cuda.is_available() is False.")
    return torch.device(requested)


def tanimoto(smiles_a: str, smiles_b: str) -> Optional[float]:
    mol_a, mol_b = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return None
    return float(DataStructs.TanimotoSimilarity(get_fp(mol_a), get_fp(mol_b)))


def make_tr_state(dim: int, batch_size: int, mode: str, best_value: float) -> TurboState:
    if mode not in TR_TOLERANCE_MODES:
        raise ValueError(f"Unknown trust-region tolerance mode: {mode}")
    failure_tolerance = (
        LITERAL_FAILURE_TOLERANCE
        if mode == "literal"
        else budget_scaled_failure_tolerance(dim, batch_size)
    )
    return TurboState(
        dim=dim,
        batch_size=batch_size,
        failure_tolerance=failure_tolerance,
        best_value=best_value,
    )


def run_lolbo_seed(
    initial_smiles: Sequence[str],
    seed: int,
    out_dir: Path,
    n_iters: int,
    batch_size: int,
    constraints: MoleculeConstraints,
    oracle_spec,
    device: torch.device,
    method: str = "lolbo_frozen",
    tr_tolerances: str = "literal",
    vae_state_dict: Path = DEFAULT_STATE_DICT,
    max_string_length: int = 1024,
    strip_stereo: bool = True,
    learning_rate: float = 0.01,
    init_n_epochs: int = 20,
    num_update_epochs: int = 2,
    n_candidates: Optional[int] = None,
    max_fill_rounds: int = 12,
    max_iteration_multiplier: float = 2.0,
    oracle_source: str = "plogp_benchmark_suite.oracles",
    verbose: bool = True,
) -> Dict[str, Any]:
    started = time.time()
    set_all_seeds(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_population = sanitize_smiles_for_constraints(initial_smiles, constraints)
    if not init_population:
        raise ValueError(f"No valid initial population for seed {seed}.")

    # identical budget formula to every other method in the suite
    max_oracle_calls = len(init_population) + int(n_iters) * int(batch_size)
    oracle = CachedMoleculeOracle(
        max_oracle_calls=max_oracle_calls,
        constraints=constraints,
        score_mol=oracle_spec.score_mol,
        score_smiles=oracle_spec.score_smiles,
        oracle_name=oracle_spec.name,
        reward_label=oracle_spec.reward_label,
        metadata=oracle_spec.metadata,
    )
    oracle.score_many(init_population, phase="init", step=0)

    vae = FrozenSelfiesVAE(
        state_dict_path=vae_state_dict,
        device=str(device),
        max_string_length=max_string_length,
        strip_stereo=strip_stereo,
    )

    # ---- initial latent data ------------------------------------------------
    init_smiles_scored = list(oracle.cache.keys())
    encoded = vae.encode(init_smiles_scored)
    if encoded.z.shape[0] < 2:
        raise RuntimeError(
            f"Only {encoded.z.shape[0]} of {len(init_smiles_scored)} initial molecules could be "
            "encoded by the frozen VAE; cannot fit a surrogate. See verify_lolbo.py."
        )

    train_x: List[str] = list(encoded.encoded_smiles)
    train_z = encoded.z.clone()
    train_y = torch.tensor(
        [oracle.cache[smiles] for smiles in train_x], dtype=torch.float32
    ).unsqueeze(-1)

    model, mll = initialize_surrogate(train_z, device)
    model = update_surrogate(model, mll, learning_rate, train_z, train_y.squeeze(-1), init_n_epochs, device)

    global_mode = tr_tolerances == "global"
    tr_state = make_tr_state(LATENT_DIM, batch_size, "literal", float(train_y.max()))

    trajectory: List[Dict[str, Any]] = [
        {
            "seed": int(seed),
            "iteration": 0,
            "oracle_calls": int(len(oracle)),
            "n_batch": int(len(oracle)),
            "best_so_far": float(oracle.best()["score"]),
            "best_smiles": oracle.best()["smiles"],
            "tr_length": float(tr_state.length),
            "n_restarts": 0,
            "n_gp_train": int(train_z.shape[0]),
        }
    ]
    diagnostics: List[Dict[str, Any]] = []

    totals = {
        "n_decoded": 0,
        "n_empty_decode": 0,
        "n_invalid_smiles": 0,
        "n_duplicate_history": 0,
        "n_duplicate_in_batch": 0,
        "n_constraint_reject": 0,
        "n_score_none": 0,
        "n_accepted": 0,
        "n_fill_rounds": 0,
    }
    restart_events: List[Dict[str, Any]] = []
    underfilled_batches = 0
    stalled_iterations = 0

    # Extra iterations are only reachable when a batch underfills; with full
    # batches this stops at exactly n_iters.
    max_iterations = max(int(n_iters), int(math.ceil(n_iters * max_iteration_multiplier)))

    iteration = 0
    while not oracle.finished and iteration < max_iterations:
        iteration += 1
        target = min(int(batch_size), oracle.remaining())
        center_idx = int(train_y.argmax())
        center_smiles = train_x[center_idx]
        center_z = train_z[center_idx]

        accepted_smiles: List[str] = []
        accepted_z: List[torch.Tensor] = []
        accepted_y: List[float] = []
        seen_this_iteration = set()
        counts = {key: 0 for key in totals}
        rounds = 0

        while len(accepted_smiles) < target and rounds < int(max_fill_rounds):
            rounds += 1
            counts["n_fill_rounds"] += 1
            need = target - len(accepted_smiles)

            if global_mode:
                z_next = generate_batch_global(
                    model=model, dim=LATENT_DIM, batch_size=need,
                    device=device, n_candidates=n_candidates,
                )
            else:
                z_next, _ = generate_batch(
                    state=tr_state,
                    model=model,
                    X=train_z,
                    Y=train_y,
                    batch_size=need,
                    device=device,
                    n_candidates=n_candidates,
                )
            z_next = z_next.detach().cpu().float()
            gp_mean, gp_std = posterior_mean_std(model, z_next, device)
            decoded = vae.decode(z_next)

            for rank, raw_smiles in enumerate(decoded):
                counts["n_decoded"] += 1
                if not raw_smiles:
                    counts["n_empty_decode"] += 1
                    continue
                canonical = canonicalize_smiles(raw_smiles)
                if canonical is None:
                    counts["n_invalid_smiles"] += 1
                    continue
                if canonical in oracle.cache:
                    counts["n_duplicate_history"] += 1
                    continue
                if canonical in seen_this_iteration:
                    counts["n_duplicate_in_batch"] += 1
                    continue
                mol = Chem.MolFromSmiles(canonical)
                if not molecule_satisfies_constraints(mol, constraints):
                    counts["n_constraint_reject"] += 1
                    seen_this_iteration.add(canonical)
                    continue

                score = oracle.score_one(canonical, phase="iter", step=iteration)
                seen_this_iteration.add(canonical)
                if score is None:
                    if oracle.finished:  # budget exhausted mid-batch
                        break
                    counts["n_score_none"] += 1  # non-finite / erroring oracle value
                    continue

                counts["n_accepted"] += 1
                accepted_smiles.append(canonical)
                accepted_z.append(z_next[rank])
                accepted_y.append(float(score))
                diagnostics.append(
                    {
                        "seed": int(seed),
                        "iteration": int(iteration),
                        "fill_round": int(rounds),
                        "rank_in_round": int(rank),
                        "oracle_call": int(len(oracle)),
                        "smiles": canonical,
                        "score": float(score),
                        # surrogate promise vs realised delivery
                        "gp_mean": float(gp_mean[rank]),
                        "gp_std": float(gp_std[rank]),
                        # how local is a latent trust region, in structure space?
                        "tanimoto_to_center": tanimoto(canonical, center_smiles),
                        "latent_dist_to_center": float(torch.norm(z_next[rank] - center_z)),
                        "tr_length": float(tr_state.length),
                        "center_smiles": center_smiles,
                        "center_score": float(train_y[center_idx]),
                        "n_gp_train": int(train_z.shape[0]),
                    }
                )
                if len(accepted_smiles) >= target or oracle.finished:
                    break

            if oracle.finished:
                break

        if len(accepted_smiles) < target:
            underfilled_batches += 1

        if accepted_smiles:
            y_next = torch.tensor(accepted_y, dtype=torch.float32).unsqueeze(-1)
            if not global_mode:
                update_state(state=tr_state, Y_next=y_next)  # vendored, unmodified
            z_next_accepted = torch.stack(accepted_z, dim=0)
            train_x.extend(accepted_smiles)
            train_z = torch.cat((train_z, z_next_accepted), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)
            model = update_surrogate(
                model,
                mll,
                learning_rate,
                train_z[-len(accepted_smiles) :],
                train_y[-len(accepted_smiles) :].squeeze(-1),
                num_update_epochs,
                device,
            )
            stalled_iterations = 0
        else:
            stalled_iterations += 1

        restarted = False
        if (not global_mode) and tr_state.restart_triggered:
            restart_events.append(
                {
                    "seed": int(seed),
                    "iteration": int(iteration),
                    "oracle_calls": int(len(oracle)),
                    "length_at_restart": float(tr_state.length),
                    "best_so_far": float(oracle.best()["score"]),
                }
            )
            tr_state = make_tr_state(LATENT_DIM, batch_size, tr_tolerances, float(train_y.max()))
            restarted = True

        for key, value in counts.items():
            totals[key] += value

        trajectory.append(
            {
                "seed": int(seed),
                "iteration": int(iteration),
                "oracle_calls": int(len(oracle)),
                "n_batch": int(len(accepted_smiles)),
                "best_so_far": float(oracle.best()["score"]),
                "best_smiles": oracle.best()["smiles"],
                "tr_length": float(tr_state.length),
                "tr_success_counter": int(tr_state.success_counter),
                "tr_failure_counter": int(tr_state.failure_counter),
                "restarted": bool(restarted),
                "n_restarts": int(len(restart_events)),
                "n_fill_rounds": int(rounds),
                "n_decoded": int(counts["n_decoded"]),
                "n_empty_decode": int(counts["n_empty_decode"]),
                "n_invalid_smiles": int(counts["n_invalid_smiles"]),
                "n_duplicate_history": int(counts["n_duplicate_history"]),
                "n_duplicate_in_batch": int(counts["n_duplicate_in_batch"]),
                "n_constraint_reject": int(counts["n_constraint_reject"]),
                "n_score_none": int(counts["n_score_none"]),
                "n_gp_train": int(train_z.shape[0]),
                "center_smiles": center_smiles,
            }
        )

        if verbose:
            print(
                f"[{oracle_spec.name}|{method}|seed {seed}] iter {iteration:>3} "
                f"calls={len(oracle):>4} best={oracle.best()['score']:.4f} "
                f"batch={len(accepted_smiles)}/{target} rounds={rounds} "
                f"decoded={counts['n_decoded']} dup={counts['n_duplicate_history']} "
                f"L={tr_state.length:.4f}",
                flush=True,
            )

        if stalled_iterations >= 3:
            print(
                f"[{oracle_spec.name}|{method}|seed {seed}] stopping: 3 consecutive iterations "
                "produced no new valid in-constraint molecule.",
                flush=True,
            )
            break

    write_rows_csv(out_dir / f"latent_diagnostics_seed_{seed}.csv", diagnostics)
    if restart_events:
        write_rows_csv(out_dir / f"restarts_seed_{seed}.csv", restart_events)

    decode_yield = totals["n_accepted"] / totals["n_decoded"] if totals["n_decoded"] else float("nan")
    summary = finish_run(
        out_dir=out_dir,
        method=method,
        seed=seed,
        oracle=oracle,
        trajectory=trajectory,
        extra_summary={
            "algorithm": (
                "Latent BO (frozen SELFIES VAE), no trust region: DKL SVGP + Thompson sampling "
                "over candidates drawn from the whole latent prior"
                if tr_tolerances == "global" else
                "LOL-BO (frozen SELFIES VAE): TuRBO trust region + DKL SVGP + Thompson sampling in a 256-d latent space"
            ),
            "vae": "InfoTransformerVAE (LOL-BO SELFIES VAE), frozen, no end-to-end fine-tuning",
            "vae_state_dict": str(vae_state_dict),
            "latent_dim": int(LATENT_DIM),
            "strip_stereo": bool(strip_stereo),
            "max_string_length": int(max_string_length),
            "tr_tolerances": tr_tolerances,
            "tr_failure_tolerance": (
                None if tr_tolerances == "global" else int(
                    LITERAL_FAILURE_TOLERANCE if tr_tolerances == "literal"
                    else budget_scaled_failure_tolerance(LATENT_DIM, batch_size)
                )
            ),
            "tr_success_tolerance": int(tr_state.success_tolerance),
            "tr_length_final": float(tr_state.length),
            "n_iters_requested": int(n_iters),
            "batch_size": int(batch_size),
            "iterations_completed": int(iteration),
            "underfilled_batches": int(underfilled_batches),
            "learning_rate": float(learning_rate),
            "init_n_epochs": int(init_n_epochs),
            "num_update_epochs": int(num_update_epochs),
            "n_candidates": int(n_candidates) if n_candidates else min(5000, max(2000, 200 * LATENT_DIM)),
            "max_fill_rounds": int(max_fill_rounds),
            "n_initial_molecules": int(len(init_population)),
            "n_initial_encoded": int(encoded.z.shape[0]),
            "n_initial_encode_failures": int(encoded.n_failed),
            "initial_encode_failures": [reason for _, reason in encoded.failed][:20],
            "n_stereo_stripped_total": int(vae.n_stereo_stripped),
            "n_restarts": int(len(restart_events)),
            "decode_counts": {key: int(value) for key, value in totals.items()},
            "decode_yield_new_valid_per_decode": float(decode_yield),
            "oracle_source": oracle_source,
            "device": str(device),
            "runtime_seconds": float(time.time() - started),
            "diagnostics_csv": str(out_dir / f"latent_diagnostics_seed_{seed}.csv"),
        },
    )
    return summary


def run_lolbo_benchmark(
    initial_smiles_by_seed: Dict[int, List[str]],
    oracle_spec,
    out_root: Path,
    seeds: Sequence[int],
    n_iters: int,
    batch_size: int,
    constraints: MoleculeConstraints,
    device: torch.device,
    method: str = "lolbo_frozen",
    **seed_kwargs,
) -> Dict[str, Any]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for seed in seeds:
        summaries.append(
            run_lolbo_seed(
                initial_smiles=initial_smiles_by_seed[int(seed)],
                seed=int(seed),
                out_dir=out_root / f"seed_{seed}",
                n_iters=n_iters,
                batch_size=batch_size,
                constraints=constraints,
                oracle_spec=oracle_spec,
                device=device,
                method=method,
                **seed_kwargs,
            )
        )

    final_rows = [
        {
            "seed": int(s["seed"]),
            "best_score": float(s["best_score"]),
            "best_smiles": s["best_smiles"],
            "n_oracle_calls": int(s["n_oracle_calls"]),
        }
        for s in summaries
    ]
    write_rows_csv(out_root / "final_scores_by_seed.csv", final_rows)

    best_scores = [float(row["best_score"]) for row in final_rows]
    overall = {
        "method": method,
        "oracle": oracle_spec.name,
        "reward_label": oracle_spec.reward_label,
        "seeds": [int(seed) for seed in seeds],
        "n_iters": int(n_iters),
        "batch_size": int(batch_size),
        "tr_tolerances": seed_kwargs.get("tr_tolerances", "literal"),
        "oracle_source": seed_kwargs.get("oracle_source", "plogp_benchmark_suite.oracles"),
        "mean_final_best": float(np.mean(best_scores)),
        "std_final_best": float(np.std(best_scores)),
        "best_score_overall": float(np.max(best_scores)),
        "mean_decode_yield": float(
            np.mean([s["decode_yield_new_valid_per_decode"] for s in summaries])
        ),
        "out_root": str(out_root),
    }
    write_json(out_root / "benchmark_summary.json", overall)
    return overall


def parse_args():
    parser = argparse.ArgumentParser(description="Run one oracle x seeds of frozen-VAE LOL-BO.")
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--initial-data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", default="lolbo_frozen")
    parser.add_argument("--tr-tolerances", choices=TR_TOLERANCE_MODES, default="literal")
    parser.add_argument("--seeds", type=int, nargs="*", default=[11, 22, 33, 44, 55])
    parser.add_argument("--n-iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=49)
    parser.add_argument("--max-sa", type=float, required=True)
    parser.add_argument("--max-smiles-length", type=int, default=None)
    parser.add_argument("--max-heavy-atoms", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vae-state-dict", default=str(DEFAULT_STATE_DICT))
    parser.add_argument("--max-string-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--init-n-epochs", type=int, default=20)
    parser.add_argument("--num-update-epochs", type=int, default=2)
    parser.add_argument("--n-candidates", type=int, default=None)
    parser.add_argument("--max-fill-rounds", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    from benchmark_oracles import read_initial_smiles_by_seed
    from plogp_benchmark_suite.oracles import get_oracle_spec

    oracle_spec = get_oracle_spec(args.oracle)
    constraints = MoleculeConstraints(
        max_sa_score=args.max_sa,
        max_smiles_length=args.max_smiles_length,
        max_heavy_atoms=args.max_heavy_atoms,
    )
    initials = read_initial_smiles_by_seed(Path(args.initial_data), list(args.seeds))
    overall = run_lolbo_benchmark(
        initial_smiles_by_seed=initials,
        oracle_spec=oracle_spec,
        out_root=Path(args.out),
        seeds=args.seeds,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        constraints=constraints,
        device=resolve_device(args.device),
        method=args.method,
        tr_tolerances=args.tr_tolerances,
        vae_state_dict=Path(args.vae_state_dict),
        max_string_length=args.max_string_length,
        learning_rate=args.learning_rate,
        init_n_epochs=args.init_n_epochs,
        num_update_epochs=args.num_update_epochs,
        n_candidates=args.n_candidates,
        max_fill_rounds=args.max_fill_rounds,
    )
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
