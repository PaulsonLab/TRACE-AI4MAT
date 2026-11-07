import argparse
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gpytorch
import numpy as np
import torch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.normal import IIDNormalSampler
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from plogp_benchmark_suite.common import (
    FP_NBITS,
    MAX_SA_SCORE,
    CachedMoleculeOracle,
    MoleculeConstraints,
    PenalizedLogPOracle,
    default_initial_smiles_by_seed,
    finish_run,
    parse_required_atomic_nums,
    sanitize_smiles_for_constraints,
    set_all_seeds,
    smiles_to_fp_array,
    str_to_bool,
    write_json,
    write_rows_csv,
)
from plogp_benchmark_suite.graph_ga_ops import (
    propose_graph_ga_offspring,
)


class TanimotoKernel(Kernel):
    has_lengthscale = False

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return torch.ones(x1.shape[:-1], device=x1.device, dtype=x1.dtype)

        if x1.dim() == 2:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 2:
            x2 = x2.unsqueeze(0)

        inter = torch.matmul(x1, x2.transpose(-1, -2))
        x1_sum = x1.sum(dim=-1).unsqueeze(-1)
        x2_sum = x2.sum(dim=-1).unsqueeze(-2)
        union = torch.clamp(x1_sum + x2_sum - inter, min=1e-6)
        return (inter / union).squeeze(0)


class TanimotoGP(SingleTaskGP):
    def __init__(self, train_X, train_Y):
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-4))
        covar_module = ScaleKernel(TanimotoKernel())
        super().__init__(train_X, train_Y, covar_module=covar_module, likelihood=likelihood)


def fit_gp(history_smiles: List[str], history_scores: List[float]):
    X = np.stack([smiles_to_fp_array(s, n_bits=FP_NBITS) for s in history_smiles]).astype(np.float32)
    train_X = torch.from_numpy(X).float()
    train_Y = torch.tensor(history_scores, dtype=torch.float32).unsqueeze(-1)

    y_mean = train_Y.mean()
    y_std = train_Y.std(unbiased=False)
    if float(y_std.item()) < 1e-6:
        y_std = torch.tensor(1.0, dtype=torch.float32)

    train_Y_norm = (train_Y - y_mean) / y_std
    model = TanimotoGP(train_X, train_Y_norm)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    try:
        fit_gpytorch_mll(mll)
        model.eval()
        model.likelihood.eval()
    except Exception:
        pass

    return model, float(y_mean.item()), float(y_std.item())


def select_gp_training_data(
    history_smiles: List[str],
    history_scores: List[float],
    max_gp_train_size: int,
) -> Tuple[List[str], List[float]]:
    if max_gp_train_size is None or int(max_gp_train_size) <= 0:
        return history_smiles, history_scores
    if len(history_smiles) <= int(max_gp_train_size):
        return history_smiles, history_scores

    max_size = int(max_gp_train_size)
    n_top = max(1, max_size // 2)
    n_recent = max_size - n_top
    top_idx = sorted(range(len(history_scores)), key=lambda i: history_scores[i], reverse=True)[:n_top]
    recent_idx = list(range(max(0, len(history_smiles) - n_recent), len(history_smiles)))

    selected_idx = []
    seen = set()
    for idx in top_idx + recent_idx:
        if idx not in seen:
            selected_idx.append(idx)
            seen.add(idx)

    if len(selected_idx) < max_size:
        for idx in range(len(history_smiles) - 1, -1, -1):
            if idx in seen:
                continue
            selected_idx.append(idx)
            seen.add(idx)
            if len(selected_idx) >= max_size:
                break

    selected_idx = selected_idx[:max_size]
    return [history_smiles[i] for i in selected_idx], [history_scores[i] for i in selected_idx]


def predict_mu_std(model, smiles_list: List[str], y_mean: float, y_std: float):
    if not smiles_list:
        return np.array([]), np.array([])
    X = np.stack([smiles_to_fp_array(s, n_bits=FP_NBITS) for s in smiles_list]).astype(np.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(torch.from_numpy(X).float())
        mu = posterior.mean.squeeze(-1).cpu().numpy()
        var = posterior.variance.squeeze(-1).cpu().numpy()
    mu = mu * y_std + y_mean
    std = np.sqrt(np.maximum(var, 1e-12)) * y_std
    return mu, std


def expected_improvement(mu: np.ndarray, std: np.ndarray, y_best: float) -> np.ndarray:
    if len(mu) == 0:
        return np.array([])

    mu_t = torch.as_tensor(mu, dtype=torch.float64)
    std_t = torch.clamp(torch.as_tensor(std, dtype=torch.float64), min=1e-12)
    improvement = mu_t - float(y_best)
    z = improvement / std_t
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    ei = improvement * normal.cdf(z) + std_t * torch.exp(normal.log_prob(z))
    return torch.clamp(ei, min=0.0).cpu().numpy()


def select_batch_qei(
    model,
    candidate_smiles: List[str],
    y_mean: float,
    y_std: float,
    y_best: float,
    batch_size: int,
    qei_samples: int = 128,
) -> List[str]:
    if not candidate_smiles:
        return []

    q = max(1, min(int(batch_size), len(candidate_smiles)))
    X = np.stack([smiles_to_fp_array(s, n_bits=FP_NBITS) for s in candidate_smiles]).astype(np.float32)
    X_cand = torch.from_numpy(X).float()
    y_std = max(float(y_std), 1e-8)
    best_f_norm = (float(y_best) - float(y_mean)) / y_std
    sampler = IIDNormalSampler(sample_shape=torch.Size([int(qei_samples)]))
    acq_func = qExpectedImprovement(model=model, best_f=best_f_norm, sampler=sampler)

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
            selected.append(candidate_smiles[matches[0].item()])
    return list(dict.fromkeys(selected))


def run_acquisition_graph_ga(
    start_smiles: List[str],
    acq_score_fn,
    constraints: MoleculeConstraints,
    population_size: int,
    offspring_size: int,
    mutation_rate: float,
    ga_generations: int,
    acq_eval_batch_size: int = 256,
) -> Tuple[List[str], Dict[str, float]]:
    population = sanitize_smiles_for_constraints(start_smiles, constraints)
    if not population:
        return [], {}

    acq_cache: Dict[str, float] = {}

    def score_acq_batch(smiles_batch: List[str]):
        missing = [s for s in dict.fromkeys(smiles_batch) if s not in acq_cache]
        batch = max(1, int(acq_eval_batch_size))
        for start in range(0, len(missing), batch):
            chunk = missing[start : start + batch]
            try:
                values = acq_score_fn(chunk)
            except Exception:
                continue
            for smiles, value in zip(chunk, values):
                value = float(value)
                if np.isfinite(value):
                    acq_cache[smiles] = value

    def trim_acq_population(candidate_smiles: List[str]) -> Tuple[List[str], List[float]]:
        unique = list(dict.fromkeys(sanitize_smiles_for_constraints(candidate_smiles, constraints)))
        score_acq_batch(unique)
        pairs = [(acq_cache[s], s) for s in unique if s in acq_cache]
        pairs.sort(reverse=True, key=lambda item: item[0])
        pairs = pairs[:population_size]
        return [smiles for _, smiles in pairs], [score for score, _ in pairs]

    population, population_scores = trim_acq_population(population)

    for _ in range(ga_generations):
        offspring = propose_graph_ga_offspring(
            population_smiles=population,
            population_scores=population_scores,
            constraints=constraints,
            offspring_size=offspring_size,
            mutation_rate=mutation_rate,
        )
        if not offspring:
            continue
        population, population_scores = trim_acq_population(population + offspring)

    all_smiles = sorted(acq_cache.keys(), key=lambda s: acq_cache[s], reverse=True)
    return all_smiles, acq_cache


def run_generator_graph_ga(
    start_smiles: List[str],
    constraints: MoleculeConstraints,
    population_size: int,
    offspring_size: int,
    mutation_rate: float,
    ga_generations: int,
    max_candidates: int,
    exclude_smiles,
) -> List[str]:
    population = sanitize_smiles_for_constraints(start_smiles, constraints)
    if not population:
        return []

    excluded = set(sanitize_smiles_for_constraints(exclude_smiles, constraints))
    generated: List[str] = []
    seen = set(population) | excluded

    for _ in range(ga_generations):
        neutral_scores = [1.0] * len(population)
        offspring = propose_graph_ga_offspring(
            population_smiles=population,
            population_scores=neutral_scores,
            constraints=constraints,
            offspring_size=offspring_size,
            mutation_rate=mutation_rate,
            exclude_smiles=seen,
        )
        if not offspring:
            continue

        for smiles in offspring:
            if smiles in seen:
                continue
            seen.add(smiles)
            generated.append(smiles)
            if max_candidates and len(generated) >= int(max_candidates):
                break

        population = list(dict.fromkeys(population + offspring))
        if len(population) > population_size:
            population = population[-population_size:]

        if max_candidates and len(generated) >= int(max_candidates):
            break

    return generated


def rank_by_acquisition(candidate_smiles: List[str], acq_score_fn, batch_size: int = 256) -> List[str]:
    scores: Dict[str, float] = {}
    unique = list(dict.fromkeys(candidate_smiles))
    batch_size = max(1, int(batch_size))

    for start in range(0, len(unique), batch_size):
        chunk = unique[start : start + batch_size]
        try:
            values = acq_score_fn(chunk)
        except Exception:
            continue
        for smiles, value in zip(chunk, values):
            value = float(value)
            if np.isfinite(value):
                scores[smiles] = value

    return sorted(scores.keys(), key=lambda s: scores[s], reverse=True)


def run_gpbo_seed(
    initial_smiles: List[str],
    seed: int,
    out_dir: Path,
    n_iters: int,
    batch_size: int,
    constraints: MoleculeConstraints,
    population_size: int = 80,
    offspring_size: int = 40,
    mutation_rate: float = 0.01,
    ga_generations: int = 8,
    acquisition: str = "qei",
    ucb_beta: float = 4.0,
    max_acq_candidates: int = 256,
    acq_eval_batch_size: int = 256,
    qei_samples: int = 64,
    max_gp_train_size: int = 400,
    graph_ga_role: str = "generator",
    max_resample_rounds: int = 12,
    oracle_spec=None,
) -> Dict:
    set_all_seeds(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_population = sanitize_smiles_for_constraints(initial_smiles, constraints)
    if not init_population:
        raise ValueError(f"No valid initial population for seed {seed}.")

    max_oracle_calls = len(init_population) + int(n_iters) * int(batch_size)
    if oracle_spec is None:
        oracle = PenalizedLogPOracle(max_oracle_calls=max_oracle_calls, constraints=constraints)
    else:
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

    trajectory = [
        {
            "seed": int(seed),
            "iteration": 0,
            "oracle_calls": int(len(oracle)),
            "n_batch": int(len(oracle)),
            "best_so_far": float(oracle.best()["score"]),
            "best_smiles": oracle.best()["smiles"],
            "n_acq_candidates": 0,
        }
    ]
    carryover: List[str] = []
    underfilled_batches = 0

    iteration = 0
    while not oracle.finished and iteration < n_iters:
        iteration += 1
        history_smiles = list(oracle.cache.keys())
        history_scores = [oracle.cache[s] for s in history_smiles]
        if len(history_smiles) < 2:
            offspring = propose_graph_ga_offspring(
                population_smiles=history_smiles,
                population_scores=history_scores,
                constraints=constraints,
                offspring_size=min(batch_size, oracle.remaining()),
                mutation_rate=mutation_rate,
            )
            if not offspring:
                break
            oracle.score_many(offspring, phase="warm_start", step=iteration)
            trajectory.append(
                {
                    "seed": int(seed),
                    "iteration": int(iteration),
                    "oracle_calls": int(len(oracle)),
                    "n_batch": int(len(offspring)),
                    "best_so_far": float(oracle.best()["score"]),
                    "best_smiles": oracle.best()["smiles"],
                    "n_acq_candidates": 0,
                    "max_acq": None,
                }
            )
            continue

        train_smiles, train_scores = select_gp_training_data(
            history_smiles,
            history_scores,
            max_gp_train_size=max_gp_train_size,
        )
        model, y_mean, y_std = fit_gp(train_smiles, train_scores)
        y_best = max(history_scores)

        def acq_score(smiles_batch: List[str]):
            mu, std = predict_mu_std(model, smiles_batch, y_mean, y_std)
            if acquisition == "ucb":
                return mu + math.sqrt(ucb_beta) * std
            return expected_improvement(mu, std, y_best)

        ranked_known = sorted(oracle.cache.items(), key=lambda item: item[1], reverse=True)
        start_smiles = [s for s, _ in ranked_known[: min(len(ranked_known), population_size)]]
        start_smiles.extend(carryover)

        acq_cache: Dict[str, float] = {}
        if graph_ga_role == "acq_optimizer":
            candidate_source, acq_cache = run_acquisition_graph_ga(
                start_smiles=start_smiles,
                acq_score_fn=acq_score,
                constraints=constraints,
                population_size=population_size,
                offspring_size=offspring_size,
                mutation_rate=mutation_rate,
                ga_generations=ga_generations,
                acq_eval_batch_size=acq_eval_batch_size,
            )
        else:
            generated_smiles = run_generator_graph_ga(
                start_smiles=start_smiles,
                constraints=constraints,
                population_size=population_size,
                offspring_size=offspring_size,
                mutation_rate=mutation_rate,
                ga_generations=ga_generations,
                max_candidates=max_acq_candidates,
                exclude_smiles=oracle.cache.keys(),
            )
            candidate_source = carryover + generated_smiles

        target = min(batch_size, oracle.remaining())
        candidate_pool = []
        for smiles in candidate_source:
            if smiles not in oracle.cache and smiles not in candidate_pool:
                candidate_pool.append(smiles)
            if max_acq_candidates and len(candidate_pool) >= int(max_acq_candidates):
                break

        # Resample fresh molecules if the generator under-produced this iteration.
        # Without this the batch (and therefore the oracle budget) silently
        # underfills, so GP-BO ends with fewer evaluations than the other methods.
        # Each round re-runs the Graph-GA generator excluding everything already
        # seen/queued and escalates the mutation rate to break out of a converged
        # population. Only triggers when len(candidate_pool) < target.
        resample_rounds = 0
        while len(candidate_pool) < target and resample_rounds < int(max_resample_rounds):
            resample_rounds += 1
            resample_mutation_rate = min(0.5, float(mutation_rate) * (1.0 + resample_rounds))
            extra = run_generator_graph_ga(
                start_smiles=start_smiles,
                constraints=constraints,
                population_size=population_size,
                offspring_size=offspring_size,
                mutation_rate=resample_mutation_rate,
                ga_generations=ga_generations,
                max_candidates=max_acq_candidates,
                exclude_smiles=set(oracle.cache.keys()) | set(candidate_pool),
            )
            added = 0
            for smiles in extra:
                if smiles in oracle.cache or smiles in candidate_pool:
                    continue
                candidate_pool.append(smiles)
                added += 1
                if max_acq_candidates and len(candidate_pool) >= int(max_acq_candidates):
                    break
            if added == 0:
                break  # generator exhausted -- cannot produce more unique valid molecules

        ranked_by_acq = rank_by_acquisition(
            candidate_pool,
            acq_score_fn=acq_score,
            batch_size=acq_eval_batch_size,
        )

        if acquisition == "qei":
            batch = select_batch_qei(
                model=model,
                candidate_smiles=candidate_pool,
                y_mean=y_mean,
                y_std=y_std,
                y_best=y_best,
                batch_size=target,
                qei_samples=qei_samples,
            )
            for smiles in ranked_by_acq:
                if len(batch) >= target:
                    break
                if smiles not in batch:
                    batch.append(smiles)
        else:
            batch = ranked_by_acq[:target]

        if not batch:
            break

        calls_before = len(oracle)
        oracle.score_many(batch[:target], phase="iter", step=iteration)
        if len(oracle) - calls_before < target:
            underfilled_batches += 1
        carryover = [
            smiles
            for smiles in candidate_source
            if smiles not in oracle.cache
        ][:population_size]

        trajectory.append(
            {
                "seed": int(seed),
                "iteration": int(iteration),
                "oracle_calls": int(len(oracle)),
                "n_batch": int(len(batch[:target])),
                "best_so_far": float(oracle.best()["score"]),
                "best_smiles": oracle.best()["smiles"],
                "graph_ga_role": graph_ga_role,
                "n_generated_candidates": int(len(candidate_source)),
                "n_acq_candidates": int(len(ranked_by_acq)),
                "n_candidate_pool": int(len(candidate_pool)),
                "n_resample_rounds": int(resample_rounds),
                "n_gp_train": int(len(train_smiles)),
                "max_acq": float(max(acq_cache.values())) if acq_cache else None,
            }
        )

    return finish_run(
        out_dir=out_dir,
        method="gpbo_graph_ga",
        seed=seed,
        oracle=oracle,
        trajectory=trajectory,
        extra_summary={
            "n_iters_requested": int(n_iters),
            "batch_size": int(batch_size),
            "population_size": int(population_size),
            "offspring_size": int(offspring_size),
            "mutation_rate": float(mutation_rate),
            "ga_generations": int(ga_generations),
            "acquisition": acquisition,
            "ucb_beta": float(ucb_beta),
            "max_acq_candidates": int(max_acq_candidates),
            "acq_eval_batch_size": int(acq_eval_batch_size),
            "qei_samples": int(qei_samples),
            "max_gp_train_size": int(max_gp_train_size),
            "graph_ga_role": graph_ga_role,
            "max_resample_rounds": int(max_resample_rounds),
            "underfilled_batches": int(underfilled_batches),
            "iterations_completed": int(iteration),
        },
    )


def run_gpbo_benchmark(
    out_root: str = "benchmark_gpbo_graph_ga_plogp",
    seeds: Optional[List[int]] = None,
    n_iters: int = 50,
    batch_size: int = 5,
    max_sa_score: float = MAX_SA_SCORE,
    max_smiles_length: Optional[int] = None,
    max_heavy_atoms: Optional[int] = None,
    required_atomic_nums: Optional[List[int]] = None,
    population_size: int = 80,
    offspring_size: int = 40,
    mutation_rate: float = 0.01,
    ga_generations: int = 8,
    acquisition: str = "qei",
    ucb_beta: float = 4.0,
    max_acq_candidates: int = 256,
    acq_eval_batch_size: int = 256,
    qei_samples: int = 64,
    max_gp_train_size: int = 400,
    graph_ga_role: str = "generator",
    max_resample_rounds: int = 12,
    initial_smiles_by_seed: Optional[Dict[int, List[str]]] = None,
    oracle_spec=None,
) -> Dict:
    if seeds is None:
        seeds = [11, 22, 33, 44, 55]

    constraints = MoleculeConstraints(
        max_sa_score=max_sa_score,
        max_smiles_length=max_smiles_length,
        max_heavy_atoms=max_heavy_atoms,
        required_atomic_nums=required_atomic_nums,
    )
    initials = initial_smiles_by_seed or default_initial_smiles_by_seed()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for seed in seeds:
        summaries.append(
            run_gpbo_seed(
                initial_smiles=initials[seed],
                seed=seed,
                out_dir=out_root / f"seed_{seed}",
                n_iters=n_iters,
                batch_size=batch_size,
                constraints=constraints,
                population_size=population_size,
                offspring_size=offspring_size,
                mutation_rate=mutation_rate,
                ga_generations=ga_generations,
                acquisition=acquisition,
                ucb_beta=ucb_beta,
                max_acq_candidates=max_acq_candidates,
                acq_eval_batch_size=acq_eval_batch_size,
                qei_samples=qei_samples,
                max_gp_train_size=max_gp_train_size,
                graph_ga_role=graph_ga_role,
                max_resample_rounds=max_resample_rounds,
                oracle_spec=oracle_spec,
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
        "method": "gpbo_graph_ga",
        "oracle": oracle_spec.name if oracle_spec is not None else "penalized_logp",
        "reward_label": oracle_spec.reward_label if oracle_spec is not None else "penalized logP",
        "seeds": [int(s) for s in seeds],
        "n_iters": int(n_iters),
        "batch_size": int(batch_size),
        "acquisition": acquisition,
        "max_acq_candidates": int(max_acq_candidates),
        "acq_eval_batch_size": int(acq_eval_batch_size),
        "qei_samples": int(qei_samples),
        "max_gp_train_size": int(max_gp_train_size),
        "graph_ga_role": graph_ga_role,
        "mean_final_best": float(np.mean(best_scores)),
        "std_final_best": float(np.std(best_scores)),
        "best_score_overall": float(np.max(best_scores)),
        "out_root": str(out_root),
    }
    write_json(out_root / "benchmark_summary.json", overall)
    return overall


def parse_args():
    parser = argparse.ArgumentParser(description="Run GP-BO with Graph-GA acquisition search for pLogP.")
    parser.add_argument("--out", default="benchmark_gpbo_graph_ga_plogp")
    parser.add_argument("--n-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=80)
    parser.add_argument("--offspring-size", type=int, default=40)
    parser.add_argument("--mutation-rate", type=float, default=0.01)
    parser.add_argument("--ga-generations", type=int, default=8)
    parser.add_argument("--acquisition", choices=["qei", "ucb"], default="qei")
    parser.add_argument("--ucb-beta", type=float, default=4.0)
    parser.add_argument("--max-acq-candidates", type=int, default=256)
    parser.add_argument("--acq-eval-batch-size", type=int, default=256)
    parser.add_argument("--qei-samples", type=int, default=64)
    parser.add_argument("--max-gp-train-size", type=int, default=400)
    parser.add_argument("--graph-ga-role", choices=["generator", "acq_optimizer"], default="generator")
    parser.add_argument("--max-resample-rounds", type=int, default=12)
    parser.add_argument("--max-sa", type=float, default=MAX_SA_SCORE)
    parser.add_argument("--max-smiles-length", type=int, default=None)
    parser.add_argument("--max-heavy-atoms", type=int, default=None)
    parser.add_argument("--require-fe", type=str_to_bool, default=False, metavar="true/false")
    parser.add_argument("--require-elements", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=[11, 22, 33, 44, 55])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_gpbo_benchmark(
        out_root=args.out,
        seeds=args.seeds,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        max_sa_score=args.max_sa,
        max_smiles_length=args.max_smiles_length,
        max_heavy_atoms=args.max_heavy_atoms,
        required_atomic_nums=parse_required_atomic_nums(args.require_elements, args.require_fe),
        population_size=args.population_size,
        offspring_size=args.offspring_size,
        mutation_rate=args.mutation_rate,
        ga_generations=args.ga_generations,
        acquisition=args.acquisition,
        ucb_beta=args.ucb_beta,
        max_acq_candidates=args.max_acq_candidates,
        acq_eval_batch_size=args.acq_eval_batch_size,
        qei_samples=args.qei_samples,
        max_gp_train_size=args.max_gp_train_size,
        graph_ga_role=args.graph_ga_role,
        max_resample_rounds=args.max_resample_rounds,
    )
