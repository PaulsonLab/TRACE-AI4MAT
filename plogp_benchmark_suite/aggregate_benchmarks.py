import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from plogp_benchmark_suite.common import write_json, write_rows_csv


def collect_seed_summaries(root: Path) -> List[Dict]:
    summaries = []
    for path in sorted(root.glob("seed_*/summary_seed_*.json")):
        with open(path) as f:
            payload = json.load(f)
        payload["_summary_path"] = str(path)
        summaries.append(payload)
    return summaries


def aggregate_roots(roots: List[Path], out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    method_rows = []
    seed_rows = []
    for root in roots:
        summaries = collect_seed_summaries(root)
        if not summaries:
            continue

        method = summaries[0].get("method", root.name)
        scores = [float(item["best_score"]) for item in summaries]
        calls = [int(item.get("n_oracle_calls", item.get("n_evaluated_total", 0))) for item in summaries]

        for item in summaries:
            seed_rows.append(
                {
                    "method": method,
                    "seed": int(item["seed"]),
                    "best_score": float(item["best_score"]),
                    "best_smiles": item["best_smiles"],
                    "n_oracle_calls": int(item.get("n_oracle_calls", item.get("n_evaluated_total", 0))),
                    "summary_path": item["_summary_path"],
                }
            )

        method_rows.append(
            {
                "method": method,
                "root": str(root),
                "n_seeds": int(len(summaries)),
                "mean_best_score": float(np.mean(scores)),
                "std_best_score": float(np.std(scores)),
                "best_score": float(np.max(scores)),
                "mean_oracle_calls": float(np.mean(calls)),
            }
        )

    write_rows_csv(out_dir / "comparison_by_seed.csv", seed_rows)
    write_rows_csv(out_dir / "comparison_by_method.csv", method_rows)
    write_json(
        out_dir / "comparison_summary.json",
        {
            "roots": [str(root) for root in roots],
            "n_methods": int(len(method_rows)),
            "methods": [row["method"] for row in method_rows],
        },
    )
    return method_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate TRACE pLogP benchmark outputs.")
    parser.add_argument("--roots", nargs="+", required=True, help="Benchmark output roots.")
    parser.add_argument("--out", default="benchmark_plogp_comparison")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    aggregate_roots([Path(root) for root in args.roots], Path(args.out))
