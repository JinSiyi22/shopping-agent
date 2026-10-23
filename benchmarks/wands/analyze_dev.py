
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COMPARISONS = (
    ("rrf", "dense"),
    ("dense_ce", "dense"),
    ("rrf_ce", "rrf"),
    ("rrf_ce", "dense_ce"),
)

METRICS = (
    "ndcg_at_10",
    "exact_mrr_at_10",
    "exact_recall_at_50",
    "relevant_recall_at_50",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-query",
        default="reports/wands_dev_rerank_per_query.csv",
    )
    parser.add_argument(
        "--labels",
        default="data/wands_processed/labels_resolved.parquet",
    )
    parser.add_argument(
        "--output",
        default="reports/wands_dev_analysis.json",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def paired_bootstrap(
    frame: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str,
    samples: int,
    seed: int,
) -> dict:
    col_a = f"{method_a}__{metric}"
    col_b = f"{method_b}__{metric}"

    paired = frame[[col_a, col_b]].dropna()
    deltas = (
        paired[col_a].to_numpy(float)
        - paired[col_b].to_numpy(float)
    )

    n = len(deltas)
    if n == 0:
        return {"n": 0}

    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)

    for start in range(0, samples, 1000):
        batch = min(1000, samples - start)
        idx = rng.integers(0, n, size=(batch, n))
        boot[start:start + batch] = deltas[idx].mean(axis=1)

    return {
        "n": int(n),
        "mean_a": float(paired[col_a].mean()),
        "mean_b": float(paired[col_b].mean()),
        "mean_delta": float(deltas.mean()),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "prob_delta_gt_0": float((boot > 0).mean()),
    }


def oracle_recall(
    labels: pd.DataFrame,
    query_ids: list[int],
    positive_labels: set[str],
    k: int,
) -> dict:
    selected = labels[labels["query_id"].isin(query_ids)]
    counts = (
        selected[selected["label"].isin(positive_labels)]
        .groupby("query_id")
        .size()
        .to_dict()
    )

    values = []
    for query_id in query_ids:
        total = int(counts.get(query_id, 0))
        if total > 0:
            values.append(min(k, total) / total)

    arr = np.asarray(values, dtype=float)
    return {
        "n": int(len(arr)),
        "mean_oracle_recall": float(arr.mean()),
        "median_oracle_recall": float(np.median(arr)),
    }


def main() -> None:
    args = parse_args()

    per_query = pd.read_csv(args.per_query)
    labels = pd.read_parquet(args.labels)
    query_ids = per_query["query_id"].astype(int).tolist()

    comparisons = {}
    counter = 0

    for method_a, method_b in COMPARISONS:
        key = f"{method_a}_vs_{method_b}"
        comparisons[key] = {}

        for metric in METRICS:
            counter += 1
            comparisons[key][metric] = paired_bootstrap(
                per_query,
                method_a,
                method_b,
                metric,
                args.bootstrap_samples,
                args.seed + counter,
            )

    ceilings = {}
    for k in (10, 50):
        ceilings[f"exact_recall_at_{k}"] = oracle_recall(
            labels,
            query_ids,
            {"Exact"},
            k,
        )
        ceilings[f"relevant_recall_at_{k}"] = oracle_recall(
            labels,
            query_ids,
            {"Exact", "Partial"},
            k,
        )

    observed_vs_oracle = {}

    for method in (
        "bm25",
        "dense",
        "rrf",
        "dense_ce",
        "rrf_ce",
    ):
        observed_vs_oracle[method] = {}

        for metric in (
            "exact_recall_at_10",
            "exact_recall_at_50",
            "relevant_recall_at_10",
            "relevant_recall_at_50",
        ):
            column = f"{method}__{metric}"
            if column not in per_query.columns:
                continue

            observed = float(per_query[column].dropna().mean())
            ceiling = ceilings[metric]["mean_oracle_recall"]

            observed_vs_oracle[method][metric] = {
                "observed": observed,
                "oracle_ceiling": ceiling,
                "fraction_of_oracle": observed / ceiling,
            }

    report = {
        "queries": len(query_ids),
        "bootstrap_samples": args.bootstrap_samples,
        "paired_bootstrap": comparisons,
        "oracle_recall_ceilings": ceilings,
        "observed_vs_oracle": observed_vs_oracle,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("WANDS DEV STATISTICAL ANALYSIS")
    print("=" * 80)

    for name, metrics in comparisons.items():
        print()
        print(name)
        for metric, result in metrics.items():
            if result.get("n", 0) == 0:
                continue
            print(
                f"  {metric}: "
                f"Δ={result['mean_delta']:+.4f}, "
                f"95% CI=[{result['ci95_low']:+.4f}, "
                f"{result['ci95_high']:+.4f}], "
                f"P(Δ>0)={result['prob_delta_gt_0']:.3f}, "
                f"n={result['n']}"
            )

    print()
    print("ORACLE RECALL CEILINGS")
    for metric, result in ceilings.items():
        print(
            f"  {metric}: "
            f"{result['mean_oracle_recall']:.4f} "
            f"(n={result['n']})"
        )

    print()
    print("report:", output_path)


if __name__ == "__main__":
    main()
