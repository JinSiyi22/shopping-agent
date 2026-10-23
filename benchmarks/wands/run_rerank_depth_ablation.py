from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from benchmarks.wands.run_benchmark import (
    build_document,
    build_judgement_maps,
    compute_query_metrics,
    labels_for_ranking,
    load_or_build_embeddings,
    rrf_fuse,
    summarize_latency,
    tokenize,
    top_k_indices,
)
from benchmarks.wands.schema import read_wands_tsv
from benchmarks.wands.metrics import summarize_metrics


RERANK_DEPTHS = (10, 20, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-dir",
        default="external/WANDS/dataset",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/wands_processed",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
    )
    parser.add_argument(
        "--output",
        default="reports/wands_dev_rerank_depth.json",
    )
    parser.add_argument(
        "--per-query-output",
        default="reports/wands_dev_rerank_depth_per_query.csv",
    )
    parser.add_argument(
        "--dense-model",
        default="intfloat/multilingual-e5-small",
    )
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-base",
    )
    parser.add_argument(
        "--document-mode",
        choices=["name_only", "name_desc"],
        default="name_desc",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=32,
    )

    return parser.parse_args()


def rerank_prefix(
    base_order: np.ndarray,
    scores: np.ndarray,
    depth: int,
) -> np.ndarray:
    prefix = base_order[:depth]

    local_order = np.argsort(
        -scores,
        kind="stable",
    )

    reranked_prefix = prefix[
        local_order
    ]

    return np.concatenate(
        [
            reranked_prefix,
            base_order[depth:],
        ]
    )


def bootstrap_delta(
    a: np.ndarray,
    b: np.ndarray,
    *,
    samples: int = 10000,
    seed: int = 42,
) -> dict:
    mask = (
        ~np.isnan(a)
        & ~np.isnan(b)
    )

    delta = (
        a[mask]
        - b[mask]
    )

    n = len(delta)

    rng = np.random.default_rng(
        seed
    )

    means = np.empty(
        samples,
        dtype=np.float64,
    )

    for start in range(
        0,
        samples,
        1000,
    ):
        batch = min(
            1000,
            samples - start,
        )

        index = rng.integers(
            0,
            n,
            size=(batch, n),
        )

        means[
            start:
            start + batch
        ] = (
            delta[index]
            .mean(axis=1)
        )

    return {
        "n": int(n),
        "mean_delta": float(
            delta.mean()
        ),
        "ci95_low": float(
            np.percentile(
                means,
                2.5,
            )
        ),
        "ci95_high": float(
            np.percentile(
                means,
                97.5,
            )
        ),
        "prob_delta_gt_0": float(
            (
                means > 0
            ).mean()
        ),
    }


def main() -> None:
    args = parse_args()

    source_dir = Path(
        args.source_dir
    )

    processed_dir = Path(
        args.processed_dir
    )

    manifest = json.loads(
        (
            processed_dir
            / "split_manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    selected_ids = set(
        int(value)
        for value in manifest[
            "split"
        ][
            f"{args.split}_query_ids"
        ]
    )

    products = read_wands_tsv(
        source_dir
        / "product.csv"
    )

    queries = read_wands_tsv(
        source_dir
        / "query.csv"
    )

    labels = (
        __import__(
            "pandas"
        )
        .read_parquet(
            processed_dir
            / "labels_resolved.parquet"
        )
    )

    queries = (
        queries[
            queries[
                "query_id"
            ].isin(
                selected_ids
            )
        ]
        .copy()
        .sort_values(
            "query_id"
        )
    )

    labels = labels[
        labels[
            "query_id"
        ].isin(
            selected_ids
        )
    ].copy()

    products[
        "document"
    ] = products.apply(
        lambda row: build_document(
            row,
            mode=args.document_mode,
        ),
        axis=1,
    )

    product_ids = (
        products[
            "product_id"
        ]
        .astype(str)
        .tolist()
    )

    documents = (
        products[
            "document"
        ]
        .tolist()
    )

    print("=" * 72)
    print("WANDS RRF -> CROSSENCODER DEPTH ABLATION")
    print("=" * 72)
    print("split:", args.split)
    print("queries:", len(queries))
    print("products:", len(products))
    print("depths:", RERANK_DEPTHS)
    print("retrieve_k:", args.retrieve_k)
    print("rrf_k:", args.rrf_k)

    # --------------------------------------------------------
    # Global BM25
    # --------------------------------------------------------

    bm25_build_start = perf_counter()

    tokenized_documents = [
        tokenize(
            document
        )
        for document
        in documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    bm25_build_seconds = (
        perf_counter()
        - bm25_build_start
    )

    # --------------------------------------------------------
    # Dense model / cached product embeddings
    # --------------------------------------------------------

    dense_model = SentenceTransformer(
        args.dense_model
    )

    (
        product_embeddings,
        embedding_info,
    ) = load_or_build_embeddings(
        dense_model,
        documents=documents,
        product_ids=product_ids,
        processed_dir=processed_dir,
        model_name=args.dense_model,
        document_mode=args.document_mode,
        batch_size=args.embedding_batch_size,
    )

    query_records = (
        queries[
            [
                "query_id",
                "query",
            ]
        ]
        .to_dict(
            orient="records"
        )
    )

    query_encode_start = perf_counter()

    query_embeddings = dense_model.encode(
        [
            (
                "query: "
                f"{record['query']}"
            )
            for record
            in query_records
        ],
        batch_size=args.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    query_embedding_seconds = (
        perf_counter()
        - query_encode_start
    )

    reranker = CrossEncoder(
        args.reranker_model
    )

    judgement_maps = build_judgement_maps(
        labels
    )

    methods = [
        "rrf",
        *[
            f"rrf_ce_{depth}"
            for depth in RERANK_DEPTHS
        ],
    ]

    metric_rows = {
        method: []
        for method in methods
    }

    latency_rows = {
        depth: []
        for depth in RERANK_DEPTHS
    }

    per_query_rows = []

    for position, record in enumerate(
        query_records,
        start=1,
    ):
        query_id = int(
            record[
                "query_id"
            ]
        )

        query = str(
            record[
                "query"
            ]
        )

        judgement_map = (
            judgement_maps.get(
                query_id,
                {},
            )
        )

        judged_labels = list(
            judgement_map.values()
        )

        bm25_scores = np.asarray(
            bm25.get_scores(
                tokenize(
                    query
                )
            ),
            dtype=np.float64,
        )

        bm25_order = top_k_indices(
            bm25_scores,
            args.retrieve_k,
        )

        dense_scores = (
            product_embeddings
            @ query_embeddings[
                position - 1
            ]
        )

        dense_order = top_k_indices(
            dense_scores,
            args.retrieve_k,
        )

        rrf_order = rrf_fuse(
            [
                bm25_order,
                dense_order,
            ],
            top_k=args.retrieve_k,
            rrf_k=args.rrf_k,
        )

        rrf_ranked_labels = (
            labels_for_ranking(
                rrf_order,
                product_ids=product_ids,
                judgement_map=judgement_map,
            )
        )

        rrf_metrics = compute_query_metrics(
            query_id,
            ranked_labels=rrf_ranked_labels,
            judged_labels=judged_labels,
        )

        metric_rows[
            "rrf"
        ].append(
            rrf_metrics
        )

        flat_row = {
            "query_id": query_id,
            "query": query,
        }

        for key, value in rrf_metrics.items():
            if key != "query_id":
                flat_row[
                    f"rrf__{key}"
                ] = value

        for depth in RERANK_DEPTHS:
            actual_depth = min(
                depth,
                len(
                    rrf_order
                ),
            )

            prefix = rrf_order[
                :actual_depth
            ]

            pairs = [
                [
                    query,
                    documents[
                        int(index)
                    ],
                ]
                for index in prefix
            ]

            start = perf_counter()

            ce_scores = np.asarray(
                reranker.predict(
                    pairs,
                    batch_size=args.rerank_batch_size,
                    show_progress_bar=False,
                ),
                dtype=np.float64,
            ).reshape(-1)

            latency_rows[
                depth
            ].append(
                (
                    perf_counter()
                    - start
                )
                * 1000
            )

            reranked_order = rerank_prefix(
                rrf_order,
                ce_scores,
                actual_depth,
            )

            ranked_labels = labels_for_ranking(
                reranked_order,
                product_ids=product_ids,
                judgement_map=judgement_map,
            )

            method = (
                f"rrf_ce_{depth}"
            )

            metrics = compute_query_metrics(
                query_id,
                ranked_labels=ranked_labels,
                judged_labels=judged_labels,
            )

            metric_rows[
                method
            ].append(
                metrics
            )

            for key, value in metrics.items():
                if key != "query_id":
                    flat_row[
                        f"{method}__{key}"
                    ] = value

        per_query_rows.append(
            flat_row
        )

        if (
            position % 25 == 0
            or position
            == len(
                query_records
            )
        ):
            print(
                f"processed "
                f"{position}/"
                f"{len(query_records)} "
                "queries"
            )

    method_summary = {
        method: summarize_metrics(
            rows
        )
        for method, rows
        in metric_rows.items()
    }

    latency_summary = {
        f"ce_{depth}": summarize_latency(
            values
        )
        for depth, values
        in latency_rows.items()
    }

    # --------------------------------------------------------
    # Pre-declared quality comparisons versus CE@50
    # --------------------------------------------------------

    import pandas as pd

    per_query_frame = pd.DataFrame(
        per_query_rows
    )

    comparisons = {}

    for depth in (
        10,
        20,
    ):
        comparisons[
            f"ce_{depth}_vs_ce_50"
        ] = {}

        for metric in (
            "ndcg_at_10",
            "exact_mrr_at_10",
        ):
            a = (
                per_query_frame[
                    (
                        f"rrf_ce_{depth}"
                        f"__{metric}"
                    )
                ]
                .to_numpy(
                    dtype=float
                )
            )

            b = (
                per_query_frame[
                    (
                        "rrf_ce_50"
                        f"__{metric}"
                    )
                ]
                .to_numpy(
                    dtype=float
                )
            )

            comparisons[
                f"ce_{depth}_vs_ce_50"
            ][
                metric
            ] = bootstrap_delta(
                a,
                b,
                samples=10000,
                seed=42 + depth,
            )

    report = {
        "benchmark": (
            "WANDS RRF CrossEncoder "
            "rerank-depth ablation"
        ),
        "split": args.split,
        "selection_rule": (
            "Choose the smallest rerank depth whose "
            "mean nDCG@10 is within 0.005 absolute of "
            "the best depth, whose exact MRR@10 is "
            "within 0.01 absolute of the best depth, "
            "and whose paired-bootstrap 95% CI for "
            "nDCG@10 difference versus CE@50 crosses "
            "zero (no clear degradation)."
        ),
        "config": {
            "retrieve_k": args.retrieve_k,
            "rrf_k": args.rrf_k,
            "rerank_depths": list(
                RERANK_DEPTHS
            ),
            "document_mode": (
                args.document_mode
            ),
            "dense_model": (
                args.dense_model
            ),
            "reranker_model": (
                args.reranker_model
            ),
        },
        "methods": method_summary,
        "latency": latency_summary,
        "paired_bootstrap": comparisons,
        "setup": {
            "bm25_build_seconds": (
                bm25_build_seconds
            ),
            "embedding_cache": (
                embedding_info
            ),
            "query_embedding_seconds": (
                query_embedding_seconds
            ),
        },
    }

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    per_query_path = Path(
        args.per_query_output
    )

    per_query_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_query_frame.to_csv(
        per_query_path,
        index=False,
    )

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)

    for method, summary in method_summary.items():
        print()
        print(method)

        for metric in (
            "ndcg_at_10",
            "exact_mrr_at_10",
            "exact_recall_at_50",
            "relevant_recall_at_50",
        ):
            value = summary.get(
                metric
            )

            if value is not None:
                print(
                    f"  {metric}: "
                    f"{value:.4f}"
                )

    print()
    print("LATENCY")

    for depth in RERANK_DEPTHS:
        summary = latency_summary[
            f"ce_{depth}"
        ]

        print(
            f"  CE@{depth}: "
            f"mean={summary['mean_ms']:.1f} ms, "
            f"p95={summary['p95_ms']:.1f} ms"
        )

    print()
    print("PAIRED BOOTSTRAP VS CE@50")

    for name, metrics in comparisons.items():
        print()
        print(name)

        for metric, result in metrics.items():
            print(
                f"  {metric}: "
                f"Δ={result['mean_delta']:+.4f}, "
                f"95% CI=["
                f"{result['ci95_low']:+.4f}, "
                f"{result['ci95_high']:+.4f}]"
            )

    print()
    print("report:", output_path)
    print(
        "per-query:",
        per_query_path,
    )


if __name__ == "__main__":
    main()
