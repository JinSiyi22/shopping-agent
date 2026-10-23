from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer,
)

from benchmarks.wands.schema import read_wands_tsv

from benchmarks.wands.metrics import (
    exact_mrr_at_k,
    exact_recall_at_k,
    ndcg_at_k,
    relevant_recall_at_k,
    summarize_metrics,
)


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
        choices=[
            "dev",
            "test",
        ],
        default="dev",
    )

    parser.add_argument(
        "--output",
        default="reports/wands_dev.json",
    )

    parser.add_argument(
        "--per-query-output",
        default="reports/wands_dev_per_query.csv",
    )

    parser.add_argument(
        "--dense-model",
        default=(
            "intfloat/"
            "multilingual-e5-small"
        ),
    )

    parser.add_argument(
        "--reranker-model",
        default=(
            "BAAI/"
            "bge-reranker-base"
        ),
    )

    parser.add_argument(
        "--document-mode",
        choices=[
            "name_only",
            "name_desc",
        ],
        default="name_desc",
    )

    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--rerank-k",
        type=int,
        default=50,
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

    parser.add_argument(
        "--skip-cross-encoder",
        action="store_true",
    )

    return parser.parse_args()


def clean_text(
    value,
) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(
            value
        ):
            return ""
    except (
        TypeError,
        ValueError,
    ):
        pass

    return (
        str(value)
        .strip()
    )


def tokenize(
    text: str,
) -> list[str]:
    return re.findall(
        r"[a-z0-9]+",
        text.lower(),
    )


def build_document(
    row: pd.Series,
    *,
    mode: str,
) -> str:
    name = clean_text(
        row.get(
            "product_name"
        )
    )

    if mode == "name_only":
        return name

    fields = [
        (
            "name",
            name,
        ),
        (
            "class",
            clean_text(
                row.get(
                    "product_class"
                )
            ),
        ),
        (
            "category",
            clean_text(
                row.get(
                    "category_hierarchy"
                )
            ),
        ),
        (
            "description",
            clean_text(
                row.get(
                    "product_description"
                )
            ),
        ),
        (
            "features",
            clean_text(
                row.get(
                    "product_features"
                )
            ),
        ),
    ]

    parts = [
        f"{key}: {value}"
        for key, value
        in fields
        if value
    ]

    return " | ".join(
        parts
    )[:2200]


def top_k_indices(
    scores: np.ndarray,
    k: int,
) -> np.ndarray:
    k = min(
        k,
        len(
            scores
        ),
    )

    if k <= 0:
        return np.asarray(
            [],
            dtype=np.int64,
        )

    if k == len(
        scores
    ):
        return np.argsort(
            -scores,
            kind="stable",
        )

    partition = np.argpartition(
        -scores,
        k - 1,
    )[:k]

    local = np.argsort(
        -scores[
            partition
        ],
        kind="stable",
    )

    return partition[
        local
    ]


def rrf_fuse(
    rankings: list[np.ndarray],
    *,
    top_k: int,
    rrf_k: int,
) -> np.ndarray:
    scores: dict[
        int,
        float,
    ] = {}

    first_seen: dict[
        int,
        int,
    ] = {}

    seen_counter = 0

    for ranking in rankings:
        for rank, item in enumerate(
            ranking,
            start=1,
        ):
            item_int = int(
                item
            )

            if item_int not in first_seen:
                first_seen[
                    item_int
                ] = seen_counter
                seen_counter += 1

            scores[
                item_int
            ] = (
                scores.get(
                    item_int,
                    0.0,
                )
                + (
                    1.0
                    / (
                        rrf_k
                        + rank
                    )
                )
            )

    ordered = sorted(
        scores,
        key=lambda item: (
            -scores[
                item
            ],
            first_seen[
                item
            ],
        ),
    )

    return np.asarray(
        ordered[
            :top_k
        ],
        dtype=np.int64,
    )


def embedding_cache_paths(
    processed_dir: Path,
    *,
    model_name: str,
    document_mode: str,
) -> tuple[
    Path,
    Path,
]:
    key = (
        f"{model_name}|"
        f"{document_mode}"
    )

    digest = (
        hashlib.sha256(
            key.encode(
                "utf-8"
            )
        )
        .hexdigest()[
            :12
        ]
    )

    return (
        processed_dir
        / (
            "product_embeddings_"
            f"{digest}.npy"
        ),
        processed_dir
        / (
            "product_embeddings_"
            f"{digest}.json"
        ),
    )


def load_or_build_embeddings(
    model: SentenceTransformer,
    *,
    documents: list[str],
    product_ids: list[str],
    processed_dir: Path,
    model_name: str,
    document_mode: str,
    batch_size: int,
) -> tuple[
    np.ndarray,
    dict,
]:
    (
        array_path,
        metadata_path,
    ) = embedding_cache_paths(
        processed_dir,
        model_name=model_name,
        document_mode=document_mode,
    )

    expected_meta = {
        "model_name": (
            model_name
        ),
        "document_mode": (
            document_mode
        ),
        "product_count": (
            len(
                product_ids
            )
        ),
        "product_ids_hash": (
            hashlib.sha256(
                "\n".join(
                    product_ids
                )
                .encode(
                    "utf-8"
                )
            )
            .hexdigest()
        ),
    }

    if (
        array_path.exists()
        and metadata_path.exists()
    ):
        current_meta = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if current_meta == expected_meta:
            return (
                np.load(
                    array_path,
                ),
                {
                    "cache_hit": (
                        True
                    ),
                    "build_seconds": (
                        0.0
                    ),
                    "path": str(
                        array_path
                    ),
                },
            )

    start = perf_counter()

    embeddings = model.encode(
        [
            f"passage: {document}"
            for document
            in documents
        ],
        batch_size=(
            batch_size
        ),
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    build_seconds = (
        perf_counter()
        - start
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        array_path,
        embeddings,
    )

    metadata_path.write_text(
        json.dumps(
            expected_meta,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        embeddings,
        {
            "cache_hit": (
                False
            ),
            "build_seconds": (
                build_seconds
            ),
            "path": str(
                array_path
            ),
        },
    )


def build_judgement_maps(
    labels: pd.DataFrame,
) -> dict[
    int,
    dict[
        str,
        str,
    ],
]:
    result = {}

    for (
        query_id,
        group,
    ) in labels.groupby(
        "query_id",
        sort=False,
    ):
        result[
            int(
                query_id
            )
        ] = {
            str(
                product_id
            ): str(
                label
            )
            for (
                product_id,
                label,
            ) in zip(
                group[
                    "product_id"
                ],
                group[
                    "label"
                ],
            )
        }

    return result


def labels_for_ranking(
    order: np.ndarray,
    *,
    product_ids: list[str],
    judgement_map: dict[
        str,
        str,
    ],
) -> list[
    str | None
]:
    return [
        judgement_map.get(
            product_ids[
                int(
                    index
                )
            ]
        )
        for index
        in order
    ]


def compute_query_metrics(
    query_id: int,
    *,
    ranked_labels: list[
        str | None
    ],
    judged_labels: list[str],
) -> dict:
    return {
        "query_id": (
            query_id
        ),
        "ndcg_at_10": (
            ndcg_at_k(
                ranked_labels,
                judged_labels,
                10,
            )
        ),
        "ndcg_at_20": (
            ndcg_at_k(
                ranked_labels,
                judged_labels,
                20,
            )
        ),
        "exact_mrr_at_10": (
            exact_mrr_at_k(
                ranked_labels,
                judged_labels,
                10,
            )
        ),
        "exact_recall_at_10": (
            exact_recall_at_k(
                ranked_labels,
                judged_labels,
                10,
            )
        ),
        "exact_recall_at_50": (
            exact_recall_at_k(
                ranked_labels,
                judged_labels,
                50,
            )
        ),
        "relevant_recall_at_10": (
            relevant_recall_at_k(
                ranked_labels,
                judged_labels,
                10,
            )
        ),
        "relevant_recall_at_50": (
            relevant_recall_at_k(
                ranked_labels,
                judged_labels,
                50,
            )
        ),
    }


def rerank_two_candidate_sets(
    reranker: CrossEncoder,
    *,
    query: str,
    documents: list[str],
    dense_order: np.ndarray,
    rrf_order: np.ndarray,
    rerank_k: int,
    batch_size: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    int,
]:
    dense_prefix = dense_order[
        : min(
            rerank_k,
            len(
                dense_order
            ),
        )
    ]

    rrf_prefix = rrf_order[
        : min(
            rerank_k,
            len(
                rrf_order
            ),
        )
    ]

    union = []

    seen = set()

    for item in np.concatenate(
        [
            dense_prefix,
            rrf_prefix,
        ]
    ):
        item_int = int(
            item
        )

        if item_int in seen:
            continue

        seen.add(
            item_int
        )

        union.append(
            item_int
        )

    pairs = [
        [
            query,
            documents[
                item
            ],
        ]
        for item
        in union
    ]

    start = perf_counter()

    scores = np.asarray(
        reranker.predict(
            pairs,
            batch_size=(
                batch_size
            ),
            show_progress_bar=False,
        ),
        dtype=np.float64,
    ).reshape(-1)

    latency_ms = (
        (
            perf_counter()
            - start
        )
        * 1000
    )

    score_by_item = {
        item: float(
            score
        )
        for item, score
        in zip(
            union,
            scores,
        )
    }

    def rerank(
        base_order: np.ndarray,
    ) -> np.ndarray:
        prefix = base_order[
            : min(
                rerank_k,
                len(
                    base_order
                ),
            )
        ]

        prefix_sorted = sorted(
            [
                int(
                    item
                )
                for item
                in prefix
            ],
            key=lambda item: (
                -score_by_item[
                    item
                ]
            ),
        )

        return np.concatenate(
            [
                np.asarray(
                    prefix_sorted,
                    dtype=np.int64,
                ),
                base_order[
                    len(
                        prefix
                    ):
                ],
            ]
        )

    return (
        rerank(
            dense_order
        ),
        rerank(
            rrf_order
        ),
        latency_ms,
        len(
            union
        ),
    )


def summarize_latency(
    values: list[float],
) -> dict:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean_ms": float(
            array.mean()
        ),
        "p95_ms": float(
            np.percentile(
                array,
                95,
            )
        ),
    }


def main() -> None:
    args = parse_args()

    if (
        args.rerank_k
        > args.retrieve_k
    ):
        raise ValueError(
            "--rerank-k must be <= "
            "--retrieve-k"
        )

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

    split_key = (
        f"{args.split}_query_ids"
    )

    selected_query_ids = set(
        int(
            query_id
        )
        for query_id
        in manifest[
            "split"
        ][
            split_key
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

    labels = pd.read_parquet(
        processed_dir
        / "labels_resolved.parquet"
    )

    queries = queries[
        queries[
            "query_id"
        ].isin(
            selected_query_ids
        )
    ].copy()

    labels = labels[
        labels[
            "query_id"
        ].isin(
            selected_query_ids
        )
    ].copy()

    products[
        "document"
    ] = products.apply(
        lambda row: (
            build_document(
                row,
                mode=(
                    args.document_mode
                ),
            )
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

    print(
        "=" * 72
    )

    print(
        "WANDS FULL-CORPUS "
        "RETRIEVAL BENCHMARK"
    )

    print(
        "=" * 72
    )

    print(
        "split:",
        args.split,
    )

    print(
        "queries:",
        len(
            queries
        ),
    )

    print(
        "products:",
        len(
            products
        ),
    )

    print(
        "document_mode:",
        args.document_mode,
    )

    print(
        "retrieve_k:",
        args.retrieve_k,
    )

    print(
        "rerank_k:",
        args.rerank_k,
    )

    # ========================================================
    # BM25 global corpus
    # ========================================================

    bm25_build_start = (
        perf_counter()
    )

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

    # ========================================================
    # Dense embeddings
    # ========================================================

    dense_load_start = (
        perf_counter()
    )

    dense_model = (
        SentenceTransformer(
            args.dense_model
        )
    )

    dense_model_load_seconds = (
        perf_counter()
        - dense_load_start
    )

    (
        product_embeddings,
        embedding_info,
    ) = (
        load_or_build_embeddings(
            dense_model,
            documents=documents,
            product_ids=(
                product_ids
            ),
            processed_dir=(
                processed_dir
            ),
            model_name=(
                args.dense_model
            ),
            document_mode=(
                args.document_mode
            ),
            batch_size=(
                args.embedding_batch_size
            ),
        )
    )

    query_records = (
        queries[
            [
                "query_id",
                "query",
            ]
        ]
        .sort_values(
            "query_id"
        )
        .to_dict(
            orient="records"
        )
    )

    query_encode_start = (
        perf_counter()
    )

    query_embeddings = (
        dense_model.encode(
            [
                (
                    "query: "
                    f"{record['query']}"
                )
                for record
                in query_records
            ],
            batch_size=(
                args.embedding_batch_size
            ),
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    )

    query_embedding_seconds = (
        perf_counter()
        - query_encode_start
    )

    reranker = None

    reranker_load_seconds = (
        0.0
    )

    if not (
        args.skip_cross_encoder
    ):
        start = (
            perf_counter()
        )

        reranker = (
            CrossEncoder(
                args.reranker_model
            )
        )

        reranker_load_seconds = (
            perf_counter()
            - start
        )

    judgement_maps = (
        build_judgement_maps(
            labels
        )
    )

    methods = [
        "bm25",
        "dense",
        "rrf",
    ]

    if (
        reranker
        is not None
    ):
        methods.extend(
            [
                "dense_ce",
                "rrf_ce",
            ]
        )

    metric_rows = {
        method: []
        for method
        in methods
    }

    latency_rows = {
        "bm25": [],
        "dense": [],
        "rrf": [],
    }

    if (
        reranker
        is not None
    ):
        latency_rows[
            "cross_encoder_shared"
        ] = []

        ce_pair_counts = []
    else:
        ce_pair_counts = []

    per_query_rows = []

    for (
        query_position,
        record,
    ) in enumerate(
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

        # ---------------- BM25 ----------------

        start = (
            perf_counter()
        )

        bm25_scores = np.asarray(
            bm25.get_scores(
                tokenize(
                    query
                )
            ),
            dtype=np.float64,
        )

        bm25_order = (
            top_k_indices(
                bm25_scores,
                args.retrieve_k,
            )
        )

        latency_rows[
            "bm25"
        ].append(
            (
                perf_counter()
                - start
            )
            * 1000
        )

        # ---------------- Dense ----------------

        start = (
            perf_counter()
        )

        dense_scores = (
            product_embeddings
            @ query_embeddings[
                query_position
                - 1
            ]
        )

        dense_order = (
            top_k_indices(
                dense_scores,
                args.retrieve_k,
            )
        )

        latency_rows[
            "dense"
        ].append(
            (
                perf_counter()
                - start
            )
            * 1000
        )

        # ---------------- RRF ----------------

        start = (
            perf_counter()
        )

        rrf_order = (
            rrf_fuse(
                [
                    bm25_order,
                    dense_order,
                ],
                top_k=(
                    args.retrieve_k
                ),
                rrf_k=(
                    args.rrf_k
                ),
            )
        )

        latency_rows[
            "rrf"
        ].append(
            (
                perf_counter()
                - start
            )
            * 1000
        )

        orders = {
            "bm25": (
                bm25_order
            ),
            "dense": (
                dense_order
            ),
            "rrf": (
                rrf_order
            ),
        }

        # -------- shared CrossEncoder scores --------

        if (
            reranker
            is not None
        ):
            (
                dense_ce_order,
                rrf_ce_order,
                ce_latency_ms,
                ce_pairs,
            ) = (
                rerank_two_candidate_sets(
                    reranker,
                    query=query,
                    documents=documents,
                    dense_order=(
                        dense_order
                    ),
                    rrf_order=(
                        rrf_order
                    ),
                    rerank_k=(
                        args.rerank_k
                    ),
                    batch_size=(
                        args.rerank_batch_size
                    ),
                )
            )

            orders[
                "dense_ce"
            ] = (
                dense_ce_order
            )

            orders[
                "rrf_ce"
            ] = (
                rrf_ce_order
            )

            latency_rows[
                "cross_encoder_shared"
            ].append(
                ce_latency_ms
            )

            ce_pair_counts.append(
                ce_pairs
            )

        flat_row = {
            "query_id": (
                query_id
            ),
            "query": (
                query
            ),
            "judged_count": (
                len(
                    judged_labels
                )
            ),
        }

        for (
            method,
            order,
        ) in orders.items():
            ranked_labels = (
                labels_for_ranking(
                    order,
                    product_ids=(
                        product_ids
                    ),
                    judgement_map=(
                        judgement_map
                    ),
                )
            )

            metrics = (
                compute_query_metrics(
                    query_id,
                    ranked_labels=(
                        ranked_labels
                    ),
                    judged_labels=(
                        judged_labels
                    ),
                )
            )

            metric_rows[
                method
            ].append(
                metrics
            )

            for (
                key,
                value,
            ) in metrics.items():
                if (
                    key
                    == "query_id"
                ):
                    continue

                flat_row[
                    (
                        f"{method}__"
                        f"{key}"
                    )
                ] = (
                    value
                )

        per_query_rows.append(
            flat_row
        )

        if (
            query_position % 25
            == 0
            or query_position
            == len(
                query_records
            )
        ):
            print(
                "processed "
                f"{query_position}/"
                f"{len(query_records)} "
                "queries"
            )

    method_summary = {
        method: (
            summarize_metrics(
                rows
            )
        )
        for method, rows
        in metric_rows.items()
    }

    latency_summary = {
        name: (
            summarize_latency(
                values
            )
        )
        for name, values
        in latency_rows.items()
    }

    if (
        ce_pair_counts
    ):
        ce_pair_array = (
            np.asarray(
                ce_pair_counts,
                dtype=np.float64,
            )
        )

        latency_summary[
            "cross_encoder_shared"
        ][
            "mean_pairs"
        ] = float(
            ce_pair_array.mean()
        )

        latency_summary[
            "cross_encoder_shared"
        ][
            "p95_pairs"
        ] = float(
            np.percentile(
                ce_pair_array,
                95,
            )
        )

    report = {
        "benchmark": (
            "Wayfair WANDS "
            "full-corpus retrieval"
        ),
        "split": (
            args.split
        ),
        "scope": {
            "queries": (
                len(
                    query_records
                )
            ),
            "products": (
                len(
                    products
                )
            ),
            "judged_pairs": (
                len(
                    labels
                )
            ),
        },
        "config": {
            "document_mode": (
                args.document_mode
            ),
            "dense_model": (
                args.dense_model
            ),
            "reranker_model": (
                None
                if reranker
                is None
                else (
                    args.reranker_model
                )
            ),
            "retrieve_k": (
                args.retrieve_k
            ),
            "rerank_k": (
                args.rerank_k
            ),
            "rrf_k": (
                args.rrf_k
            ),
        },
        "evaluation_note": (
            "All retrieval methods search the same "
            "full 42,994-product corpus. nDCG uses "
            "graded WANDS judgments: Exact=2, "
            "Partial=1, Irrelevant=0. Unjudged "
            "retrieved products receive zero gain, "
            "so recall metrics should be interpreted "
            "as recall of known judged positives, "
            "not complete true relevance."
        ),
        "split_note": (
            "Architecture choices must be made on "
            "the dev query split only. The test split "
            "is reserved for one final evaluation."
        ),
        "methods": (
            method_summary
        ),
        "latency": (
            latency_summary
        ),
        "setup": {
            "bm25_build_seconds": (
                bm25_build_seconds
            ),
            "dense_model_load_seconds": (
                dense_model_load_seconds
            ),
            "embedding_cache": (
                embedding_info
            ),
            "query_embedding_seconds": (
                query_embedding_seconds
            ),
            "mean_query_embedding_ms": (
                query_embedding_seconds
                * 1000
                / max(
                    len(
                        query_records
                    ),
                    1,
                )
            ),
            "reranker_load_seconds": (
                reranker_load_seconds
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

    pd.DataFrame(
        per_query_rows
    ).to_csv(
        per_query_path,
        index=False,
    )

    print()

    print(
        "=" * 72
    )

    print(
        "RESULTS"
    )

    print(
        "=" * 72
    )

    for (
        method,
        summary,
    ) in method_summary.items():
        print()

        print(
            method
        )

        for metric_name in (
            "ndcg_at_10",
            "ndcg_at_20",
            "exact_mrr_at_10",
            "exact_recall_at_10",
            "exact_recall_at_50",
            "relevant_recall_at_10",
            "relevant_recall_at_50",
        ):
            value = (
                summary.get(
                    metric_name
                )
            )

            count = (
                summary.get(
                    f"{metric_name}__n"
                )
            )

            if (
                value
                is None
            ):
                continue

            print(
                f"  {metric_name}: "
                f"{value:.4f} "
                f"(n={count})"
            )

    print()

    print(
        "latency:",
        json.dumps(
            latency_summary,
            ensure_ascii=False,
            indent=2,
        ),
    )

    print()

    print(
        "report:",
        output_path,
    )

    print(
        "per-query:",
        per_query_path,
    )


if __name__ == "__main__":
    main()
