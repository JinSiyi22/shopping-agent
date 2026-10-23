from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from benchmarks.esci.metrics import (
    exact_hit_at_k,
    ndcg,
    recall_non_irrelevant_at_k,
    reciprocal_rank_exact,
    summarize_query_metrics,
)
from benchmarks.esci.ranking import (
    deterministic_random_order,
    rank_desc,
    reciprocal_rank_fusion,
    rerank_prefix,
)

METHODS = (
    "random",
    "bm25",
    "dense",
    "rrf",
    "dense_cross_encoder",
    "rrf_cross_encoder",
    "cross_encoder_all",
)

LATENCY_SEMANTICS = {
    "random": "Random permutation generation only.",
    "bm25": (
        "Per-query BM25 index construction, candidate scoring, and sorting."
    ),
    "dense": (
        "Dense similarity scoring and sorting only; query/document encoding "
        "are excluded and reported separately."
    ),
    "rrf": (
        "RRF fusion and sorting only; BM25 and Dense prerequisite work "
        "are excluded."
    ),
    "dense_cross_encoder": (
        "CrossEncoder inference and prefix reorder for Dense top-k only; "
        "Dense candidate generation is excluded."
    ),
    "rrf_cross_encoder": (
        "CrossEncoder inference and prefix reorder for RRF top-k only; "
        "BM25, Dense, and RRF prerequisite work are excluded."
    ),
    "cross_encoder_all": (
        "CrossEncoder inference and sorting across all ESCI candidates."
    ),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/esci_processed/us_small_test.parquet",
    )
    parser.add_argument(
        "--output",
        default="reports/esci_benchmark.json",
    )
    parser.add_argument(
        "--per-query-output",
        default="reports/esci_benchmark_per_query.csv",
    )
    parser.add_argument(
        "--dense-model",
        default="intfloat/multilingual-e5-small",
    )
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-base",
    )
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument(
        "--skip-ce-all",
        action="store_true",
        help="Skip the expensive CrossEncoder-All ablation.",
    )
    return parser.parse_args()

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def clean_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()

def build_document(row: pd.Series) -> str:
    fields = [
        ("title", clean_text(row.get("product_title"))),
        ("brand", clean_text(row.get("product_brand"))),
        ("color", clean_text(row.get("product_color"))),
        ("bullet", clean_text(row.get("product_bullet_point"))),
        ("description", clean_text(row.get("product_description"))),
    ]
    parts = [f"{name}: {value}" for name, value in fields if value]
    return " | ".join(parts)[:2500]

def labels_from_order(labels: list[str], order: np.ndarray) -> list[str]:
    return [labels[int(index)] for index in order]

def query_metrics(query_id, ranked_labels: list[str]) -> dict:
    return {
        "query_id": query_id,
        "ndcg": ndcg(ranked_labels),
        "mrr_exact": reciprocal_rank_exact(ranked_labels),
        "exact_hit_at_1": exact_hit_at_k(ranked_labels, 1),
        "exact_hit_at_3": exact_hit_at_k(ranked_labels, 3),
        "recall_non_i_at_5": recall_non_irrelevant_at_k(ranked_labels, 5),
        "recall_non_i_at_10": recall_non_irrelevant_at_k(ranked_labels, 10),
    }

def choose_query_ids(
    frame: pd.DataFrame,
    *,
    max_queries: int,
    seed: int,
) -> list:
    query_ids = sorted(frame["query_id"].unique().tolist())
    if max_queries <= 0 or max_queries >= len(query_ids):
        return query_ids
    rng = np.random.default_rng(seed)
    selected = rng.choice(query_ids, size=max_queries, replace=False)
    return sorted(selected.tolist())

def predict_cross_encoder(
    reranker: CrossEncoder,
    *,
    query: str,
    documents: list[str],
    batch_size: int,
) -> tuple[np.ndarray, float]:
    pairs = [[query, document] for document in documents]
    start = perf_counter()
    scores = np.asarray(
        reranker.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        ),
        dtype=np.float64,
    ).reshape(-1)
    latency_ms = (perf_counter() - start) * 1000
    return scores, latency_ms

def summarize_latency(latencies: list[float]) -> dict[str, float]:
    values = np.asarray(latencies, dtype=np.float64)
    if len(values) == 0:
        return {"mean_method_ms": 0.0, "p95_method_ms": 0.0}
    return {
        "mean_method_ms": float(values.mean()),
        "p95_method_ms": float(np.percentile(values, 95)),
    }

def summarize_pair_counts(pair_counts: list[int]) -> dict[str, float]:
    if not pair_counts:
        return {}
    values = np.asarray(pair_counts, dtype=np.float64)
    return {
        "mean_pairs_scored": float(values.mean()),
        "p95_pairs_scored": float(np.percentile(values, 95)),
    }

def main() -> None:
    args = parse_args()

    if args.candidate_k <= 0:
        raise ValueError("--candidate-k must be > 0")
    if args.rrf_k <= 0:
        raise ValueError("--rrf-k must be > 0")

    frame = pd.read_parquet(Path(args.data))
    query_ids = choose_query_ids(
        frame,
        max_queries=args.max_queries,
        seed=args.seed,
    )
    frame = frame[frame["query_id"].isin(query_ids)].copy()
    frame["document"] = frame.apply(build_document, axis=1)

    product_table = (
        frame[["product_id", "document"]]
        .drop_duplicates(subset=["product_id"], keep="first")
        .reset_index(drop=True)
    )
    product_ids = product_table["product_id"].astype(str).tolist()
    documents = product_table["document"].tolist()
    product_to_position = {
        product_id: index
        for index, product_id in enumerate(product_ids)
    }

    print("=" * 72)
    print("ESCI TASK-1 CANDIDATE-RANKING ABLATION V2")
    print("=" * 72)
    print("queries:", len(query_ids))
    print("judgements:", len(frame))
    print("unique products:", len(product_ids))
    print("candidate_k:", args.candidate_k)
    print("rrf_k:", args.rrf_k)
    print(
        "cross_encoder_all:",
        "disabled" if args.skip_ce_all else "enabled",
    )

    dense_load_start = perf_counter()
    dense_model = SentenceTransformer(args.dense_model)
    dense_model_load_seconds = perf_counter() - dense_load_start

    document_encode_start = perf_counter()
    document_embeddings = dense_model.encode(
        [f"passage: {document}" for document in documents],
        batch_size=args.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    document_embedding_seconds = perf_counter() - document_encode_start

    query_text_by_id = (
        frame[["query_id", "query"]]
        .drop_duplicates(subset=["query_id"], keep="first")
        .set_index("query_id")["query"]
        .to_dict()
    )
    query_texts = [str(query_text_by_id[qid]) for qid in query_ids]

    query_encode_start = perf_counter()
    query_embeddings = dense_model.encode(
        [f"query: {query}" for query in query_texts],
        batch_size=args.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    query_embedding_seconds = perf_counter() - query_encode_start

    query_embedding_by_id = {
        query_id: query_embeddings[index]
        for index, query_id in enumerate(query_ids)
    }

    reranker_load_start = perf_counter()
    reranker = CrossEncoder(args.reranker_model)
    reranker_model_load_seconds = perf_counter() - reranker_load_start

    active_methods = [
        method
        for method in METHODS
        if not (args.skip_ce_all and method == "cross_encoder_all")
    ]
    method_metric_rows = {method: [] for method in active_methods}
    method_latencies = {method: [] for method in active_methods}
    method_pair_counts = {
        "dense_cross_encoder": [],
        "rrf_cross_encoder": [],
    }
    if not args.skip_ce_all:
        method_pair_counts["cross_encoder_all"] = []

    per_query_rows = []
    grouped = {
        query_id: group.copy()
        for query_id, group in frame.groupby("query_id", sort=False)
    }

    for query_position, query_id in enumerate(query_ids, start=1):
        group = grouped[query_id].reset_index(drop=True)
        query = str(group.loc[0, "query"])
        labels = group["esci_label"].astype(str).tolist()
        candidate_product_ids = group["product_id"].astype(str).tolist()
        candidate_docs = group["document"].tolist()
        n_candidates = len(candidate_docs)

        start = perf_counter()
        random_order = deterministic_random_order(
            n_candidates,
            seed=args.seed,
            query_position=query_position,
        )
        method_latencies["random"].append(
            (perf_counter() - start) * 1000
        )

        start = perf_counter()
        tokenized_docs = [tokenize(document) for document in candidate_docs]
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = np.asarray(
            bm25.get_scores(tokenize(query)),
            dtype=np.float64,
        )
        bm25_order = rank_desc(bm25_scores)
        method_latencies["bm25"].append(
            (perf_counter() - start) * 1000
        )

        start = perf_counter()
        global_positions = [
            product_to_position[product_id]
            for product_id in candidate_product_ids
        ]
        candidate_embeddings = document_embeddings[global_positions]
        query_embedding = query_embedding_by_id[query_id]
        dense_scores = candidate_embeddings @ query_embedding
        dense_order = rank_desc(dense_scores)
        method_latencies["dense"].append(
            (perf_counter() - start) * 1000
        )

        start = perf_counter()
        rrf_scores = reciprocal_rank_fusion(
            [bm25_order, dense_order],
            n_items=n_candidates,
            rrf_k=args.rrf_k,
        )
        rrf_order = rank_desc(rrf_scores)
        method_latencies["rrf"].append(
            (perf_counter() - start) * 1000
        )

        dense_k = min(args.candidate_k, n_candidates)
        dense_top_indices = dense_order[:dense_k]
        dense_top_docs = [
            candidate_docs[int(index)]
            for index in dense_top_indices
        ]
        dense_ce_scores, dense_ce_ms = predict_cross_encoder(
            reranker,
            query=query,
            documents=dense_top_docs,
            batch_size=args.rerank_batch_size,
        )
        dense_ce_order = rerank_prefix(
            dense_order,
            dense_ce_scores,
            top_k=dense_k,
        )
        method_latencies["dense_cross_encoder"].append(dense_ce_ms)
        method_pair_counts["dense_cross_encoder"].append(dense_k)

        rrf_candidate_k = min(args.candidate_k, n_candidates)
        rrf_top_indices = rrf_order[:rrf_candidate_k]
        rrf_top_docs = [
            candidate_docs[int(index)]
            for index in rrf_top_indices
        ]
        rrf_ce_scores, rrf_ce_ms = predict_cross_encoder(
            reranker,
            query=query,
            documents=rrf_top_docs,
            batch_size=args.rerank_batch_size,
        )
        rrf_ce_order = rerank_prefix(
            rrf_order,
            rrf_ce_scores,
            top_k=rrf_candidate_k,
        )
        method_latencies["rrf_cross_encoder"].append(rrf_ce_ms)
        method_pair_counts["rrf_cross_encoder"].append(rrf_candidate_k)

        ce_all_order = None
        if not args.skip_ce_all:
            ce_all_scores, ce_all_ms = predict_cross_encoder(
                reranker,
                query=query,
                documents=candidate_docs,
                batch_size=args.rerank_batch_size,
            )
            ce_all_order = rank_desc(ce_all_scores)
            method_latencies["cross_encoder_all"].append(ce_all_ms)
            method_pair_counts["cross_encoder_all"].append(n_candidates)

        orders = {
            "random": random_order,
            "bm25": bm25_order,
            "dense": dense_order,
            "rrf": rrf_order,
            "dense_cross_encoder": dense_ce_order,
            "rrf_cross_encoder": rrf_ce_order,
        }
        if ce_all_order is not None:
            orders["cross_encoder_all"] = ce_all_order

        flat_row = {
            "query_id": query_id,
            "query": query,
            "candidate_count": n_candidates,
        }

        for method, order in orders.items():
            ranked_labels = labels_from_order(labels, order)
            metrics = query_metrics(query_id, ranked_labels)
            method_metric_rows[method].append(metrics)

            for metric_name, metric_value in metrics.items():
                if metric_name == "query_id":
                    continue
                flat_row[f"{method}__{metric_name}"] = metric_value

        per_query_rows.append(flat_row)

        if (
            query_position % 100 == 0
            or query_position == len(query_ids)
        ):
            print(
                f"processed {query_position}/{len(query_ids)} queries"
            )

    method_summaries = {}

    for method in active_methods:
        summary = summarize_query_metrics(method_metric_rows[method])
        summary.update(summarize_latency(method_latencies[method]))

        if method in method_pair_counts:
            summary.update(
                summarize_pair_counts(method_pair_counts[method])
            )

        summary["latency_semantics"] = LATENCY_SEMANTICS[method]
        method_summaries[method] = summary

    average_candidates = float(
        frame.groupby("query_id").size().mean()
    )

    report = {
        "benchmark": "Amazon ESCI Task 1 candidate ranking",
        "benchmark_version": "day19_v2",
        "scope": {
            "locale": "us",
            "version": "small",
            "split": "test",
            "queries": len(query_ids),
            "judgements": len(frame),
            "unique_products": len(product_ids),
            "average_candidates_per_query": average_candidates,
        },
        "quality_metric_note": (
            "Primary nDCG follows the official ESCI Task-1 gain "
            "configuration: E=1, C=0.1, S=0.01, I=0. "
            "MRR/Hit treat Exact as target; Recall treats all non-I "
            "labels as relevant. Supplemental metrics are not "
            "substitutes for official nDCG."
        ),
        "latency_note": (
            "Timings are component-level, not full end-to-end latency. "
            "Dense scoring excludes offline query/document encoding. "
            "CrossEncoder timings include model inference for the "
            "reported number of pairs but exclude upstream generation."
        ),
        "config": {
            "dense_model": args.dense_model,
            "reranker_model": args.reranker_model,
            "candidate_k": args.candidate_k,
            "rrf_k": args.rrf_k,
            "seed": args.seed,
            "skip_ce_all": args.skip_ce_all,
        },
        "setup": {
            "dense_model_load_seconds": dense_model_load_seconds,
            "dense_document_embedding_seconds": (
                document_embedding_seconds
            ),
            "dense_query_embedding_seconds": query_embedding_seconds,
            "dense_query_embedding_mean_ms": (
                query_embedding_seconds * 1000
                / max(len(query_ids), 1)
            ),
            "reranker_model_load_seconds": reranker_model_load_seconds,
        },
        "methods": method_summaries,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    per_query_path = Path(args.per_query_output)
    per_query_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_query_rows).to_csv(per_query_path, index=False)

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)

    for method, summary in method_summaries.items():
        print()
        print(method)
        print("  nDCG:", round(summary["ndcg"], 4))
        print("  MRR(E):", round(summary["mrr_exact"], 4))
        print("  Hit@1(E):", round(summary["exact_hit_at_1"], 4))
        print(
            "  Recall@10(non-I):",
            round(summary["recall_non_i_at_10"], 4),
        )
        print(
            "  mean method ms:",
            round(summary["mean_method_ms"], 3),
        )
        if "mean_pairs_scored" in summary:
            print(
                "  mean CE pairs:",
                round(summary["mean_pairs_scored"], 2),
            )

    print()
    print("Dense setup:")
    print(
        "  document embedding s:",
        round(document_embedding_seconds, 3),
    )
    print(
        "  query embedding s:",
        round(query_embedding_seconds, 3),
    )
    print(
        "  mean query embedding ms:",
        round(report["setup"]["dense_query_embedding_mean_ms"], 3),
    )
    print()
    print("report:", output_path)
    print("per-query:", per_query_path)

if __name__ == "__main__":
    main()
