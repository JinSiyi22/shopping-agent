from __future__ import annotations
import numpy as np

def rank_desc(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, kind="stable")

def reciprocal_rank_fusion(
    rankings: list[np.ndarray],
    *,
    n_items: int,
    rrf_k: int,
) -> np.ndarray:
    scores = np.zeros(n_items, dtype=np.float64)
    for ranking in rankings:
        for rank, item_index in enumerate(ranking, start=1):
            scores[int(item_index)] += 1.0 / (rrf_k + rank)
    return scores

def deterministic_random_order(
    n_items: int,
    *,
    seed: int,
    query_position: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + query_position)
    return rng.permutation(n_items)

def rerank_prefix(
    base_order: np.ndarray,
    rerank_scores: np.ndarray,
    *,
    top_k: int,
) -> np.ndarray:
    candidate_k = min(top_k, len(base_order))
    prefix = base_order[:candidate_k]
    scores = np.asarray(rerank_scores, dtype=np.float64).reshape(-1)
    if len(scores) != candidate_k:
        raise ValueError(
            "rerank_scores length must equal the reranked prefix length."
        )
    local_order = rank_desc(scores)
    reranked_prefix = np.asarray(
        [prefix[int(i)] for i in local_order],
        dtype=np.int64,
    )
    return np.concatenate([reranked_prefix, base_order[candidate_k:]])
