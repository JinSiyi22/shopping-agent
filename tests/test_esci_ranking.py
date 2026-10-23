import numpy as np

from benchmarks.esci.ranking import (
    deterministic_random_order,
    reciprocal_rank_fusion,
    rerank_prefix,
)

def test_random_order_is_reproducible():
    first = deterministic_random_order(
        10,
        seed=42,
        query_position=7,
    )
    second = deterministic_random_order(
        10,
        seed=42,
        query_position=7,
    )
    assert np.array_equal(first, second)

def test_random_order_is_permutation():
    order = deterministic_random_order(
        8,
        seed=42,
        query_position=1,
    )
    assert sorted(order.tolist()) == list(range(8))

def test_rrf_prefers_item_ranked_high_by_both():
    a = np.array([0, 1, 2])
    b = np.array([0, 2, 1])
    scores = reciprocal_rank_fusion(
        [a, b],
        n_items=3,
        rrf_k=60,
    )
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]

def test_rerank_prefix_only_changes_prefix():
    base = np.array([2, 0, 3, 1])
    scores = np.array([0.1, 0.9])
    final = rerank_prefix(
        base,
        scores,
        top_k=2,
    )
    assert final.tolist() == [0, 2, 3, 1]

def test_rerank_prefix_preserves_all_items():
    base = np.array([4, 1, 3, 0, 2])
    scores = np.array([0.2, 0.9, 0.3])
    final = rerank_prefix(
        base,
        scores,
        top_k=3,
    )
    assert sorted(final.tolist()) == sorted(base.tolist())

def test_rerank_prefix_rejects_bad_score_length():
    base = np.array([0, 1, 2])
    scores = np.array([1.0])

    try:
        rerank_prefix(
            base,
            scores,
            top_k=2,
        )
    except ValueError:
        return

    raise AssertionError("Expected ValueError")
