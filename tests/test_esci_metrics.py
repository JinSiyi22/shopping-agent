from benchmarks.esci.metrics import (
    exact_hit_at_k,
    ndcg,
    recall_non_irrelevant_at_k,
    reciprocal_rank_exact,
)


def test_ndcg_ideal_is_one():
    labels = [
        "E",
        "C",
        "S",
        "I",
    ]

    assert ndcg(labels) == 1.0


def test_ndcg_penalizes_bad_order():
    ideal = [
        "E",
        "C",
        "S",
        "I",
    ]

    bad = [
        "I",
        "S",
        "C",
        "E",
    ]

    assert (
        ndcg(bad)
        < ndcg(ideal)
    )


def test_exact_mrr():
    labels = [
        "I",
        "C",
        "E",
        "S",
    ]

    assert (
        reciprocal_rank_exact(
            labels
        )
        == 1 / 3
    )


def test_exact_hit_at_k():
    labels = [
        "I",
        "C",
        "E",
        "S",
    ]

    assert (
        exact_hit_at_k(
            labels,
            2,
        )
        == 0.0
    )

    assert (
        exact_hit_at_k(
            labels,
            3,
        )
        == 1.0
    )


def test_recall_non_i():
    labels = [
        "E",
        "I",
        "C",
        "I",
        "S",
    ]

    assert (
        recall_non_irrelevant_at_k(
            labels,
            2,
        )
        == 1 / 3
    )

    assert (
        recall_non_irrelevant_at_k(
            labels,
            5,
        )
        == 1.0
    )
