from benchmarks.wands.metrics import (
    exact_mrr_at_k,
    exact_recall_at_k,
    ndcg_at_k,
    relevant_recall_at_k,
)


def test_ndcg_ideal_is_one():

    judged = [
        "Exact",
        "Partial",
        "Irrelevant",
    ]

    ranked = [
        "Exact",
        "Partial",
        "Irrelevant",
    ]

    assert (
        ndcg_at_k(
            ranked,
            judged,
            3,
        )
        == 1.0
    )


def test_ndcg_treats_unjudged_as_zero():

    judged = [
        "Exact",
        "Partial",
    ]

    ranked = [
        None,
        "Exact",
    ]

    score = (
        ndcg_at_k(
            ranked,
            judged,
            2,
        )
    )

    assert (
        0.0
        < score
        < 1.0
    )


def test_exact_metrics_skip_query_without_exact():

    judged = [
        "Partial",
        "Irrelevant",
    ]

    ranked = [
        "Partial",
        None,
    ]

    assert (
        exact_mrr_at_k(
            ranked,
            judged,
            10,
        )
        is None
    )

    assert (
        exact_recall_at_k(
            ranked,
            judged,
            10,
        )
        is None
    )


def test_exact_recall():

    judged = [
        "Exact",
        "Exact",
        "Partial",
    ]

    ranked = [
        "Exact",
        "Partial",
        None,
    ]

    assert (
        exact_recall_at_k(
            ranked,
            judged,
            3,
        )
        == 0.5
    )


def test_relevant_recall():

    judged = [
        "Exact",
        "Partial",
        "Irrelevant",
    ]

    ranked = [
        "Partial",
        None,
        "Irrelevant",
    ]

    assert (
        relevant_recall_at_k(
            ranked,
            judged,
            3,
        )
        == 0.5
    )
