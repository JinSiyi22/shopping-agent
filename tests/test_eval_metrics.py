from app.evaluation.metrics import (
    accuracy,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_accuracy():

    result = accuracy(
        [
            True,
            True,
            False,
        ]
    )

    assert (
        result
        == 2 / 3
    )


def test_hit_at_k():

    result = hit_at_k(
        retrieved_ids=[
            "laptop_003",
            "laptop_001",
        ],
        relevant_ids=[
            "laptop_001",
        ],
        k=2,
    )

    assert (
        result
        == 1.0
    )


def test_recall_at_k():

    result = recall_at_k(
        retrieved_ids=[
            "laptop_001",
            "laptop_003",
        ],
        relevant_ids=[
            "laptop_001",
            "laptop_002",
        ],
        k=2,
    )

    assert (
        result
        == 0.5
    )


def test_reciprocal_rank():

    result = reciprocal_rank(
        retrieved_ids=[
            "laptop_003",
            "laptop_001",
            "laptop_002",
        ],
        relevant_ids=[
            "laptop_001",
        ],
    )

    assert (
        result
        == 0.5
    )


def test_reciprocal_rank_missing():

    result = reciprocal_rank(
        retrieved_ids=[
            "laptop_003",
        ],
        relevant_ids=[
            "laptop_001",
        ],
    )

    assert (
        result
        == 0.0
    )