from __future__ import annotations


def accuracy(
    values: list[bool],
) -> float:

    if not values:
        return 0.0

    return (
        sum(values)
        / len(values)
    )


def hit_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:

    if not relevant_ids:
        return 0.0

    top_k = set(
        retrieved_ids[:k]
    )

    relevant = set(
        relevant_ids
    )

    return float(
        bool(
            top_k
            & relevant
        )
    )


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:

    if not relevant_ids:
        return 0.0

    top_k = set(
        retrieved_ids[:k]
    )

    relevant = set(
        relevant_ids
    )

    matched = (
        top_k
        & relevant
    )

    return (
        len(matched)
        / len(relevant)
    )


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:

    relevant = set(
        relevant_ids
    )

    if not relevant:
        return 0.0

    for rank, product_id in enumerate(
        retrieved_ids,
        start=1,
    ):

        if product_id in relevant:

            return (
                1.0
                / rank
            )

    return 0.0


def mean(
    values: list[float],
) -> float:

    if not values:
        return 0.0

    return (
        sum(values)
        / len(values)
    )


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:

    if not values:
        return 0.0

    ordered = sorted(
        values
    )

    index = int(
        round(
            (
                len(ordered)
                - 1
            )
            * percentile_value
        )
    )

    return ordered[
        index
    ]