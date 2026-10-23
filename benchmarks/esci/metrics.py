from __future__ import annotations

import math
from statistics import mean


# Official ESCI Task-1 gain mapping used by Amazon's trec_eval setup:
# I -> 0, S -> 0.01, C -> 0.1, E -> 1.0
LABEL_GAIN = {
    "E": 1.0,
    "C": 0.1,
    "S": 0.01,
    "I": 0.0,
}


def dcg(labels: list[str]) -> float:
    total = 0.0

    for rank, label in enumerate(labels, start=1):
        gain = LABEL_GAIN[label]
        total += gain / math.log2(rank + 1)

    return total


def ndcg(labels: list[str]) -> float:
    ideal = sorted(
        labels,
        key=lambda label: LABEL_GAIN[label],
        reverse=True,
    )

    ideal_dcg = dcg(ideal)

    if ideal_dcg == 0:
        return 0.0

    return dcg(labels) / ideal_dcg


def reciprocal_rank_exact(
    labels: list[str],
) -> float:
    for rank, label in enumerate(labels, start=1):
        if label == "E":
            return 1.0 / rank

    return 0.0


def exact_hit_at_k(
    labels: list[str],
    k: int,
) -> float:
    return float(
        any(
            label == "E"
            for label in labels[:k]
        )
    )


def recall_non_irrelevant_at_k(
    labels: list[str],
    k: int,
) -> float:
    relevant_total = sum(
        label != "I"
        for label in labels
    )

    if relevant_total == 0:
        return 0.0

    relevant_top_k = sum(
        label != "I"
        for label in labels[:k]
    )

    return (
        relevant_top_k
        / relevant_total
    )


def summarize_query_metrics(
    rows: list[dict],
) -> dict[str, float]:
    if not rows:
        return {}

    keys = [
        key
        for key in rows[0]
        if key != "query_id"
    ]

    return {
        key: mean(
            float(row[key])
            for row in rows
        )
        for key in keys
    }
