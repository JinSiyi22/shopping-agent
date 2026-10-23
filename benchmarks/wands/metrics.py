from __future__ import annotations

import math
from statistics import mean


LABEL_GAIN = {
    "Exact": 2.0,
    "Partial": 1.0,
    "Irrelevant": 0.0,
}


def dcg_at_k(
    ranked_labels: list[str | None],
    k: int,
) -> float:
    total = 0.0

    for rank, label in enumerate(
        ranked_labels[:k],
        start=1,
    ):
        gain = (
            LABEL_GAIN.get(
                label,
                0.0,
            )
            if label is not None
            else 0.0
        )

        total += (
            gain
            / math.log2(
                rank + 1
            )
        )

    return total


def ndcg_at_k(
    ranked_labels: list[str | None],
    judged_labels: list[str],
    k: int,
) -> float:
    ideal_labels = sorted(
        judged_labels,
        key=lambda label: LABEL_GAIN[label],
        reverse=True,
    )

    ideal = dcg_at_k(
        ideal_labels,
        k,
    )

    if ideal == 0:
        return 0.0

    return (
        dcg_at_k(
            ranked_labels,
            k,
        )
        / ideal
    )


def exact_mrr_at_k(
    ranked_labels: list[str | None],
    judged_labels: list[str],
    k: int,
) -> float | None:
    if "Exact" not in judged_labels:
        return None

    for rank, label in enumerate(
        ranked_labels[:k],
        start=1,
    ):
        if label == "Exact":
            return 1.0 / rank

    return 0.0


def exact_recall_at_k(
    ranked_labels: list[str | None],
    judged_labels: list[str],
    k: int,
) -> float | None:
    exact_total = sum(
        label == "Exact"
        for label in judged_labels
    )

    if exact_total == 0:
        return None

    exact_retrieved = sum(
        label == "Exact"
        for label in ranked_labels[:k]
    )

    return (
        exact_retrieved
        / exact_total
    )


def relevant_recall_at_k(
    ranked_labels: list[str | None],
    judged_labels: list[str],
    k: int,
) -> float | None:
    relevant_total = sum(
        label in {
            "Exact",
            "Partial",
        }
        for label in judged_labels
    )

    if relevant_total == 0:
        return None

    relevant_retrieved = sum(
        label in {
            "Exact",
            "Partial",
        }
        for label in ranked_labels[:k]
    )

    return (
        relevant_retrieved
        / relevant_total
    )


def mean_optional(
    values: list[float | None],
) -> float | None:
    clean = [
        float(value)
        for value in values
        if value is not None
    ]

    if not clean:
        return None

    return mean(clean)


def summarize_metrics(
    rows: list[dict],
) -> dict[str, float | int | None]:
    if not rows:
        return {}

    metric_names = [
        key
        for key in rows[0]
        if key != "query_id"
    ]

    summary: dict[
        str,
        float | int | None,
    ] = {}

    for metric_name in metric_names:
        values = [
            row[metric_name]
            for row in rows
        ]

        summary[metric_name] = (
            mean_optional(
                values
            )
        )

        summary[
            f"{metric_name}__n"
        ] = sum(
            value is not None
            for value in values
        )

    return summary
