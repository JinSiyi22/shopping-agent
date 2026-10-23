import pandas as pd

from benchmarks.wands.prepare_data import (
    allocate_stratified_dev_counts,
    resolve_labels,
    stratified_query_split,
)


def test_resolve_labels_collapses_identical_duplicates():

    labels = pd.DataFrame(
        [
            {
                "query_id": 1,
                "product_id": 10,
                "label": "Exact",
            },
            {
                "query_id": 1,
                "product_id": 10,
                "label": "Exact",
            },
        ]
    )

    resolved, audit = resolve_labels(
        labels
    )

    assert len(resolved) == 1

    assert (
        resolved.iloc[0]["label"]
        == "Exact"
    )

    assert (
        audit[
            "identical_duplicate_pair_groups"
        ]
        == 1
    )


def test_resolve_labels_drops_conflicts():

    labels = pd.DataFrame(
        [
            {
                "query_id": 1,
                "product_id": 10,
                "label": "Exact",
            },
            {
                "query_id": 1,
                "product_id": 10,
                "label": "Partial",
            },
        ]
    )

    resolved, audit = resolve_labels(
        labels
    )

    assert len(resolved) == 0

    assert (
        audit[
            "conflicting_pair_groups_dropped"
        ]
        == 1
    )


def test_allocate_stratified_dev_counts_matches_total():

    allocation = (
        allocate_stratified_dev_counts(
            total_queries=480,
            dev_queries=96,
            stratum_sizes={
                True: 379,
                False: 101,
            },
        )
    )

    assert (
        sum(
            allocation.values()
        )
        == 96
    )

    assert (
        allocation[True]
        + allocation[False]
        == 96
    )


def test_stratified_query_split_is_reproducible():

    queries = list(
        range(10)
    )

    labels = pd.DataFrame(
        [
            {
                "query_id": query_id,
                "product_id": query_id,
                "label": (
                    "Exact"
                    if query_id < 6
                    else "Partial"
                ),
            }
            for query_id in queries
        ]
    )

    first = stratified_query_split(
        queries,
        labels=labels,
        dev_queries=4,
        seed=42,
    )

    second = stratified_query_split(
        queries,
        labels=labels,
        dev_queries=4,
        seed=42,
    )

    assert first == second


def test_stratified_split_preserves_exact_eligibility_mix():

    queries = list(
        range(10)
    )

    labels = pd.DataFrame(
        [
            {
                "query_id": query_id,
                "product_id": query_id,
                "label": (
                    "Exact"
                    if query_id < 8
                    else "Partial"
                ),
            }
            for query_id in queries
        ]
    )

    (
        dev_ids,
        test_ids,
        audit,
    ) = stratified_query_split(
        queries,
        labels=labels,
        dev_queries=5,
        seed=42,
    )

    assert len(dev_ids) == 5
    assert len(test_ids) == 5

    assert (
        audit["dev_has_exact"]
        == 4
    )

    assert (
        audit["dev_without_exact"]
        == 1
    )
