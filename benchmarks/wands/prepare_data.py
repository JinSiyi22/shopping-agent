from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from benchmarks.wands.schema import read_wands_tsv


REQUIRED_PRODUCT_COLUMNS = {
    "product_id",
    "product_name",
    "product_class",
    "category_hierarchy",
    "product_description",
    "product_features",
}

REQUIRED_QUERY_COLUMNS = {
    "query_id",
    "query",
    "query_class",
}

REQUIRED_LABEL_COLUMNS = {
    "id",
    "query_id",
    "product_id",
    "label",
}

VALID_LABELS = {
    "Exact",
    "Partial",
    "Irrelevant",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-dir",
        default="external/WANDS/dataset",
    )

    parser.add_argument(
        "--output-dir",
        default="data/wands_processed",
    )

    parser.add_argument(
        "--dev-queries",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def validate_columns(
    frame: pd.DataFrame,
    *,
    required: set[str],
    name: str,
) -> None:
    missing = required - set(frame.columns)

    if missing:
        raise RuntimeError(
            f"{name} missing columns: {sorted(missing)}"
        )


def resolve_labels(
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Clean duplicate query-product rows without inventing
    a second annotation aggregation layer.

    WANDS documentation/paper states that the published
    relevance label for a query-product pair is already an
    aggregated final label from up to three annotators.

    Therefore:
    - duplicate rows with the SAME label collapse to one row;
    - duplicate rows with CONFLICTING labels are excluded
      from evaluation ground truth.

    This is intentionally conservative and avoids applying
    an unsupported second majority-vote rule.
    """

    grouped = labels.groupby(
        [
            "query_id",
            "product_id",
        ],
        sort=False,
        dropna=False,
    )

    resolved_rows: list[dict] = []

    duplicate_groups = 0
    identical_duplicate_groups = 0
    conflicting_groups = 0
    dropped_conflicting_rows = 0

    for (
        query_id,
        product_id,
    ), group in grouped:

        raw_labels = (
            group["label"]
            .astype(str)
            .tolist()
        )

        if len(raw_labels) > 1:
            duplicate_groups += 1

        unique_labels = sorted(
            set(raw_labels)
        )

        if len(unique_labels) == 1:
            if len(raw_labels) > 1:
                identical_duplicate_groups += 1

            resolved_rows.append(
                {
                    "query_id": query_id,
                    "product_id": product_id,
                    "label": unique_labels[0],
                }
            )
            continue

        conflicting_groups += 1
        dropped_conflicting_rows += len(raw_labels)

    resolved = pd.DataFrame(
        resolved_rows,
        columns=[
            "query_id",
            "product_id",
            "label",
        ],
    )

    audit = {
        "raw_label_rows": int(
            len(labels)
        ),
        "raw_unique_query_product_pairs": int(
            labels[
                [
                    "query_id",
                    "product_id",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "resolved_query_product_pairs": int(
            len(resolved)
        ),
        "duplicate_pair_groups": int(
            duplicate_groups
        ),
        "identical_duplicate_pair_groups": int(
            identical_duplicate_groups
        ),
        "conflicting_pair_groups_dropped": int(
            conflicting_groups
        ),
        "raw_rows_in_conflicting_groups_dropped": int(
            dropped_conflicting_rows
        ),
        "conflict_policy": (
            "same-label duplicates collapse; "
            "conflicting duplicate query-product pairs "
            "are excluded from ground truth"
        ),
    }

    return resolved, audit


def allocate_stratified_dev_counts(
    *,
    total_queries: int,
    dev_queries: int,
    stratum_sizes: dict[bool, int],
) -> dict[bool, int]:
    """
    Allocate the requested dev size across:
      has_exact=True / has_exact=False

    Uses proportional allocation with deterministic
    largest-remainder correction so totals match exactly.
    """

    if not (0 < dev_queries < total_queries):
        raise ValueError(
            "--dev-queries must be between "
            "1 and total_queries-1"
        )

    raw = {
        key: (
            dev_queries
            * size
            / total_queries
        )
        for key, size in stratum_sizes.items()
    }

    allocated = {
        key: int(value)
        for key, value in raw.items()
    }

    remaining = (
        dev_queries
        - sum(allocated.values())
    )

    remainder_order = sorted(
        stratum_sizes,
        key=lambda key: (
            raw[key] - allocated[key]
        ),
        reverse=True,
    )

    for key in remainder_order:
        if remaining <= 0:
            break

        if allocated[key] < stratum_sizes[key]:
            allocated[key] += 1
            remaining -= 1

    if remaining != 0:
        raise RuntimeError(
            "Could not allocate requested dev split."
        )

    return allocated


def stratified_query_split(
    query_ids: list[int],
    *,
    labels: pd.DataFrame,
    dev_queries: int,
    seed: int,
) -> tuple[list[int], list[int], dict]:
    """
    Stratify only on whether a query has at least one
    Exact judgment.

    This keeps Exact-only metric eligibility balanced
    across dev/test without trying to stratify 480
    queries over a highly sparse category taxonomy.
    """

    exact_query_ids = set(
        labels.loc[
            labels["label"] == "Exact",
            "query_id",
        ].astype(int)
    )

    strata = {
        True: [
            int(query_id)
            for query_id in query_ids
            if int(query_id) in exact_query_ids
        ],
        False: [
            int(query_id)
            for query_id in query_ids
            if int(query_id) not in exact_query_ids
        ],
    }

    allocation = allocate_stratified_dev_counts(
        total_queries=len(query_ids),
        dev_queries=dev_queries,
        stratum_sizes={
            key: len(value)
            for key, value in strata.items()
        },
    )

    rng = random.Random(seed)

    dev_ids: list[int] = []

    for has_exact in (True, False):
        members = sorted(
            strata[has_exact]
        )

        sampled = rng.sample(
            members,
            allocation[has_exact],
        )

        dev_ids.extend(sampled)

    dev_ids = sorted(dev_ids)
    dev_set = set(dev_ids)

    test_ids = sorted(
        int(query_id)
        for query_id in query_ids
        if int(query_id) not in dev_set
    )

    split_audit = {
        "strategy": (
            "query-level stratified split by "
            "has_exact_label"
        ),
        "dev_total": len(dev_ids),
        "test_total": len(test_ids),
        "dev_has_exact": sum(
            query_id in exact_query_ids
            for query_id in dev_ids
        ),
        "dev_without_exact": sum(
            query_id not in exact_query_ids
            for query_id in dev_ids
        ),
        "test_has_exact": sum(
            query_id in exact_query_ids
            for query_id in test_ids
        ),
        "test_without_exact": sum(
            query_id not in exact_query_ids
            for query_id in test_ids
        ),
    }

    return dev_ids, test_ids, split_audit


def label_distribution_for_queries(
    labels: pd.DataFrame,
    query_ids: list[int],
) -> dict[str, int]:
    subset = labels[
        labels["query_id"].isin(
            set(query_ids)
        )
    ]

    counts = (
        subset["label"]
        .value_counts()
        .to_dict()
    )

    return {
        label: int(
            counts.get(
                label,
                0,
            )
        )
        for label in (
            "Exact",
            "Partial",
            "Irrelevant",
        )
    }


def main() -> None:
    args = parse_args()

    source_dir = Path(
        args.source_dir
    )

    output_dir = Path(
        args.output_dir
    )

    products_path = (
        source_dir
        / "product.csv"
    )

    queries_path = (
        source_dir
        / "query.csv"
    )

    labels_path = (
        source_dir
        / "label.csv"
    )

    for path in (
        products_path,
        queries_path,
        labels_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing WANDS file: {path}"
            )

    products = read_wands_tsv(
        products_path
    )

    queries = read_wands_tsv(
        queries_path
    )

    labels = read_wands_tsv(
        labels_path
    )

    validate_columns(
        products,
        required=REQUIRED_PRODUCT_COLUMNS,
        name="product.csv",
    )

    validate_columns(
        queries,
        required=REQUIRED_QUERY_COLUMNS,
        name="query.csv",
    )

    validate_columns(
        labels,
        required=REQUIRED_LABEL_COLUMNS,
        name="label.csv",
    )

    invalid_labels = sorted(
        set(
            labels["label"]
            .dropna()
            .astype(str)
        )
        - VALID_LABELS
    )

    if invalid_labels:
        raise RuntimeError(
            "Unexpected WANDS labels: "
            f"{invalid_labels}"
        )

    resolved_labels, label_audit = (
        resolve_labels(
            labels
        )
    )

    product_ids = set(
        products["product_id"]
        .astype(int)
        .tolist()
    )

    query_ids = set(
        queries["query_id"]
        .astype(int)
        .tolist()
    )

    missing_products = int(
        (
            ~resolved_labels[
                "product_id"
            ]
            .astype(int)
            .isin(product_ids)
        ).sum()
    )

    missing_queries = int(
        (
            ~resolved_labels[
                "query_id"
            ]
            .astype(int)
            .isin(query_ids)
        ).sum()
    )

    if missing_products or missing_queries:
        raise RuntimeError(
            "Broken WANDS references: "
            f"missing_products={missing_products}, "
            f"missing_queries={missing_queries}"
        )

    sorted_query_ids = sorted(
        int(query_id)
        for query_id in query_ids
    )

    (
        dev_ids,
        test_ids,
        split_audit,
    ) = stratified_query_split(
        sorted_query_ids,
        labels=resolved_labels,
        dev_queries=args.dev_queries,
        seed=args.seed,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved_path = (
        output_dir
        / "labels_resolved.parquet"
    )

    resolved_labels.to_parquet(
        resolved_path,
        index=False,
    )

    label_distribution = {
        label: int(
            (
                resolved_labels["label"]
                == label
            ).sum()
        )
        for label in (
            "Exact",
            "Partial",
            "Irrelevant",
        )
    }

    queries_without_exact = int(
        split_audit[
            "dev_without_exact"
        ]
        + split_audit[
            "test_without_exact"
        ]
    )

    manifest = {
        "dataset": (
            "Wayfair WANDS"
        ),
        "seed": args.seed,
        "total_products": int(
            products["product_id"]
            .nunique()
        ),
        "total_queries": int(
            queries["query_id"]
            .nunique()
        ),
        "label_audit": (
            label_audit
        ),
        "label_distribution": (
            label_distribution
        ),
        "queries_without_exact": (
            queries_without_exact
        ),
        "split": {
            **split_audit,
            "dev_label_distribution": (
                label_distribution_for_queries(
                    resolved_labels,
                    dev_ids,
                )
            ),
            "test_label_distribution": (
                label_distribution_for_queries(
                    resolved_labels,
                    test_ids,
                )
            ),
            "dev_query_ids": dev_ids,
            "test_query_ids": test_ids,
        },
        "evaluation_note": (
            "Split is by query only and stratified "
            "by whether the query has at least one "
            "Exact judgment. The product corpus is "
            "shared across dev and test."
        ),
    }

    manifest_path = (
        output_dir
        / "split_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("WANDS AUDIT + STRATIFIED QUERY SPLIT V3")
    print("=" * 72)

    print(
        "products:",
        manifest["total_products"],
    )

    print(
        "queries:",
        manifest["total_queries"],
    )

    print(
        "raw label rows:",
        label_audit["raw_label_rows"],
    )

    print(
        "raw unique judged pairs:",
        label_audit[
            "raw_unique_query_product_pairs"
        ],
    )

    print(
        "resolved judged pairs:",
        label_audit[
            "resolved_query_product_pairs"
        ],
    )

    print(
        "duplicate pair groups:",
        label_audit[
            "duplicate_pair_groups"
        ],
    )

    print(
        "identical duplicate groups:",
        label_audit[
            "identical_duplicate_pair_groups"
        ],
    )

    print(
        "conflicting pair groups dropped:",
        label_audit[
            "conflicting_pair_groups_dropped"
        ],
    )

    print(
        "raw rows in conflicts dropped:",
        label_audit[
            "raw_rows_in_conflicting_groups_dropped"
        ],
    )

    print(
        "label distribution:",
        manifest["label_distribution"],
    )

    print(
        "queries without Exact:",
        queries_without_exact,
    )

    print(
        "dev queries:",
        split_audit["dev_total"],
    )

    print(
        "  dev has Exact:",
        split_audit["dev_has_exact"],
    )

    print(
        "  dev without Exact:",
        split_audit["dev_without_exact"],
    )

    print(
        "  dev label distribution:",
        manifest["split"][
            "dev_label_distribution"
        ],
    )

    print(
        "test queries:",
        split_audit["test_total"],
    )

    print(
        "  test has Exact:",
        split_audit["test_has_exact"],
    )

    print(
        "  test without Exact:",
        split_audit["test_without_exact"],
    )

    print(
        "  test label distribution:",
        manifest["split"][
            "test_label_distribution"
        ],
    )

    print(
        "resolved labels:",
        resolved_path,
    )

    print(
        "manifest:",
        manifest_path,
    )


if __name__ == "__main__":
    main()
