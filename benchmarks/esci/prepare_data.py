from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


EXAMPLE_COLUMNS = [
    "example_id",
    "query",
    "query_id",
    "product_id",
    "product_locale",
    "esci_label",
    "small_version",
    "split",
]

PRODUCT_COLUMNS = [
    "product_id",
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
    "product_locale",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-dir",
        default=(
            "external/esci-data/"
            "shopping_queries_dataset"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/esci_processed/"
            "us_small_test.parquet"
        ),
    )

    parser.add_argument(
        "--locale",
        default="us",
    )

    parser.add_argument(
        "--split",
        default="test",
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help=(
            "0 means all matching queries. "
            "Use 300 for a smoke benchmark."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16384,
    )

    return parser.parse_args()


def choose_query_ids(
    query_ids: list[int],
    *,
    max_queries: int,
    seed: int,
) -> list[int]:

    unique_ids = sorted(
        set(query_ids)
    )

    if (
        max_queries <= 0
        or max_queries >= len(unique_ids)
    ):
        return unique_ids

    rng = random.Random(seed)

    return sorted(
        rng.sample(
            unique_ids,
            max_queries,
        )
    )


def load_relevant_products(
    products_path: Path,
    *,
    locale: str,
    product_ids: set[str],
    batch_size: int,
) -> pd.DataFrame:

    parquet_file = pq.ParquetFile(
        products_path
    )

    chunks: list[pd.DataFrame] = []

    for batch in parquet_file.iter_batches(
        columns=PRODUCT_COLUMNS,
        batch_size=batch_size,
    ):
        frame = batch.to_pandas()

        frame = frame[
            (
                frame["product_locale"]
                == locale
            )
            & (
                frame["product_id"]
                .isin(product_ids)
            )
        ]

        if not frame.empty:
            chunks.append(frame)

    if not chunks:
        raise RuntimeError(
            "No ESCI products matched selected examples."
        )

    products = pd.concat(
        chunks,
        ignore_index=True,
    )

    products = products.drop_duplicates(
        subset=[
            "product_locale",
            "product_id",
        ],
        keep="first",
    )

    return products


def main() -> None:
    args = parse_args()

    source_dir = Path(
        args.source_dir
    )

    examples_path = (
        source_dir
        / "shopping_queries_dataset_examples.parquet"
    )

    products_path = (
        source_dir
        / "shopping_queries_dataset_products.parquet"
    )

    if not examples_path.exists():
        raise FileNotFoundError(
            f"Missing: {examples_path}"
        )

    if not products_path.exists():
        raise FileNotFoundError(
            f"Missing: {products_path}"
        )

    examples = pd.read_parquet(
        examples_path,
        columns=EXAMPLE_COLUMNS,
    )

    examples = examples[
        (
            examples["small_version"]
            == 1
        )
        & (
            examples["split"]
            == args.split
        )
        & (
            examples["product_locale"]
            == args.locale
        )
    ].copy()

    selected_query_ids = choose_query_ids(
        examples[
            "query_id"
        ].tolist(),
        max_queries=args.max_queries,
        seed=args.seed,
    )

    examples = examples[
        examples["query_id"].isin(
            selected_query_ids
        )
    ].copy()

    needed_product_ids = set(
        examples[
            "product_id"
        ].astype(str)
    )

    products = load_relevant_products(
        products_path,
        locale=args.locale,
        product_ids=needed_product_ids,
        batch_size=args.batch_size,
    )

    merged = examples.merge(
        products,
        how="left",
        on=[
            "product_locale",
            "product_id",
        ],
        validate="many_to_one",
    )

    missing_titles = int(
        merged[
            "product_title"
        ].isna().sum()
    )

    if missing_titles:
        raise RuntimeError(
            "Missing product metadata for "
            f"{missing_titles} judged rows."
        )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_parquet(
        output_path,
        index=False,
    )

    label_counts = (
        merged[
            "esci_label"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print("=" * 72)
    print("ESCI TASK-1 DATA PREP")
    print("=" * 72)
    print(
        "locale:",
        args.locale,
    )
    print(
        "split:",
        args.split,
    )
    print(
        "queries:",
        merged["query_id"].nunique(),
    )
    print(
        "judgements:",
        len(merged),
    )
    print(
        "unique products:",
        merged["product_id"].nunique(),
    )
    print(
        "label counts:",
        label_counts,
    )
    print(
        "output:",
        output_path,
    )


if __name__ == "__main__":
    main()
