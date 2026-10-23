import pandas as pd

from benchmarks.wands.schema import (
    normalize_wands_columns,
)


def test_normalize_official_category_hierarchy_header():
    frame = pd.DataFrame(
        columns=[
            "product_id",
            "product_name",
            "category hierarchy",
        ]
    )

    normalized = normalize_wands_columns(
        frame
    )

    assert "category_hierarchy" in normalized.columns
    assert "category hierarchy" not in normalized.columns


def test_normalize_strips_surrounding_whitespace():
    frame = pd.DataFrame(
        columns=[
            " product_id ",
            "query class ",
        ]
    )

    normalized = normalize_wands_columns(
        frame
    )

    assert normalized.columns.tolist() == [
        "product_id",
        "query_class",
    ]
