from __future__ import annotations

from pathlib import Path

import pandas as pd


def normalize_wands_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize WANDS column names to stable snake_case.

    The official README documents ``category_hierarchy`` while the
    released product.tsv-style file uses ``category hierarchy``.
    Normalizing at the I/O boundary keeps downstream benchmark code
    independent of that source-schema inconsistency.
    """

    normalized = frame.copy()

    normalized.columns = [
        str(column)
        .strip()
        .replace(" ", "_")
        for column in normalized.columns
    ]

    return normalized


def read_wands_tsv(
    path: Path,
) -> pd.DataFrame:
    """
    Read one official WANDS ``.csv`` file.

    Despite the ``.csv`` extension, the official notebook reads all
    three released files with a tab separator.
    """

    frame = pd.read_csv(
        path,
        sep="\t",
    )

    return normalize_wands_columns(
        frame
    )
