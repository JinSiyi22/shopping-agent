from __future__ import annotations

import math
import re
from collections import defaultdict
from statistics import mean


def extract_product_ids(
    result,
) -> list[str]:
    if result is None:
        return []

    if (
        isinstance(result, dict)
        and set(result.keys())
        == {"result"}
    ):
        return extract_product_ids(
            result["result"]
        )

    if isinstance(
        result,
        list,
    ):
        ids = []

        for item in result:
            if not isinstance(
                item,
                dict,
            ):
                continue

            product_id = item.get(
                "id"
            )

            if product_id is not None:
                ids.append(
                    str(
                        product_id
                    )
                )

        return ids

    if isinstance(
        result,
        dict,
    ):
        products = (
            result.get(
                "products"
            )
            or result.get(
                "selected"
            )
        )

        if isinstance(
            products,
            list,
        ):
            return extract_product_ids(
                products
            )

        product_id = result.get(
            "id"
        )

        if product_id is not None:
            return [
                str(
                    product_id
                )
            ]

    return []


def product_satisfies_constraints(
    product: dict,
    constraints: dict,
) -> bool:
    if not constraints:
        return True

    brand = constraints.get(
        "brand"
    )

    if (
        brand is not None
        and str(
            product.get(
                "brand",
                "",
            )
        ).lower()
        != str(
            brand
        ).lower()
    ):
        return False

    category = constraints.get(
        "category"
    )

    if (
        category is not None
        and str(
            product.get(
                "category",
                "",
            )
        ).lower()
        != str(
            category
        ).lower()
    ):
        return False

    price = float(
        product.get(
            "price",
            math.inf,
        )
    )

    max_price = constraints.get(
        "max_price"
    )

    if (
        max_price is not None
        and price
        > float(
            max_price
        )
        + 1e-9
    ):
        return False

    min_price = constraints.get(
        "min_price"
    )

    if (
        min_price is not None
        and price
        < float(
            min_price
        )
        - 1e-9
    ):
        return False

    rating = float(
        product.get(
            "rating",
            -math.inf,
        )
    )

    min_rating = constraints.get(
        "min_rating"
    )

    if (
        min_rating is not None
        and rating
        < float(
            min_rating
        )
        - 1e-9
    ):
        return False

    return True


def constraint_pass(
    retrieved_ids: list[str],
    product_map: dict[str, dict],
    constraints: dict,
    min_results: int = 0,
) -> bool:
    if not constraints:
        return True

    if (
        len(
            retrieved_ids
        )
        < min_results
    ):
        return False

    if not retrieved_ids:
        return (
            min_results == 0
        )

    for product_id in retrieved_ids:
        product = product_map.get(
            str(
                product_id
            )
        )

        if product is None:
            return False

        if not product_satisfies_constraints(
            product,
            constraints,
        ):
            return False

    return True


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float | None:
    relevant = set(
        str(value)
        for value in relevant_ids
    )

    if not relevant:
        return None

    retrieved = set(
        str(value)
        for value
        in retrieved_ids[:k]
    )

    return (
        len(
            relevant
            & retrieved
        )
        / len(
            relevant
        )
    )


def hit_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float | None:
    relevant = set(
        str(value)
        for value
        in relevant_ids
    )

    if not relevant:
        return None

    retrieved = set(
        str(value)
        for value
        in retrieved_ids[:k]
    )

    return float(
        bool(
            relevant
            & retrieved
        )
    )


def semantic_tag_hit_at_k(
    retrieved_ids: list[str],
    product_map: dict[str, dict],
    semantic_tags: list[str],
    k: int,
) -> float | None:
    if not semantic_tags:
        return None

    required = set(
        str(tag)
        for tag in semantic_tags
    )

    for product_id in (
        retrieved_ids[
            :k
        ]
    ):
        product = product_map.get(
            str(
                product_id
            )
        )

        if product is None:
            continue

        tags = set(
            str(tag)
            for tag
            in product.get(
                "tags",
                [],
            )
        )

        if required.issubset(
            tags
        ):
            return 1.0

    return 0.0


def normalize_text(
    text: str,
) -> str:
    text = str(text).lower()
    text = re.sub(
        r"[^\w]+",
        " ",
        text,
        flags=re.UNICODE,
    )
    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _candidate_aliases(
    product: dict,
) -> set[str]:
    product_id = normalize_text(
        str(product.get("id", ""))
    )
    name = normalize_text(
        str(product.get("name", ""))
    )

    aliases: set[str] = set()

    if len(product_id) >= 6:
        aliases.add(product_id)

    if len(name) >= 10:
        aliases.add(name)

    tokens = name.split()

    for prefix_len in (8, 6, 5, 4):
        if len(tokens) < prefix_len:
            continue

        alias = " ".join(
            tokens[:prefix_len]
        )

        if len(alias) >= 12:
            aliases.add(alias)

    return aliases


def build_unique_alias_index(
    products: list[dict],
) -> dict[str, str]:
    candidates: dict[str, set[str]] = defaultdict(set)

    for product in products:
        product_id = str(product["id"])

        for alias in _candidate_aliases(product):
            candidates[alias].add(product_id)

    return {
        alias: next(iter(product_ids))
        for alias, product_ids in candidates.items()
        if len(product_ids) == 1
    }


def mentioned_catalog_product_ids(
    answer: str,
    alias_index: dict[str, str],
) -> list[str]:
    normalized_answer = normalize_text(answer)
    padded_answer = f" {normalized_answer} "

    mentioned: list[str] = []
    seen: set[str] = set()

    for alias, product_id in sorted(
        alias_index.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if f" {alias} " not in padded_answer:
            continue

        if product_id in seen:
            continue

        seen.add(product_id)
        mentioned.append(product_id)

    return mentioned


def grounding_metrics(
    answer: str,
    alias_index: dict[str, str],
    evidence_ids: set[str],
) -> dict:
    mentioned = mentioned_catalog_product_ids(
        answer,
        alias_index,
    )
    mentioned_set = set(mentioned)

    unsupported = [
        product_id
        for product_id in mentioned
        if product_id not in evidence_ids
    ]

    supported = [
        product_id
        for product_id in mentioned
        if product_id in evidence_ids
    ]

    grounding_rate = None
    grounding_pass = None

    if mentioned:
        grounding_rate = (
            len(supported)
            / len(mentioned)
        )
        grounding_pass = (
            len(unsupported) == 0
        )

    evidence_mentioned = None

    if evidence_ids:
        evidence_mentioned = float(
            bool(
                mentioned_set
                & evidence_ids
            )
        )

    return {
        "mentioned_product_ids": mentioned,
        "unsupported_product_ids": unsupported,
        "catalog_product_grounding_rate": grounding_rate,
        "evidence_product_mentioned": evidence_mentioned,
        "catalog_product_grounding_pass": grounding_pass,
    }
def accuracy(
    values: list[bool],
) -> float:
    if not values:
        return 0.0

    return (
        sum(
            bool(value)
            for value in values
        )
        / len(
            values
        )
    )


def mean_optional(
    values: list[
        float | None
    ],
) -> float | None:
    clean = [
        float(value)
        for value
        in values
        if value is not None
    ]

    if not clean:
        return None

    return mean(
        clean
    )


def percentile(
    values: list[float],
    q: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(
        float(value)
        for value in values
    )

    index = int(
        round(
            (
                len(ordered)
                - 1
            )
            * q
        )
    )

    return ordered[
        index
    ]
