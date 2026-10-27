from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


CATEGORY_ZH = {
    "camera": "相机",
    "headphone": "耳机",
    "keyboard": "键盘",
    "laptop": "笔记本电脑",
    "monitor": "显示器",
    "mouse": "鼠标",
    "router": "路由器",
    "smartwatch": "智能手表",
    "speaker": "音箱",
    "storage": "存储设备",
    "tablet": "平板电脑",
}

SEMANTIC_TAG_PHRASES = {
    "wireless": "无线连接",
    "bluetooth": "蓝牙连接",
    "lightweight": "轻薄便携",
    "gaming": "游戏性能",
    "noise_cancelling": "主动降噪",
    "OLED": "OLED 屏幕",
    "4K": "4K 显示",
    "mechanical": "机械键盘手感",
    "USB-C": "USB-C 连接",
    "SSD": "SSD 固态存储",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--catalog",
        default="data/products_real.json",
    )

    parser.add_argument(
        "--output",
        default="evals/shopping_real_eval.json",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def load_catalog(path: Path) -> list[dict]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        products = json.load(file)

    if not isinstance(products, list):
        raise RuntimeError(
            "Catalog must be a JSON list."
        )

    required = {
        "id",
        "name",
        "category",
        "brand",
        "price",
        "rating",
        "tags",
        "description",
    }

    cleaned = []

    for item in products:
        if not isinstance(item, dict):
            continue

        if not required.issubset(
            item.keys()
        ):
            continue

        cleaned.append(item)

    if len(cleaned) < 100:
        raise RuntimeError(
            "Real-agent evaluation expects "
            "a substantial catalog."
        )

    return cleaned


def round_budget(
    price: float,
    multiplier: float,
) -> float:
    raw = max(
        price,
        price * multiplier,
    )

    if raw < 50:
        return round(
            math.ceil(raw * 2) / 2,
            2,
        )

    return float(
        math.ceil(raw)
    )


def safe_brand(
    product: dict,
) -> bool:
    brand = str(
        product.get(
            "brand",
            "",
        )
    ).strip()

    return (
        1 < len(brand) <= 50
        and brand.lower()
        not in {
            "unknown",
            "none",
            "n/a",
            "na",
        }
    )


def category_label(
    category: str,
) -> str:
    zh = CATEGORY_ZH.get(
        category,
        category,
    )

    return (
        f"{category}（{zh}）"
    )


def make_structured_cases(
    products: list[dict],
    *,
    count: int,
    rng: random.Random,
) -> list[dict]:
    groups: dict[
        str,
        list[dict],
    ] = defaultdict(list)

    for product in products:
        if safe_brand(product):
            groups[
                product["category"]
            ].append(product)

    categories = sorted(
        groups
    )

    for items in groups.values():
        rng.shuffle(items)

    cases = []
    cursors = {
        category: 0
        for category in categories
    }

    while (
        len(cases) < count
        and categories
    ):
        progressed = False

        for category in categories:
            items = groups[
                category
            ]

            cursor = cursors[
                category
            ]

            if cursor >= len(items):
                continue

            product = items[
                cursor
            ]

            cursors[
                category
            ] += 1

            progressed = True

            max_price = round_budget(
                float(
                    product["price"]
                ),
                1.08,
            )

            min_rating = max(
                0.0,
                round(
                    float(
                        product["rating"]
                    )
                    - 0.2,
                    1,
                ),
            )

            brand = str(
                product["brand"]
            ).strip()

            message = (
                f"只看品牌 {brand} 的"
                f"{category_label(category)}，"
                f"预算 {max_price:g} 美元以内，"
                f"评分至少 {min_rating:g}。"
            )

            cases.append(
                {
                    "id": (
                        f"structured_"
                        f"{len(cases)+1:02d}"
                    ),
                    "type": (
                        "structured_search"
                    ),
                    "description": (
                        "真实商品目录上的结构化硬条件查询"
                    ),
                    "turns": [
                        {
                            "message": message,
                            "expected_route": "research",
                            "expected_tool": (
                                "search_products"
                            ),
                            "hard_constraints": {
                                "brand": brand,
                                "category": category,
                                "max_price": (
                                    max_price
                                ),
                                "min_rating": (
                                    min_rating
                                ),
                            },
                            "min_results": 1,
                        }
                    ],
                }
            )

            if len(cases) >= count:
                break

        if not progressed:
            break

    if len(cases) < count:
        raise RuntimeError(
            "Could not build enough "
            "structured cases."
        )

    return cases


def canonical_semantic_tag(
    product: dict,
) -> str | None:
    tags = set(
        str(tag)
        for tag
        in product.get(
            "tags",
            [],
        )
    )

    for tag in (
        SEMANTIC_TAG_PHRASES
    ):
        if tag in tags:
            return tag

    return None


def semantic_relevant_ids(
    products: list[dict],
    *,
    category: str,
    max_price: float,
    tag: str,
) -> list[str]:
    result = []

    for product in products:
        if (
            product["category"]
            != category
        ):
            continue

        if (
            float(
                product["price"]
            )
            > max_price
        ):
            continue

        tags = set(
            str(value)
            for value
            in product.get(
                "tags",
                [],
            )
        )

        if tag not in tags:
            continue

        result.append(
            str(
                product["id"]
            )
        )

    return result


def semantic_pool(
    products: list[dict],
) -> list[tuple[dict, str]]:
    pool = []

    for product in products:
        tag = canonical_semantic_tag(
            product
        )

        if tag is None:
            continue

        pool.append(
            (
                product,
                tag,
            )
        )

    return pool


def make_semantic_turn(
    products: list[dict],
    *,
    product: dict,
    tag: str,
) -> dict:
    category = str(
        product["category"]
    )

    max_price = round_budget(
        float(
            product["price"]
        ),
        1.15,
    )

    phrase = (
        SEMANTIC_TAG_PHRASES[
            tag
        ]
    )

    relevant_ids = (
        semantic_relevant_ids(
            products,
            category=category,
            max_price=max_price,
            tag=tag,
        )
    )

    return {
        "message": (
            f"我想买一款"
            f"{category_label(category)}，"
            f"主要关注{phrase}，"
            f"预算不超过 {max_price:g} 美元，"
            f"请根据这些需求推荐。"
        ),
        "expected_route": "research",
        "expected_tool": (
            "rerank_search_products"
        ),
        "hard_constraints": {
            "category": category,
            "max_price": max_price,
        },
        "semantic_tags": [
            tag
        ],
        "relevant_product_ids": (
            relevant_ids
        ),
        "eval_k": 5,
        "min_results": 1,
    }


def make_semantic_cases(
    products: list[dict],
    *,
    count: int,
    rng: random.Random,
) -> list[dict]:
    pool = semantic_pool(
        products
    )

    rng.shuffle(
        pool
    )

    chosen = []
    used_pairs = set()
    used_categories = defaultdict(int)

    # First pass: encourage category diversity.
    for product, tag in pool:
        pair = (
            product["category"],
            tag,
        )

        if pair in used_pairs:
            continue

        if (
            used_categories[
                product["category"]
            ]
            >= 2
        ):
            continue

        turn = make_semantic_turn(
            products,
            product=product,
            tag=tag,
        )

        if not (
            turn[
                "relevant_product_ids"
            ]
        ):
            continue

        chosen.append(
            (
                product,
                tag,
                turn,
            )
        )

        used_pairs.add(
            pair
        )

        used_categories[
            product["category"]
        ] += 1

        if len(chosen) >= count:
            break

    # Second pass: fill if tags/categories are sparse.
    if len(chosen) < count:
        used_products = {
            str(item[0]["id"])
            for item in chosen
        }

        for product, tag in pool:
            if (
                str(
                    product["id"]
                )
                in used_products
            ):
                continue

            turn = make_semantic_turn(
                products,
                product=product,
                tag=tag,
            )

            if not (
                turn[
                    "relevant_product_ids"
                ]
            ):
                continue

            chosen.append(
                (
                    product,
                    tag,
                    turn,
                )
            )

            used_products.add(
                str(
                    product["id"]
                )
            )

            if len(chosen) >= count:
                break

    if len(chosen) < count:
        raise RuntimeError(
            "Not enough tagged real products "
            "to build semantic cases."
        )

    cases = []

    for index, (
        product,
        tag,
        turn,
    ) in enumerate(
        chosen[:count],
        start=1,
    ):
        cases.append(
            {
                "id": (
                    f"semantic_"
                    f"{index:02d}"
                ),
                "type": (
                    "semantic_recommendation"
                ),
                "description": (
                    "真实商品目录上的语义推荐"
                ),
                "seed_product_id": (
                    str(
                        product["id"]
                    )
                ),
                "seed_tag": tag,
                "turns": [
                    turn
                ],
            }
        )

    return cases


def diverse_products(
    products: list[dict],
    *,
    count: int,
    rng: random.Random,
) -> list[dict]:
    groups = defaultdict(
        list
    )

    for product in products:
        groups[
            product["category"]
        ].append(product)

    for items in groups.values():
        rng.shuffle(items)

    categories = sorted(
        groups
    )

    selected = []

    while (
        len(selected) < count
        and categories
    ):
        progressed = False

        for category in categories:
            items = groups[
                category
            ]

            if not items:
                continue

            selected.append(
                items.pop()
            )

            progressed = True

            if len(selected) >= count:
                break

        if not progressed:
            break

    return selected


def make_detail_cases(
    products: list[dict],
    *,
    count: int,
    rng: random.Random,
) -> list[dict]:
    selected = diverse_products(
        products,
        count=count,
        rng=rng,
    )

    return [
        {
            "id": (
                f"detail_{index:02d}"
            ),
            "type": (
                "product_detail"
            ),
            "description": (
                "已知真实商品 ID 查询详情"
            ),
            "turns": [
                {
                    "message": (
                        "请告诉我商品 "
                        f"{product['id']} "
                        "的详细信息。"
                    ),
                    "expected_route": (
                        "research"
                    ),
                    "expected_tool": (
                        "get_product_detail"
                    ),
                    "expected_product_ids": [
                        str(
                            product["id"]
                        )
                    ],
                }
            ],
        }
        for index, product
        in enumerate(
            selected,
            start=1,
        )
    ]


def make_compare_cases(
    products: list[dict],
    *,
    count: int,
    rng: random.Random,
) -> list[dict]:
    groups = defaultdict(
        list
    )

    for product in products:
        groups[
            product["category"]
        ].append(product)

    valid_categories = [
        category
        for category, items
        in groups.items()
        if len(items) >= 2
    ]

    rng.shuffle(
        valid_categories
    )

    cases = []

    for category in valid_categories:
        items = list(
            groups[
                category
            ]
        )

        rng.shuffle(
            items
        )

        first, second = (
            items[0],
            items[1],
        )

        cases.append(
            {
                "id": (
                    f"compare_"
                    f"{len(cases)+1:02d}"
                ),
                "type": (
                    "product_compare"
                ),
                "description": (
                    "比较同类真实商品"
                ),
                "turns": [
                    {
                        "message": (
                            "请比较商品 "
                            f"{first['id']} "
                            "和 "
                            f"{second['id']}，"
                            "说明各自优缺点。"
                        ),
                        "expected_route": (
                            "research"
                        ),
                        "expected_tool": (
                            "compare_products"
                        ),
                        "expected_product_ids": [
                            str(
                                first["id"]
                            ),
                            str(
                                second["id"]
                            ),
                        ],
                    }
                ],
            }
        )

        if len(cases) >= count:
            break

    if len(cases) < count:
        raise RuntimeError(
            "Could not build enough "
            "comparison cases."
        )

    return cases


def feasible_budget_for_semantic_tag(
    products: list[dict],
    *,
    category: str,
    tag: str | None,
) -> float:
    candidates = []

    for product in products:
        if product["category"] != category:
            continue

        if tag is not None:
            tags = {
                str(value)
                for value
                in product.get(
                    "tags",
                    [],
                )
            }

            if tag not in tags:
                continue

        candidates.append(
            float(
                product["price"]
            )
        )

    if not candidates:
        raise RuntimeError(
            "No feasible products for "
            "follow-up constraint."
        )

    prices = sorted(
        candidates
    )

    index = max(
        0,
        min(
            len(prices) - 1,
            int(
                0.40
                * (
                    len(prices)
                    - 1
                )
            ),
        ),
    )

    return float(
        math.ceil(
            prices[
                index
            ]
        )
    )


def make_followup_cases(
    products: list[dict],
    semantic_cases: list[dict],
    *,
    no_research_count: int,
    new_constraint_count: int,
) -> list[dict]:
    result = []

    base_cases = (
        semantic_cases[
            :(
                no_research_count
                + new_constraint_count
            )
        ]
    )

    if len(base_cases) < (
        no_research_count
        + new_constraint_count
    ):
        raise RuntimeError(
            "Not enough semantic cases "
            "for follow-up evaluation."
        )

    for index in range(
        no_research_count
    ):
        first_turn = json.loads(
            json.dumps(
                base_cases[
                    index
                ][
                    "turns"
                ][0]
            )
        )

        result.append(
            {
                "id": (
                    "followup_reuse_"
                    f"{index+1:02d}"
                ),
                "type": (
                    "multi_turn_reuse"
                ),
                "description": (
                    "第二轮应复用历史结果，"
                    "不重新研究"
                ),
                "turns": [
                    first_turn,
                    {
                        "message": (
                            "基于刚才的结果，"
                            "为什么你更推荐第一款？"
                            "请直接解释，不要重新搜索。"
                        ),
                        "expected_route": (
                            "answer"
                        ),
                    },
                ],
            }
        )

    offset = no_research_count

    for index in range(
        new_constraint_count
    ):
        source = base_cases[
            offset
            + index
        ]

        first_turn = json.loads(
            json.dumps(
                source[
                    "turns"
                ][0]
            )
        )

        category = (
            first_turn[
                "hard_constraints"
            ][
                "category"
            ]
        )

        semantic_tags = (
            first_turn.get(
                "semantic_tags",
                [],
            )
        )

        semantic_tag = (
            semantic_tags[0]
            if semantic_tags
            else None
        )

        new_budget = (
            feasible_budget_for_semantic_tag(
                products,
                category=category,
                tag=semantic_tag,
            )
        )

        phrase = (
            SEMANTIC_TAG_PHRASES.get(
                semantic_tags[0],
                "之前的偏好",
            )
            if semantic_tags
            else "之前的偏好"
        )

        relevant_ids = []

        if semantic_tags:
            relevant_ids = (
                semantic_relevant_ids(
                    products,
                    category=category,
                    max_price=new_budget,
                    tag=semantic_tags[0],
                )
            )

        second_turn = {
            "message": (
                f"预算改成 {new_budget:g} 美元以内，"
                f"仍然想要{phrase}，"
                "请按照新预算重新筛选。"
            ),
            "expected_route": (
                "research"
            ),
            "expected_tool": (
                "rerank_search_products"
            ),
            "hard_constraints": {
                "category": category,
                "max_price": (
                    new_budget
                ),
            },
            "semantic_tags": (
                semantic_tags
            ),
            "relevant_product_ids": (
                relevant_ids
            ),
            "eval_k": 5,
            "min_results": 1,
        }

        result.append(
            {
                "id": (
                    "followup_new_constraint_"
                    f"{index+1:02d}"
                ),
                "type": (
                    "multi_turn_new_constraint"
                ),
                "description": (
                    "第二轮新增预算限制，"
                    "必须重新研究"
                ),
                "turns": [
                    first_turn,
                    second_turn,
                ],
            }
        )

    return result


def main() -> None:
    args = parse_args()

    catalog_path = Path(
        args.catalog
    )

    output_path = Path(
        args.output
    )

    products = load_catalog(
        catalog_path
    )

    rng = random.Random(
        args.seed
    )

    structured = (
        make_structured_cases(
            products,
            count=12,
            rng=rng,
        )
    )

    semantic = (
        make_semantic_cases(
            products,
            count=12,
            rng=rng,
        )
    )

    details = (
        make_detail_cases(
            products,
            count=4,
            rng=rng,
        )
    )

    comparisons = (
        make_compare_cases(
            products,
            count=4,
            rng=rng,
        )
    )

    followups = (
        make_followup_cases(
            products,
            semantic,
            no_research_count=4,
            new_constraint_count=4,
        )
    )

    scenarios = [
        *structured,
        *semantic,
        *details,
        *comparisons,
        *followups,
    ]

    turn_count = sum(
        len(
            scenario[
                "turns"
            ]
        )
        for scenario
        in scenarios
    )

    payload = {
        "metadata": {
            "name": (
                "Day 20 Real Catalog "
                "Agent Evaluation"
            ),
            "seed": (
                args.seed
            ),
            "catalog": str(
                catalog_path
            ),
            "catalog_products": (
                len(
                    products
                )
            ),
            "scenario_count": (
                len(
                    scenarios
                )
            ),
            "turn_count": (
                turn_count
            ),
            "scenario_mix": {
                "structured_search": 12,
                "semantic_recommendation": 12,
                "product_detail": 4,
                "product_compare": 4,
                "multi_turn_reuse": 4,
                "multi_turn_new_constraint": 4,
            },
        },
        "scenarios": (
            scenarios
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("DAY 20 REAL AGENT EVAL BUILDER")
    print("=" * 72)
    print(
        "catalog products:",
        len(
            products
        ),
    )
    print(
        "scenarios:",
        len(
            scenarios
        ),
    )
    print(
        "turns:",
        turn_count,
    )
    print(
        "output:",
        output_path,
    )


if __name__ == "__main__":
    main()
