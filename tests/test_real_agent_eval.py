from app.evaluation.real_agent_eval import (
    build_unique_alias_index,
    constraint_pass,
    grounding_metrics,
    hit_at_k,
    recall_at_k,
    semantic_tag_hit_at_k,
)


PRODUCTS = [
    {
        "id": "B000000001",
        "name": (
            "Example Wireless Headphone "
            "Model Alpha Bluetooth"
        ),
        "category": "headphone",
        "brand": "BrandA",
        "price": 99.0,
        "rating": 4.5,
        "tags": [
            "wireless",
            "bluetooth",
        ],
        "description": "Wireless headphone.",
    },
    {
        "id": "B000000002",
        "name": (
            "Example Gaming Mouse "
            "Model Beta Wireless"
        ),
        "category": "mouse",
        "brand": "BrandB",
        "price": 49.0,
        "rating": 4.2,
        "tags": [
            "gaming",
        ],
        "description": "Gaming mouse.",
    },
]

PRODUCT_MAP = {
    product["id"]: product
    for product in PRODUCTS
}

ALIAS_INDEX = build_unique_alias_index(
    PRODUCTS
)


def test_constraint_pass_success():
    assert constraint_pass(
        ["B000000001"],
        PRODUCT_MAP,
        {
            "brand": "BrandA",
            "category": "headphone",
            "max_price": 100,
            "min_rating": 4.0,
        },
        min_results=1,
    )


def test_constraint_pass_rejects_wrong_category():
    assert not constraint_pass(
        ["B000000002"],
        PRODUCT_MAP,
        {
            "category": "headphone",
        },
        min_results=1,
    )


def test_recall_and_hit():
    assert (
        recall_at_k(
            ["B000000001"],
            [
                "B000000001",
                "B000000002",
            ],
            5,
        )
        == 0.5
    )

    assert (
        hit_at_k(
            ["B000000001"],
            ["B000000002"],
            5,
        )
        == 0.0
    )


def test_semantic_tag_hit():
    assert (
        semantic_tag_hit_at_k(
            ["B000000001"],
            PRODUCT_MAP,
            ["bluetooth"],
            5,
        )
        == 1.0
    )


def test_grounding_shortened_unique_title_detected():
    result = grounding_metrics(
        (
            "推荐 Example Wireless Headphone "
            "Model Alpha。"
        ),
        ALIAS_INDEX,
        {"B000000001"},
    )

    assert (
        result[
            "evidence_product_mentioned"
        ]
        == 1.0
    )

    assert (
        result[
            "catalog_product_grounding_pass"
        ]
        is True
    )


def test_grounding_detects_unsupported_product():
    result = grounding_metrics(
        (
            "推荐 Example Wireless Headphone "
            "Model Alpha，也可以看看 "
            "Example Gaming Mouse Model Beta。"
        ),
        ALIAS_INDEX,
        {"B000000001"},
    )

    assert (
        result[
            "catalog_product_grounding_pass"
        ]
        is False
    )

    assert (
        "B000000002"
        in result[
            "unsupported_product_ids"
        ]
    )


def test_no_identifiable_product_not_auto_passed():
    result = grounding_metrics(
        "这几款都值得考虑。",
        ALIAS_INDEX,
        {"B000000001"},
    )

    assert (
        result[
            "catalog_product_grounding_pass"
        ]
        is None
    )

    assert (
        result[
            "evidence_product_mentioned"
        ]
        == 0.0
    )


def test_grounding_does_not_match_generic_three_token_prefix():

    products = [
        {
            "id": "target",
            "name": (
                "1080P 60FPS Webcam with Microphone, "
                "Angetube 920H Pro USB HD Computer Web Camera"
            ),
            "category": "camera",
            "brand": "Angetube",
            "price": 46.99,
            "rating": 3.7,
            "tags": [],
            "description": "",
        },
        {
            "id": "other",
            "name": (
                "Webcam With Microphone for Desktop, "
                "1080p Webcam Built-in Mic"
            ),
            "category": "camera",
            "brand": "Other",
            "price": 20.0,
            "rating": 4.0,
            "tags": [],
            "description": "",
        },
    ]

    alias_index = (
        build_unique_alias_index(
            products
        )
    )

    result = grounding_metrics(
        (
            "推荐 1080P 60FPS Webcam "
            "with Microphone, "
            "Angetube 920H Pro。"
        ),
        alias_index,
        {"target"},
    )

    assert (
        "target"
        in result[
            "mentioned_product_ids"
        ]
    )

    assert (
        "other"
        not in result[
            "mentioned_product_ids"
        ]
    )

    assert (
        result[
            "catalog_product_grounding_pass"
        ]
        is True
    )
