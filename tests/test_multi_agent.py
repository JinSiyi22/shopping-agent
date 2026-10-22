from app.agents.multi_agent import (
    enrich_tool_arguments,
    infer_explicit_category,
)


def test_infer_laptop_category():

    result = infer_explicit_category(
        "预算1000美元以内，"
        "想买一台轻薄便携的笔记本电脑。"
    )

    assert result == "laptop"


def test_infer_headphone_category():

    result = infer_explicit_category(
        "我想买一副无线蓝牙耳机，"
        "希望支持主动降噪。"
    )

    assert result == "headphone"


def test_constraint_guard_adds_missing_category():

    result = enrich_tool_arguments(
        user_query=(
            "预算1000美元以内，"
            "想买轻薄笔记本电脑。"
        ),
        tool_name=(
            "rerank_search_products"
        ),
        arguments={
            "query": "轻薄 便携",
            "max_price": 1000,
        },
    )

    assert (
        result["category"]
        == "laptop"
    )


def test_constraint_guard_does_not_override_category():

    result = enrich_tool_arguments(
        user_query=(
            "给我推荐一台笔记本电脑"
        ),
        tool_name=(
            "rerank_search_products"
        ),
        arguments={
            "query": "test",
            "category": "desktop",
        },
    )

    assert (
        result["category"]
        == "desktop"
    )


def test_constraint_guard_ignores_ambiguous_multi_category():

    result = enrich_tool_arguments(
        user_query=(
            "笔记本电脑和平板电脑"
            "都给我看看"
        ),
        tool_name=(
            "rerank_search_products"
        ),
        arguments={
            "query": "便携办公",
        },
    )

    assert (
        "category"
        not in result
    )