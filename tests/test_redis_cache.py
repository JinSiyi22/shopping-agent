from app.services.redis_cache import (
    build_tool_cache_key,
)


def test_tool_cache_key_is_order_independent():

    key_a = build_tool_cache_key(
        "rerank_search_products",
        {
            "max_price": 10000,
            "query": "轻薄 编程",
        },
    )

    key_b = build_tool_cache_key(
        "rerank_search_products",
        {
            "query": "轻薄 编程",
            "max_price": 10000,
        },
    )

    assert (
        key_a
        == key_b
    )


def test_tool_cache_key_changes_with_arguments():

    key_a = build_tool_cache_key(
        "rerank_search_products",
        {
            "max_price": 10000,
            "query": "轻薄 编程",
        },
    )

    key_b = build_tool_cache_key(
        "rerank_search_products",
        {
            "max_price": 8000,
            "query": "轻薄 编程",
        },
    )

    assert (
        key_a
        != key_b
    )
