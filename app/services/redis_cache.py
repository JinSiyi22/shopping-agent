import hashlib
import json
import os

from app.infrastructure.redis_client import (
    get_redis_client,
)


TOOL_CACHE_TTL_SECONDS = int(
    os.getenv(
        "TOOL_CACHE_TTL_SECONDS",
        "1800",
    )
)


CACHEABLE_TOOLS = {
    "search_products",
    "semantic_search_products",
    "hybrid_search_products",
    "rerank_search_products",
    "get_product_detail",
    "compare_products",
}


# ============================================================
# Tool Default Arguments
# ============================================================

TOOL_DEFAULT_ARGUMENTS = {

    "search_products": {
        "top_k": 5,
    },

    "semantic_search_products": {
        "top_k": 5,
    },

    "hybrid_search_products": {
        "top_k": 5,
    },

    "rerank_search_products": {
        "top_k": 5,
    },
}


# ============================================================
# Argument Canonicalization
# ============================================================

def canonicalize_tool_arguments(
    tool_name: str,
    arguments: dict,
) -> dict:
    """
    将语义等价的 Tool Arguments 规范化为相同形式。

    例如：

    {
        "query": "轻薄 编程",
        "max_price": 10000
    }

    和：

    {
        "query": "轻薄 编程",
        "max_price": 10000,
        "top_k": 5
    }

    对 rerank_search_products 来说执行效果相同，
    因此应该生成相同的 Cache Key。
    """

    normalized = {}

    # ----------------------------------------
    # 1. 清理无意义的 optional values
    # ----------------------------------------

    for key, value in arguments.items():

        if value is None:
            continue

        if (
            isinstance(value, str)
            and not value.strip()
        ):
            continue

        if (
            isinstance(value, list)
            and len(value) == 0
        ):
            continue

        normalized[
            key
        ] = value

    # ----------------------------------------
    # 2. 补齐 Tool 默认参数
    # ----------------------------------------

    defaults = (
        TOOL_DEFAULT_ARGUMENTS.get(
            tool_name,
            {},
        )
    )

    for key, default_value in (
        defaults.items()
    ):

        normalized.setdefault(
            key,
            default_value,
        )

    return normalized


# ============================================================
# Cache Key
# ============================================================

def build_tool_cache_key(
    tool_name: str,
    arguments: dict,
) -> str:

    canonical_arguments = (
        canonicalize_tool_arguments(
            tool_name,
            arguments,
        )
    )

    payload = {
        "tool": tool_name,
        "arguments": (
            canonical_arguments
        ),
    }

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    digest = hashlib.sha256(
        serialized.encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        f"tool_cache:{digest}"
    )


# ============================================================
# Cache GET
# ============================================================

async def get_cached_tool_result(
    tool_name: str,
    arguments: dict,
):

    if (
        tool_name
        not in CACHEABLE_TOOLS
    ):
        return None

    client = (
        get_redis_client()
    )

    key = (
        build_tool_cache_key(
            tool_name,
            arguments,
        )
    )

    raw_value = await (
        client.get(
            key
        )
    )

    if raw_value is None:
        return None

    try:

        return json.loads(
            raw_value
        )

    except json.JSONDecodeError:

        await client.delete(
            key
        )

        return None


# ============================================================
# Cache SET
# ============================================================

async def set_cached_tool_result(
    tool_name: str,
    arguments: dict,
    result,
) -> None:

    if (
        tool_name
        not in CACHEABLE_TOOLS
    ):
        return

    client = (
        get_redis_client()
    )

    key = (
        build_tool_cache_key(
            tool_name,
            arguments,
        )
    )

    payload = json.dumps(
        result,
        ensure_ascii=False,
    )

    await client.set(
        key,
        payload,
        ex=TOOL_CACHE_TTL_SECONDS,
    )