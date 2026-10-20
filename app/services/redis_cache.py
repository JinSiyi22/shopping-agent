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


def build_tool_cache_key(
    tool_name: str,
    arguments: dict,
) -> str:
    """
    使用：
        tool_name + 排序后的 JSON arguments
    构造稳定缓存键。

    sort_keys=True 保证参数字典顺序不同，
    但语义相同时得到相同 hash。
    """

    payload = {
        "tool": tool_name,
        "arguments": arguments,
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

        # 缓存值如果意外损坏，
        # 删除该 Key，让本次请求回退为 Cache Miss。
        await client.delete(
            key
        )

        return None


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
