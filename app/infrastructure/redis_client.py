import os

import redis.asyncio as redis
from dotenv import load_dotenv


load_dotenv()


REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0",
)


_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    """
    返回进程级共享 Redis Client。

    redis-py 内部会维护连接池，因此不要为每个 HTTP 请求
    都重新创建一个 Redis Client。
    """

    global _redis_client

    if _redis_client is None:

        _redis_client = (
            redis.Redis.from_url(
                REDIS_URL,
                decode_responses=True,
                max_connections=20,
            )
        )

    return _redis_client


async def ping_redis() -> bool:

    client = (
        get_redis_client()
    )

    result = await (
        client.ping()
    )

    return bool(
        result
    )


async def close_redis_client() -> None:

    global _redis_client

    if _redis_client is None:
        return

    await (
        _redis_client
        .aclose()
    )

    _redis_client = None
