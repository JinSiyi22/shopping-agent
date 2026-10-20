import asyncio
import os
from uuid import uuid4

from app.infrastructure.redis_client import (
    get_redis_client,
)


SESSION_TTL_SECONDS = int(
    os.getenv(
        "SESSION_TTL_SECONDS",
        "86400",
    )
)


class SessionStore:
    """
    Redis:
        保存 Session Registry 和 TTL，可跨进程共享。

    asyncio.Lock:
        仍然只负责当前 Python 进程内，
        同一 session 的请求互斥。

    注意：
        这还不是 Distributed Lock。
        多 Uvicorn Worker 的跨进程互斥会在后续单独处理。
    """

    def __init__(
        self,
        redis_client=None,
    ):

        # 测试时允许注入 Fake Redis；
        # 正常运行时使用共享 Redis Client。
        self._redis_client = (
            redis_client
        )

        self._locks: dict[
            str,
            asyncio.Lock,
        ] = {}

    def _client(self):

        if self._redis_client is not None:
            return self._redis_client

        return (
            get_redis_client()
        )

    @staticmethod
    def _key(
        session_id: str,
    ) -> str:

        return (
            f"session:{session_id}"
        )

    async def create_session(
        self,
    ) -> str:

        session_id = str(
            uuid4()
        )

        client = (
            self._client()
        )

        await client.set(
            self._key(
                session_id
            ),
            "active",
            ex=SESSION_TTL_SECONDS,
        )

        self._locks[
            session_id
        ] = asyncio.Lock()

        return session_id

    async def register_session(
        self,
        session_id: str,
    ) -> None:

        client = (
            self._client()
        )

        await client.set(
            self._key(
                session_id
            ),
            "active",
            ex=SESSION_TTL_SECONDS,
        )

        self._locks.setdefault(
            session_id,
            asyncio.Lock(),
        )

    async def exists(
        self,
        session_id: str,
    ) -> bool:

        client = (
            self._client()
        )

        exists = await client.exists(
            self._key(
                session_id
            )
        )

        return bool(
            exists
        )

    async def touch(
        self,
        session_id: str,
    ) -> None:

        client = (
            self._client()
        )

        await client.expire(
            self._key(
                session_id
            ),
            SESSION_TTL_SECONDS,
        )

    def get_lock(
        self,
        session_id: str,
    ) -> asyncio.Lock:

        if (
            session_id
            not in self._locks
        ):

            self._locks[
                session_id
            ] = asyncio.Lock()

        return (
            self._locks[
                session_id
            ]
        )

    async def clear(
        self,
        session_id: str,
    ) -> None:

        client = (
            self._client()
        )

        await client.delete(
            self._key(
                session_id
            )
        )

        self._locks.pop(
            session_id,
            None,
        )


session_store = (
    SessionStore()
)
