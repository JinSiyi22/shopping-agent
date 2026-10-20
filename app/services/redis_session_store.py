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


class RedisSessionStore:

    def __init__(self):

        # 注意：
        # Redis负责Session Registry。
        #
        # asyncio.Lock仍然是当前进程内锁。
        # Day15暂时不做Distributed Lock。
        self._locks: dict[
            str,
            asyncio.Lock
        ] = {}

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
            get_redis_client()
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
            get_redis_client()
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
            get_redis_client()
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
            get_redis_client()
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

        return self._locks[
            session_id
        ]

    async def clear(
        self,
        session_id: str,
    ) -> None:

        client = (
            get_redis_client()
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
    RedisSessionStore()
)