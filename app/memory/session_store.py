import asyncio
from uuid import uuid4


class SessionStore:

    def __init__(self):
        self._session_ids: set[str] = set()

        self._locks: dict[
            str,
            asyncio.Lock
        ] = {}

    def create_session(self) -> str:
        session_id = str(uuid4())

        self._session_ids.add(
            session_id
        )

        self._locks[session_id] = (
            asyncio.Lock()
        )

        return session_id

    def exists(
        self,
        session_id: str,
    ) -> bool:

        return (
            session_id
            in self._session_ids
        )

    def get_lock(
        self,
        session_id: str,
    ) -> asyncio.Lock:

        if (
            session_id
            not in self._locks
        ):
            self._locks[session_id] = (
                asyncio.Lock()
            )

        return self._locks[
            session_id
        ]

    def clear(
        self,
        session_id: str,
    ) -> None:

        self._session_ids.discard(
            session_id
        )

        self._locks.pop(
            session_id,
            None,
        )


session_store = SessionStore()