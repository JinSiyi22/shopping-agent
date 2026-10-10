import asyncio
from uuid import uuid4


class SessionStore:

    def __init__(self):
        self._sessions: dict[
            str,
            list[dict]
        ] = {}

        self._locks: dict[
            str,
            asyncio.Lock
        ] = {}

    def create_session(self) -> str:
        session_id = str(uuid4())

        self._sessions[session_id] = []
        self._locks[session_id] = (
            asyncio.Lock()
        )

        return session_id

    def exists(
        self,
        session_id: str,
    ) -> bool:
        return session_id in self._sessions

    def get_messages(
        self,
        session_id: str,
    ) -> list[dict]:

        return self._sessions.setdefault(
            session_id,
            [],
        )

    def get_lock(
        self,
        session_id: str,
    ) -> asyncio.Lock:

        if session_id not in self._locks:
            self._locks[session_id] = (
                asyncio.Lock()
            )

        return self._locks[session_id]

    def clear(
        self,
        session_id: str,
    ) -> None:

        self._sessions.pop(
            session_id,
            None,
        )

        self._locks.pop(
            session_id,
            None,
        )


session_store = SessionStore()