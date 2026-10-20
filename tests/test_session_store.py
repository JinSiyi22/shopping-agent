import asyncio

from app.memory.session_store import (
    SessionStore,
)


class FakeRedis:

    def __init__(self):

        self.data = {}

        self.ttls = {}

    async def set(
        self,
        key,
        value,
        ex=None,
    ):

        self.data[
            key
        ] = value

        if ex is not None:

            self.ttls[
                key
            ] = ex

        return True

    async def exists(
        self,
        key,
    ):

        return int(
            key in self.data
        )

    async def expire(
        self,
        key,
        seconds,
    ):

        if key not in self.data:
            return False

        self.ttls[
            key
        ] = seconds

        return True

    async def delete(
        self,
        key,
    ):

        existed = int(
            key in self.data
        )

        self.data.pop(
            key,
            None,
        )

        self.ttls.pop(
            key,
            None,
        )

        return existed


def test_create_session():

    async def run():

        store = SessionStore(
            redis_client=FakeRedis()
        )

        session_id = await (
            store.create_session()
        )

        assert await (
            store.exists(
                session_id
            )
        )

    asyncio.run(
        run()
    )


def test_sessions_are_unique():

    async def run():

        store = SessionStore(
            redis_client=FakeRedis()
        )

        session_a = await (
            store.create_session()
        )

        session_b = await (
            store.create_session()
        )

        assert (
            session_a
            != session_b
        )

    asyncio.run(
        run()
    )


def test_session_locks_are_independent():

    async def run():

        store = SessionStore(
            redis_client=FakeRedis()
        )

        session_a = await (
            store.create_session()
        )

        session_b = await (
            store.create_session()
        )

        lock_a = (
            store.get_lock(
                session_a
            )
        )

        lock_b = (
            store.get_lock(
                session_b
            )
        )

        assert (
            lock_a
            is not lock_b
        )

    asyncio.run(
        run()
    )


def test_touch_session():

    async def run():

        fake_redis = FakeRedis()

        store = SessionStore(
            redis_client=fake_redis
        )

        session_id = await (
            store.create_session()
        )

        await (
            store.touch(
                session_id
            )
        )

        assert (
            f"session:{session_id}"
            in fake_redis.ttls
        )

    asyncio.run(
        run()
    )


def test_clear_session():

    async def run():

        store = SessionStore(
            redis_client=FakeRedis()
        )

        session_id = await (
            store.create_session()
        )

        await (
            store.clear(
                session_id
            )
        )

        assert not await (
            store.exists(
                session_id
            )
        )

    asyncio.run(
        run()
    )
