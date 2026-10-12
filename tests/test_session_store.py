from app.memory.session_store import (
    SessionStore,
)


def test_create_session():

    store = SessionStore()

    session_id = (
        store.create_session()
    )

    assert store.exists(
        session_id
    )


def test_session_isolation():

    store = SessionStore()

    session_a = (
        store.create_session()
    )

    session_b = (
        store.create_session()
    )

    messages_a = (
        store.get_messages(
            session_a
        )
    )

    messages_b = (
        store.get_messages(
            session_b
        )
    )

    messages_a.append(
        {
            "role": "user",
            "content": "预算10000",
        }
    )

    assert len(messages_a) == 1
    assert len(messages_b) == 0


def test_session_locks_are_independent():

    store = SessionStore()

    session_a = (
        store.create_session()
    )

    session_b = (
        store.create_session()
    )

    lock_a = store.get_lock(
        session_a
    )

    lock_b = store.get_lock(
        session_b
    )

    assert lock_a is not lock_b


def test_clear_session():

    store = SessionStore()

    session_id = (
        store.create_session()
    )

    store.clear(session_id)

    assert not store.exists(
        session_id
    )