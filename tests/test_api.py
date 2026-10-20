from fastapi.testclient import (
    TestClient,
)

from app.api.main import (
    app,
    session_store,
)


client = TestClient(
    app
)


def test_health():

    response = client.get(
        "/health"
    )

    assert (
        response.status_code
        == 200
    )

    assert response.json() == {
        "status": "ok"
    }


def test_empty_message():

    response = client.post(
        "/chat",
        json={
            "message": ""
        },
    )

    assert (
        response.status_code
        == 422
    )


def test_chat(
    monkeypatch,
):

    async def fake_run_agent(
        user_query,
        session_id,
        max_steps=8,
    ):

        return (
            f"mock answer: "
            f"{user_query}"
        )

    async def fake_exists(
        session_id,
    ):

        return False

    async def fake_create_session():

        return (
            "test-session"
        )

    async def fake_touch(
        session_id,
    ):

        return None

    monkeypatch.setattr(
        "app.api.main.run_agent",
        fake_run_agent,
    )

    monkeypatch.setattr(
        session_store,
        "exists",
        fake_exists,
    )

    monkeypatch.setattr(
        session_store,
        "create_session",
        fake_create_session,
    )

    monkeypatch.setattr(
        session_store,
        "touch",
        fake_touch,
    )

    response = client.post(
        "/chat",
        json={
            "message": (
                "推荐电脑"
            )
        },
    )

    assert (
        response.status_code
        == 200
    )

    data = (
        response.json()
    )

    assert (
        data["session_id"]
        == "test-session"
    )

    assert (
        data["answer"]
        == "mock answer: 推荐电脑"
    )
