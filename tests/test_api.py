from fastapi.testclient import (
    TestClient,
)

from app.api.main import app


client = TestClient(app)


def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

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

    monkeypatch.setattr(
        "app.api.main.run_agent",
        fake_run_agent,
    )

    response = client.post(
        "/chat",
        json={
            "message": "推荐电脑"
        },
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    assert "session_id" in data

    assert (
        data["answer"]
        == "mock answer: 推荐电脑"
    )