from fastapi import (
    FastAPI,
    HTTPException,
)

from app.agents.multi_agent import (
    clear_agent_session,
    get_agent_session_state,
    run_agent,
)

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
)

from app.memory.session_store import (
    session_store,
)

from app.observability.logging_config import (
    configure_logging,
)


# ---------------------------------
# 初始化 Logging
# ---------------------------------

configure_logging()


# ---------------------------------
# FastAPI App
# ---------------------------------

app = FastAPI(
    title="Multi-Agent Shopping Copilot",
    description=(
        "An AI agent system for "
        "product research and "
        "purchase decision support."
    ),
    version="0.1.0",
)


# ---------------------------------
# Root
# ---------------------------------

@app.get("/")
def root():
    return {
        "name": (
            "Multi-Agent "
            "Shopping Copilot"
        ),
        "version": "0.1.0",
        "docs": "/docs",
    }


# ---------------------------------
# Health Check
# ---------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


# ---------------------------------
# Chat
# ---------------------------------

@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
) -> ChatResponse:

    session_id = (
        request.session_id
    )

    if (
        session_id is None
        or not session_store.exists(
            session_id
        )
    ):
        session_id = (
            session_store
            .create_session()
        )

    lock = (
        session_store
        .get_lock(
            session_id
        )
    )

    async with lock:

        answer = await run_agent(
            user_query=request.message,
            session_id=session_id,
        )

    return ChatResponse(
        session_id=session_id,
        answer=answer,
    )


# ---------------------------------
# Delete Session
# ---------------------------------

@app.delete(
    "/sessions/{session_id}"
)
async def delete_session(
    session_id: str,
):

    if not session_store.exists(
        session_id
    ):
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        )

    await clear_agent_session(
        session_id
    )

    session_store.clear(
        session_id
    )

    return {
        "message": (
            "Session deleted"
        ),
        "session_id": session_id,
    }


@app.get(
    "/sessions/{session_id}/state"
)
async def get_session_state(
    session_id: str,
):

    if not session_store.exists(
        session_id
    ):
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        )

    return await (
        get_agent_session_state(
            session_id
        )
    )