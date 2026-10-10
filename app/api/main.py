from fastapi import FastAPI
from fastapi import HTTPException

from app.agents.shopping_agent import run_agent
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
)
from app.memory.session_store import (
    session_store,
)


app = FastAPI(
    title="Multi-Agent Shopping Copilot",
    description=(
        "An AI agent system for product research "
        "and purchase decision support."
    ),
    version="0.1.0",
)

@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }

@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
) -> ChatResponse:

    session_id = request.session_id

    if (
        session_id is None
        or not session_store.exists(
            session_id
        )
    ):
        session_id = (
            session_store.create_session()
        )

    conversation = (
        session_store.get_messages(
            session_id
        )
    )

    lock = session_store.get_lock(
        session_id
    )

    async with lock:

        answer = await run_agent(
            user_query=request.message,
            conversation=conversation,
        )

    return ChatResponse(
        session_id=session_id,
        answer=answer,
    )

@app.get("/")
def root():
    return {
        "name": "Multi-Agent Shopping Copilot",
        "version": "0.1.0",
        "docs": "/docs",
    }

@app.delete(
    "/sessions/{session_id}"
)
def delete_session(
    session_id: str,
):

    if not session_store.exists(
        session_id
    ):
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        )

    session_store.clear(
        session_id
    )

    return {
        "message": "Session deleted",
        "session_id": session_id,
    }