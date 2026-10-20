import logging
from contextlib import (
    asynccontextmanager,
)

from fastapi import (
    FastAPI,
    HTTPException,
)

from app.agents.multi_agent import (
    clear_agent_session,
    close_multi_agent_storage,
    get_agent_session_state,
    run_agent,
    setup_multi_agent_storage,
)

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
)

from app.infrastructure.redis_client import (
    close_redis_client,
    ping_redis,
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

logger = logging.getLogger(
    __name__
)


# ---------------------------------
# Application Lifespan
# ---------------------------------

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    logger.info(
        "application_storage_starting"
    )

    redis_ok = await (
        ping_redis()
    )

    if not redis_ok:

        raise RuntimeError(
            "Redis is unavailable"
        )

    await (
        setup_multi_agent_storage()
    )

    logger.info(
        "application_storage_ready"
    )

    try:

        yield

    finally:

        logger.info(
            "application_storage_closing"
        )

        # Checkpointer 使用独立 Redis Client，
        # 所以单独关闭。
        await (
            close_multi_agent_storage()
        )

        # Session / Tool Cache 共用该 Client。
        await (
            close_redis_client()
        )

        logger.info(
            "application_storage_closed"
        )


# ---------------------------------
# FastAPI App
# ---------------------------------

app = FastAPI(
    title=(
        "Multi-Agent "
        "Shopping Copilot"
    ),
    description=(
        "An AI agent system for "
        "product research and "
        "purchase decision support."
    ),
    version="0.2.0",
    lifespan=lifespan,
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
        "version": "0.2.0",
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
        or not await (
            session_store.exists(
                session_id
            )
        )
    ):

        session_id = await (
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
            user_query=(
                request.message
            ),
            session_id=(
                session_id
            ),
        )

        # 活跃会话刷新 TTL。
        await (
            session_store
            .touch(
                session_id
            )
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

    if not await (
        session_store.exists(
            session_id
        )
    ):

        raise HTTPException(
            status_code=404,
            detail="Session not found",
        )

    await (
        clear_agent_session(
            session_id
        )
    )

    await (
        session_store.clear(
            session_id
        )
    )

    return {
        "message": (
            "Session deleted"
        ),
        "session_id": session_id,
    }


# ---------------------------------
# Session State
# ---------------------------------

@app.get(
    "/sessions/{session_id}/state"
)
async def get_session_state(
    session_id: str,
):

    if not await (
        session_store.exists(
            session_id
        )
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
