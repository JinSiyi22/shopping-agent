from fastapi import FastAPI

from app.agents.shopping_agent import run_agent
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
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
def chat(
    request: ChatRequest,
) -> ChatResponse:

    answer = run_agent(
        user_query=request.message
    )

    return ChatResponse(
        answer=answer
    )