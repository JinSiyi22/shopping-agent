import json
import logging
import os
from operator import add
from time import perf_counter
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from openai import AsyncOpenAI

from app.observability.trace import (
    AgentTrace,
    TraceEvent,
)
from app.tools.product_tools import (
    calculate_shipping_tool,
    compare_products_tool,
    get_product_detail_tool,
    search_products_tool,
)


load_dotenv()


logger = logging.getLogger(__name__)


MODEL = os.getenv("LLM_MODEL")


# ============================================================
# LLM Client
# ============================================================

_llm_client: AsyncOpenAI | None = None


def get_llm_client() -> AsyncOpenAI:
    global _llm_client

    if _llm_client is not None:
        return _llm_client

    api_key = os.getenv(
        "LLM_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY is not configured"
        )

    _llm_client = AsyncOpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "LLM_BASE_URL"
        ) or None,
    )

    return _llm_client


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
你是一个专业的商品研究与购买决策助手。

你的任务是根据用户需求，
使用商品搜索、详情查询、商品比较和运费计算工具，
帮助用户完成商品研究和购买决策。

工作原则：

1. 用户要求推荐、搜索或比较商品时，
   优先调用工具获取真实商品数据。

2. 不要编造商品数据库中不存在的信息。

3. 如果完成任务需要多个步骤，
   可以连续调用多个工具。

4. 搜索后如果还需要进一步分析，
   可以继续查询商品详情或进行商品比较。

5. 用户自然语言中的需求可以转换为标准标签，例如：
   - 写代码、开发 -> 编程
   - AI、机器学习、小模型训练 -> GPU、高性能
   - 出差、便携 -> 轻薄
   - 日常工作 -> 办公

6. 如果工具返回错误，
   根据错误信息决定下一步，而不是编造数据。

7. calculate_shipping 返回的是开发阶段模拟运费，
   最终回答时需要明确说明。

8. 如果已有信息已经足够回答问题，
   不需要为了调用工具而调用工具。
"""


# ============================================================
# Tool Schema
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "根据用户预算、品牌、类别、评分和用途标签"
                "搜索商品。当用户需要推荐、搜索或筛选商品时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_price": {
                        "type": "number",
                        "description": "用户可接受的最高价格",
                    },
                    "min_price": {
                        "type": "number",
                        "description": "用户要求的最低价格",
                    },
                    "brand": {
                        "type": "string",
                        "description": "指定商品品牌",
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "商品类别，例如 laptop"
                        ),
                    },
                    "tags": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": (
                            "商品需求标签，例如："
                            "编程、轻薄、办公、商务、"
                            "游戏、GPU、高性能"
                        ),
                    },
                    "min_rating": {
                        "type": "number",
                        "description": (
                            "最低商品评分，范围0到5"
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": (
                            "最多返回多少个商品"
                        ),
                        "default": 5,
                    },
                },
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_product_detail",
            "description": (
                "根据商品ID获取一个商品的完整详细信息。"
                "当已经知道商品ID，需要进一步查看商品时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "商品唯一ID",
                    },
                },
                "required": [
                    "product_id"
                ],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "compare_products",
            "description": (
                "获取多个商品的结构化信息，"
                "用于比较价格、评分、标签和用途。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": (
                            "需要比较的商品ID列表"
                        ),
                    },
                },
                "required": [
                    "product_ids"
                ],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "calculate_shipping",
            "description": (
                "根据商品价格和目的地估算运费与总价格。"
                "当前返回开发阶段的模拟运费。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "商品价格",
                    },
                    "destination": {
                        "type": "string",
                        "description": (
                            "用户的收货国家或地区"
                        ),
                    },
                },
                "required": [
                    "price",
                    "destination",
                ],
            },
        },
    },
]


# ============================================================
# Tool Registry
# ============================================================

TOOL_REGISTRY = {
    "search_products": (
        search_products_tool
    ),
    "get_product_detail": (
        get_product_detail_tool
    ),
    "compare_products": (
        compare_products_tool
    ),
    "calculate_shipping": (
        calculate_shipping_tool
    ),
}


def execute_tool(
    tool_name: str,
    arguments: dict,
):
    tool_function = (
        TOOL_REGISTRY.get(
            tool_name
        )
    )

    if tool_function is None:
        return {
            "error": (
                f"Unknown tool: {tool_name}"
            )
        }

    try:
        return tool_function(
            **arguments
        )

    except Exception as exc:
        logger.exception(
            "tool_execution_failed "
            "tool=%s",
            tool_name,
        )

        return {
            "error": str(exc),
            "tool": tool_name,
        }


# ============================================================
# LangGraph State
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[
        list[dict],
        add,
    ]

    step: int
    max_steps: int

    trace: AgentTrace


# ============================================================
# Node 1: Agent Node
# ============================================================

async def agent_node(
    state: AgentState,
) -> dict:

    step = state["step"] + 1
    trace = state["trace"]

    logger.info(
        "agent_node_started "
        "trace_id=%s "
        "step=%s",
        trace.trace_id,
        step,
    )

    client = get_llm_client()

    if not MODEL:
        raise RuntimeError(
            "LLM_MODEL is not configured"
        )

    llm_start = perf_counter()

    try:
        response = await (
            client
            .chat
            .completions
            .create(
                model=MODEL,
                messages=state["messages"],
                tools=TOOLS,
                tool_choice="auto",
            )
        )

    except Exception:

        llm_latency = (
            perf_counter()
            - llm_start
        ) * 1000

        trace.add_event(
            TraceEvent(
                step=step,
                event_type="llm",
                name="chat_completion",
                latency_ms=llm_latency,
                success=False,
            )
        )

        logger.exception(
            "llm_call_failed "
            "trace_id=%s "
            "step=%s "
            "latency_ms=%.2f",
            trace.trace_id,
            step,
            llm_latency,
        )

        raise

    llm_latency = (
        perf_counter()
        - llm_start
    ) * 1000

    trace.add_event(
        TraceEvent(
            step=step,
            event_type="llm",
            name="chat_completion",
            latency_ms=llm_latency,
            success=True,
        )
    )

    logger.info(
        "llm_call_finished "
        "trace_id=%s "
        "step=%s "
        "latency_ms=%.2f",
        trace.trace_id,
        step,
        llm_latency,
    )

    assistant_message = (
        response
        .choices[0]
        .message
    )

    message_dict = (
        assistant_message
        .model_dump(
            exclude_none=True
        )
    )

    return {
        "messages": [
            message_dict
        ],
        "step": step,
    }


# ============================================================
# Router 1:
# Agent → Tool / END
# ============================================================

def route_after_agent(
    state: AgentState,
):
    last_message = (
        state["messages"][-1]
    )

    tool_calls = (
        last_message.get(
            "tool_calls"
        )
    )

    if tool_calls:
        logger.info(
            "route "
            "trace_id=%s "
            "from=agent "
            "to=tools",
            state["trace"].trace_id,
        )

        return "tools"

    logger.info(
        "route "
        "trace_id=%s "
        "from=agent "
        "to=end",
        state["trace"].trace_id,
    )

    return END


# ============================================================
# Node 2: Tool Node
# ============================================================

async def tool_node(
    state: AgentState,
) -> dict:

    trace = state["trace"]
    step = state["step"]

    last_message = (
        state["messages"][-1]
    )

    tool_calls = (
        last_message.get(
            "tool_calls"
        )
        or []
    )

    tool_messages = []

    for tool_call in tool_calls:

        tool_name = (
            tool_call["function"]["name"]
        )

        try:
            arguments = json.loads(
                tool_call[
                    "function"
                ]["arguments"]
            )

        except json.JSONDecodeError:

            tool_result = {
                "error": (
                    "Invalid JSON arguments"
                )
            }

            trace.add_event(
                TraceEvent(
                    step=step,
                    event_type="tool",
                    name=tool_name,
                    success=False,
                )
            )

            logger.warning(
                "invalid_tool_arguments "
                "trace_id=%s "
                "step=%s "
                "tool=%s",
                trace.trace_id,
                step,
                tool_name,
            )

        else:

            logger.info(
                "tool_node_started "
                "trace_id=%s "
                "step=%s "
                "tool=%s "
                "arguments=%s",
                trace.trace_id,
                step,
                tool_name,
                arguments,
            )

            tool_start = (
                perf_counter()
            )

            tool_result = (
                execute_tool(
                    tool_name,
                    arguments,
                )
            )

            tool_latency = (
                perf_counter()
                - tool_start
            ) * 1000

            tool_success = not (
                isinstance(
                    tool_result,
                    dict,
                )
                and "error"
                in tool_result
            )

            trace.add_event(
                TraceEvent(
                    step=step,
                    event_type="tool",
                    name=tool_name,
                    latency_ms=(
                        tool_latency
                    ),
                    success=(
                        tool_success
                    ),
                )
            )

            logger.info(
                "tool_node_finished "
                "trace_id=%s "
                "step=%s "
                "tool=%s "
                "success=%s "
                "latency_ms=%.2f",
                trace.trace_id,
                step,
                tool_name,
                tool_success,
                tool_latency,
            )

        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": (
                    tool_call["id"]
                ),
                "content": json.dumps(
                    tool_result,
                    ensure_ascii=False,
                ),
            }
        )

    return {
        "messages": tool_messages
    }


# ============================================================
# Router 2:
# Tool → Agent / Limit
# ============================================================

def route_after_tools(
    state: AgentState,
):
    if (
        state["step"]
        >= state["max_steps"]
    ):
        logger.warning(
            "route "
            "trace_id=%s "
            "from=tools "
            "to=limit "
            "step=%s",
            state["trace"].trace_id,
            state["step"],
        )

        return "limit"

    logger.info(
        "route "
        "trace_id=%s "
        "from=tools "
        "to=agent",
        state["trace"].trace_id,
    )

    return "agent"


# ============================================================
# Node 3: Limit Node
# ============================================================

def limit_node(
    state: AgentState,
) -> dict:

    message = {
        "role": "assistant",
        "content": (
            "达到最大 Agent 执行步数，"
            "任务未能正常结束。"
        ),
    }

    return {
        "messages": [
            message
        ]
    }


# ============================================================
# Build LangGraph
# ============================================================

def build_shopping_graph():

    builder = StateGraph(
        AgentState
    )

    # Nodes
    builder.add_node(
        "agent",
        agent_node,
    )

    builder.add_node(
        "tools",
        tool_node,
    )

    builder.add_node(
        "limit",
        limit_node,
    )

    # START → Agent
    builder.add_edge(
        START,
        "agent",
    )

    # Agent → Tools / END
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        [
            "tools",
            END,
        ],
    )

    # Tools → Agent / Limit
    builder.add_conditional_edges(
        "tools",
        route_after_tools,
        [
            "agent",
            "limit",
        ],
    )

    # Limit → END
    builder.add_edge(
        "limit",
        END,
    )

    return builder.compile()


shopping_graph = (
    build_shopping_graph()
)


# ============================================================
# Public Agent Interface
# ============================================================

async def run_agent(
    user_query: str,
    conversation: list[dict],
    max_steps: int = 8,
) -> str:

    trace = AgentTrace()

    logger.info(
        "agent_started "
        "trace_id=%s",
        trace.trace_id,
    )

    # ----------------------------------
    # 准备原有 Session History
    # ----------------------------------

    initial_messages = list(
        conversation
    )

    if not initial_messages:
        initial_messages.append(
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        )

    initial_messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    # ----------------------------------
    # 执行 LangGraph
    # ----------------------------------

    result = await (
        shopping_graph.ainvoke(
            {
                "messages": (
                    initial_messages
                ),
                "step": 0,
                "max_steps": (
                    max_steps
                ),
                "trace": trace,
            }
        )
    )

    final_messages = (
        result["messages"]
    )

    # ----------------------------------
    # 更新当前 Session History
    # ----------------------------------

    conversation.clear()

    conversation.extend(
        final_messages
    )

    # ----------------------------------
    # 获取最终 Answer
    # ----------------------------------

    last_message = (
        final_messages[-1]
    )

    answer = (
        last_message.get(
            "content"
        )
        or ""
    )

    logger.info(
        "agent_finished "
        "trace_id=%s "
        "steps=%s "
        "events=%s "
        "total_latency_ms=%.2f",
        trace.trace_id,
        result["step"],
        len(trace.events),
        trace.total_latency_ms(),
    )

    return answer