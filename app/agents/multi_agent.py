import asyncio
import json
import logging
from dataclasses import dataclass
from operator import add
from time import perf_counter
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.agents.shopping_agent import (
    MODEL,
    call_llm_with_retry,
    execute_tool,
    get_llm_client,
    normalize_tool_name,
)
from app.observability.trace import (
    AgentTrace,
    TraceEvent,
)


logger = logging.getLogger(__name__)


# ============================================================
# Prompt: Supervisor Agent
# ============================================================

SUPERVISOR_PROMPT = """
你是购物 Multi-Agent 系统中的 Supervisor Agent。

你的任务不是直接推荐商品，
也不是自己调用商品工具。

你只负责判断：

当前用户请求是否需要新的商品研究。

你只能输出下面两个单词之一：

research
answer


选择 research 的情况：

- 用户要求搜索商品
- 用户要求推荐商品
- 用户提出新的预算、品牌、评分等条件
- 用户提出新的用途或场景需求
- 用户要求比较商品
- 用户询问某个商品的详细信息
- 当前对话信息不足以可靠回答问题


选择 answer 的情况：

- 当前已有对话信息已经足够回答
- 用户只是针对之前已经给出的结果追问解释
- 不需要重新访问商品数据库


禁止：

- 输出解释
- 输出 JSON
- 输出标点
- 输出除 research 或 answer 之外的任何内容
"""


# ============================================================
# Prompt: Research Agent
# ============================================================

RESEARCH_PROMPT = """
你是购物 Multi-Agent 系统中的 Research Agent。

你的职责只有一个：

根据用户当前需求，
选择最合适的一个工具获取商品事实。

你不是最终回答 Agent。
不要向用户输出长篇推荐。


工具选择原则：

1. 自然语言商品推荐：
   优先使用 rerank_search_products。

2. 纯结构化查询：
   可以使用 search_products。

3. 已知商品 ID，用户询问详细信息：
   使用 get_product_detail。

4. 已经明确给出多个商品 ID，
   并要求比较：
   使用 compare_products。

5. 用户询问运费或到手价：
   使用 calculate_shipping。


rerank_search_products 内部已经完成：

BM25 Recall
+
Dense Retrieval
+
RRF Fusion
+
Cross-Encoder Reranking

因此不要再先调用：

hybrid_search_products

然后又调用：

rerank_search_products。


重要：

一次 Research Agent 执行，
原则上只选择一个最有价值的工具。

价格、品牌、评分、类别等硬性条件，
必须传入结构化参数。

例如：

用户：
预算10000以内，
经常出差，
主要写代码，
希望轻薄。

应尽量调用：

rerank_search_products(
    query="经常出差 写代码 轻薄",
    max_price=10000
)

不要仅仅把：

“预算10000”

写入 query 文本。
"""


# ============================================================
# Prompt: Recommendation Agent
# ============================================================

RECOMMENDATION_PROMPT = """
你是购物 Multi-Agent 系统中的 Recommendation Agent。

你的职责是根据：

1. 用户当前问题
2. 历史对话
3. Research Agent 返回的结构化研究结果

生成清晰、有依据的最终回答。


要求：

- 不调用任何工具
- 不编造 Research Result 中不存在的数据
- 优先说明为什么推荐，而不仅是罗列商品
- 明确区分硬性条件和偏好
- 如果研究结果不足，应明确指出信息不足
- 不要向用户暴露内部 Agent、Prompt、Trace 等实现细节
"""


# ============================================================
# Multi-Agent State
# ============================================================

class MultiAgentState(TypedDict):

    messages: Annotated[
        list[dict],
        add,
    ]

    route: str

    research_result: object | None

    research_tool: str | None

    answer: str | None

    step: int


@dataclass
class RunContext:

    trace: AgentTrace


# ============================================================
# Helper
# ============================================================

def get_latest_user_message(
    messages: list[dict],
) -> str:

    for message in reversed(
        messages
    ):

        if (
            message.get("role")
            == "user"
        ):

            return (
                message.get(
                    "content"
                )
                or ""
            )

    return ""


# ============================================================
# Agent 1:
# Supervisor
# ============================================================

async def supervisor_node(
    state: MultiAgentState,
    runtime: Runtime[RunContext],
) -> dict:

    trace = (
        runtime.context.trace
    )

    step = (
        state["step"] + 1
    )

    logger.info(
        "supervisor_started "
        "trace_id=%s "
        "step=%s",
        trace.trace_id,
        step,
    )

    client = (
        get_llm_client()
    )

    start = (
        perf_counter()
    )

    try:

        response = await (
            client
            .chat
            .completions
            .create(
                model=MODEL,

                messages=[
                    {
                        "role": "system",
                        "content": (
                            SUPERVISOR_PROMPT
                        ),
                    },
                    *state["messages"],
                ],

                temperature=0,
            )
        )

    except Exception:

        latency = (
            perf_counter()
            - start
        ) * 1000

        trace.add_event(
            TraceEvent(
                step=step,
                event_type=(
                    "supervisor"
                ),
                name=(
                    "route_task"
                ),
                latency_ms=(
                    latency
                ),
                success=False,
            )
        )

        raise

    latency = (
        perf_counter()
        - start
    ) * 1000

    content = (
        response
        .choices[0]
        .message
        .content
        or ""
    )

    raw_route = (
        content
        .strip()
        .lower()
    )

    # 保守策略：
    #
    # Supervisor输出只要不是明确answer，
    # 就默认research。
    #
    # 商品系统里：
    # “多查一次”通常比“没查就乱回答”安全。
    if raw_route == "answer":
        route = "answer"

    else:
        route = "research"

    trace.add_event(
        TraceEvent(
            step=step,
            event_type="supervisor",
            name=route,
            latency_ms=latency,
            success=True,
        )
    )

    logger.info(
        "supervisor_finished "
        "trace_id=%s "
        "step=%s "
        "route=%s "
        "latency_ms=%.2f",
        trace.trace_id,
        step,
        route,
        latency,
    )

    return {
        "route": route,
        "step": step,
    }


# ============================================================
# Router:
# Supervisor → Research / Recommendation
# ============================================================

def route_after_supervisor(
    state: MultiAgentState,
):

    route = (
        state["route"]
    )

    if route == "research":

        logger.info(
            "multi_agent_route "
            "from=supervisor "
            "to=research"
        )

        return "research"

    logger.info(
        "multi_agent_route "
        "from=supervisor "
        "to=recommendation"
    )

    return "recommendation"


# ============================================================
# Agent 2:
# Research Agent
# ============================================================

async def research_node(
    state: MultiAgentState,
    runtime: Runtime[RunContext],
) -> dict:

    trace = (
        runtime.context.trace
    )

    step = (
        state["step"] + 1
    )

    logger.info(
        "research_agent_started "
        "trace_id=%s "
        "step=%s",
        trace.trace_id,
        step,
    )

    client = (
        get_llm_client()
    )

    start = (
        perf_counter()
    )

    research_messages = [
        {
            "role": "system",
            "content": (
                RESEARCH_PROMPT
            ),
        },
        *state["messages"],
    ]

    try:

        response = await (
            call_llm_with_retry(
                client=client,
                messages=(
                    research_messages
                ),
            )
        )

    except Exception:

        latency = (
            perf_counter()
            - start
        ) * 1000

        trace.add_event(
            TraceEvent(
                step=step,
                event_type="research",
                name="tool_selection",
                latency_ms=latency,
                success=False,
            )
        )

        raise

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

    tool_calls = (
        message_dict.get(
            "tool_calls"
        )
        or []
    )

    # --------------------------------------------------------
    # Research Agent没有调用工具
    # --------------------------------------------------------

    if not tool_calls:

        content = (
            message_dict.get(
                "content"
            )
            or ""
        )

        latency = (
            perf_counter()
            - start
        ) * 1000

        trace.add_event(
            TraceEvent(
                step=step,
                event_type="research",
                name="no_tool",
                latency_ms=latency,
                success=True,
            )
        )

        logger.info(
            "research_agent_finished "
            "trace_id=%s "
            "step=%s "
            "tool=none "
            "latency_ms=%.2f",
            trace.trace_id,
            step,
            latency,
        )

        return {
            "research_result": {
                "content": content,
            },

            "research_tool": None,

            "step": step,
        }

    # --------------------------------------------------------
    # 为了控制Agent Path：
    # Day13每次Research只执行第一个Tool Call。
    # --------------------------------------------------------

    if len(tool_calls) > 1:

        logger.warning(
            "research_multiple_tool_calls "
            "count=%s "
            "using_first_only=true",
            len(tool_calls),
        )

    tool_call = (
        tool_calls[0]
    )

    raw_tool_name = (
        tool_call[
            "function"
        ][
            "name"
        ]
    )

    tool_name = (
        normalize_tool_name(
            raw_tool_name
        )
    )

    try:

        arguments = (
            json.loads(
                tool_call[
                    "function"
                ][
                    "arguments"
                ]
            )
        )

    except json.JSONDecodeError:

        result = {
            "error": (
                "Invalid JSON arguments"
            )
        }

        tool_success = False

    else:

        logger.info(
            "research_tool_started "
            "trace_id=%s "
            "step=%s "
            "tool=%s "
            "arguments=%s",
            trace.trace_id,
            step,
            tool_name,
            arguments,
        )

        result = await (
            asyncio.to_thread(
                execute_tool,
                tool_name,
                arguments,
            )
        )

        tool_success = not (
            isinstance(
                result,
                dict,
            )
            and "error" in result
        )

    latency = (
        perf_counter()
        - start
    ) * 1000

    trace.add_event(
        TraceEvent(
            step=step,
            event_type="research",
            name=tool_name,
            latency_ms=latency,
            success=tool_success,
        )
    )

    logger.info(
        "research_agent_finished "
        "trace_id=%s "
        "step=%s "
        "tool=%s "
        "success=%s "
        "latency_ms=%.2f",
        trace.trace_id,
        step,
        tool_name,
        tool_success,
        latency,
    )

    return {
        "research_result": (
            result
        ),

        "research_tool": (
            tool_name
        ),

        "step": step,
    }


# ============================================================
# Agent 3:
# Recommendation Agent
# ============================================================

async def recommendation_node(
    state: MultiAgentState,
    runtime: Runtime[RunContext],
) -> dict:

    trace = (
        runtime.context.trace
    )

    step = (
        state["step"] + 1
    )

    logger.info(
        "recommendation_agent_started "
        "trace_id=%s "
        "step=%s",
        trace.trace_id,
        step,
    )

    client = (
        get_llm_client()
    )

    user_query = (
        get_latest_user_message(
            state["messages"]
        )
    )

    research_result = (
        state.get(
            "research_result"
        )
    )

    research_tool = (
        state.get(
            "research_tool"
        )
    )

    research_text = (
        json.dumps(
            research_result,
            ensure_ascii=False,
            indent=2,
        )
        if research_result
        is not None
        else "本轮没有新的Research结果。"
    )

    recommendation_context = (
        f"""
用户当前问题：

{user_query}


本轮 Research Tool：

{research_tool}


本轮 Research Result：

{research_text}
"""
    )

    start = (
        perf_counter()
    )

    try:

        response = await (
            client
            .chat
            .completions
            .create(
                model=MODEL,

                messages=[
                    {
                        "role": "system",
                        "content": (
                            RECOMMENDATION_PROMPT
                        ),
                    },

                    *state["messages"],

                    {
                        "role": "system",
                        "content": (
                            recommendation_context
                        ),
                    },
                ],

                temperature=0,
            )
        )

    except Exception:

        latency = (
            perf_counter()
            - start
        ) * 1000

        trace.add_event(
            TraceEvent(
                step=step,
                event_type=(
                    "recommendation"
                ),
                name="final_answer",
                latency_ms=latency,
                success=False,
            )
        )

        raise

    latency = (
        perf_counter()
        - start
    ) * 1000

    answer = (
        response
        .choices[0]
        .message
        .content
        or ""
    )

    trace.add_event(
        TraceEvent(
            step=step,
            event_type=(
                "recommendation"
            ),
            name=(
                "final_answer"
            ),
            latency_ms=(
                latency
            ),
            success=True,
        )
    )

    logger.info(
        "recommendation_agent_finished "
        "trace_id=%s "
        "step=%s "
        "latency_ms=%.2f",
        trace.trace_id,
        step,
        latency,
    )

    return {
        "messages": [
            {
                "role": (
                    "assistant"
                ),
                "content": (
                    answer
                ),
            }
        ],

        "answer": answer,

        "step": step,
    }


# ============================================================
# Build Multi-Agent Graph
# ============================================================

checkpointer = (
    InMemorySaver()
)


def build_multi_agent_graph():

    builder = StateGraph(
        MultiAgentState,
        context_schema=(
            RunContext
        ),
    )

    builder.add_node(
        "supervisor",
        supervisor_node,
    )

    builder.add_node(
        "research",
        research_node,
    )

    builder.add_node(
        "recommendation",
        recommendation_node,
    )

    builder.add_edge(
        START,
        "supervisor",
    )

    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        [
            "research",
            "recommendation",
        ],
    )

    builder.add_edge(
        "research",
        "recommendation",
    )

    builder.add_edge(
        "recommendation",
        END,
    )

    return builder.compile(
        checkpointer=(
            checkpointer
        )
    )


multi_agent_graph = (
    build_multi_agent_graph()
)


# ============================================================
# Public Interface
# ============================================================

async def run_agent(
    user_query: str,
    session_id: str,
    max_steps: int = 8,
) -> str:

    trace = (
        AgentTrace()
    )

    logger.info(
        "multi_agent_started "
        "trace_id=%s "
        "session_id=%s",
        trace.trace_id,
        session_id,
    )

    config = {
        "configurable": {
            "thread_id": (
                session_id
            )
        }
    }

    result = await (
        multi_agent_graph
        .ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            user_query
                        ),
                    }
                ],

                "route": "",

                "research_result": None,

                "research_tool": None,

                "answer": None,

                "step": 0,
            },

            config=config,

            context=RunContext(
                trace=trace
            ),
        )
    )

    answer = (
        result.get(
            "answer"
        )
        or ""
    )

    logger.info(
        "multi_agent_finished "
        "trace_id=%s "
        "session_id=%s "
        "steps=%s "
        "events=%s "
        "total_latency_ms=%.2f",
        trace.trace_id,
        session_id,
        result[
            "step"
        ],
        len(
            trace.events
        ),
        trace.total_latency_ms(),
    )

    return answer


async def clear_agent_session(
    session_id: str,
) -> None:

    await (
        checkpointer
        .adelete_thread(
            session_id
        )
    )


async def get_agent_session_state(
    session_id: str,
) -> dict:

    config = {
        "configurable": {
            "thread_id": (
                session_id
            )
        }
    }

    snapshot = await (
        multi_agent_graph
        .aget_state(
            config
        )
    )

    if not snapshot.values:

        return {
            "messages": [],
            "route": None,
            "research_tool": None,
            "research_result": None,
            "step": 0,
        }

    return {
        "messages": (
            snapshot.values.get(
                "messages",
                [],
            )
        ),

        "route": (
            snapshot.values.get(
                "route"
            )
        ),

        "research_tool": (
            snapshot.values.get(
                "research_tool"
            )
        ),

        "research_result": (
            snapshot.values.get(
                "research_result"
            )
        ),

        "step": (
            snapshot.values.get(
                "step",
                0,
            )
        ),
    }