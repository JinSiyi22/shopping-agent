from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from operator import add
from time import perf_counter
from typing import Annotated, TypedDict

import redis.asyncio as redis
from langgraph.checkpoint.redis.aio import (
    AsyncRedisSaver,
)
from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.runtime import Runtime

from app.agents.shopping_agent import (
    MODEL,
    call_chat_with_retry,
    call_llm_with_retry,
    get_llm_client,
    normalize_tool_arguments,
    normalize_tool_name,
)
from app.mcp.product_client import (
    product_mcp_client,
)
from app.observability.trace import (
    AgentTrace,
    TraceEvent,
)
from app.services.redis_cache import (
    get_cached_tool_result,
    set_cached_tool_result,
)


logger = logging.getLogger(
    __name__
)


# ============================================================
# Config
# ============================================================

PARALLEL_RESEARCH_TOP_N = int(
    os.getenv(
        "PARALLEL_RESEARCH_TOP_N",
        "2",
    )
)


RESEARCH_CONCURRENCY = int(
    os.getenv(
        "RESEARCH_CONCURRENCY",
        "2",
    )
)


REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0",
)


CHECKPOINT_TTL_MINUTES = int(
    os.getenv(
        "CHECKPOINT_TTL_MINUTES",
        "1440",
    )
)


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

1. 纯结构化条件查询：
   必须使用 search_products。

   当用户需求只包含以下明确硬条件时：
   - 品牌
   - 价格范围
   - 商品类别
   - 最低评分
   - 明确标签

   且没有用途、场景、语义偏好时，
   禁止使用 rerank_search_products。


2. 自然语言商品推荐：
   必须优先使用 rerank_search_products。

   当用户需求中包含用途、场景、
   模糊偏好或语义描述时，例如：

   - 经常出差
   - 写代码
   - 希望轻薄
   - 续航好
   - AI 开发
   - 高性能
   - 游戏

   应使用 rerank_search_products。


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

因此不要再先调用 hybrid_search_products，
然后又调用 rerank_search_products。


重要：

一次 Research Agent 执行，
原则上只选择一个最有价值的工具。

价格、品牌、评分、类别等硬性条件，
必须传入结构化参数，
不要只写入 query 文本。

如果用户明确提到了商品类别，
必须尽量传 category，例如：

- 笔记本电脑 / laptop -> category="laptop"
- 台式机 / desktop PC -> category="desktop"
- 显示器 / monitor -> category="monitor"
- 耳机 / headphones / earbuds -> category="headphone"
- 手机 / smartphone -> category="phone"
- 平板 / tablet -> category="tablet"

不要仅把“笔记本”“耳机”“显示器”等类别词
留在 query 文本中。
"""


# ============================================================
# Prompt: Candidate Research Worker
# ============================================================

CANDIDATE_RESEARCH_PROMPT = """
你是商品候选分析 Worker。

你不会调用任何工具。

你只负责分析一个候选商品与用户需求的匹配程度。

要求：

1. 只使用输入中提供的商品字段，不得编造参数。
2. 明确指出这个商品满足用户的哪些需求。
3. 明确指出可能不满足、无法确认或存在风险的部分。
4. 不要因为评分高就默认所有方面都更好。
5. 输出简洁，控制在 120 个中文字符左右。
6. 不要输出最终购买结论，因为最终决策由 Recommendation Agent 完成。
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
4. 并行 Candidate Research Workers 返回的候选分析

生成清晰、有依据的最终回答。


要求：

- 不调用任何工具
- 不编造 Research Result 中不存在的数据
- 优先说明为什么推荐，而不仅是罗列商品
- 明确区分硬性条件和偏好
- 如果研究结果不足，应明确指出信息不足
- Candidate Analysis 只是辅助判断，原始 Research Result 是事实依据
- 不要向用户暴露内部 Agent、Prompt、Trace 等实现细节

输出约束：

- 只使用 Research Result 中明确存在的事实。
- 禁止补充数据库没有提供的重量、续航、CPU、
  GPU、材质、认证、品牌口碑等外部信息。
- 必须用商品名称或商品 ID 明确指代推荐对象。
- 普通搜索/推荐尽量控制在 220 个中文字符以内。
- 单商品详情尽量控制在 180 个中文字符以内。
- 商品比较尽量控制在 300 个中文字符以内。
- 优先给结论和最关键依据，不写长篇通用购物科普。
"""


# ============================================================
# Multi-Agent State
# ============================================================

class MultiAgentState(
    TypedDict
):

    messages: Annotated[
        list[dict],
        add,
    ]

    route: str

    research_result: (
        object
        | None
    )

    research_tool: (
        str
        | None
    )

    candidate_analyses: (
        list[dict]
    )

    answer: (
        str
        | None
    )

    step: int


@dataclass
class RunContext:

    trace: AgentTrace


# ============================================================
# Basic Helpers
# ============================================================

def get_latest_user_message(
    messages: list[dict],
) -> str:

    for message in reversed(
        messages
    ):

        if (
            message.get(
                "role"
            )
            == "user"
        ):

            return (
                message.get(
                    "content"
                )
                or ""
            )

    return ""


def extract_candidate_products(
    research_result: (
        object
        | None
    ),
) -> list[dict]:
    """
    支持：
    1. search / semantic / hybrid / rerank -> list[dict]
    2. compare_products -> {"products": [...]}
    3. 旧格式 -> {"selected": [...]}
    4. get_product_detail -> 单个商品 dict
    """

    if (
        research_result
        is None
    ):
        return []

    if isinstance(
        research_result,
        list,
    ):

        return [
            item
            for item
            in research_result
            if (
                isinstance(
                    item,
                    dict,
                )
                and (
                    "id" in item
                    or "name" in item
                )
            )
        ]

    if isinstance(
        research_result,
        dict,
    ):

        products = (
            research_result.get(
                "products"
            )
            or research_result.get(
                "selected"
            )
        )

        if isinstance(
            products,
            list,
        ):

            return [
                item
                for item
                in products
                if isinstance(
                    item,
                    dict,
                )
            ]

        if (
            "id"
            in research_result
            or "name"
            in research_result
        ):

            return [
                research_result
            ]

    return []


# ============================================================
# Day 18:
# Deterministic Constraint Guard
# ============================================================
#
# LLM 负责开放语义理解与工具选择；
# 对“商品类别”这种高置信度、确定性硬约束，
# 在真正访问 Cache/MCP 之前做一次验证与补全。
#
# 原则：
# 1. LLM 已经给出 category -> 完全尊重，不覆盖；
# 2. 只对 retrieval/search 类工具补 category；
# 3. 必须唯一匹配到一个类别才补；
# 4. "电脑" 这种过于宽泛的词不做推断；
# 5. 多类别请求（例如“笔记本和平板都看看”）不强行补一个类别。
# ============================================================

SEARCH_TOOL_NAMES = {
    "search_products",
    "semantic_search_products",
    "hybrid_search_products",
    "rerank_search_products",
}


CATEGORY_PATTERNS: dict[
    str,
    tuple[str, ...],
] = {
    "laptop": (
        r"笔记本电脑",
        r"笔记本",
        r"\blaptops?\b",
        r"\bnotebook(?: computer| pc)?\b",
        r"\bchromebooks?\b",
        r"\bmacbooks?\b",
    ),

    "desktop": (
        r"台式电脑",
        r"台式机",
        r"\bdesktop(?: computer| pc)?\b",
        r"\bgaming desktops?\b",
        r"\bgaming pc\b",
        r"\bmini pc\b",
        r"\ball[- ]in[- ]one pc\b",
    ),

    "monitor": (
        r"显示器",
        r"\bmonitors?\b",
    ),

    "headphone": (
        r"无线耳机",
        r"蓝牙耳机",
        r"降噪耳机",
        r"耳机",
        r"耳麦",
        r"\bheadphones?\b",
        r"\bheadsets?\b",
        r"\bearbuds?\b",
        r"\bearphones?\b",
    ),

    "phone": (
        r"智能手机",
        r"手机",
        r"\bsmartphones?\b",
        r"\bcell phones?\b",
        r"\bmobile phones?\b",
        r"\biphones?\b",
    ),

    "tablet": (
        r"平板电脑",
        r"平板",
        r"\btablets?\b",
        r"\bipads?\b",
    ),

    "camera": (
        r"数码相机",
        r"相机",
        r"摄像头",
        r"\bcameras?\b",
        r"\bwebcams?\b",
        r"\bdslr\b",
    ),

    "keyboard": (
        r"键盘",
        r"\bkeyboards?\b",
    ),

    "mouse": (
        r"鼠标",
        r"\bmice\b",
        r"\bmouse\b",
    ),

    "storage": (
        r"固态硬盘",
        r"移动硬盘",
        r"硬盘",
        r"\bssd\b",
        r"\bhard drives?\b",
        r"\bflash drives?\b",
    ),

    "router": (
        r"路由器",
        r"\brouters?\b",
    ),

    "speaker": (
        r"蓝牙音箱",
        r"音箱",
        r"扬声器",
        r"\bspeakers?\b",
    ),

    "smartwatch": (
        r"智能手表",
        r"\bsmart\s?watches?\b",
        r"\bapple watch\b",
    ),

    "printer": (
        r"打印机",
        r"\bprinters?\b",
    ),
}


def infer_explicit_category(
    user_query: str,
) -> str | None:
    """
    从原始用户请求中只提取“明确出现”的商品类别。

    不做语义猜测：
    - "办公设备" -> None
    - "适合写代码的电脑" -> None
    - "轻薄笔记本电脑" -> laptop
    """

    text_value = (
        user_query
        .strip()
        .lower()
    )

    if not text_value:
        return None

    matched_categories = []

    for (
        category,
        patterns,
    ) in (
        CATEGORY_PATTERNS
        .items()
    ):

        matched = any(
            re.search(
                pattern,
                text_value,
                flags=re.IGNORECASE,
            )
            is not None
            for pattern
            in patterns
        )

        if matched:

            matched_categories.append(
                category
            )

    matched_categories = list(
        dict.fromkeys(
            matched_categories
        )
    )

    if (
        len(
            matched_categories
        )
        != 1
    ):

        return None

    return (
        matched_categories[0]
    )


def enrich_tool_arguments(
    *,
    user_query: str,
    tool_name: str,
    arguments: dict,
) -> dict:
    """
    Constraint Guard。

    只补“LLM漏掉”的 category；
    不覆盖模型明确给出的结构化约束。
    """

    enriched = dict(
        arguments
    )

    if (
        tool_name
        not in SEARCH_TOOL_NAMES
    ):

        return enriched

    existing_category = (
        enriched.get(
            "category"
        )
    )

    if (
        isinstance(
            existing_category,
            str,
        )
        and existing_category.strip()
    ):

        return enriched

    inferred_category = (
        infer_explicit_category(
            user_query
        )
    )

    if (
        inferred_category
        is None
    ):

        return enriched

    enriched[
        "category"
    ] = (
        inferred_category
    )

    return enriched


# ============================================================
# Agent 1: Supervisor
# ============================================================

async def supervisor_node(
    state: MultiAgentState,
    runtime: Runtime[
        RunContext
    ],
) -> dict:

    trace = (
        runtime.context.trace
    )

    step = (
        state["step"]
        + 1
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
            call_chat_with_retry(
                client,
                model=MODEL,

                messages=[
                    {
                        "role": (
                            "system"
                        ),
                        "content": (
                            SUPERVISOR_PROMPT
                        ),
                    },
                    *state[
                        "messages"
                    ],
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

    if (
        raw_route
        == "answer"
    ):
        route = "answer"

    else:
        route = "research"

    trace.add_event(
        TraceEvent(
            step=step,
            event_type=(
                "supervisor"
            ),
            name=route,
            latency_ms=(
                latency
            ),
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


def route_after_supervisor(
    state: MultiAgentState,
):

    route = (
        state["route"]
    )

    if (
        route
        == "research"
    ):

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
# Agent 2: Research Agent
# ============================================================

async def research_node(
    state: MultiAgentState,
    runtime: Runtime[
        RunContext
    ],
) -> dict:

    trace = (
        runtime.context.trace
    )

    step = (
        state["step"]
        + 1
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
            "role": (
                "system"
            ),
            "content": (
                RESEARCH_PROMPT
            ),
        },
        *state[
            "messages"
        ],
    ]

    mcp_tools = await (
        product_mcp_client
        .get_openai_tools()
    )

    try:

        response = await (
            call_llm_with_retry(
                client=client,
                messages=(
                    research_messages
                ),
                tools=(
                    mcp_tools
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
                event_type=(
                    "research"
                ),
                name=(
                    "tool_selection"
                ),
                latency_ms=(
                    latency
                ),
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
                event_type=(
                    "research"
                ),
                name=(
                    "no_tool"
                ),
                latency_ms=(
                    latency
                ),
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
                "content": (
                    content
                ),
            },
            "research_tool": (
                None
            ),
            "step": step,
        }

    if (
        len(
            tool_calls
        )
        > 1
    ):

        logger.warning(
            "research_multiple_tool_calls "
            "count=%s "
            "using_first_only=true",
            len(
                tool_calls
            ),
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

        arguments = (
            normalize_tool_arguments(
                arguments
            )
        )

        user_query = (
            get_latest_user_message(
                state[
                    "messages"
                ]
            )
        )

        before_guard = dict(
            arguments
        )

        arguments = (
            enrich_tool_arguments(
                user_query=(
                    user_query
                ),
                tool_name=(
                    tool_name
                ),
                arguments=(
                    arguments
                ),
            )
        )

        # 只有真正发生补全时才打印，
        # 便于 Day 18/20 做 Trace。
        if (
            before_guard
            .get(
                "category"
            )
            != arguments
            .get(
                "category"
            )
        ):

            logger.info(
                "constraint_guard_category_added "
                "trace_id=%s "
                "step=%s "
                "tool=%s "
                "category=%s",
                trace.trace_id,
                step,
                tool_name,
                arguments.get(
                    "category"
                ),
            )

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

        cached_result = await (
            get_cached_tool_result(
                tool_name,
                arguments,
            )
        )

        if (
            cached_result
            is not None
        ):

            logger.info(
                "tool_cache_hit "
                "trace_id=%s "
                "step=%s "
                "tool=%s",
                trace.trace_id,
                step,
                tool_name,
            )

            result = (
                cached_result
            )

        else:

            logger.info(
                "tool_cache_miss "
                "trace_id=%s "
                "step=%s "
                "tool=%s",
                trace.trace_id,
                step,
                tool_name,
            )

            result = await (
                product_mcp_client
                .call_tool(
                    tool_name,
                    arguments,
                )
            )

            tool_has_error = (
                isinstance(
                    result,
                    dict,
                )
                and "error"
                in result
            )

            if not (
                tool_has_error
            ):

                await (
                    set_cached_tool_result(
                        tool_name,
                        arguments,
                        result,
                    )
                )

        tool_success = not (
            isinstance(
                result,
                dict,
            )
            and "error"
            in result
        )

    latency = (
        perf_counter()
        - start
    ) * 1000

    trace.add_event(
        TraceEvent(
            step=step,
            event_type=(
                "research"
            ),
            name=(
                tool_name
            ),
            latency_ms=(
                latency
            ),
            success=(
                tool_success
            ),
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
# Parallel Candidate Research
# ============================================================

async def analyze_candidate(
    *,
    client,
    semaphore: (
        asyncio.Semaphore
    ),
    user_query: str,
    candidate: dict,
    candidate_index: int,
    trace: AgentTrace,
    step: int,
) -> dict:

    product_id = str(
        candidate.get(
            "id",
            (
                "candidate_"
                f"{candidate_index}"
            ),
        )
    )

    product_name = str(
        candidate.get(
            "name",
            product_id,
        )
    )

    async with semaphore:

        logger.info(
            "candidate_research_started "
            "trace_id=%s "
            "step=%s "
            "candidate=%s",
            trace.trace_id,
            step,
            product_id,
        )

        start = (
            perf_counter()
        )

        worker_input = (
            "用户需求：\n"
            f"{user_query}"
            "\n\n候选商品：\n"
            f"{json.dumps(candidate, ensure_ascii=False, indent=2)}"
        )

        try:

            response = await (
                call_chat_with_retry(
                    client,
                    model=MODEL,

                    messages=[
                        {
                            "role": (
                                "system"
                            ),
                            "content": (
                                CANDIDATE_RESEARCH_PROMPT
                            ),
                        },
                        {
                            "role": (
                                "user"
                            ),
                            "content": (
                                worker_input
                            ),
                        },
                    ],

                    temperature=0,
                )
            )

            analysis = (
                response
                .choices[0]
                .message
                .content
                or ""
            )

            success = True

        except Exception as exc:

            logger.exception(
                "candidate_research_failed "
                "trace_id=%s "
                "step=%s "
                "candidate=%s",
                trace.trace_id,
                step,
                product_id,
            )

            analysis = (
                "候选分析失败："
                f"{exc}"
            )

            success = False

        latency = (
            perf_counter()
            - start
        ) * 1000

        trace.add_event(
            TraceEvent(
                step=step,
                event_type=(
                    "candidate_research"
                ),
                name=(
                    product_id
                ),
                latency_ms=(
                    latency
                ),
                success=(
                    success
                ),
            )
        )

        logger.info(
            "candidate_research_finished "
            "trace_id=%s "
            "step=%s "
            "candidate=%s "
            "success=%s "
            "latency_ms=%.2f",
            trace.trace_id,
            step,
            product_id,
            success,
            latency,
        )

        return {
            "product_id": (
                product_id
            ),
            "product_name": (
                product_name
            ),
            "analysis": (
                analysis
            ),
            "success": (
                success
            ),
            "latency_ms": (
                latency
            ),
        }


async def parallel_research_node(
    state: MultiAgentState,
    runtime: Runtime[
        RunContext
    ],
) -> dict:

    trace = (
        runtime.context.trace
    )

    step = (
        state["step"]
        + 1
    )

    candidates = (
        extract_candidate_products(
            state.get(
                "research_result"
            )
        )
    )

    candidates = (
        candidates[
            :PARALLEL_RESEARCH_TOP_N
        ]
    )

    if not candidates:

        logger.info(
            "parallel_research_skipped "
            "trace_id=%s "
            "step=%s "
            "reason=no_candidates",
            trace.trace_id,
            step,
        )

        return {
            "candidate_analyses": [],
            "step": step,
        }

    client = (
        get_llm_client()
    )

    user_query = (
        get_latest_user_message(
            state[
                "messages"
            ]
        )
    )

    safe_concurrency = max(
        1,
        RESEARCH_CONCURRENCY,
    )

    semaphore = (
        asyncio.Semaphore(
            safe_concurrency
        )
    )

    logger.info(
        "parallel_research_started "
        "trace_id=%s "
        "step=%s "
        "candidates=%s "
        "concurrency=%s",
        trace.trace_id,
        step,
        len(
            candidates
        ),
        safe_concurrency,
    )

    start = (
        perf_counter()
    )

    tasks = [
        analyze_candidate(
            client=client,
            semaphore=(
                semaphore
            ),
            user_query=(
                user_query
            ),
            candidate=(
                candidate
            ),
            candidate_index=(
                index
            ),
            trace=trace,
            step=step,
        )

        for (
            index,
            candidate,
        )
        in enumerate(
            candidates,
            start=1,
        )
    ]

    analyses = await (
        asyncio.gather(
            *tasks
        )
    )

    wall_latency = (
        perf_counter()
        - start
    ) * 1000

    sum_worker_latency = sum(
        item.get(
            "latency_ms",
            0.0,
        )
        for item
        in analyses
    )

    logger.info(
        "parallel_research_finished "
        "trace_id=%s "
        "step=%s "
        "candidates=%s "
        "concurrency=%s "
        "wall_latency_ms=%.2f "
        "sum_worker_latency_ms=%.2f",
        trace.trace_id,
        step,
        len(
            candidates
        ),
        safe_concurrency,
        wall_latency,
        sum_worker_latency,
    )

    return {
        "candidate_analyses": (
            analyses
        ),
        "step": step,
    }


# ============================================================
# Agent 3: Recommendation Agent
# ============================================================

async def recommendation_node(
    state: MultiAgentState,
    runtime: Runtime[
        RunContext
    ],
) -> dict:

    trace = (
        runtime.context.trace
    )

    step = (
        state["step"]
        + 1
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
            state[
                "messages"
            ]
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

    candidate_analyses = (
        state.get(
            "candidate_analyses"
        )
        or []
    )

    research_text = (
        json.dumps(
            research_result,
            ensure_ascii=False,
            indent=2,
        )
        if (
            research_result
            is not None
        )
        else (
            "本轮没有新的"
            "Research结果。"
        )
    )

    candidate_analysis_text = (
        json.dumps(
            candidate_analyses,
            ensure_ascii=False,
            indent=2,
        )
        if candidate_analyses
        else (
            "本轮没有Candidate"
            "并行分析结果。"
        )
    )

    recommendation_context = (
        f"""
用户当前问题：

{user_query}


本轮 Research Tool：

{research_tool}


本轮 Research Result：

{research_text}


本轮 Candidate Parallel Analyses：

{candidate_analysis_text}
"""
    )

    start = (
        perf_counter()
    )

    try:

        response = await (
            call_chat_with_retry(
                client,
                model=MODEL,

                messages=[
                    {
                        "role": (
                            "system"
                        ),
                        "content": (
                            RECOMMENDATION_PROMPT
                        ),
                    },

                    *state[
                        "messages"
                    ],

                    {
                        "role": (
                            "system"
                        ),
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
                name=(
                    "final_answer"
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
        "answer": (
            answer
        ),
        "step": (
            step
        ),
    }


# ============================================================
# Build Multi-Agent Graph
# ============================================================

checkpoint_redis_client = (
    redis.Redis.from_url(
        REDIS_URL,
        decode_responses=False,
        max_connections=20,
    )
)


checkpointer = (
    AsyncRedisSaver(
        redis_client=(
            checkpoint_redis_client
        ),
        ttl={
            "default_ttl": (
                CHECKPOINT_TTL_MINUTES
            ),
            "refresh_on_read": (
                True
            ),
        },
    )
)



def route_after_research(
    state: MultiAgentState,
):
    """
    Day 20:
    只有语义推荐 rerank_search_products
    需要进入 Candidate Research Workers。

    结构化搜索、商品详情、商品比较等结果
    可直接交给 Recommendation Agent。
    """

    if (
        state.get(
            "research_tool"
        )
        == "rerank_search_products"
    ):

        logger.info(
            "multi_agent_route "
            "from=research "
            "to=parallel_research"
        )

        return "parallel_research"

    logger.info(
        "multi_agent_route "
        "from=research "
        "to=recommendation"
    )

    return "recommendation"


def build_multi_agent_graph():

    builder = (
        StateGraph(
            MultiAgentState,
            context_schema=(
                RunContext
            ),
        )
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
        "parallel_research",
        parallel_research_node,
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

    # Day 18这里只加入 Constraint Guard，
    # 暂不提前做 Day 20 的 conditional parallel optimization，
    # 保持当前实验主线稳定。
    builder.add_conditional_edges(
        "research",
        route_after_research,
        [
            "parallel_research",
            "recommendation",
        ],
    )

    builder.add_edge(
        "parallel_research",
        "recommendation",
    )

    builder.add_edge(
        "recommendation",
        END,
    )

    return (
        builder.compile(
            checkpointer=(
                checkpointer
            )
        )
    )


multi_agent_graph = (
    build_multi_agent_graph()
)


# ============================================================
# Redis Checkpoint Lifecycle
# ============================================================

async def setup_multi_agent_storage() -> None:

    logger.info(
        "redis_checkpointer_setup_started"
    )

    await (
        checkpointer
        .asetup()
    )

    logger.info(
        "redis_checkpointer_setup_finished"
    )


async def close_multi_agent_storage() -> None:

    await (
        checkpoint_redis_client
        .aclose()
    )


# ============================================================
# Public Interface
# ============================================================

async def run_agent(
    user_query: str,
    session_id: str,
    max_steps: int = 8,
) -> str:
    """
    max_steps 参数继续保留，
    与现有 API / baseline 接口兼容。
    """

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
                        "role": (
                            "user"
                        ),
                        "content": (
                            user_query
                        ),
                    }
                ],
                "route": "",
                "research_result": (
                    None
                ),
                "research_tool": (
                    None
                ),
                "candidate_analyses": [],
                "answer": (
                    None
                ),
                "step": 0,
            },

            config=config,

            context=(
                RunContext(
                    trace=trace
                )
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

    # Optional demo-only persistence: existing events, no extra LLM calls.
    try:
        from app.observability.trace_export import export_agent_trace_if_enabled
        export_agent_trace_if_enabled(
            trace=trace,
            user_query=user_query,
            final_answer=answer,
            steps=result.get("step"),
        )
    except Exception:
        logger.exception("multi_agent_trace_export_failed")

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

    if not (
        snapshot.values
    ):

        return {
            "messages": [],
            "route": (
                None
            ),
            "research_tool": (
                None
            ),
            "research_result": (
                None
            ),
            "candidate_analyses": [],
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

        "candidate_analyses": (
            snapshot.values.get(
                "candidate_analyses",
                [],
            )
        ),

        "step": (
            snapshot.values.get(
                "step",
                0,
            )
        ),
    }
