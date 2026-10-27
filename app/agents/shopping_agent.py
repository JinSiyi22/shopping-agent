import asyncio
import json
import logging
import os
from dataclasses import dataclass
from operator import add
from time import perf_counter
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    RateLimitError,
)

from app.observability.trace import AgentTrace, TraceEvent
from app.tools.product_tools import (
    calculate_shipping_tool,
    compare_products_tool,
    get_product_detail_tool,
    hybrid_search_products_tool,
    rerank_search_products_tool,
    search_products_tool,
    semantic_search_products_tool,
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

    api_key = os.getenv("LLM_API_KEY")

    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY is not configured"
        )

    _llm_client = AsyncOpenAI(
        api_key=api_key,
        base_url=(
            os.getenv("LLM_BASE_URL")
            or None
        ),
        timeout=30.0,
    )

    return _llm_client


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
你是一个专业的商品研究与购买决策助手。

你的任务是根据用户需求，
使用商品搜索、语义检索、混合检索、详情查询、
商品比较和运费计算工具，
帮助用户完成商品研究和购买决策。

工作原则：

1. 所有商品结论都必须基于工具返回的数据，
   不要编造数据库中不存在的商品或属性。

2. 优先使用最少的工具调用完成任务。
   如果已有工具结果足够回答用户，
   应立即生成最终答案，不要为了“确认”而重复查询。

3. 不要连续调用功能重复的工具。


关于商品检索工具：

- search_products：
  用于纯结构化条件查询。

- semantic_search_products：
  主要用于实验和纯语义检索测试，
  一般用户请求不优先使用。

- hybrid_search_products：
  主要用于 Hybrid Retrieval 实验和调试，
  一般最终商品推荐不优先直接调用。

- rerank_search_products：
  是自然语言商品推荐的默认高级检索工具。

  该工具内部已经完成：
  BM25 Recall
  + Dense Recall
  + RRF Fusion
  + Cross-Encoder Reranking。

  如果用户只是要求推荐商品，
  调用本工具之后应直接生成最终答案，
  不要继续调用 hybrid_search_products
  或 get_product_detail。

关于 get_product_detail：

- 只有以下情况才调用：
  1. 用户明确询问某一个已知商品的详细信息；
  2. 当前已有结果确实缺少回答用户所必需的信息。

- 如果 hybrid_search_products 已经返回了
  名称、品牌、价格、评分、标签和描述，
  不要再次调用 get_product_detail。


关于 compare_products：

- 当用户明确要求比较两个或多个商品时使用。

- 如果需要比较：
  先通过 search_products 或 hybrid_search_products
  找到候选商品 ID，
  然后直接调用 compare_products。

- 不要采用：
  search
  → get_product_detail A
  → get_product_detail B
  → compare

- 正确路径是：
  search
  → compare

- 如果用户没有明确要求比较，
  不要主动调用 compare_products。


关于 calculate_shipping：

- 只有用户询问运费、到手价或配送成本时才使用。
- 返回的是开发阶段模拟运费，
  最终回答中必须说明。


自然语言需求映射示例：

- 写代码、开发 -> 编程
- AI、机器学习、小模型训练 -> GPU、高性能
- 出差、便携 -> 轻薄
- 日常工作 -> 办公


参数规则：

- 用户未明确提供的可选参数优先省略。
- 不要使用空字符串代替缺失参数。
- 必须显式表示缺失值时可以使用 null。
- 普通推荐的 top_k 使用 5 或更少，
  除非用户明确要求大量候选，否则不要超过 5。
- 价格、品牌、评分、类别等明确硬性条件，
  必须优先放入对应的结构化参数，
  不要只写进 query 文本。
- 例如用户说“预算10000以内”，
  应传 max_price=10000。
- query 主要保留用途、场景和偏好描述，
  例如“经常出差、写代码、轻薄、续航好”。


工具调用规则：

- function.name 必须严格使用系统提供的工具名。
- 禁止修改工具名。
- 禁止添加前缀、后缀、channel、analysis、
  commentary 或其他控制标记。


最重要的执行原则：

优先选择最短的有效执行路径。

普通自然语言推荐：

User
→ hybrid_search_products
→ Final Answer

明确要求比较：

User
→ hybrid_search_products
→ compare_products
→ Final Answer

询问已知单个商品详情：

User
→ get_product_detail
→ Final Answer

纯结构化查询：

User
→ search_products
→ Final Answer

不要为了增加步骤而调用工具。
"""


TOOL_NAME_GUARD_PROMPT = """
重要工具调用规则：

当你决定调用工具时，
function.name 必须严格等于系统提供的工具名称之一：

- search_products
- semantic_search_products
- hybrid_search_products
- rerank_search_products
- get_product_detail
- compare_products
- calculate_shipping

禁止：
- 修改工具名
- 添加任何前缀或后缀
- 添加空格
- 添加解释文本
- 添加 <|...|> 一类控制标记
- 将 channel、analysis、commentary 等内容拼接到工具名

例如：

正确：
compare_products

错误：
compare_products<|channel|>commentary
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
                "根据明确的结构化条件搜索商品。"
                "适合价格、品牌、类别、评分和明确标签等条件。"
                "如果返回结果已经足够回答用户，应直接生成最终答案。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_price": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "null"},
                        ],
                        "description": (
                            "用户可接受的最高价格；"
                            "未指定时省略或传 null"
                        ),
                    },
                    "min_price": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "null"},
                        ],
                        "description": (
                            "用户要求的最低价格；"
                            "未指定时省略或传 null"
                        ),
                    },
                    "brand": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ],
                        "description": (
                            "指定商品品牌；"
                            "未指定时省略或传 null，"
                            "不要传空字符串"
                        ),
                    },
                    "category": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ],
                        "description": (
                            "商品类别，例如 laptop；"
                            "未指定时省略或传 null"
                        ),
                    },
                    "tags": {
                        "anyOf": [
                            {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                            },
                            {"type": "null"},
                        ],
                        "description": (
                            "商品需求标签，例如："
                            "编程、轻薄、办公、商务、"
                            "游戏、GPU、高性能；"
                            "没有明确标签时省略或传 null"
                        ),
                    },
                    "min_rating": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "null"},
                        ],
                        "description": (
                            "最低商品评分，范围0到5；"
                            "未指定时省略或传 null"
                        ),
                    },
                    "top_k": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "null"},
                        ],
                        "description": (
                            "最多返回多少个商品。"
                            "普通推荐默认使用 5 或更少；"
                            "除非用户明确要求大量候选，"
                            "否则不要超过 5。"
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
            "name": "semantic_search_products",
            "description": (
                "根据用户自然语言需求进行纯语义商品检索。"
                "主要用于纯语义检索场景或检索效果验证；"
                "一般商品推荐优先使用 hybrid_search_products。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "用户完整的自然语言商品需求，"
                            "尽量保留使用场景和偏好信息"
                        ),
                    },
                    "max_price": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "null"},
                        ],
                    "description": (
                        "用户明确给出的最高预算。"
                        "例如用户说预算10000以内时，"
                        "必须传 max_price=10000，"
                        "不要只把预算写进 query。"
                    ),
                    },
                    "min_price": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "null"},
                        ],
                        "description": (
                            "最低价格；未指定时省略或传 null"
                        ),
                    },
                    "brand": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ],
                        "description": (
                            "指定品牌；未指定时省略或传 null，"
                            "不要传空字符串"
                        ),
                    },
                    "category": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ],
                        "description": (
                            "商品类别，例如 laptop；"
                            "未指定时省略或传 null"
                        ),
                    },
                    "min_rating": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "null"},
                        ],
                        "description": (
                            "最低商品评分；"
                            "未指定时省略或传 null"
                        ),
                    },
                    "top_k": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "null"},
                        ],
                        "description": (
                            "返回最相关的商品数量。"
                            "普通场景默认使用 5 或更少。"
                        ),
                        "default": 5,
                    },
                },
                "required": [
                    "query"
                ],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "hybrid_search_products",
            "description": (
                "默认的自然语言商品检索工具。"
                "使用 BM25 + Dense Embedding 双路召回，"
                "并通过 RRF 融合排序。"
                "返回商品ID、名称、品牌、价格、评分、标签、"
                "描述以及检索排名信息，"
                "这些信息通常已经足够直接完成普通商品推荐。"
                "调用本工具后，如果用户没有明确要求进一步比较"
                "或查询缺失字段，应直接生成最终答案，"
                "不要再逐个调用 get_product_detail。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "用户完整的自然语言商品需求"
                        ),
                    },
                    "max_price": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "null"},
                        ],
                        "description": (
                            "最高预算；"
                            "未指定时省略或传 null"
                        ),
                    },
                    "min_price": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "null"},
                        ],
                        "description": (
                            "最低价格；"
                            "未指定时省略或传 null"
                        ),
                    },
                    "brand": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ],
                        "description": (
                            "品牌；未指定时省略"
                            "或传 null，不要传空字符串"
                        ),
                    },
                    "category": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ],
                        "description": (
                            "商品类别，例如 laptop；"
                            "未指定时省略或传 null"
                        ),
                    },
                    "min_rating": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "null"},
                        ],
                        "description": (
                            "最低评分；"
                            "未指定时省略或传 null"
                        ),
                    },
                    "top_k": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "null"},
                        ],
                        "description": (
                            "最终返回的候选商品数量。"
                            "普通推荐默认使用 5；"
                            "除非用户明确要求大量候选，"
                            "否则不要超过 5。"
                        ),
                        "default": 5,
                    },
                },
                "required": [
                    "query"
                ],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": (
                "rerank_search_products"
            ),

            "description": (
                "高级自然语言商品检索工具。"
                "内部首先使用 BM25 + Dense Embedding "
                "进行双路召回，再使用 RRF 融合候选，"
                "最后通过 Cross-Encoder 对候选进行精排。"
                "对于自然语言商品推荐任务，"
                "优先使用本工具。"
                "返回结果已经包含完成普通推荐所需的信息，"
                "调用后通常应直接回答用户。"
            ),

            "parameters": {
                "type": "object",

                "properties": {

                    "query": {
                        "type": "string",

                        "description": (
                            "用户的用途、场景和偏好需求。"
                            "例如轻薄、编程、经常出差、"
                            "续航好、AI开发等。"
                            "价格等硬约束应使用单独参数。"
                        ),
                    },

                    "max_price": {
                        "anyOf": [
                            {
                                "type": "number"
                            },
                            {
                                "type": "null"
                            },
                        ],

                        "description": (
                            "用户明确给出的最高预算。"
                            "例如用户说预算10000以内，"
                            "必须传 max_price=10000，"
                            "不要只把预算写进 query。"
                        ),
                    },

                    "min_price": {
                        "anyOf": [
                            {
                                "type": "number"
                            },
                            {
                                "type": "null"
                            },
                        ],
                    },

                    "brand": {
                        "anyOf": [
                            {
                                "type": "string"
                            },
                            {
                                "type": "null"
                            },
                        ],
                    },

                    "category": {
                        "anyOf": [
                            {
                                "type": "string"
                            },
                            {
                                "type": "null"
                            },
                        ],
                    },

                    "min_rating": {
                        "anyOf": [
                            {
                                "type": "number"
                            },
                            {
                                "type": "null"
                            },
                        ],
                    },

                    "top_k": {
                        "anyOf": [
                            {
                                "type": "integer"
                            },
                            {
                                "type": "null"
                            },
                        ],

                        "default": 5,
                    },
                },

                "required": [
                    "query"
                ],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_product_detail",
            "description": (
                "查询一个已知商品ID的详细信息。"
                "仅在用户明确询问某一个商品的详细信息，"
                "或者现有检索结果缺少回答所必需的字段时使用。"
                "如果 search_products 或 hybrid_search_products "
                "已经返回了名称、品牌、价格、评分、标签和描述，"
                "不要为了确认信息重复调用本工具。"
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
                "比较多个已经确定商品ID的商品。"
                "仅在用户明确要求比较、对比、二选一或询问"
                "哪个更合适时使用。"
                "如果前一步搜索已经得到商品ID，"
                "应直接调用本工具进行比较，"
                "不要先逐个调用 get_product_detail。"
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
                "只有用户询问运费、到手价或配送成本时使用。"
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
    "semantic_search_products": (
        semantic_search_products_tool
    ),
    "hybrid_search_products": (
        hybrid_search_products_tool
    ),
    "rerank_search_products": (
        rerank_search_products_tool
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


# ============================================================
# Tool Name / Tool Argument Normalization
# ============================================================

def normalize_tool_name(
    raw_tool_name: str,
) -> str:

    tool_name = (
        raw_tool_name.strip()
    )

    if tool_name in TOOL_REGISTRY:
        return tool_name

    if "<|" in tool_name:

        candidate = (
            tool_name
            .split(
                "<|",
                1,
            )[0]
            .strip()
        )

        if candidate in TOOL_REGISTRY:

            logger.warning(
                "tool_name_normalized "
                "original=%s "
                "normalized=%s",
                raw_tool_name,
                candidate,
            )

            return candidate

    return tool_name


CATEGORY_ALIASES = {
    "ssd": "storage",
    "solid state drive": "storage",
    "solid state drives": "storage",
    "hdd": "storage",
    "hard drive": "storage",
    "hard drives": "storage",
    "flash drive": "storage",
    "flash drives": "storage",
    "memory card": "storage",
    "memory cards": "storage",
    "earbud": "headphone",
    "earbuds": "headphone",
    "headphones": "headphone",
    "headset": "headphone",
    "headsets": "headphone",
    "notebook": "laptop",
    "notebook computer": "laptop",
    "laptops": "laptop",
    "monitors": "monitor",
    "keyboards": "keyboard",
    "mice": "mouse",
    "routers": "router",
    "smart watch": "smartwatch",
    "smart watches": "smartwatch",
    "speakers": "speaker",
    "tablets": "tablet",
    "digital camera": "camera",
    "digital cameras": "camera",
}


def normalize_category_alias(value: str) -> str:
    normalized = " ".join(
        value.strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )
    return CATEGORY_ALIASES.get(normalized, normalized)


def normalize_tool_arguments(
    arguments: dict,
) -> dict:

    cleaned_arguments = {}

    for key, value in arguments.items():

        if value is None:
            continue

        if (
            isinstance(value, str)
            and not value.strip()
        ):
            continue

        if (
            isinstance(value, list)
            and len(value) == 0
        ):
            continue

        cleaned_arguments[
            key
        ] = value

    category = cleaned_arguments.get("category")
    if isinstance(category, str):
        cleaned_arguments["category"] = normalize_category_alias(category)

    return cleaned_arguments


def execute_tool(
    tool_name: str,
    arguments: dict,
):

    tool_name = (
        normalize_tool_name(
            tool_name
        )
    )

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

    cleaned_arguments = (
        normalize_tool_arguments(
            arguments
        )
    )

    logger.info(
        "tool_arguments_normalized "
        "tool=%s "
        "original=%s "
        "cleaned=%s",
        tool_name,
        arguments,
        cleaned_arguments,
    )

    try:

        return tool_function(
            **cleaned_arguments
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
# LLM Call + Retry
# ============================================================

async def call_chat_with_retry(
    client: AsyncOpenAI,
    *,
    max_retries: int = 4,
    **kwargs,
):
    """
    通用 LLM 调用重试层。

    处理：
    1. 429 Rate Limit
    2. 请求超时
    3. 临时网络连接失败
    """

    # Day 20 latency / TPM guard.
    # GPT-OSS defaults to medium reasoning on Groq; these agent
    # sub-tasks are narrow, so use low reasoning and cap output.
    if MODEL and MODEL.startswith("openai/gpt-oss"):
        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body.setdefault("reasoning_effort", "low")
        extra_body.setdefault("include_reasoning", False)
        kwargs["extra_body"] = extra_body
        kwargs.setdefault("max_completion_tokens", 512)

    for attempt in range(
        max_retries + 1
    ):

        try:

            return await (
                client
                .chat
                .completions
                .create(
                    **kwargs
                )
            )

        # ----------------------------------
        # 429 Rate Limit
        # ----------------------------------
        except RateLimitError as exc:

            if attempt >= max_retries:

                logger.error(
                    "llm_rate_limit_retry_exhausted "
                    "attempt=%s "
                    "error=%s",
                    attempt + 1,
                    exc,
                )

                raise

            # 3, 6, 12, 20 秒
            delay = min(
                3 * (2 ** attempt),
                20,
            )

            logger.warning(
                "llm_rate_limit_retry "
                "attempt=%s "
                "sleep_seconds=%s "
                "error=%s",
                attempt + 1,
                delay,
                exc,
            )

            await asyncio.sleep(
                delay
            )

        # ----------------------------------
        # 网络超时 / 连接失败
        # ----------------------------------
        except (
            APITimeoutError,
            APIConnectionError,
        ) as exc:

            if attempt >= max_retries:

                logger.error(
                    "llm_network_retry_exhausted "
                    "attempt=%s "
                    "error=%s",
                    attempt + 1,
                    exc,
                )

                raise

            # 1, 2, 4, 8 秒
            delay = min(
                2 ** attempt,
                8,
            )

            logger.warning(
                "llm_network_retry "
                "attempt=%s "
                "sleep_seconds=%s "
                "error_type=%s "
                "error=%s",
                attempt + 1,
                delay,
                type(exc).__name__,
                exc,
            )

            await asyncio.sleep(
                delay
            )

    raise RuntimeError(
        "LLM retry loop "
        "exited unexpectedly"
    )

async def call_llm_with_retry(
    client: AsyncOpenAI,
    messages: list[dict],
    max_retries: int = 2,
    tools: list[dict] | None = None,
):

    tool_definitions = (
        tools
        if tools is not None
        else TOOLS
    )

    current_messages = (
        messages
    )

    for attempt in range(
        1,
        max_retries + 2,
    ):

        try:

            return await (
                call_chat_with_retry(
                    client,
                    model=MODEL,
                    messages=(
                        current_messages
                    ),
                    tools=(
                        tool_definitions
                    ),
                    tool_choice="auto",
                    temperature=0,
                    parallel_tool_calls=False,
                    extra_body={
                        "disable_tool_validation": True
                    },
                )
            )

        except BadRequestError as exc:

            error_text = (
                str(exc)
            )

            is_tool_call_error = (
                "tool_use_failed"
                in error_text
                or
                "Tool call validation failed"
                in error_text
                or
                "attempted to call tool"
                in error_text
            )

            if not is_tool_call_error:
                raise

            if (
                attempt
                >= max_retries + 1
            ):

                logger.error(
                    "tool_call_retry_exhausted "
                    "attempt=%s "
                    "error=%s",
                    attempt,
                    error_text,
                )

                raise

            logger.warning(
                "tool_call_retry "
                "attempt=%s "
                "error=%s",
                attempt,
                error_text,
            )

            current_messages = [
                {
                    "role": "system",
                    "content": (
                        TOOL_NAME_GUARD_PROMPT
                    ),
                },
                *messages,
            ]

    raise RuntimeError(
        "LLM retry loop exited unexpectedly"
    )


# ============================================================
# LangGraph State / Runtime Context
# ============================================================

class AgentState(TypedDict):

    messages: Annotated[
        list[dict],
        add,
    ]

    step: int
    max_steps: int


@dataclass
class RunContext:

    trace: AgentTrace


# ============================================================
# Node 1: Agent Node
# ============================================================

async def agent_node(
    state: AgentState,
    runtime: Runtime[RunContext],
) -> dict:

    step = (
        state["step"] + 1
    )

    trace = (
        runtime.context.trace
    )

    logger.info(
        "agent_node_started "
        "trace_id=%s "
        "step=%s",
        trace.trace_id,
        step,
    )

    client = (
        get_llm_client()
    )

    if not MODEL:

        raise RuntimeError(
            "LLM_MODEL is not configured"
        )

    llm_start = (
        perf_counter()
    )

    try:

        response = await (
            call_llm_with_retry(
                client=client,
                messages=(
                    state["messages"]
                ),
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
                latency_ms=(
                    llm_latency
                ),
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
            latency_ms=(
                llm_latency
            ),
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

    # 立即清洗异常 Tool Name，
    # 避免脏名字进入 LangGraph Checkpoint。
    tool_calls = (
        message_dict.get(
            "tool_calls"
        )
        or []
    )

    for tool_call in tool_calls:

        function_info = (
            tool_call.get(
                "function"
            )
            or {}
        )

        raw_tool_name = (
            function_info.get(
                "name",
                "",
            )
        )

        function_info[
            "name"
        ] = (
            normalize_tool_name(
                raw_tool_name
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
        state[
            "messages"
        ][-1]
    )

    tool_calls = (
        last_message.get(
            "tool_calls"
        )
    )

    if tool_calls:

        logger.info(
            "route "
            "from=agent "
            "to=tools"
        )

        return "tools"

    logger.info(
        "route "
        "from=agent "
        "to=end"
    )

    return END


# ============================================================
# Node 2: Tool Node
# ============================================================

async def tool_node(
    state: AgentState,
    runtime: Runtime[RunContext],
) -> dict:

    trace = (
        runtime.context.trace
    )

    step = (
        state["step"]
    )

    last_message = (
        state[
            "messages"
        ][-1]
    )

    tool_calls = (
        last_message.get(
            "tool_calls"
        )
        or []
    )

    tool_messages = []

    for tool_call in tool_calls:

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

            tool_result = await (
                asyncio.to_thread(
                    execute_tool,
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
                    tool_call[
                        "id"
                    ]
                ),

                "content": (
                    json.dumps(
                        tool_result,
                        ensure_ascii=False,
                    )
                ),
            }
        )

    return {
        "messages": (
            tool_messages
        )
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
            "from=tools "
            "to=limit "
            "step=%s",
            state["step"],
        )

        return "limit"

    logger.info(
        "route "
        "from=tools "
        "to=agent"
    )

    return "agent"


# ============================================================
# Node 3: Limit Node
# ============================================================

def limit_node(
    state: AgentState,
) -> dict:

    return {
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "达到最大 Agent 执行步数，"
                    "任务未能正常结束。"
                ),
            }
        ]
    }


# ============================================================
# Build LangGraph
# ============================================================

checkpointer = (
    InMemorySaver()
)


def build_shopping_graph():

    builder = StateGraph(
        AgentState,
        context_schema=(
            RunContext
        ),
    )

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

    builder.add_edge(
        START,
        "agent",
    )

    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        [
            "tools",
            END,
        ],
    )

    builder.add_conditional_edges(
        "tools",
        route_after_tools,
        [
            "agent",
            "limit",
        ],
    )

    builder.add_edge(
        "limit",
        END,
    )

    return builder.compile(
        checkpointer=(
            checkpointer
        )
    )


shopping_graph = (
    build_shopping_graph()
)


# ============================================================
# Public Agent Interface
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
        "agent_started "
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

    snapshot = await (
        shopping_graph
        .aget_state(
            config
        )
    )

    existing_messages = (
        snapshot.values.get(
            "messages",
            [],
        )
        if snapshot.values
        else []
    )

    new_messages = []

    if not existing_messages:

        new_messages.append(
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                ),
            }
        )

    new_messages.append(
        {
            "role": "user",
            "content": (
                user_query
            ),
        }
    )

    result = await (
        shopping_graph
        .ainvoke(
            {
                "messages": (
                    new_messages
                ),
                "step": 0,
                "max_steps": (
                    max_steps
                ),
            },
            config=config,
            context=RunContext(
                trace=trace
            ),
        )
    )

    final_messages = (
        result[
            "messages"
        ]
    )

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
        "session_id=%s "
        "steps=%s "
        "events=%s "
        "total_latency_ms=%.2f",
        trace.trace_id,
        session_id,
        result["step"],
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
        shopping_graph
        .aget_state(
            config
        )
    )

    if not snapshot.values:

        return {
            "messages": [],
            "step": 0,
            "max_steps": None,
        }

    return {
        "messages": (
            snapshot.values.get(
                "messages",
                [],
            )
        ),

        "step": (
            snapshot.values.get(
                "step",
                0,
            )
        ),

        "max_steps": (
            snapshot.values.get(
                "max_steps"
            )
        ),
    }