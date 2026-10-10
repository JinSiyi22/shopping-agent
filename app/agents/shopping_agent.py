import json
import os

from dotenv import load_dotenv
#from openai import OpenAI
from openai import AsyncOpenAI

from app.tools.product_tools import (
    calculate_shipping_tool,
    compare_products_tool,
    get_product_detail_tool,
    search_products_tool,
)

load_dotenv()


client = AsyncOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL") or None,
)


MODEL = os.getenv("LLM_MODEL")

#Tool Schema
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "根据用户的预算、品牌、商品类别、评分和用途标签"
                "搜索商品。当用户要求推荐、搜索、比较或选择商品时使用。"
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
                        "description": "指定品牌",
                    },
                    "category": {
                        "type": "string",
                        "description": "商品类别，例如 laptop",
                    },
                    "tags": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": (
                            "商品需求标签。当前常见标签包括："
                            "编程、轻薄、办公、商务、游戏、"
                            "GPU、高性能。"
                        ),
                    },
                    "min_rating": {
                        "type": "number",
                        "description": "最低用户评分，范围0到5",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "最多返回多少个商品",
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
                "当你已经知道商品ID并需要进一步了解商品时使用。"
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
                "获取多个指定商品的结构化信息，"
                "用于比较价格、评分、标签和用途。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {
                            "type": "string",
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
    }
]

SYSTEM_PROMPT = """
你是一个专业的商品研究与购买决策助手。

你可以自主使用多个工具完成用户的商品研究任务。

可用能力包括：

1. 搜索符合条件的商品。
2. 根据商品ID查询详细信息。
3. 获取多个商品的信息用于比较。
4. 估算商品运费和总成本。

工作原则：

- 用户提出商品推荐、搜索或比较需求时，优先使用工具获取真实商品数据。
- 不要编造商品库中不存在的商品事实。
- 如果完成任务需要多个步骤，可以连续调用多个工具。
- 搜索后如果需要更深入分析，可以继续查询详情或比较商品。
- 不要因为已经调用过一次工具就立即结束。
- 用户自然语言可以映射为标准需求标签，例如：
  - 写代码、开发 -> 编程
  - AI、机器学习、小模型训练 -> GPU、高性能
  - 出差、便携 -> 轻薄
- 工具出错时，根据错误信息决定是否需要换一种方法。
- 最终回答需要说明推荐依据和重要限制。
- calculate_shipping 返回的是开发阶段模拟运费，需要向用户说明这一点。
"""

async def run_agent(
    user_query: str,
    conversation: list[dict],
    max_steps: int = 8,
) -> str:

    if not conversation:
        conversation.append(
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        )

    conversation.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    for step in range(
        1,
        max_steps + 1,
    ):

        print(
            f"\n{'=' * 20} "
            f"Agent Step {step} "
            f"{'=' * 20}"
        )

        response = await (
            client
            .chat
            .completions
            .create(
                model=MODEL,
                messages=conversation,
                tools=TOOLS,
                tool_choice="auto",
            )
        )

        assistant_message = (
            response
            .choices[0]
            .message
        )

        conversation.append(
            assistant_message.model_dump(
                exclude_none=True
            )
        )

        if not assistant_message.tool_calls:

            return (
                assistant_message.content
                or ""
            )

        for tool_call in (
            assistant_message.tool_calls
        ):

            tool_name = (
                tool_call
                .function
                .name
            )

            try:
                arguments = json.loads(
                    tool_call
                    .function
                    .arguments
                )

            except json.JSONDecodeError:

                tool_result = {
                    "error": (
                        "Invalid JSON arguments"
                    )
                }

            else:
                print(
                    "\n[Agent 调用 Tool]"
                )

                print(
                    f"Tool: {tool_name}"
                )

                print(
                    json.dumps(
                        arguments,
                        ensure_ascii=False,
                        indent=2,
                    )
                )

                tool_result = execute_tool(
                    tool_name,
                    arguments,
                )

            print(
                "\n[Tool Observation]"
            )

            print(
                json.dumps(
                    tool_result,
                    ensure_ascii=False,
                    indent=2,
                )
            )

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": (
                        tool_call.id
                    ),
                    "content": json.dumps(
                        tool_result,
                        ensure_ascii=False,
                    ),
                }
            )

    final_message = (
        "达到最大 Agent 执行步数，"
        "任务未能正常结束。"
    )

    conversation.append(
        {
            "role": "assistant",
            "content": final_message,
        }
    )

    return final_message

TOOL_REGISTRY = {
    "search_products": search_products_tool,
    "get_product_detail": get_product_detail_tool,
    "compare_products": compare_products_tool,
    "calculate_shipping": calculate_shipping_tool,
}

def execute_tool(   #工具调度器
    tool_name: str,
    arguments: dict,
):

    tool_function = TOOL_REGISTRY.get(
        tool_name
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
        return {
            "error": str(exc),
            "tool": tool_name,
        }