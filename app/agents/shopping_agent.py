import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from app.tools.product_tools import search_products_tool


load_dotenv()


client = OpenAI(
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
    }
]

SYSTEM_PROMPT = """
你是一个专业的商品研究与购买决策助手。

你的任务是理解用户的购买需求，并使用提供的商品搜索工具获取真实商品数据。

规则：

1. 用户要求推荐、搜索、比较商品时，优先调用 search_products 工具。
2. 不要编造商品数据库中不存在的商品信息。
3. 用户自然语言中的需求可以转换成最接近的商品标签。
4. 例如：
   - 写代码、开发 -> 编程
   - AI计算、机器学习、小模型训练 -> GPU、高性能
   - 经常出差、便携 -> 轻薄
5. 推荐时需要解释商品为什么符合用户需求。
6. 如果现有商品无法充分满足用户要求，要明确说明。
"""

def run_agent(user_query: str) -> str:

    messages = [    #这次要发给大模型的对话内容
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]

    response = client.chat.completions.create(  #把请求发给大模型
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message

    if not assistant_message.tool_calls:
        return assistant_message.content or ""

    messages.append(
        assistant_message.model_dump(   #把 Pydantic / SDK 对象转换成普通 Python dict
            exclude_none=True
        )
    )

    for tool_call in assistant_message.tool_calls:

        tool_name = tool_call.function.name

        arguments = json.loads( #字符串 → Python对象
            tool_call.function.arguments
        )

        print("\n[Agent 决定调用 Tool]")
        print(f"Tool: {tool_name}")
        print(
            "Arguments:",
            json.dumps(
                arguments,
                ensure_ascii=False,
                indent=2,
            ),
        )

        if tool_name == "search_products":

            tool_result = search_products_tool(
                **arguments
            )

        else:
            tool_result = {
                "error": f"Unknown tool: {tool_name}"
            }

        print("\n[Tool 执行结果]")
        print(
            json.dumps( #Python 对象 → JSON 字符串
                tool_result,
                ensure_ascii=False,
                indent=2,
            )
        )

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    tool_result,
                    ensure_ascii=False,
                ),
            }
        )

    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
    )

    return (
        final_response
        .choices[0]
        .message
        .content
        or ""
    )