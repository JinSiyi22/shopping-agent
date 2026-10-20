import asyncio

from app.mcp.product_client import (
    product_mcp_client,
)


async def main():

    try:

        await (
            product_mcp_client
            .start()
        )

        tools = await (
            product_mcp_client
            .list_tools()
        )

        print(
            "\n=== MCP Tools ==="
        )

        for tool in tools:

            print(
                f"- {tool.name}"
            )

        print(
            "\n=== MCP Call ==="
        )

        result = await (
            product_mcp_client
            .call_tool(
                "search_products",
                {
                    "max_price": 10000,
                    "category": "laptop",
                    "top_k": 3,
                },
            )
        )

        print(
            result
        )

    finally:

        await (
            product_mcp_client
            .close()
        )


if __name__ == "__main__":

    asyncio.run(
        main()
    )