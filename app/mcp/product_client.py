import asyncio
import json
import os
import sys

from mcp import (
    Client,
    StdioServerParameters,
)
from mcp.types import (
    TextContent,
)


class ProductMCPClient:

    def __init__(
        self,
    ):

        self._client = None

        self._start_lock = (
            asyncio.Lock()
        )

        self._tools = []

    async def start(
        self,
    ) -> None:

        async with self._start_lock:

            if self._client is not None:
                return

            child_env = {}

            for key in (
                "HF_HUB_OFFLINE",
                "HF_HOME",
                "HF_HUB_CACHE",
                "TRANSFORMERS_CACHE",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "http_proxy",
                "https_proxy",
                "NO_PROXY",
                "no_proxy",
            ):

                value = os.getenv(
                    key
                )

                if value is not None:

                    child_env[key] = value

            server_params = (
                StdioServerParameters(
                    command=sys.executable,

                    args=[
                        "-m",
                        "app.mcp.product_server",
                    ],

                    env=child_env,
                )
            )

            client = Client(
                server_params
            )

            await (
                client.__aenter__()
            )

            self._client = (
                client
            )

            tools_result = await (
                client.list_tools()
            )

            self._tools = (
                tools_result.tools
            )

    async def close(
        self,
    ) -> None:

        if self._client is None:
            return

        await (
            self._client
            .__aexit__(
                None,
                None,
                None,
            )
        )

        self._client = None
        self._tools = []

    async def ensure_started(
        self,
    ) -> None:

        if self._client is None:

            await (
                self.start()
            )

    async def list_tools(
        self,
    ):

        await (
            self.ensure_started()
        )

        return list(
            self._tools
        )

    async def get_openai_tools(
        self,
    ) -> list[dict]:

        tools = await (
            self.list_tools()
        )

        return [
            {
                "type": "function",

                "function": {
                    "name": (
                        tool.name
                    ),

                    "description": (
                        tool.description
                        or ""
                    ),

                    "parameters": (
                        tool.input_schema
                    ),
                },
            }

            for tool in tools
        ]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ):

        await (
            self.ensure_started()
        )

        result = await (
            self._client
            .call_tool(
                tool_name,
                arguments,
            )
        )

        if result.is_error:

            error_text = (
                self._extract_text(
                    result.content
                )
            )

            return {
                "error": (
                    error_text
                    or
                    f"MCP tool failed: "
                    f"{tool_name}"
                )
            }

        structured = (
            result
            .structured_content
        )

        if structured is not None:

            # MCPServer 对 list / primitive
            # 等返回值可能包装为：
            #
            # {"result": ...}
            if (
                isinstance(
                    structured,
                    dict,
                )
                and set(
                    structured.keys()
                )
                == {"result"}
            ):

                return (
                    structured[
                        "result"
                    ]
                )

            return structured

        text = (
            self._extract_text(
                result.content
            )
        )

        if not text:
            return None

        try:

            return json.loads(
                text
            )

        except json.JSONDecodeError:

            return {
                "content": text
            }

    @staticmethod
    def _extract_text(
        content,
    ) -> str:

        texts = []

        for block in content:

            if isinstance(
                block,
                TextContent,
            ):

                texts.append(
                    block.text
                )

        return "\n".join(
            texts
        )


product_mcp_client = (
    ProductMCPClient()
)