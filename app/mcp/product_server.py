from mcp.server import MCPServer

from app.tools.product_tools import (
    calculate_shipping_tool,
    compare_products_tool,
    get_product_detail_tool,
    hybrid_search_products_tool,
    rerank_search_products_tool,
    search_products_tool,
    semantic_search_products_tool,
)


mcp = MCPServer(
    "shopping-product-tools"
)


@mcp.tool()
def search_products(
    max_price: float | None = None,
    min_price: float | None = None,
    brand: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    min_rating: float | None = None,
    top_k: int = 5,
) -> list[dict]:
    """根据明确的结构化条件搜索商品。"""

    return search_products_tool(
        max_price=max_price,
        min_price=min_price,
        brand=brand,
        category=category,
        tags=tags,
        min_rating=min_rating,
        top_k=top_k,
    )


@mcp.tool()
def semantic_search_products(
    query: str,
    max_price: float | None = None,
    min_price: float | None = None,
    brand: str | None = None,
    category: str | None = None,
    min_rating: float | None = None,
    top_k: int = 5,
) -> list[dict]:
    """执行 Dense Embedding 语义商品检索。"""

    return semantic_search_products_tool(
        query=query,
        max_price=max_price,
        min_price=min_price,
        brand=brand,
        category=category,
        min_rating=min_rating,
        top_k=top_k,
    )


@mcp.tool()
def hybrid_search_products(
    query: str,
    max_price: float | None = None,
    min_price: float | None = None,
    brand: str | None = None,
    category: str | None = None,
    min_rating: float | None = None,
    top_k: int = 5,
) -> list[dict]:
    """执行 BM25 + Dense + RRF 混合检索。"""

    return hybrid_search_products_tool(
        query=query,
        max_price=max_price,
        min_price=min_price,
        brand=brand,
        category=category,
        min_rating=min_rating,
        top_k=top_k,
    )


@mcp.tool()
def rerank_search_products(
    query: str,
    max_price: float | None = None,
    min_price: float | None = None,
    brand: str | None = None,
    category: str | None = None,
    min_rating: float | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    高级自然语言商品检索。

    内部执行：
    BM25 + Dense Retrieval
    + RRF Fusion
    + Cross-Encoder Reranking。
    """

    return rerank_search_products_tool(
        query=query,
        max_price=max_price,
        min_price=min_price,
        brand=brand,
        category=category,
        min_rating=min_rating,
        top_k=top_k,
    )


@mcp.tool()
def get_product_detail(
    product_id: str,
) -> dict:
    """根据商品 ID 查询完整商品信息。"""

    return get_product_detail_tool(
        product_id=product_id
    )


@mcp.tool()
def compare_products(
    product_ids: list[str],
) -> dict:
    """比较多个已知商品。"""

    return compare_products_tool(
        product_ids=product_ids
    )


@mcp.tool()
def calculate_shipping(
    price: float,
    destination: str,
) -> dict:
    """计算开发阶段模拟运费。"""

    return calculate_shipping_tool(
        price=price,
        destination=destination,
    )


if __name__ == "__main__":

    mcp.run(
        transport="stdio"
    )