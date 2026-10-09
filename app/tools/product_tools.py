from app.services.product_search import (
    load_products,
    search_products,
)


def search_products_tool(
    max_price: float | None = None,
    min_price: float | None = None,
    brand: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    min_rating: float | None = None,
    top_k: int = 5,
) -> list[dict]:    #LLM 特别喜欢：JSON / Dictionary
    """
    根据用户给出的搜索条件筛选并排序商品。

    主要作用：
    1. 加载全部商品数据；
    2. 调用底层 search_products() 完成价格、品牌、类别、
       标签和评分筛选；
    3. 按搜索得分排序；
    4. 将 Pydantic SearchResult 对象转换成普通 dict，
       便于 LLM Tool Calling 使用。

    返回：
        商品搜索结果列表，每个元素都是一个 dict。
    """

    products = load_products()

    results = search_products(
        products=products,
        max_price=max_price,
        min_price=min_price,
        brand=brand,
        category=category,
        tags=tags,
        min_rating=min_rating,
        top_k=top_k,
    )

    return [
        {
            "id": result.product.id,
            "name": result.product.name,
            "brand": result.product.brand,
            "price": result.product.price,
            "currency": result.product.currency,
            "rating": result.product.rating,
            "tags": result.product.tags,
            "description": result.product.description,
            "matched_tags": result.matched_tags,
            "search_score": result.score,
        }
        for result in results
    ]

def get_product_detail_tool(
    product_id: str,
) -> dict:
    """
    根据商品 ID 查询某一个商品的完整详细信息。

    主要用途：
    当 Agent 已经通过搜索得到商品 ID 后，
    可以进一步获取该商品的完整字段。

    参数：
        product_id: 商品唯一 ID。

    返回：
        找到商品时返回商品 dict；
        找不到时返回 error 信息。
    """

    products = load_products()

    for product in products:
        if product.id == product_id:
            return product.model_dump()

    return {
        "error": f"Product not found: {product_id}"
    }

def compare_products_tool(
    product_ids: list[str],
) -> dict:
    """
    根据多个商品 ID 获取商品信息，用于商品之间的横向比较。

    主要作用：
    1. 根据 product_ids 找出对应商品；
    2. 返回所有找到的商品；
    3. 记录不存在的商品 ID；
    4. 告诉 Agent 推荐比较哪些字段。

    参数：
        product_ids: 要比较的商品 ID 列表。

    返回：
        products:
            成功找到的商品数据；
        missing_ids:
            没找到的商品 ID；
        comparison_fields:
            建议比较的字段。
    """

    products = load_products()

    product_map = {
        product.id: product
        for product in products
    }

    selected_products = []

    missing_ids = []

    for product_id in product_ids:

        product = product_map.get(product_id)

        if product is None:
            missing_ids.append(product_id)
            continue

        selected_products.append(
            product.model_dump()
        )

    return {
        "products": selected_products,
        "missing_ids": missing_ids,
        "comparison_fields": [
            "price",
            "rating",
            "brand",
            "tags",
            "description",
        ],
    }

def calculate_shipping_tool(
    price: float,
    destination: str,
) -> dict:
    """
    根据商品价格和目的地计算模拟运费及预计总价。

    当前只是开发测试用的 Mock Tool，
    并没有连接真实物流 API。

    运费规则：
    - 中国大陆：0 元
    - 日本：80 元
    - 其他地区：150 元

    参数：
        price:
            商品价格。
        destination:
            收货国家或地区。

    返回：
        商品价格、目的地、运费以及预计总价。
    """
    
    destination_lower = destination.lower()

    if destination_lower in {
        "china",
        "cn",
        "中国",
        "中国大陆",
    }:
        shipping_fee = 0.0

    elif destination_lower in {
        "japan",
        "jp",
        "日本",
    }:
        shipping_fee = 80.0

    else:
        shipping_fee = 150.0

    return {
        "product_price": price,
        "destination": destination,
        "shipping_fee": shipping_fee,
        "estimated_total": price + shipping_fee,
        "note": (
            "This is a mock shipping estimate "
            "for development and testing."
        ),
    }