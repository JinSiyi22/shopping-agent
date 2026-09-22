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