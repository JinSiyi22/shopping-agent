import json
from pathlib import Path
from pydantic import BaseModel
from app.models.product import Product



DATA_PATH = (
    Path(__file__).resolve().parents[2] #__file__代表当前 Python 文件的位置；.parents[2]向上两层
    / "data"
    / "products.json"
)


def load_products() -> list[Product]:
    with open(
        DATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        raw_products = json.load(file)

    return [
        Product(**item)
        for item in raw_products
    ]

class SearchResult(BaseModel):
    product: Product
    score: float
    matched_tags: list[str]

def search_products(
    products: list[Product],
    max_price: float | None = None, #max_price 可以是一个 float，也可以什么都不传。
    min_price: float | None = None,
    brand: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    min_rating: float | None = None,
    top_k: int = 5,
) -> list[SearchResult]:

    results: list[SearchResult] = []

    for product in products:

        if max_price is not None:
            if product.price > max_price:
                continue

        if min_price is not None:
            if product.price < min_price:
                continue

        if brand is not None:
            if product.brand.lower() != brand.lower():
                continue

        if category is not None:
            if product.category.lower() != category.lower():
                continue

        if min_rating is not None:
            if product.rating < min_rating:
                continue

        score = 0.0
        matched_tags: list[str] = []

        if tags:
            for tag in tags:
                if tag in product.tags:
                    matched_tags.append(tag)
                    score += 1.0

        score += product.rating / 5.0

        results.append(
            SearchResult(
                product=product,
                score=score,
                matched_tags=matched_tags,
            )
        )

    results.sort(
        key=lambda item: item.score,
        reverse=True,
    )

    return results[:top_k]