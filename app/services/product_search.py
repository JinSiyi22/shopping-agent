import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from app.models.product import Product


load_dotenv()


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


def get_data_path() -> Path:
    """
    Runtime catalog 可通过环境变量切换：

        PRODUCT_DATA_PATH=data/products.json
        PRODUCT_DATA_PATH=data/products_real.json

    Unit Test 会在 tests/conftest.py 中强制使用 mock catalog。
    """
    configured_path = os.getenv(
        "PRODUCT_DATA_PATH",
        "data/products.json",
    )

    path = Path(configured_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_products() -> list[Product]:
    data_path = get_data_path()

    with data_path.open(
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
    max_price: float | None = None,
    min_price: float | None = None,
    brand: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    min_rating: float | None = None,
    top_k: int = 5,
) -> list[SearchResult]:
    results: list[SearchResult] = []

    for product in products:
        if (
            max_price is not None
            and product.price > max_price
        ):
            continue

        if (
            min_price is not None
            and product.price < min_price
        ):
            continue

        if (
            brand is not None
            and product.brand.lower()
            != brand.lower()
        ):
            continue

        if (
            category is not None
            and product.category.lower()
            != category.lower()
        ):
            continue

        if (
            min_rating is not None
            and product.rating < min_rating
        ):
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

    safe_top_k = max(1, top_k)

    return results[:safe_top_k]
