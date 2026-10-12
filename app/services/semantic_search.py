from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from app.models.product import Product
from app.services.product_search import load_products


EMBEDDING_MODEL_NAME = (
    "BAAI/bge-small-zh-v1.5"
)


class SemanticSearchResult(BaseModel):
    product: Product
    semantic_score: float


_embedding_model: (
    SentenceTransformer | None
) = None

_indexed_products: list[Product] = []

_product_embeddings: (
    np.ndarray | None
) = None


def build_product_document(
    product: Product,
) -> str:
    """
    把结构化商品数据转换成适合做 Embedding 的文本。
    """

    tags_text = "、".join(
        product.tags
    )

    return (
        f"商品名称：{product.name}。"
        f"品牌：{product.brand}。"
        f"商品类别：{product.category}。"
        f"价格：{product.price} "
        f"{product.currency}。"
        f"评分：{product.rating}。"
        f"标签：{tags_text}。"
        f"商品描述：{product.description}"
    )


def get_embedding_model() -> (
    SentenceTransformer
):
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = (
            SentenceTransformer(
                EMBEDDING_MODEL_NAME
            )
        )

    return _embedding_model


def reset_semantic_index() -> None:
    """
    开发和测试时用于清空内存中的向量索引。
    """

    global _indexed_products
    global _product_embeddings

    _indexed_products = []
    _product_embeddings = None


def _build_index_if_needed() -> None:
    global _indexed_products
    global _product_embeddings

    if _product_embeddings is not None:
        return

    products = load_products()

    documents = [
        build_product_document(
            product
        )
        for product in products
    ]

    model = get_embedding_model()

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    _indexed_products = products

    _product_embeddings = (
        np.asarray(
            embeddings,
            dtype=np.float32,
        )
    )


def _matches_filters(
    product: Product,
    max_price: float | None,
    min_price: float | None,
    brand: str | None,
    category: str | None,
    min_rating: float | None,
) -> bool:

    if (
        max_price is not None
        and product.price > max_price
    ):
        return False

    if (
        min_price is not None
        and product.price < min_price
    ):
        return False

    if (
        brand is not None
        and product.brand.lower()
        != brand.lower()
    ):
        return False

    if (
        category is not None
        and product.category.lower()
        != category.lower()
    ):
        return False

    if (
        min_rating is not None
        and product.rating < min_rating
    ):
        return False

    return True


def semantic_search_products(
    query: str,
    max_price: float | None = None,
    min_price: float | None = None,
    brand: str | None = None,
    category: str | None = None,
    min_rating: float | None = None,
    top_k: int = 5,
) -> list[SemanticSearchResult]:

    query = query.strip()

    if not query:
        return []

    _build_index_if_needed()

    if _product_embeddings is None:
        return []

    model = get_embedding_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    candidate_indices = [
        index
        for index, product
        in enumerate(
            _indexed_products
        )
        if _matches_filters(
            product=product,
            max_price=max_price,
            min_price=min_price,
            brand=brand,
            category=category,
            min_rating=min_rating,
        )
    ]

    if not candidate_indices:
        return []

    scored_results: list[
        SemanticSearchResult
    ] = []

    for index in candidate_indices:

        product_embedding = (
            _product_embeddings[
                index
            ]
        )

        semantic_score = float(
            np.dot(
                query_embedding,
                product_embedding,
            )
        )

        scored_results.append(
            SemanticSearchResult(
                product=(
                    _indexed_products[
                        index
                    ]
                ),
                semantic_score=(
                    semantic_score
                ),
            )
        )

    scored_results.sort(
        key=lambda item: (
            item.semantic_score
        ),
        reverse=True,
    )

    safe_top_k = max(
        1,
        min(
            top_k,
            20,
        ),
    )

    return scored_results[
        :safe_top_k
    ]