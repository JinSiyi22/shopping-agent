from __future__ import annotations

from collections import defaultdict

import jieba
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from app.models.product import Product
from app.services.product_search import (
    load_products,
)
from app.services.semantic_search import (
    build_product_document,
    semantic_search_products,
)


class HybridSearchResult(BaseModel):
    product: Product

    hybrid_score: float

    bm25_rank: int | None = None
    semantic_rank: int | None = None

    bm25_score: float | None = None
    semantic_score: float | None = None


_bm25_products: list[Product] = []
_bm25_documents: list[str] = []
_bm25_model: BM25Okapi | None = None


def tokenize(
    text: str,
) -> list[str]:
    """
    中文 / 英文混合文本分词。
    """

    return [
        token.strip().lower()
        for token in jieba.lcut(text)
        if token.strip()
    ]


def reset_bm25_index() -> None:
    global _bm25_products
    global _bm25_documents
    global _bm25_model

    _bm25_products = []
    _bm25_documents = []
    _bm25_model = None


def _build_bm25_index_if_needed() -> None:
    global _bm25_products
    global _bm25_documents
    global _bm25_model

    if _bm25_model is not None:
        return

    products = load_products()

    documents = [
        build_product_document(
            product
        )
        for product in products
    ]

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    _bm25_products = products
    _bm25_documents = documents

    _bm25_model = BM25Okapi(
        tokenized_documents
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


def bm25_search_products(
    query: str,
    max_price: float | None = None,
    min_price: float | None = None,
    brand: str | None = None,
    category: str | None = None,
    min_rating: float | None = None,
    top_k: int = 10,
) -> list[tuple[Product, float]]:

    query = query.strip()

    if not query:
        return []

    _build_bm25_index_if_needed()

    if _bm25_model is None:
        return []

    query_tokens = tokenize(
        query
    )

    scores = (
        _bm25_model
        .get_scores(
            query_tokens
        )
    )

    candidates: list[
        tuple[Product, float]
    ] = []

    for index, product in enumerate(
        _bm25_products
    ):

        if not _matches_filters(
            product=product,
            max_price=max_price,
            min_price=min_price,
            brand=brand,
            category=category,
            min_rating=min_rating,
        ):
            continue

        candidates.append(
            (
                product,
                float(scores[index]),
            )
        )

    candidates.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    safe_top_k = max(
        1,
        min(top_k, 50),
    )

    return candidates[
        :safe_top_k
    ]


def hybrid_search_products(
    query: str,
    max_price: float | None = None,
    min_price: float | None = None,
    brand: str | None = None,
    category: str | None = None,
    min_rating: float | None = None,
    top_k: int = 5,
    candidate_k: int = 10,
    rrf_k: int = 60,
    bm25_weight: float = 1.0,
    semantic_weight: float = 1.0,
) -> list[HybridSearchResult]:

    query = query.strip()

    if not query:
        return []

    # ==================================================
    # 1. BM25 Recall
    # ==================================================

    bm25_results = (
        bm25_search_products(
            query=query,
            max_price=max_price,
            min_price=min_price,
            brand=brand,
            category=category,
            min_rating=min_rating,
            top_k=candidate_k,
        )
    )

    # ==================================================
    # 2. Dense Recall
    # ==================================================

    semantic_results = (
        semantic_search_products(
            query=query,
            max_price=max_price,
            min_price=min_price,
            brand=brand,
            category=category,
            min_rating=min_rating,
            top_k=candidate_k,
        )
    )

    # ==================================================
    # 3. Candidate Union
    # ==================================================

    product_map: dict[
        str,
        Product
    ] = {}

    bm25_rank_map: dict[
        str,
        int
    ] = {}

    semantic_rank_map: dict[
        str,
        int
    ] = {}

    bm25_score_map: dict[
        str,
        float
    ] = {}

    semantic_score_map: dict[
        str,
        float
    ] = {}

    # ==================================================
    # 4. RRF Scores
    # ==================================================

    fused_scores: dict[
        str,
        float
    ] = defaultdict(float)

    for rank, (
        product,
        bm25_score,
    ) in enumerate(
        bm25_results,
        start=1,
    ):

        product_id = product.id

        product_map[
            product_id
        ] = product

        bm25_rank_map[
            product_id
        ] = rank

        bm25_score_map[
            product_id
        ] = bm25_score

        fused_scores[
            product_id
        ] += (
            bm25_weight
            / (rrf_k + rank)
        )

    for rank, result in enumerate(
        semantic_results,
        start=1,
    ):

        product_id = (
            result.product.id
        )

        product_map[
            product_id
        ] = result.product

        semantic_rank_map[
            product_id
        ] = rank

        semantic_score_map[
            product_id
        ] = (
            result.semantic_score
        )

        fused_scores[
            product_id
        ] += (
            semantic_weight
            / (rrf_k + rank)
        )

    # ==================================================
    # 5. Final Ranking
    # ==================================================

    ranked_product_ids = sorted(
        fused_scores,
        key=lambda product_id: (
            fused_scores[
                product_id
            ]
        ),
        reverse=True,
    )

    safe_top_k = max(
        1,
        min(top_k, 20),
    )

    results: list[
        HybridSearchResult
    ] = []

    for product_id in (
        ranked_product_ids[
            :safe_top_k
        ]
    ):

        results.append(
            HybridSearchResult(
                product=product_map[
                    product_id
                ],
                hybrid_score=(
                    fused_scores[
                        product_id
                    ]
                ),
                bm25_rank=(
                    bm25_rank_map.get(
                        product_id
                    )
                ),
                semantic_rank=(
                    semantic_rank_map.get(
                        product_id
                    )
                ),
                bm25_score=(
                    bm25_score_map.get(
                        product_id
                    )
                ),
                semantic_score=(
                    semantic_score_map.get(
                        product_id
                    )
                ),
            )
        )

    return results