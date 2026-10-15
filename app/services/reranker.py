from __future__ import annotations

from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from app.models.product import Product
from app.services.hybrid_search import (
    HybridSearchResult,
    hybrid_search_products,
)
from app.services.semantic_search import (
    build_product_document,
)


RERANKER_MODEL_NAME = (
    "BAAI/bge-reranker-base"
)


class RerankResult(BaseModel):
    product: Product

    rerank_score: float

    hybrid_score: float | None = None
    bm25_rank: int | None = None
    semantic_rank: int | None = None


_reranker_model: CrossEncoder | None = None


def get_reranker_model() -> CrossEncoder:
    global _reranker_model

    if _reranker_model is None:

        _reranker_model = CrossEncoder(
            RERANKER_MODEL_NAME
        )

    return _reranker_model


def reset_reranker() -> None:
    global _reranker_model

    _reranker_model = None


def rerank_products(
    query: str,
    candidates: list[
        HybridSearchResult
    ],
    top_k: int = 5,
) -> list[RerankResult]:

    query = query.strip()

    if not query:
        return []

    if not candidates:
        return []

    model = (
        get_reranker_model()
    )

    # -------------------------------------
    # Cross-Encoder输入：
    #
    # [
    #   [query, document1],
    #   [query, document2],
    #   ...
    # ]
    # -------------------------------------

    pairs = []

    for candidate in candidates:

        document = (
            build_product_document(
                candidate.product
            )
        )

        pairs.append(
            [
                query,
                document,
            ]
        )

    scores = (
        model.predict(
            pairs
        )
    )

    results = []

    for candidate, score in zip(
        candidates,
        scores,
    ):

        results.append(
            RerankResult(
                product=(
                    candidate.product
                ),
                rerank_score=(
                    float(score)
                ),
                hybrid_score=(
                    candidate.hybrid_score
                ),
                bm25_rank=(
                    candidate.bm25_rank
                ),
                semantic_rank=(
                    candidate.semantic_rank
                ),
            )
        )

    results.sort(
        key=lambda item: (
            item.rerank_score
        ),
        reverse=True,
    )

    safe_top_k = max(
        1,
        min(
            top_k,
            len(results),
        ),
    )

    return results[
        :safe_top_k
    ]


def retrieve_and_rerank_products(
    query: str,
    max_price: float | None = None,
    min_price: float | None = None,
    brand: str | None = None,
    category: str | None = None,
    min_rating: float | None = None,
    top_k: int = 5,
    candidate_k: int = 10,
) -> list[RerankResult]:

    # =====================================
    # Stage 1:
    # Hybrid Retrieval
    # =====================================

    candidates = (
        hybrid_search_products(
            query=query,
            max_price=max_price,
            min_price=min_price,
            brand=brand,
            category=category,
            min_rating=min_rating,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )
    )

    # =====================================
    # Stage 2:
    # Cross-Encoder Reranking
    # =====================================

    return rerank_products(
        query=query,
        candidates=candidates,
        top_k=top_k,
    )