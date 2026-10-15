import numpy as np

from app.services import (
    reranker,
)
from app.services.hybrid_search import (
    HybridSearchResult,
)


class FakeCrossEncoder:

    def predict(
        self,
        pairs,
    ):

        scores = []

        for query, document in pairs:

            score = 0.0

            if (
                "轻薄" in document
                or "商务" in document
            ):
                score += 2.0

            if "编程" in document:
                score += 1.0

            if "GPU" in document:
                score += 0.5

            scores.append(
                score
            )

        return np.array(
            scores,
            dtype=np.float32,
        )


def test_rerank_products(
    monkeypatch,
):

    products = (
        reranker
        .hybrid_search_products(
            query="电脑",
            top_k=3,
            candidate_k=3,
        )
    )

    monkeypatch.setattr(
        reranker,
        "get_reranker_model",
        lambda: FakeCrossEncoder(),
    )

    results = (
        reranker
        .rerank_products(
            query=(
                "轻薄 编程"
            ),
            candidates=products,
            top_k=2,
        )
    )

    assert len(
        results
    ) <= 2

    assert all(
        hasattr(
            result,
            "rerank_score",
        )
        for result in results
    )