import numpy as np

from app.services import (
    semantic_search,
)


class FakeEmbeddingModel:

    def encode(
        self,
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ):
        vectors = []

        for text in texts:

            text_lower = (
                text.lower()
            )

            programming_score = 0.0
            performance_score = 0.0
            office_score = 0.0

            if (
                "编程" in text_lower
                or "写代码"
                in text_lower
            ):
                programming_score = 1.0

            if (
                "gpu" in text_lower
                or "高性能"
                in text_lower
                or "ai"
                in text_lower
                or "小模型"
                in text_lower
            ):
                performance_score = 1.0

            if (
                "办公" in text_lower
                or "商务"
                in text_lower
            ):
                office_score = 1.0

            vector = np.array(
                [
                    programming_score,
                    performance_score,
                    office_score,
                ],
                dtype=np.float32,
            )

            norm = np.linalg.norm(
                vector
            )

            if norm > 0:
                vector = (
                    vector / norm
                )

            vectors.append(
                vector
            )

        return np.stack(
            vectors
        )


def test_build_product_document():

    products = (
        semantic_search
        .load_products()
    )

    document = (
        semantic_search
        .build_product_document(
            products[0]
        )
    )

    assert products[0].name in document
    assert products[0].brand in document


def test_semantic_search(
    monkeypatch,
):

    fake_model = (
        FakeEmbeddingModel()
    )

    monkeypatch.setattr(
        semantic_search,
        "get_embedding_model",
        lambda: fake_model,
    )

    semantic_search.reset_semantic_index()

    results = (
        semantic_search
        .semantic_search_products(
            query="我想写代码",
            top_k=3,
        )
    )

    assert len(results) > 0

    assert any(
        "编程"
        in result.product.tags
        for result in results
    )

    semantic_search.reset_semantic_index()


def test_semantic_search_price_filter(
    monkeypatch,
):

    fake_model = (
        FakeEmbeddingModel()
    )

    monkeypatch.setattr(
        semantic_search,
        "get_embedding_model",
        lambda: fake_model,
    )

    semantic_search.reset_semantic_index()

    results = (
        semantic_search
        .semantic_search_products(
            query="写代码",
            max_price=8000,
            top_k=5,
        )
    )

    for result in results:
        assert (
            result.product.price
            <= 8000
        )

    semantic_search.reset_semantic_index()