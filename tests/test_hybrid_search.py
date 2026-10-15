from app.services import (
    hybrid_search,
)
from app.services.semantic_search import (
    SemanticSearchResult,
)


def test_tokenize():
    tokens = hybrid_search.tokenize(
        "ThinkPad X1 适合编程"
    )

    assert len(tokens) > 0
    assert "thinkpad" in tokens


def test_bm25_price_filter():

    results = (
        hybrid_search
        .bm25_search_products(
            query="电脑",
            max_price=8000,
            top_k=10,
        )
    )

    for product, _ in results:
        assert product.price <= 8000


def test_hybrid_search(
    monkeypatch,
):

    products = (
        hybrid_search
        .load_products()
    )

    fake_dense_results = [
        SemanticSearchResult(
            product=products[0],
            semantic_score=0.95,
        ),
        SemanticSearchResult(
            product=products[1],
            semantic_score=0.90,
        ),
    ]

    monkeypatch.setattr(
        hybrid_search,
        "semantic_search_products",
        lambda **kwargs: (
            fake_dense_results
        ),
    )

    hybrid_search.reset_bm25_index()

    results = (
        hybrid_search
        .hybrid_search_products(
            query="编程电脑",
            top_k=5,
        )
    )

    assert len(results) > 0

    assert all(
        result.hybrid_score > 0
        for result in results
    )

    hybrid_search.reset_bm25_index()


def test_hybrid_respects_price_filter(
    monkeypatch,
):

    products = (
        hybrid_search
        .load_products()
    )

    fake_dense_results = [
        SemanticSearchResult(
            product=product,
            semantic_score=0.9,
        )
        for product in products
        if product.price <= 8000
    ]

    monkeypatch.setattr(
        hybrid_search,
        "semantic_search_products",
        lambda **kwargs: (
            fake_dense_results
        ),
    )

    hybrid_search.reset_bm25_index()

    results = (
        hybrid_search
        .hybrid_search_products(
            query="轻薄编程电脑",
            max_price=8000,
            top_k=10,
        )
    )

    for result in results:
        assert (
            result.product.price
            <= 8000
        )

    hybrid_search.reset_bm25_index()