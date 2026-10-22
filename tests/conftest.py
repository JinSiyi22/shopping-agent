import pytest


@pytest.fixture(autouse=True)
def use_test_product_catalog(monkeypatch):
    """
    所有 Unit Tests 默认使用仓库内的小型确定性 mock catalog。

    即使开发环境 .env 配置了：
        PRODUCT_DATA_PATH=data/products_real.json

    原有 tests/ 也不会受真实 Amazon catalog 影响。
    """
    monkeypatch.setenv(
        "PRODUCT_DATA_PATH",
        "data/products.json",
    )

    from app.services import hybrid_search, semantic_search

    semantic_search.reset_semantic_index()
    hybrid_search.reset_bm25_index()

    yield

    semantic_search.reset_semantic_index()
    hybrid_search.reset_bm25_index()
