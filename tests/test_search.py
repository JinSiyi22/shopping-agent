from app.services.product_search import (
    load_products,
    search_products,
)


def test_load_products():
    products = load_products()

    assert len(products) > 0


def test_max_price_filter():
    products = load_products()

    results = search_products(
        products=products,
        max_price=8000,
    )

    assert len(results) > 0

    for result in results:
        assert result.product.price <= 8000


def test_brand_filter():
    products = load_products()

    results = search_products(
        products=products,
        brand="Lenovo",
    )

    for result in results:
        assert (
            result.product.brand.lower()
            == "lenovo"
        )


def test_top_k():
    products = load_products()

    results = search_products(
        products=products,
        top_k=2,
    )

    assert len(results) <= 2