from app.tools.product_tools import (
    calculate_shipping_tool,
    compare_products_tool,
    get_product_detail_tool,
)


def test_get_existing_product():
    result = get_product_detail_tool(
        "laptop_001"
    )

    assert "error" not in result
    assert result["id"] == "laptop_001"


def test_get_missing_product():
    result = get_product_detail_tool(
        "not_exist"
    )

    assert "error" in result


def test_compare_products():
    result = compare_products_tool(
        [
            "laptop_001",
            "laptop_002",
        ]
    )

    assert len(
        result["products"]
    ) == 2

    assert result["missing_ids"] == []


def test_compare_with_missing_product():
    result = compare_products_tool(
        [
            "laptop_001",
            "not_exist",
        ]
    )

    assert "not_exist" in (
        result["missing_ids"]
    )


def test_shipping_to_japan():
    result = calculate_shipping_tool(
        price=8000,
        destination="日本",
    )

    assert (
        result["shipping_fee"]
        == 80.0
    )

    assert (
        result["estimated_total"]
        == 8080.0
    )