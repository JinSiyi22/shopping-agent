from app.data.amazon_reviews_adapter import (
    adapt_amazon_product,
    extract_primary_title,
    infer_category,
    infer_taxonomy_category,
    title_supports_category,
)


def build_base_raw(
    *,
    title: str,
    parent_asin: str,
    categories: list[str],
) -> dict:

    return {
        "main_category": "Electronics",
        "title": title,
        "average_rating": 4.6,
        "rating_number": 120,
        "features": [],
        "description": [],
        "price": 899.99,
        "store": "Example Brand",
        "categories": categories,
        "details": {
            "Brand": "Example Brand"
        },
        "parent_asin": parent_asin,
    }


def test_valid_laptop():

    raw = build_base_raw(
        title=(
            "Dell Inspiron 15 Laptop"
        ),
        parent_asin="LAPTOP_001",
        categories=[
            "Electronics",
            "Computers & Accessories",
            "Laptops",
        ],
    )

    product = adapt_amazon_product(
        raw
    )

    assert product is not None
    assert product.category == "laptop"


def test_reject_laptop_charger_by_taxonomy():

    raw = build_base_raw(
        title=(
            "65W AC Charger "
            "for ASUS Laptop"
        ),
        parent_asin="ACCESSORY_001",
        categories=[
            "Electronics",
            "Computers & Accessories",
            "Laptops",
            "Laptop Accessories",
            "Chargers & Adapters",
        ],
    )

    raw["price"] = 19.99

    assert (
        infer_taxonomy_category(
            raw
        )
        is None
    )

    assert (
        adapt_amazon_product(
            raw
        )
        is None
    )


def test_reject_laptop_sticker_by_taxonomy():

    raw = build_base_raw(
        title=(
            "Football Stickers "
            "for Water Bottle Laptop"
        ),
        parent_asin="ACCESSORY_002",
        categories=[
            "Electronics",
            "Accessories",
            "Stickers & Decals",
        ],
    )

    raw["price"] = 5.99

    assert (
        adapt_amazon_product(
            raw
        )
        is None
    )


def test_reject_monitor_mount():

    raw = build_base_raw(
        title=(
            "VIVO Monitor Mount "
            "for 27 Inch Monitor"
        ),
        parent_asin="MONITOR_ACCESSORY_001",
        categories=[
            "Electronics",
            "Computers & Accessories",
            "Monitors",
            "Monitor Accessories",
            "Mounts",
        ],
    )

    raw["price"] = 39.99

    assert (
        adapt_amazon_product(
            raw
        )
        is None
    )


def test_monitor_headphones_stay_headphones():

    raw = build_base_raw(
        title=(
            "Professional Studio "
            "Monitor Headphones"
        ),
        parent_asin="HEADPHONE_001",
        categories=[
            "Electronics",
            "Headphones",
            "Over-Ear Headphones",
        ],
    )

    raw["price"] = 149.99

    assert (
        infer_taxonomy_category(
            raw
        )
        == "headphone"
    )

    assert title_supports_category(
        raw["title"],
        "headphone",
    )

    product = adapt_amazon_product(
        raw
    )

    assert product is not None
    assert product.category == "headphone"


def test_reject_headphone_adapter():

    raw = build_base_raw(
        title=(
            "Lightning to 3.5mm "
            "Headphone Jack Adapter"
        ),
        parent_asin="HEADPHONE_ACCESSORY_001",
        categories=[
            "Electronics",
            "Headphone Accessories",
            "Adapters",
        ],
    )

    raw["price"] = 12.99

    assert (
        adapt_amazon_product(
            raw
        )
        is None
    )


def test_valid_phone():

    raw = build_base_raw(
        title=(
            "Unlocked Android Smartphone "
            "128GB 5G"
        ),
        parent_asin="PHONE_001",
        categories=[
            "Electronics",
            "Cell Phones & Accessories",
            "Cell Phones",
        ],
    )

    raw["price"] = 399.99

    product = adapt_amazon_product(
        raw
    )

    assert product is not None
    assert product.category == "phone"


def test_reject_phone_tripod():

    raw = build_base_raw(
        title=(
            "Flexible Tripod "
            "with Smartphone Clamp"
        ),
        parent_asin="PHONE_ACCESSORY_001",
        categories=[
            "Electronics",
            "Cell Phones & Accessories",
            "Accessories",
            "Tripods",
        ],
    )

    raw["price"] = 24.99

    assert (
        adapt_amazon_product(
            raw
        )
        is None
    )


def test_reject_desktop_memory():

    raw = build_base_raw(
        title=(
            "128GB DDR4 Desktop "
            "Computer Memory"
        ),
        parent_asin="DESKTOP_ACCESSORY_001",
        categories=[
            "Electronics",
            "Computers & Accessories",
            "Computer Components",
            "Memory",
            "Memory Modules",
        ],
    )

    raw["price"] = 229.99

    assert (
        adapt_amazon_product(
            raw
        )
        is None
    )


def test_valid_desktop():

    raw = build_base_raw(
        title=(
            "Gaming Desktop PC "
            "32GB RAM 1TB SSD"
        ),
        parent_asin="DESKTOP_001",
        categories=[
            "Electronics",
            "Computers & Accessories",
            "Desktop Computers",
        ],
    )

    product = adapt_amazon_product(
        raw
    )

    assert product is not None
    assert product.category == "desktop"


def test_storage_not_misclassified_as_desktop():

    raw = build_base_raw(
        title=(
            "1TB SSD for "
            "Desktop PC Laptop"
        ),
        parent_asin="STORAGE_001",
        categories=[
            "Electronics",
            "Computers & Accessories",
            "Data Storage",
            "Internal Solid State Drives",
        ],
    )

    raw["price"] = 89.99

    product = adapt_amazon_product(
        raw
    )

    assert product is not None
    assert product.category == "storage"


def test_primary_title_removes_for_compatibility_tail():

    primary_title = extract_primary_title(
        "USB Microphone for Laptop and PC"
    )

    assert (
        primary_title
        == "usb microphone"
    )


def test_primary_title_removes_compatible_with_tail():

    primary_title = extract_primary_title(
        (
            "Wireless Headphones "
            "Compatible with iPhone"
        )
    )

    assert (
        primary_title
        == "wireless headphones"
    )


def test_primary_title_keeps_real_laptop():

    title = (
        "Dell Inspiron Laptop "
        "for Students and Business"
    )

    assert (
        extract_primary_title(
            title
        )
        == "dell inspiron laptop"
    )

    assert title_supports_category(
        title,
        "laptop",
    )


def test_compatibility_target_is_not_product_type():

    assert not title_supports_category(
        (
            "USB Microphone "
            "Compatible with Laptop"
        ),
        "laptop",
    )


def test_real_headphone_with_compatibility_tail():

    raw = build_base_raw(
        title=(
            "Wireless Bluetooth Headphones "
            "Compatible with iPhone"
        ),
        parent_asin="HEADPHONE_COMPAT_001",
        categories=[
            "Electronics",
            "Headphones",
        ],
    )

    raw["price"] = 79.99

    product = adapt_amazon_product(
        raw
    )

    assert product is not None
    assert (
        product.category
        == "headphone"
    )


def test_reject_taxonomy_title_disagreement():

    raw = build_base_raw(
        title=(
            "USB Microphone "
            "for Laptop and PC"
        ),
        parent_asin="CONFLICT_001",
        categories=[
            "Electronics",
            "Computers & Accessories",
            "Laptops",
        ],
    )

    assert (
        infer_taxonomy_category(
            raw
        )
        == "laptop"
    )

    assert not title_supports_category(
        raw["title"],
        "laptop",
    )

    assert (
        adapt_amazon_product(
            raw
        )
        is None
    )


def test_details_json_string_supported():

    raw = build_base_raw(
        title=(
            "Wireless Bluetooth Headphones"
        ),
        parent_asin="HEADPHONE_002",
        categories=[
            "Electronics",
            "Headphones",
        ],
    )

    raw["details"] = (
        '{"Brand": "Audio Brand", '
        '"Color": "Black"}'
    )

    raw["price"] = 79.99

    product = adapt_amazon_product(
        raw
    )

    assert product is not None
    assert product.brand == "Audio Brand"


def test_reject_product_without_price():

    raw = build_base_raw(
        title="Example Laptop",
        parent_asin="NO_PRICE_001",
        categories=[
            "Electronics",
            "Computers & Accessories",
            "Laptops",
        ],
    )

    raw["price"] = None

    assert (
        adapt_amazon_product(
            raw
        )
        is None
    )
