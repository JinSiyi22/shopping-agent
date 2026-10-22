from __future__ import annotations

import json
import re
from typing import Any

from app.models.product import Product


# ============================================================
# Day 18 - Amazon Reviews 2023 Adapter: Taxonomy Agreement V3
# ============================================================
#
# 目标：
# 1. Amazon hierarchical categories 是第一信号；
# 2. title 只做 product-type corroboration；
# 3. taxonomy/title 不一致时直接 abstain；
# 4. precision-first，不追求 catalog recall；
# 5. 不修改现有 Product schema。
# ============================================================


TAXONOMY_CATEGORY_PATTERNS: dict[
    str,
    tuple[str, ...],
] = {
    "laptop": (
        r"laptops?",
        r"laptop computers?",
        r"traditional laptops?",
        r"2[- ]in[- ]1 laptops?",
        r"chromebooks?",
    ),
    "desktop": (
        r"desktop computers?",
        r"desktop pcs?",
        r"all[- ]in[- ]one computers?",
        r"mini computers?",
        r"mini pcs?",
        r"tower computers?",
    ),
    "monitor": (
        r"monitors?",
        r"computer monitors?",
    ),
    "tablet": (
        r"tablets?",
        r"tablet computers?",
    ),
    "phone": (
        r"cell phones?",
        r"mobile phones?",
        r"smartphones?",
        r"unlocked cell phones?",
        r"unlocked phones?",
    ),
    "headphone": (
        r"headphones?",
        r"headsets?",
        r"earbuds?",
        r"in[- ]ear headphones?",
        r"over[- ]ear headphones?",
        r"on[- ]ear headphones?",
        r"earbuds? & in[- ]ear headphones?",
    ),
    "keyboard": (
        r"keyboards?",
        r"computer keyboards?",
    ),
    "mouse": (
        r"computer mice",
        r"mice",
        r"computer mouse",
    ),
    "storage": (
        r"solid state drives?",
        r"internal solid state drives?",
        r"external solid state drives?",
        r"internal hard drives?",
        r"external hard drives?",
        r"hard disk drives?",
        r"usb flash drives?",
        r"flash drives?",
        r"memory cards?",
    ),
    "router": (
        r"routers?",
        r"wireless routers?",
        r"wi[- ]fi routers?",
        r"mesh routers?",
    ),
    "camera": (
        r"digital cameras?",
        r"mirrorless cameras?",
        r"dslr cameras?",
        r"action cameras?",
        r"webcams?",
    ),
    "speaker": (
        r"bluetooth speakers?",
        r"portable bluetooth speakers?",
        r"wireless speakers?",
        r"computer speakers?",
        r"smart speakers?",
    ),
    "smartwatch": (
        r"smartwatches?",
        r"smart watches?",
    ),
    "printer": (
        r"printers?",
        r"laser printers?",
        r"inkjet printers?",
        r"photo printers?",
        r"thermal printers?",
    ),
}


TAXONOMY_BLOCKER_PATTERNS = (
    r".*\baccessor(?:y|ies)\b.*",
    r".*\bchargers?\b.*",
    r".*\badapters?\b.*",
    r".*\badaptors?\b.*",
    r".*\bcables?\b.*",
    r".*\bcords?\b.*",
    r".*\bcases?\b.*",
    r".*\bcovers?\b.*",
    r".*\bsleeves?\b.*",
    r".*\bbags?\b.*",
    r".*\bstands?\b.*",
    r".*\bmounts?\b.*",
    r".*\bbrackets?\b.*",
    r".*\bprotectors?\b.*",
    r".*\bskins?\b.*",
    r".*\bstickers?\b.*",
    r".*\bdecals?\b.*",
    r".*\breplacement\b.*",
    r".*\bparts?\b.*",
    r".*\bbatter(?:y|ies)\b.*",
    r".*\bmemory upgrades?\b.*",
    r".*\bram modules?\b.*",
    r".*\bmemory modules?\b.*",
    r".*\bdocking stations?\b.*",
    r".*\bdocks?\b.*",
    r".*\benclosures?\b.*",
    r".*\bcard readers?\b.*",
    r".*\bdrive bays?\b.*",
    r".*\boptical drives?\b.*",
    r".*\bdvd drives?\b.*",
    r".*\bblu[- ]?ray drives?\b.*",
)


TITLE_CORROBORATION_PATTERNS: dict[
    str,
    tuple[str, ...],
] = {
    "laptop": (
        r"\blaptop(?:s)?\b",
        r"\bnotebook computer\b",
        r"\bnotebook pc\b",
        r"\bchromebook(?:s)?\b",
        r"\bmacbook(?: air| pro)?\b",
    ),
    "desktop": (
        r"\bdesktop computer\b",
        r"\bdesktop pc\b",
        r"\bgaming desktop\b",
        r"\bgaming pc\b",
        r"\bmini pc\b",
        r"\btower pc\b",
        r"\ball[- ]in[- ]one(?: desktop| computer| pc)?\b",
    ),
    "monitor": (
        r"\bmonitor(?:s)?\b",
        r"\bcomputer display\b",
    ),
    "tablet": (
        r"\btablet(?:s)?\b",
        r"\bipad(?:s)?\b",
    ),
    "phone": (
        r"\bsmartphone(?:s)?\b",
        r"\bcell phone(?:s)?\b",
        r"\bmobile phone(?:s)?\b",
        r"\bunlocked phone(?:s)?\b",
        r"\biphone(?:s)?\b",
    ),
    "headphone": (
        r"\bheadphone(?:s)?\b",
        r"\bheadset(?:s)?\b",
        r"\bearbud(?:s)?\b",
        r"\bearphone(?:s)?\b",
    ),
    "keyboard": (
        r"\bkeyboard(?:s)?\b",
    ),
    "mouse": (
        r"\bcomputer mouse\b",
        r"\bgaming mouse\b",
        r"\bwireless mouse\b",
        r"\bbluetooth mouse\b",
        r"\bmouse\b",
    ),
    "storage": (
        r"\bsolid state drive(?:s)?\b",
        r"\bssd\b",
        r"\bnvme\b",
        r"\bhard drive(?:s)?\b",
        r"\bhard disk drive(?:s)?\b",
        r"\bflash drive(?:s)?\b",
        r"\bmemory card(?:s)?\b",
    ),
    "router": (
        r"\brouter(?:s)?\b",
        r"\bmesh wi[- ]?fi\b",
    ),
    "camera": (
        r"\bdigital camera(?:s)?\b",
        r"\bmirrorless camera(?:s)?\b",
        r"\bdslr\b",
        r"\baction camera(?:s)?\b",
        r"\bwebcam(?:s)?\b",
    ),
    "speaker": (
        r"\bspeaker(?:s)?\b",
    ),
    "smartwatch": (
        r"\bsmartwatch(?:es)?\b",
        r"\bsmart watch(?:es)?\b",
        r"\bapple watch\b",
    ),
    "printer": (
        r"\bprinter(?:s)?\b",
    ),
}


# ============================================================
# Compatibility-tail Split
# ============================================================
#
# title 中这些短语后面的内容通常是在描述兼容对象或使用对象，
# 不应该被拿来证明商品本体类别。
#
# Examples:
# USB Microphone for Laptop and PC
# -> USB Microphone
#
# Wireless Headphones Compatible with iPhone
# -> Wireless Headphones
#
# Dell Inspiron Laptop for Students
# -> Dell Inspiron Laptop
# ============================================================

COMPATIBILITY_SPLIT_PATTERNS = (
    r"\s+compatible\s+with\s+",
    r"\s+compatible\s+for\s+",
    r"\s+works\s+with\s+",
    r"\s+work\s+with\s+",
    r"\s+designed\s+for\s+",
    r"\s+made\s+for\s+",
    r"\s+fits\s+",
    r"\s+fit\s+for\s+",
    r"\s+for\s+",
)


TAG_RULES: list[
    tuple[tuple[str, ...], tuple[str, ...]]
] = [
    (
        ("wireless", "wi-fi", "wifi"),
        ("wireless", "无线"),
    ),
    (
        ("bluetooth",),
        ("bluetooth", "蓝牙"),
    ),
    (
        (
            "lightweight",
            "ultralight",
            "ultra-light",
            "thin and light",
            "ultra thin",
            "slim laptop",
        ),
        ("lightweight", "轻薄"),
    ),
    (
        ("gaming", "gamer"),
        ("gaming", "游戏"),
    ),
    (
        (
            "noise cancelling",
            "noise-cancelling",
            "noise canceling",
            "active noise cancellation",
            "active noise cancelling",
        ),
        ("noise_cancelling", "降噪"),
    ),
    (
        ("oled",),
        ("OLED",),
    ),
    (
        ("4k", "2160p"),
        ("4K",),
    ),
    (
        ("mechanical keyboard", "mechanical switch"),
        ("mechanical", "机械键盘"),
    ),
    (
        ("usb-c", "usb c", "type-c", "type c"),
        ("USB-C",),
    ),
    (
        ("solid state drive", "ssd", "nvme"),
        ("SSD", "固态硬盘"),
    ),
]


def _string_list(
    value: Any,
) -> list[str]:

    if value is None:
        return []

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    text = str(value).strip()

    if not text:
        return []

    return [text]


def _normalise_text(
    value: str,
) -> str:

    value = value.strip().lower()

    return re.sub(
        r"\s+",
        " ",
        value,
    )


def _flatten_taxonomy(
    value: Any,
) -> list[str]:
    """
    官方 metadata 的 categories 是 hierarchical categories。
    常见情况已经是 list[str]；
    若某个导出格式把 path 写在一个字符串里，也兼容 > / ›。
    """

    flattened: list[str] = []

    for item in _string_list(
        value
    ):
        parts = re.split(
            r"\s*(?:>|›)\s*",
            item,
        )

        for part in parts:
            normalised = _normalise_text(
                part
            )

            if normalised:
                flattened.append(
                    normalised
                )

    return flattened


def parse_details(
    value: Any,
) -> dict:

    if value is None:
        return {}

    if isinstance(
        value,
        dict,
    ):
        return value

    if isinstance(
        value,
        str,
    ):
        value = value.strip()

        if not value:
            return {}

        try:
            parsed = json.loads(
                value
            )
        except json.JSONDecodeError:
            return {}

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    return {}


def parse_price(
    value: Any,
) -> float | None:

    if value is None:
        return None

    try:
        price = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if (
        price <= 0
        or price > 5000
    ):
        return None

    return price


def parse_rating(
    value: Any,
) -> float | None:

    if value is None:
        return None

    try:
        rating = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not (
        0 <= rating <= 5
    ):
        return None

    return rating


def get_brand(
    raw: dict,
    details: dict,
) -> str | None:

    candidates = [
        details.get("Brand"),
        details.get("Manufacturer"),
        raw.get("store"),
    ]

    for value in candidates:
        if value is None:
            continue

        brand = str(value).strip()

        if (
            brand
            and brand.lower()
            not in {
                "none",
                "unknown",
                "n/a",
                "na",
            }
        ):
            return brand

    return None


def _matches_any_full_pattern(
    text_value: str,
    patterns: tuple[str, ...],
) -> bool:

    return any(
        re.fullmatch(
            pattern,
            text_value,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in patterns
    )


def _is_taxonomy_blocker(
    taxonomy_node: str,
) -> bool:

    return _matches_any_full_pattern(
        taxonomy_node,
        TAXONOMY_BLOCKER_PATTERNS,
    )


def _category_from_taxonomy_node(
    taxonomy_node: str,
) -> str | None:

    matched_categories = []

    for (
        category,
        patterns,
    ) in TAXONOMY_CATEGORY_PATTERNS.items():

        if _matches_any_full_pattern(
            taxonomy_node,
            patterns,
        ):
            matched_categories.append(
                category
            )

    if len(
        matched_categories
    ) != 1:
        return None

    return matched_categories[0]


def infer_taxonomy_category(
    raw: dict,
) -> str | None:
    """
    从 taxonomy leaf 向 root 扫描。

    - 先遇到强 product node -> 返回类别
    - 先遇到 accessory/part node -> abstain
    - 没有明确节点 -> abstain
    """

    taxonomy_path = _flatten_taxonomy(
        raw.get("categories")
    )

    if not taxonomy_path:
        return None

    for taxonomy_node in reversed(
        taxonomy_path
    ):

        category = _category_from_taxonomy_node(
            taxonomy_node
        )

        if category is not None:
            return category

        if _is_taxonomy_blocker(
            taxonomy_node
        ):
            return None

    return None


def extract_primary_title(
    title: str,
) -> str:
    """
    截断标题中的兼容对象尾部，只保留商品主体。

    例如：

    USB Microphone for Laptop and PC
    -> usb microphone

    Wireless Headphones Compatible with iPhone
    -> wireless headphones

    Dell Inspiron Laptop for Students
    -> dell inspiron laptop
    """

    title_lower = _normalise_text(
        title
    )

    split_positions: list[int] = []

    for pattern in COMPATIBILITY_SPLIT_PATTERNS:

        match = re.search(
            pattern,
            title_lower,
            flags=re.IGNORECASE,
        )

        if match is not None:

            split_positions.append(
                match.start()
            )

    if not split_positions:
        return title_lower

    earliest_split = min(
        split_positions
    )

    return (
        title_lower[
            :earliest_split
        ]
        .strip()
    )


def title_supports_category(
    title: str,
    category: str,
) -> bool:
    """
    title 只验证商品主体，不把兼容对象当成 product-type evidence。

    例如：
    - for laptop
    - compatible with iPhone
    - works with monitor

    这些尾部不会参与类别确认。
    """

    primary_title = extract_primary_title(
        title
    )

    patterns = TITLE_CORROBORATION_PATTERNS.get(
        category,
        (),
    )

    return any(
        re.search(
            pattern,
            primary_title,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in patterns
    )


def infer_category(
    raw: dict,
) -> str | None:
    """
    taxonomy 与 title 必须达成一致。

    任意一侧不能确认，就返回 None。
    """

    title = str(
        raw.get("title")
        or ""
    ).strip()

    if not title:
        return None

    taxonomy_category = infer_taxonomy_category(
        raw
    )

    if taxonomy_category is None:
        return None

    if not title_supports_category(
        title,
        taxonomy_category,
    ):
        return None

    return taxonomy_category


def build_search_text(
    raw: dict,
    details: dict,
) -> str:

    parts: list[str] = []

    parts.extend(
        _string_list(
            raw.get("title")
        )
    )

    parts.extend(
        _string_list(
            raw.get("main_category")
        )
    )

    parts.extend(
        _string_list(
            raw.get("categories")
        )
    )

    parts.extend(
        _string_list(
            raw.get("features")
        )
    )

    parts.extend(
        _string_list(
            raw.get("description")
        )
    )

    detail_keys = (
        "Brand",
        "Model Name",
        "Model Number",
        "Color",
        "Screen Size",
        "RAM",
        "Memory Storage Capacity",
        "Hard Disk Size",
        "Connectivity Technology",
        "Special Feature",
        "Special Features",
    )

    for key in detail_keys:
        value = details.get(
            key
        )

        if value:
            parts.append(
                f"{key}: {value}"
            )

    return " ".join(
        parts
    )


def build_tags(
    category: str,
    search_text: str,
) -> list[str]:

    lowered = search_text.lower()

    tags: list[str] = [
        category
    ]

    for (
        keywords,
        canonical_tags,
    ) in TAG_RULES:

        if any(
            keyword in lowered
            for keyword in keywords
        ):
            tags.extend(
                canonical_tags
            )

    capacity_matches = re.findall(
        (
            r"\b("
            r"4|8|12|16|24|32|48|64|96|128|256|512"
            r")\s*gb\b"
        ),
        lowered,
    )

    for value in capacity_matches[:4]:
        tags.append(
            f"{value}GB"
        )

    tb_matches = re.findall(
        r"\b(1|2|4|8)\s*tb\b",
        lowered,
    )

    for value in tb_matches[:2]:
        tags.append(
            f"{value}TB"
        )

    return list(
        dict.fromkeys(
            tag
            for tag in tags
            if tag
        )
    )[:12]


def build_description(
    raw: dict,
    details: dict,
) -> str:

    parts: list[str] = []

    parts.extend(
        _string_list(
            raw.get("description")
        )
    )

    parts.extend(
        _string_list(
            raw.get("features")
        )[:8]
    )

    detail_keys = (
        "Model Name",
        "Model Number",
        "Color",
        "Screen Size",
        "RAM",
        "Memory Storage Capacity",
        "Hard Disk Size",
        "Connectivity Technology",
        "Special Feature",
        "Special Features",
    )

    for key in detail_keys:
        value = details.get(
            key
        )

        if value:
            parts.append(
                f"{key}: {value}"
            )

    text = ". ".join(
        part.strip()
        for part in parts
        if part.strip()
    )

    return text[:1800]


def adapt_amazon_product(
    raw: dict,
    *,
    min_rating_count: int = 20,
) -> Product | None:
    """
    Amazon Reviews 2023 metadata -> Product。

    严格条件：
    1. parent_asin/title 有效
    2. taxonomy + title product type 一致
    3. price/rating 有效
    4. rating_number 达阈值
    5. brand 可解析

    低置信商品直接丢弃。
    """

    parent_asin = str(
        raw.get("parent_asin")
        or ""
    ).strip()

    title = str(
        raw.get("title")
        or ""
    ).strip()

    if (
        not parent_asin
        or not title
    ):
        return None

    category = infer_category(
        raw
    )

    if category is None:
        return None

    price = parse_price(
        raw.get("price")
    )

    rating = parse_rating(
        raw.get("average_rating")
    )

    if (
        price is None
        or rating is None
    ):
        return None

    try:
        rating_count = int(
            raw.get("rating_number")
            or 0
        )
    except (
        TypeError,
        ValueError,
    ):
        rating_count = 0

    if (
        rating_count
        < min_rating_count
    ):
        return None

    details = parse_details(
        raw.get("details")
    )

    brand = get_brand(
        raw,
        details,
    )

    if brand is None:
        return None

    search_text = build_search_text(
        raw,
        details,
    )

    tags = build_tags(
        category,
        search_text,
    )

    description = (
        build_description(
            raw,
            details,
        )
        or title
    )

    return Product(
        id=parent_asin,
        name=title,
        category=category,
        brand=brand,
        price=price,
        currency="USD",
        rating=rating,
        tags=tags,
        description=description,
    )
