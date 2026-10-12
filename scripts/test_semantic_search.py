from app.services.semantic_search import (
    semantic_search_products,
)


def run_query(
    query: str,
) -> None:

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Query: {query}"
    )

    print(
        "=" * 70
    )

    results = (
        semantic_search_products(
            query=query,
            top_k=5,
        )
    )

    for index, result in enumerate(
        results,
        start=1,
    ):
        product = result.product

        print(
            f"\n{index}. "
            f"{product.name}"
        )

        print(
            f"   品牌: "
            f"{product.brand}"
        )

        print(
            f"   价格: "
            f"{product.price} "
            f"{product.currency}"
        )

        print(
            f"   标签: "
            f"{product.tags}"
        )

        print(
            f"   Semantic Score: "
            f"{result.semantic_score:.4f}"
        )


def main():

    queries = [
        (
            "我经常带着电脑出去，"
            "主要用来写代码，"
            "希望不要太重"
        ),
        (
            "我想偶尔跑AI模型，"
            "需要比较强的计算性能"
        ),
        (
            "主要用于商务办公和出差"
        ),
    ]

    for query in queries:
        run_query(query)


if __name__ == "__main__":
    main()