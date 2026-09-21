from app.services.product_search import (
    load_products,
    search_products,
)


def main():
    products = load_products()

    results = search_products(
        products=products,
        max_price=10000,
        min_rating=4.5,
        tags=["编程", "轻薄"],
        top_k=3,
    )

    print("\n搜索条件")
    print("-" * 40)
    print("最高预算: 10000")
    print("最低评分: 4.5")
    print("需求标签: 编程、轻薄")

    print("\n搜索结果")
    print("=" * 40)

    for index, result in enumerate(
        results,
        start=1,
    ):
        product = result.product

        print(f"\n{index}. {product.name}")
        print(f"   品牌: {product.brand}")
        print(
            f"   价格: "
            f"{product.price} {product.currency}"
        )
        print(f"   评分: {product.rating}")
        print(f"   匹配标签: {result.matched_tags}")
        print(f"   搜索得分: {result.score:.2f}")


if __name__ == "__main__":
    main()