import json

from app.models.product import Product


def load_products() -> list[Product]:
    with open(
        "data/products.json",
        "r",
        encoding="utf-8",
    ) as file:
        raw_products = json.load(file)  #JSON array → Python list ； JSON object → Python dict

    products = [
        Product(**item)
        for item in raw_products
    ]

    return products

def main():
    products = load_products()

    for product in products:
        print("=" * 40)
        print(f"商品名称：{product.name}")
        print(f"品牌：{product.brand}")
        print(f"价格: {product.price} {product.currency}")
        print(f"评分: {product.rating}")
        print(f"标签: {product.tags}")
        print(f"描述: {product.description}")


if __name__ == "__main__":
    main()