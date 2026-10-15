from app.services.hybrid_search import (
    hybrid_search_products,
)
from app.services.reranker import (
    rerank_products,
)


def main():

    query = (
        "经常出差，主要写代码，"
        "希望电脑轻薄并且续航好"
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"Query: {query}"
    )

    print(
        "=" * 80
    )

    hybrid_results = (
        hybrid_search_products(
            query=query,
            top_k=10,
            candidate_k=10,
        )
    )

    print(
        "\n[Before Rerank]"
    )

    for rank, result in enumerate(
        hybrid_results,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{result.product.name}"
        )

        print(
            f"   hybrid="
            f"{result.hybrid_score:.6f}"
        )

    reranked_results = (
        rerank_products(
            query=query,
            candidates=hybrid_results,
            top_k=5,
        )
    )

    print(
        "\n[After Cross-Encoder]"
    )

    for rank, result in enumerate(
        reranked_results,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{result.product.name}"
        )

        print(
            f"   rerank="
            f"{result.rerank_score:.6f}"
        )

        print(
            f"   hybrid="
            f"{result.hybrid_score:.6f}"
        )


if __name__ == "__main__":
    main()