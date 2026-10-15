from app.services.hybrid_search import (
    bm25_search_products,
    hybrid_search_products,
)
from app.services.semantic_search import (
    semantic_search_products,
)


def print_separator(
    title: str,
) -> None:

    print(
        "\n"
        + "=" * 80
    )

    print(title)

    print(
        "=" * 80
    )


def run_query(
    query: str,
) -> None:

    print_separator(
        f"Query: {query}"
    )

    # -----------------------------------
    # BM25
    # -----------------------------------

    print("\n[BM25]")

    bm25_results = (
        bm25_search_products(
            query=query,
            top_k=5,
        )
    )

    for rank, (
        product,
        score,
    ) in enumerate(
        bm25_results,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{product.name} | "
            f"score={score:.4f}"
        )

    # -----------------------------------
    # Dense
    # -----------------------------------

    print("\n[Dense]")

    dense_results = (
        semantic_search_products(
            query=query,
            top_k=5,
        )
    )

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{result.product.name} | "
            f"score="
            f"{result.semantic_score:.4f}"
        )

    # -----------------------------------
    # Hybrid
    # -----------------------------------

    print("\n[Hybrid / RRF]")

    hybrid_results = (
        hybrid_search_products(
            query=query,
            top_k=5,
        )
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

        print(
            f"   bm25_rank="
            f"{result.bm25_rank}"
        )

        print(
            f"   semantic_rank="
            f"{result.semantic_rank}"
        )


def main():

    queries = [
        "ThinkPad X1",
        "ROG G14",
        (
            "我经常出差，"
            "主要写代码，"
            "希望轻便"
        ),
        (
            "偶尔跑AI小模型，"
            "想要性能强一点"
        ),
    ]

    for query in queries:
        run_query(query)


if __name__ == "__main__":
    main()