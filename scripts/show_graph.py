from app.agents.shopping_agent import (
    shopping_graph,
)


def main():
    mermaid = (
        shopping_graph
        .get_graph()
        .draw_mermaid()
    )

    print(mermaid)


if __name__ == "__main__":
    main()