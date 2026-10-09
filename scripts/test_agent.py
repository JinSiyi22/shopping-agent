from app.agents.shopping_agent import (
    run_agent,
)


def main():

    query = (
        #"帮我推荐一台10000元以内适合编程的电脑，"
        #"我要寄到日本。"
        #"请把商品价格、模拟运费和总费用一起告诉我。"
        "帮我比较 MacBook Air 和 ABC Super Laptop。"
    )

    print("\n用户需求：")
    print(query)

    answer = run_agent(
        user_query=query,
        max_steps=8,
    )

    print("\n" + "=" * 60)

    print("\nAgent最终回答：")
    print(answer)


if __name__ == "__main__":
    main()