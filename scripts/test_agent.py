from app.agents.shopping_agent import run_agent


def main():

    query = (
        "我预算一万以内，主要用于编程，"
        "平时经常带电脑出去，希望轻一点。"
        "偶尔还想跑点机器学习任务，"
        "你给我推荐几台合适的电脑。"
    )

    print("\n用户需求：")
    print(query)

    print("\n" + "=" * 60)

    answer = run_agent(query)

    print("\nAgent 最终回答：")
    print(answer)


if __name__ == "__main__":
    main()