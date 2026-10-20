import asyncio
import json
import os
from pathlib import Path
from time import perf_counter

import httpx

from app.evaluation.metrics import (
    accuracy,
    mean,
    percentile,
    recall_at_k,
    reciprocal_rank,
)


BASE_URL = os.getenv(
    "EVAL_BASE_URL",
    "http://127.0.0.1:8000",
)


EVAL_FILE = Path(
    "evals/shopping_eval.json"
)


REPORT_FILE = Path(
    "reports/eval_latest.json"
)


def extract_product_ids(
    result,
) -> list[str]:

    if result is None:
        return []

    # MCP返回有时可能存在一层result wrapper
    if (
        isinstance(result, dict)
        and set(result.keys())
        == {"result"}
    ):

        return extract_product_ids(
            result["result"]
        )

    # Search / Rerank
    if isinstance(
        result,
        list,
    ):

        product_ids = []

        for item in result:

            if not isinstance(
                item,
                dict,
            ):
                continue

            product_id = (
                item.get("id")
            )

            if product_id:

                product_ids.append(
                    str(
                        product_id
                    )
                )

        return product_ids

    if isinstance(
        result,
        dict,
    ):

        # Compare
        products = (
            result.get(
                "products"
            )
            or
            result.get(
                "selected"
            )
        )

        if isinstance(
            products,
            list,
        ):

            return (
                extract_product_ids(
                    products
                )
            )

        # Detail
        product_id = (
            result.get("id")
        )

        if product_id:

            return [
                str(
                    product_id
                )
            ]

    return []


def load_eval_cases():

    with EVAL_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


async def evaluate_turn(
    client: httpx.AsyncClient,
    *,
    scenario_id: str,
    turn_index: int,
    turn: dict,
    session_id: str | None,
) -> tuple[
    dict,
    str,
]:

    request_payload = {
        "message": (
            turn["message"]
        )
    }

    if session_id is not None:

        request_payload[
            "session_id"
        ] = session_id

    start = (
        perf_counter()
    )

    response = await client.post(
        "/chat",
        json=request_payload,
    )

    latency_ms = (
        perf_counter()
        - start
    ) * 1000

    response.raise_for_status()

    chat_data = (
        response.json()
    )

    session_id = (
        chat_data[
            "session_id"
        ]
    )

    state_response = await client.get(
        (
            f"/sessions/"
            f"{session_id}"
            f"/state"
        )
    )

    state_response.raise_for_status()

    state = (
        state_response.json()
    )

    actual_route = (
        state.get(
            "route"
        )
    )

    actual_tool = (
        state.get(
            "research_tool"
        )
    )

    research_result = (
        state.get(
            "research_result"
        )
    )

    retrieved_ids = (
        extract_product_ids(
            research_result
        )
    )

    expected_route = (
        turn.get(
            "expected_route"
        )
    )

    route_correct = (
        actual_route
        == expected_route
    )

    # expected_tool字段不存在时，
    # 当前Turn不参加Tool Accuracy。
    tool_applicable = (
        "expected_tool"
        in turn
    )

    expected_tool = (
        turn.get(
            "expected_tool"
        )
    )

    tool_correct = None

    if tool_applicable:

        tool_correct = (
            actual_tool
            == expected_tool
        )

    relevant_ids = (
        turn.get(
            "relevant_product_ids",
            [],
        )
    )

    retrieval_applicable = bool(
        relevant_ids
    )

    eval_k = int(
        turn.get(
            "eval_k",
            5,
        )
    )

    recall_score = None
    rr_score = None

    retrieval_success = True

    if retrieval_applicable:

        recall_score = (
            recall_at_k(
                retrieved_ids=(
                    retrieved_ids
                ),
                relevant_ids=(
                    relevant_ids
                ),
                k=eval_k,
            )
        )

        rr_score = (
            reciprocal_rank(
                retrieved_ids=(
                    retrieved_ids
                ),
                relevant_ids=(
                    relevant_ids
                ),
            )
        )

        min_recall = float(
            turn.get(
                "min_recall",
                1.0,
            )
        )

        retrieval_success = (
            recall_score
            >= min_recall
        )

    task_success = (
        route_correct
        and (
            tool_correct
            if tool_applicable
            else True
        )
        and retrieval_success
    )

    result = {
        "scenario_id": (
            scenario_id
        ),

        "turn_index": (
            turn_index
        ),

        "message": (
            turn["message"]
        ),

        "session_id": (
            session_id
        ),

        "expected_route": (
            expected_route
        ),

        "actual_route": (
            actual_route
        ),

        "route_correct": (
            route_correct
        ),

        "expected_tool": (
            expected_tool
        ),

        "actual_tool": (
            actual_tool
        ),

        "tool_applicable": (
            tool_applicable
        ),

        "tool_correct": (
            tool_correct
        ),

        "relevant_product_ids": (
            relevant_ids
        ),

        "retrieved_product_ids": (
            retrieved_ids
        ),

        "recall_at_k": (
            recall_score
        ),

        "reciprocal_rank": (
            rr_score
        ),

        "task_success": (
            task_success
        ),

        "latency_ms": (
            latency_ms
        ),

        "answer": (
            chat_data.get(
                "answer",
                "",
            )
        ),
    }

    return (
        result,
        session_id,
    )


async def main():

    cases = (
        load_eval_cases()
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=120.0,
    ) as client:

        health_response = await client.get(
            "/health"
        )

        health_response.raise_for_status()

        print(
            "Backend health: OK"
        )

        for scenario in cases:

            scenario_id = (
                scenario["id"]
            )

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"Scenario: "
                f"{scenario_id}"
            )

            print(
                "=" * 70
            )

            session_id = None

            for turn_index, turn in enumerate(
                scenario["turns"],
                start=1,
            ):

                print(
                    f"\nTurn "
                    f"{turn_index}: "
                    f"{turn['message']}"
                )

                try:

                    (
                        turn_result,
                        session_id,
                    ) = await evaluate_turn(
                        client,
                        scenario_id=(
                            scenario_id
                        ),
                        turn_index=(
                            turn_index
                        ),
                        turn=turn,
                        session_id=(
                            session_id
                        ),
                    )

                except Exception as exc:

                    print(
                        f"ERROR: {exc}"
                    )

                    results.append(
                        {
                            "scenario_id": (
                                scenario_id
                            ),
                            "turn_index": (
                                turn_index
                            ),
                            "error": (
                                str(exc)
                            ),
                            "task_success": False,
                        }
                    )

                    continue

                results.append(
                    turn_result
                )

                print(
                    "route: "
                    f"{turn_result['actual_route']} "
                    f"(expected="
                    f"{turn_result['expected_route']})"
                )

                if (
                    turn_result[
                        "tool_applicable"
                    ]
                ):

                    print(
                        "tool: "
                        f"{turn_result['actual_tool']} "
                        f"(expected="
                        f"{turn_result['expected_tool']})"
                    )

                if (
                    turn_result[
                        "recall_at_k"
                    ]
                    is not None
                ):

                    print(
                        "Recall@K: "
                        f"{turn_result['recall_at_k']:.3f}"
                    )

                    print(
                        "MRR: "
                        f"{turn_result['reciprocal_rank']:.3f}"
                    )

                print(
                    "Task Success: "
                    f"{turn_result['task_success']}"
                )

                print(
                    "Latency: "
                    f"{turn_result['latency_ms']:.2f} ms"
                )

    valid_results = [
        item
        for item in results
        if "error" not in item
    ]

    route_values = [
        item[
            "route_correct"
        ]
        for item in valid_results
    ]

    tool_values = [
        item[
            "tool_correct"
        ]
        for item in valid_results
        if (
            item.get(
                "tool_applicable"
            )
            and
            item.get(
                "tool_correct"
            )
            is not None
        )
    ]

    recall_values = [
        item[
            "recall_at_k"
        ]
        for item in valid_results
        if (
            item.get(
                "recall_at_k"
            )
            is not None
        )
    ]

    rr_values = [
        item[
            "reciprocal_rank"
        ]
        for item in valid_results
        if (
            item.get(
                "reciprocal_rank"
            )
            is not None
        )
    ]

    task_values = [
        item[
            "task_success"
        ]
        for item in results
    ]

    latencies = [
        item[
            "latency_ms"
        ]
        for item in valid_results
        if (
            "latency_ms"
            in item
        )
    ]

    summary = {
        "total_turns": (
            len(results)
        ),

        "routing_accuracy": (
            accuracy(
                route_values
            )
        ),

        "tool_selection_accuracy": (
            accuracy(
                tool_values
            )
        ),

        "mean_recall_at_k": (
            mean(
                recall_values
            )
        ),

        "mrr": (
            mean(
                rr_values
            )
        ),

        "task_success_rate": (
            accuracy(
                task_values
            )
        ),

        "average_latency_ms": (
            mean(
                latencies
            )
        ),

        "p95_latency_ms": (
            percentile(
                latencies,
                0.95,
            )
        ),
    }

    report = {
        "summary": summary,
        "results": results,
    }

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EVALUATION SUMMARY"
    )

    print(
        "=" * 70
    )

    for key, value in (
        summary.items()
    ):

        if isinstance(
            value,
            float,
        ):

            print(
                f"{key}: "
                f"{value:.4f}"
            )

        else:

            print(
                f"{key}: "
                f"{value}"
            )

    print(
        "\nReport saved to:"
    )

    print(
        REPORT_FILE
    )


if __name__ == "__main__":

    asyncio.run(
        main()
    )