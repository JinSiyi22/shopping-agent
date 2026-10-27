from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import httpx

from app.evaluation.real_agent_eval import (
    accuracy,
    build_unique_alias_index,
    constraint_pass,
    extract_product_ids,
    grounding_metrics,
    hit_at_k,
    mean_optional,
    percentile,
    recall_at_k,
    semantic_tag_hit_at_k,
)


BALANCED_SMOKE_TYPES = (
    "structured_search",
    "semantic_recommendation",
    "product_detail",
    "product_compare",
    "multi_turn_reuse",
    "multi_turn_new_constraint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval-file",
        default=(
            "evals/"
            "shopping_real_eval.json"
        ),
    )

    parser.add_argument(
        "--catalog",
        default="data/products_real.json",
    )

    parser.add_argument(
        "--output",
        default=(
            "reports/"
            "agent_eval_real.json"
        ),
    )

    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "EVAL_BASE_URL",
            "http://127.0.0.1:8000",
        ),
    )

    parser.add_argument(
        "--limit-scenarios",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--balanced-smoke",
        action="store_true",
        help=(
            "Run one scenario from each "
            "Day-20 scenario type."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )

    parser.add_argument(
        "--skip-warmup",
        action="store_true",
    )

    return parser.parse_args()


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def select_balanced_smoke(
    scenarios: list[dict],
) -> list[dict]:
    selected = []

    for scenario_type in BALANCED_SMOKE_TYPES:
        match = next(
            (
                scenario
                for scenario in scenarios
                if scenario.get("type")
                == scenario_type
            ),
            None,
        )

        if match is None:
            raise RuntimeError(
                "Missing balanced-smoke scenario type: "
                f"{scenario_type}"
            )

        selected.append(match)

    return selected


async def warmup(
    client: httpx.AsyncClient,
) -> None:
    print(
        "Warmup: starting"
    )

    response = await client.post(
        "/chat",
        json={
            "message": (
                "我想买一副 headphone（耳机），"
                "希望支持蓝牙连接，"
                "请推荐几款。"
            )
        },
    )

    response.raise_for_status()

    session_id = (
        response.json()[
            "session_id"
        ]
    )

    delete_response = await client.delete(
        f"/sessions/{session_id}"
    )

    if (
        delete_response.status_code
        not in {
            200,
            404,
        }
    ):
        delete_response.raise_for_status()

    print(
        "Warmup: finished"
    )


async def evaluate_turn(
    client: httpx.AsyncClient,
    *,
    scenario_id: str,
    scenario_type: str,
    turn_index: int,
    turn: dict,
    session_id: str | None,
    product_map: dict[str, dict],
    alias_index: dict[str, str],
    evidence_ids: set[str],
) -> tuple[
    dict,
    str,
    set[str],
]:
    request_payload = {
        "message": (
            turn["message"]
        )
    }

    if session_id is not None:
        request_payload[
            "session_id"
        ] = (
            session_id
        )

    start = perf_counter()

    response = await client.post(
        "/chat",
        json=request_payload,
    )

    latency_ms = (
        perf_counter()
        - start
    ) * 1000

    response.raise_for_status()

    chat_data = response.json()

    session_id = str(
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

    state = state_response.json()

    actual_route = state.get(
        "route"
    )

    actual_tool = state.get(
        "research_tool"
    )

    research_result = state.get(
        "research_result"
    )

    retrieved_ids = (
        extract_product_ids(
            research_result
        )
    )

    new_evidence_ids = set(
        evidence_ids
    )

    new_evidence_ids.update(
        retrieved_ids
    )

    expected_route = turn.get(
        "expected_route"
    )

    route_correct = (
        actual_route
        == expected_route
    )

    tool_applicable = (
        "expected_tool"
        in turn
    )

    expected_tool = turn.get(
        "expected_tool"
    )

    tool_correct = None

    if tool_applicable:
        tool_correct = (
            actual_tool
            == expected_tool
        )

    hard_constraints = turn.get(
        "hard_constraints",
        {},
    )

    min_results = int(
        turn.get(
            "min_results",
            0,
        )
    )

    constraints_applicable = bool(
        hard_constraints
    )

    constraints_ok = None

    if constraints_applicable:
        constraints_ok = (
            constraint_pass(
                retrieved_ids,
                product_map,
                hard_constraints,
                min_results=(
                    min_results
                ),
            )
        )

    relevant_ids = [
        str(value)
        for value
        in turn.get(
            "relevant_product_ids",
            [],
        )
    ]

    eval_k = int(
        turn.get(
            "eval_k",
            5,
        )
    )

    retrieval_recall = (
        recall_at_k(
            retrieved_ids,
            relevant_ids,
            eval_k,
        )
    )

    retrieval_hit = (
        hit_at_k(
            retrieved_ids,
            relevant_ids,
            eval_k,
        )
    )

    semantic_tags = [
        str(value)
        for value
        in turn.get(
            "semantic_tags",
            [],
        )
    ]

    semantic_hit = (
        semantic_tag_hit_at_k(
            retrieved_ids,
            product_map,
            semantic_tags,
            eval_k,
        )
    )

    expected_product_ids = [
        str(value)
        for value
        in turn.get(
            "expected_product_ids",
            [],
        )
    ]

    expected_product_coverage = None

    if expected_product_ids:
        expected_set = set(
            expected_product_ids
        )

        actual_set = set(
            retrieved_ids
        )

        expected_product_coverage = (
            len(
                expected_set
                & actual_set
            )
            / len(
                expected_set
            )
        )

    answer = str(
        chat_data.get(
            "answer",
            "",
        )
    )

    grounding = grounding_metrics(
        answer,
        alias_index,
        new_evidence_ids,
    )

    mentioned_ids = set(
        grounding[
            "mentioned_product_ids"
        ]
    )

    answer_expected_product_coverage = None

    if expected_product_ids:
        expected_set = set(
            expected_product_ids
        )

        answer_expected_product_coverage = (
            len(
                expected_set
                & mentioned_ids
            )
            / len(
                expected_set
            )
        )

    success_parts = [
        route_correct,
    ]

    if tool_applicable:
        success_parts.append(
            bool(
                tool_correct
            )
        )

    if constraints_ok is not None:
        success_parts.append(
            bool(
                constraints_ok
            )
        )

    if semantic_hit is not None:
        success_parts.append(
            semantic_hit >= 1.0
        )

    if expected_product_coverage is not None:
        success_parts.append(
            expected_product_coverage
            >= 1.0
        )

    if retrieval_hit is not None:
        success_parts.append(
            retrieval_hit >= 1.0
        )

    if (
        actual_route == "research"
        and retrieved_ids
    ):
        success_parts.append(
            grounding[
                "evidence_product_mentioned"
            ]
            == 1.0
        )

    if (
        answer_expected_product_coverage
        is not None
    ):
        success_parts.append(
            answer_expected_product_coverage
            >= 1.0
        )

    if (
        grounding[
            "catalog_product_grounding_pass"
        ]
        is not None
    ):
        success_parts.append(
            bool(
                grounding[
                    "catalog_product_grounding_pass"
                ]
            )
        )

    task_success = all(
        success_parts
    )

    result = {
        "scenario_id": scenario_id,
        "scenario_type": scenario_type,
        "turn_index": turn_index,
        "message": turn["message"],
        "session_id": session_id,
        "expected_route": expected_route,
        "actual_route": actual_route,
        "route_correct": (
            route_correct
        ),
        "expected_tool": expected_tool,
        "actual_tool": actual_tool,
        "tool_applicable": (
            tool_applicable
        ),
        "tool_correct": (
            tool_correct
        ),
        "hard_constraints": (
            hard_constraints
        ),
        "constraints_applicable": (
            constraints_applicable
        ),
        "constraint_pass": (
            constraints_ok
        ),
        "semantic_tags": (
            semantic_tags
        ),
        "semantic_hit_at_k": (
            semantic_hit
        ),
        "relevant_product_ids": (
            relevant_ids
        ),
        "retrieved_product_ids": (
            retrieved_ids
        ),
        "retrieval_hit_at_k": (
            retrieval_hit
        ),
        "retrieval_recall_at_k": (
            retrieval_recall
        ),
        "expected_product_ids": (
            expected_product_ids
        ),
        "expected_product_coverage": (
            expected_product_coverage
        ),
        "answer_expected_product_coverage": (
            answer_expected_product_coverage
        ),
        **grounding,
        "task_success": (
            task_success
        ),
        "latency_ms": (
            latency_ms
        ),
        "answer": (
            answer
        ),
    }

    return (
        result,
        session_id,
        new_evidence_ids,
    )


def summarize(
    results: list[dict],
) -> dict:
    valid = [
        item
        for item in results
        if "error" not in item
    ]

    route_values = [
        bool(
            item[
                "route_correct"
            ]
        )
        for item in valid
    ]

    tool_values = [
        bool(
            item[
                "tool_correct"
            ]
        )
        for item in valid
        if (
            item.get(
                "tool_applicable"
            )
            and item.get(
                "tool_correct"
            )
            is not None
        )
    ]

    constraint_values = [
        bool(
            item[
                "constraint_pass"
            ]
        )
        for item in valid
        if (
            item.get(
                "constraint_pass"
            )
            is not None
        )
    ]

    semantic_values = [
        item[
            "semantic_hit_at_k"
        ]
        for item in valid
        if (
            item.get(
                "semantic_hit_at_k"
            )
            is not None
        )
    ]

    expected_coverage_values = [
        item[
            "expected_product_coverage"
        ]
        for item in valid
        if (
            item.get(
                "expected_product_coverage"
            )
            is not None
        )
    ]

    answer_expected_values = [
        item[
            "answer_expected_product_coverage"
        ]
        for item in valid
        if (
            item.get(
                "answer_expected_product_coverage"
            )
            is not None
        )
    ]

    grounding_pass_values = [
        bool(
            item[
                "catalog_product_grounding_pass"
            ]
        )
        for item in valid
        if (
            item.get(
                "catalog_product_grounding_pass"
            )
            is not None
        )
    ]

    grounding_rate_values = [
        item[
            "catalog_product_grounding_rate"
        ]
        for item in valid
        if (
            item.get(
                "catalog_product_grounding_rate"
            )
            is not None
        )
    ]

    evidence_mention_values = [
        item[
            "evidence_product_mentioned"
        ]
        for item in valid
        if (
            item.get(
                "evidence_product_mentioned"
            )
            is not None
        )
    ]

    task_values = [
        bool(
            item.get(
                "task_success",
                False,
            )
        )
        for item in results
    ]

    latencies = [
        float(
            item[
                "latency_ms"
            ]
        )
        for item in valid
        if "latency_ms" in item
    ]

    return {
        "total_turns": (
            len(
                results
            )
        ),
        "error_turns": (
            len(
                results
            )
            - len(
                valid
            )
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
        "hard_constraint_pass_rate": (
            accuracy(
                constraint_values
            )
        ),
        "semantic_hit_rate_at_k": (
            mean_optional(
                semantic_values
            )
        ),
        "expected_product_coverage": (
            mean_optional(
                expected_coverage_values
            )
        ),
        "answer_expected_product_coverage": (
            mean_optional(
                answer_expected_values
            )
        ),
        "catalog_product_grounding_pass_rate": (
            accuracy(
                grounding_pass_values
            )
            if grounding_pass_values
            else None
        ),
        "mean_catalog_product_grounding_rate": (
            mean_optional(
                grounding_rate_values
            )
        ),
        "evidence_product_mention_rate": (
            mean_optional(
                evidence_mention_values
            )
        ),
        "task_success_rate": (
            accuracy(
                task_values
            )
        ),
        "average_latency_ms": (
            mean_optional(
                latencies
            )
        ),
        "p50_latency_ms": (
            percentile(
                latencies,
                0.50,
            )
        ),
        "p95_latency_ms": (
            percentile(
                latencies,
                0.95,
            )
        ),
    }


def summarize_by_type(
    results: list[dict],
) -> dict:
    groups = defaultdict(
        list
    )

    for result in results:
        groups[
            result.get(
                "scenario_type",
                "unknown",
            )
        ].append(
            result
        )

    return {
        scenario_type: (
            summarize(
                items
            )
        )
        for scenario_type, items
        in sorted(
            groups.items()
        )
    }


async def main() -> None:
    args = parse_args()

    if (
        args.balanced_smoke
        and args.limit_scenarios > 0
    ):
        raise ValueError(
            "Use either --balanced-smoke "
            "or --limit-scenarios, not both."
        )

    eval_payload = load_json(
        Path(
            args.eval_file
        )
    )

    products = load_json(
        Path(
            args.catalog
        )
    )

    if isinstance(
        eval_payload,
        dict,
    ):
        scenarios = list(
            eval_payload[
                "scenarios"
            ]
        )
        metadata = (
            eval_payload.get(
                "metadata",
                {},
            )
        )
    else:
        scenarios = list(
            eval_payload
        )
        metadata = {}

    if args.balanced_smoke:
        scenarios = select_balanced_smoke(
            scenarios
        )
    elif args.limit_scenarios > 0:
        scenarios = scenarios[
            :args.limit_scenarios
        ]

    product_map = {
        str(
            product["id"]
        ): product
        for product in products
    }

    alias_index = build_unique_alias_index(
        products
    )

    print(
        "Grounding aliases:",
        len(
            alias_index
        ),
    )

    results = []
    scenario_results = []

    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=args.timeout,
    ) as client:
        health = await client.get(
            "/health"
        )

        health.raise_for_status()

        print(
            "Backend health: OK"
        )

        if not args.skip_warmup:
            await warmup(
                client
            )

        for scenario_index, scenario in enumerate(
            scenarios,
            start=1,
        ):
            scenario_id = str(
                scenario[
                    "id"
                ]
            )

            scenario_type = str(
                scenario.get(
                    "type",
                    "unknown",
                )
            )

            print()
            print("=" * 72)
            print(
                f"[{scenario_index}/"
                f"{len(scenarios)}] "
                f"{scenario_id} "
                f"({scenario_type})"
            )
            print("=" * 72)

            session_id = None
            evidence_ids: set[str] = set()
            scenario_turns = []

            for turn_index, turn in enumerate(
                scenario[
                    "turns"
                ],
                start=1,
            ):
                print()
                print(
                    f"Turn {turn_index}: "
                    f"{turn['message']}"
                )

                try:
                    (
                        result,
                        session_id,
                        evidence_ids,
                    ) = await evaluate_turn(
                        client,
                        scenario_id=scenario_id,
                        scenario_type=(
                            scenario_type
                        ),
                        turn_index=(
                            turn_index
                        ),
                        turn=turn,
                        session_id=(
                            session_id
                        ),
                        product_map=(
                            product_map
                        ),
                        alias_index=(
                            alias_index
                        ),
                        evidence_ids=(
                            evidence_ids
                        ),
                    )

                except Exception as exc:
                    result = {
                        "scenario_id": (
                            scenario_id
                        ),
                        "scenario_type": (
                            scenario_type
                        ),
                        "turn_index": (
                            turn_index
                        ),
                        "message": (
                            turn[
                                "message"
                            ]
                        ),
                        "error": str(
                            exc
                        ),
                        "task_success": (
                            False
                        ),
                    }

                    print(
                        "ERROR:",
                        exc,
                    )

                results.append(
                    result
                )

                scenario_turns.append(
                    result
                )

                if "error" not in result:
                    print(
                        "route:",
                        result[
                            "actual_route"
                        ],
                        "expected=",
                        result[
                            "expected_route"
                        ],
                    )

                    if result[
                        "tool_applicable"
                    ]:
                        print(
                            "tool:",
                            result[
                                "actual_tool"
                            ],
                            "expected=",
                            result[
                                "expected_tool"
                            ],
                        )

                    if (
                        result[
                            "constraint_pass"
                        ]
                        is not None
                    ):
                        print(
                            "constraints:",
                            result[
                                "constraint_pass"
                            ],
                        )

                    if (
                        result[
                            "semantic_hit_at_k"
                        ]
                        is not None
                    ):
                        print(
                            "semantic_hit:",
                            result[
                                "semantic_hit_at_k"
                            ],
                        )

                    print(
                        "evidence_mentioned:",
                        result[
                            "evidence_product_mentioned"
                        ],
                    )

                    print(
                        "grounding:",
                        result[
                            "catalog_product_grounding_pass"
                        ],
                    )

                    print(
                        "task_success:",
                        result[
                            "task_success"
                        ],
                    )

                    print(
                        "latency_ms:",
                        round(
                            result[
                                "latency_ms"
                            ],
                            1,
                        ),
                    )

            scenario_success = all(
                bool(
                    turn_result.get(
                        "task_success",
                        False,
                    )
                )
                for turn_result
                in scenario_turns
            )

            scenario_results.append(
                {
                    "scenario_id": (
                        scenario_id
                    ),
                    "scenario_type": (
                        scenario_type
                    ),
                    "success": (
                        scenario_success
                    ),
                    "turns": (
                        len(
                            scenario_turns
                        )
                    ),
                }
            )

            if session_id is not None:
                delete_response = (
                    await client.delete(
                        (
                            f"/sessions/"
                            f"{session_id}"
                        )
                    )
                )

                if (
                    delete_response.status_code
                    not in {
                        200,
                        404,
                    }
                ):
                    delete_response.raise_for_status()

    summary = summarize(
        results
    )

    summary[
        "scenario_success_rate"
    ] = accuracy(
        [
            bool(
                item[
                    "success"
                ]
            )
            for item
            in scenario_results
        ]
    )

    report = {
        "evaluation": (
            "Day 20 Real Catalog "
            "Agent Evaluation"
        ),
        "eval_metadata": (
            metadata
        ),
        "run_config": {
            "base_url": (
                args.base_url
            ),
            "limit_scenarios": (
                args.limit_scenarios
            ),
            "balanced_smoke": (
                args.balanced_smoke
            ),
            "grounding_aliases": (
                len(
                    alias_index
                )
            ),
            "warmup": (
                not args.skip_warmup
            ),
        },
        "summary": (
            summary
        ),
        "by_scenario_type": (
            summarize_by_type(
                results
            )
        ),
        "scenario_results": (
            scenario_results
        ),
        "results": (
            results
        ),
        "metric_notes": {
            "task_success": (
                "Deterministic composite of "
                "route/tool correctness, hard "
                "constraints, semantic tag hit "
                "when applicable, expected product "
                "coverage when applicable, and "
                "retrieval hit when a generated "
                "relevant set exists."
            ),
            "catalog_product_grounding": (
                "Checks whether full catalog "
                "product names mentioned in the "
                "answer belong to accumulated "
                "retrieval evidence. It is a "
                "product-identity grounding metric, "
                "not a complete factuality judge."
            ),
            "latency": (
                "End-to-end /chat wall-clock "
                "latency after an unscored warmup."
            ),
        },
    }

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("DAY 20 EVALUATION SUMMARY")
    print("=" * 72)

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

    print()
    print(
        "Report:",
        output_path,
    )


if __name__ == "__main__":
    asyncio.run(
        main()
    )
