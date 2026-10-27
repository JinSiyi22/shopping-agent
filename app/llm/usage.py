"""Token usage and optional cost accounting for LLM calls."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


def _read_field(obj, *names):
    if obj is None:
        return None

    for name in names:
        if isinstance(obj, dict):
            value = obj.get(name)
        else:
            value = getattr(obj, name, None)

        if value is not None:
            return value

    return None


def extract_token_usage(response) -> TokenUsage:
    """Extract usage from an OpenAI-compatible response."""
    usage = _read_field(response, "usage")

    if usage is None:
        return TokenUsage()

    input_tokens = int(
        _read_field(
            usage,
            "prompt_tokens",
            "input_tokens",
        )
        or 0
    )

    output_tokens = int(
        _read_field(
            usage,
            "completion_tokens",
            "output_tokens",
        )
        or 0
    )

    total_tokens = _read_field(
        usage,
        "total_tokens",
    )

    if total_tokens is None:
        total_tokens = (
            input_tokens
            + output_tokens
        )

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=int(total_tokens),
    )


def estimate_cost_usd(
    usage: TokenUsage,
    *,
    input_cost_per_1m: float | None,
    output_cost_per_1m: float | None,
) -> float | None:
    """Estimate request cost from per-million-token prices."""
    if (
        input_cost_per_1m is None
        or output_cost_per_1m is None
    ):
        return None

    cost = (
        usage.input_tokens
        / 1_000_000
        * input_cost_per_1m
        +
        usage.output_tokens
        / 1_000_000
        * output_cost_per_1m
    )

    return round(cost, 8)


def get_configured_cost_rates() -> tuple[
    float | None,
    float | None,
]:
    """Read optional token prices from environment variables."""

    def parse(name: str) -> float | None:
        raw = os.getenv(name)

        if raw is None or not raw.strip():
            return None

        try:
            value = float(raw)
        except ValueError:
            return None

        if value < 0:
            return None

        return value

    return (
        parse("LLM_INPUT_COST_PER_1M"),
        parse("LLM_OUTPUT_COST_PER_1M"),
    )
