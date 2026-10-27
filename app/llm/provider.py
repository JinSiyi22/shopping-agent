"""OpenAI-compatible LLM runtime.

This module owns:
- client construction
- model-specific request defaults
- provider/network retry behavior
- quota classification

Agent-specific tool-call recovery stays in the agent layer.
"""

from __future__ import annotations

import asyncio
import logging
import os

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from app.llm.errors import (
    LLMDailyQuotaExceeded,
    classify_rate_limit_error,
)
from app.llm.usage import (
    estimate_cost_usd,
    extract_token_usage,
    get_configured_cost_rates,
)

logger = logging.getLogger(__name__)

_llm_client: AsyncOpenAI | None = None


def get_llm_client() -> AsyncOpenAI:
    """Return the process-wide OpenAI-compatible async client."""
    global _llm_client

    if _llm_client is not None:
        return _llm_client

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is not configured")

    _llm_client = AsyncOpenAI(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL") or None,
        timeout=30.0,
    )
    return _llm_client


def apply_model_defaults(kwargs: dict) -> dict:
    """Apply model-specific runtime defaults without mutating the caller."""
    request = dict(kwargs)
    model = request.get("model")

    if model and model.startswith("openai/gpt-oss"):
        extra_body = dict(request.get("extra_body") or {})
        extra_body.setdefault("reasoning_effort", "low")
        extra_body.setdefault("include_reasoning", False)
        request["extra_body"] = extra_body
        request.setdefault("max_completion_tokens", 512)

    return request


async def call_chat_with_retry(
    client: AsyncOpenAI,
    *,
    max_retries: int = 4,
    **kwargs,
):
    """Call an OpenAI-compatible chat completion with runtime retries.

    Retry:
    - transient/unknown rate limits
    - timeout
    - connection failures

    Fail fast:
    - daily quota exhaustion
    """
    request = apply_model_defaults(kwargs)

    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                **request
            )

            usage = extract_token_usage(response)

            (
                input_cost_per_1m,
                output_cost_per_1m,
            ) = get_configured_cost_rates()

            estimated_cost_usd = estimate_cost_usd(
                usage,
                input_cost_per_1m=input_cost_per_1m,
                output_cost_per_1m=output_cost_per_1m,
            )

            logger.info(
                "llm_usage "
                "model=%s "
                "input_tokens=%s "
                "output_tokens=%s "
                "total_tokens=%s "
                "estimated_cost_usd=%s",
                request.get("model"),
                usage.input_tokens,
                usage.output_tokens,
                usage.total_tokens,
                (
                    estimated_cost_usd
                    if estimated_cost_usd is not None
                    else "n/a"
                ),
            )

            return response

        except RateLimitError as exc:
            rate_limit_kind = classify_rate_limit_error(exc)

            if rate_limit_kind == "daily":
                logger.error(
                    "llm_daily_quota_exhausted "
                    "attempt=%s retry=false error=%s",
                    attempt + 1,
                    exc,
                )
                raise LLMDailyQuotaExceeded(
                    "LLM daily quota is exhausted"
                ) from exc

            if attempt >= max_retries:
                logger.error(
                    "llm_rate_limit_retry_exhausted "
                    "kind=%s attempt=%s error=%s",
                    rate_limit_kind,
                    attempt + 1,
                    exc,
                )
                raise

            # 3, 6, 12, 20 seconds.
            delay = min(3 * (2 ** attempt), 20)

            logger.warning(
                "llm_rate_limit_retry "
                "kind=%s attempt=%s sleep_seconds=%s error=%s",
                rate_limit_kind,
                attempt + 1,
                delay,
                exc,
            )
            await asyncio.sleep(delay)

        except (
            APITimeoutError,
            APIConnectionError,
        ) as exc:
            if attempt >= max_retries:
                logger.error(
                    "llm_network_retry_exhausted "
                    "attempt=%s error=%s",
                    attempt + 1,
                    exc,
                )
                raise

            # 1, 2, 4, 8 seconds.
            delay = min(2 ** attempt, 8)

            logger.warning(
                "llm_network_retry "
                "attempt=%s sleep_seconds=%s "
                "error_type=%s error=%s",
                attempt + 1,
                delay,
                type(exc).__name__,
                exc,
            )
            await asyncio.sleep(delay)

    raise RuntimeError("LLM retry loop exited unexpectedly")
