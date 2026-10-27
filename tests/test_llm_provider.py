from unittest.mock import AsyncMock

import pytest

from app.llm import provider
from app.llm.errors import (
    LLMDailyQuotaExceeded,
    classify_rate_limit_error,
)


class FakeRateLimitError(Exception):
    pass


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        outcome = self.outcomes.pop(0)

        if isinstance(outcome, Exception):
            raise outcome

        return outcome


class FakeChat:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)


class FakeClient:
    def __init__(self, outcomes):
        self.chat = FakeChat(outcomes)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "Rate limit reached: tokens per day (TPD)",
            "daily",
        ),
        (
            "Rate limit reached: tokens per minute (TPM)",
            "minute",
        ),
        (
            "HTTP 429 Too Many Requests",
            "unknown",
        ),
    ],
)
def test_classify_rate_limit_error(message, expected):
    assert (
        classify_rate_limit_error(Exception(message))
        == expected
    )


def test_apply_model_defaults_for_gpt_oss_without_mutating_input():
    original = {
        "model": "openai/gpt-oss-20b",
        "extra_body": {
            "disable_tool_validation": True,
        },
    }

    result = provider.apply_model_defaults(original)

    assert result["extra_body"]["disable_tool_validation"] is True
    assert result["extra_body"]["reasoning_effort"] == "low"
    assert result["extra_body"]["include_reasoning"] is False
    assert result["max_completion_tokens"] == 512

    assert original == {
        "model": "openai/gpt-oss-20b",
        "extra_body": {
            "disable_tool_validation": True,
        },
    }


@pytest.mark.asyncio
async def test_daily_quota_fails_fast_without_retry(monkeypatch):
    monkeypatch.setattr(
        provider,
        "RateLimitError",
        FakeRateLimitError,
    )

    sleep = AsyncMock()
    monkeypatch.setattr(
        provider.asyncio,
        "sleep",
        sleep,
    )

    client = FakeClient(
        [
            FakeRateLimitError(
                "Rate limit reached: tokens per day (TPD)"
            )
        ]
    )

    with pytest.raises(LLMDailyQuotaExceeded):
        await provider.call_chat_with_retry(
            client,
            model="test-model",
            messages=[],
            max_retries=4,
        )

    assert len(client.chat.completions.calls) == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_minute_rate_limit_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(
        provider,
        "RateLimitError",
        FakeRateLimitError,
    )

    sleep = AsyncMock()
    monkeypatch.setattr(
        provider.asyncio,
        "sleep",
        sleep,
    )

    expected_response = object()

    client = FakeClient(
        [
            FakeRateLimitError(
                "Rate limit reached: tokens per minute (TPM)"
            ),
            expected_response,
        ]
    )

    result = await provider.call_chat_with_retry(
        client,
        model="test-model",
        messages=[],
        max_retries=4,
    )

    assert result is expected_response
    assert len(client.chat.completions.calls) == 2
    sleep.assert_awaited_once_with(3)


def test_extract_token_usage_from_openai_style_response():
    class Usage:
        prompt_tokens = 120
        completion_tokens = 30
        total_tokens = 150

    class Response:
        usage = Usage()

    usage = provider.extract_token_usage(Response())

    assert usage.input_tokens == 120
    assert usage.output_tokens == 30
    assert usage.total_tokens == 150


def test_extract_token_usage_falls_back_to_sum():
    class Usage:
        prompt_tokens = 100
        completion_tokens = 25

    class Response:
        usage = Usage()

    usage = provider.extract_token_usage(Response())

    assert usage.input_tokens == 100
    assert usage.output_tokens == 25
    assert usage.total_tokens == 125


def test_estimate_cost_usd():
    usage = provider.extract_token_usage(
        {
            "usage": {
                "prompt_tokens": 1_000_000,
                "completion_tokens": 500_000,
                "total_tokens": 1_500_000,
            }
        }
    )

    cost = provider.estimate_cost_usd(
        usage,
        input_cost_per_1m=1.0,
        output_cost_per_1m=2.0,
    )

    assert cost == 2.0


def test_cost_is_none_when_prices_are_not_configured():
    usage = provider.extract_token_usage(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            }
        }
    )

    cost = provider.estimate_cost_usd(
        usage,
        input_cost_per_1m=None,
        output_cost_per_1m=None,
    )

    assert cost is None
