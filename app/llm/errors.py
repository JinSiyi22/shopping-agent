"""LLM runtime error classification."""


class LLMDailyQuotaExceeded(RuntimeError):
    """Raised when the provider's daily quota is exhausted."""


def classify_rate_limit_error(exc: Exception) -> str:
    """Classify a provider rate-limit error.

    Returns:
        "daily": daily quota exhaustion such as TPD/RPD
        "minute": transient per-minute limit such as TPM/RPM
        "unknown": other 429/rate-limit errors
    """
    text = str(exc).lower()

    daily_markers = (
        "tokens per day",
        "requests per day",
        "tpd",
        "rpd",
        "daily quota",
    )
    if any(marker in text for marker in daily_markers):
        return "daily"

    minute_markers = (
        "tokens per minute",
        "requests per minute",
        "tpm",
        "rpm",
    )
    if any(marker in text for marker in minute_markers):
        return "minute"

    return "unknown"
