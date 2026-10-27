"""Minimal live API adapter. Adjust only this file if your FastAPI schema differs."""
from __future__ import annotations

from typing import Any
import requests


def _session() -> requests.Session:
    s = requests.Session()
    # Keep localhost out of host-system HTTP proxies (common with WSL + Clash).
    s.trust_env = False
    return s


def health_check(base_url: str) -> tuple[bool, str]:
    try:
        with _session() as s:
            r = s.get(f"{base_url.rstrip('/')}/health", timeout=5)
        return r.ok, f"HTTP {r.status_code}: {r.text[:240]}"
    except requests.RequestException as exc:
        return False, str(exc)


def send_chat(base_url: str, payload: dict[str, Any], timeout: int = 240) -> dict[str, Any]:
    """POST /chat. The actual request shape must match your /openapi.json."""
    with _session() as s:
        response = s.post(f"{base_url.rstrip('/')}/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {"result": data}


def get_answer(data: dict[str, Any]) -> str | None:
    """Try common response keys; use raw JSON if not found."""
    for key in ("answer", "response", "reply", "final_answer", "output", "content"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    result = data.get("result")
    if isinstance(result, dict):
        return get_answer(result)
    return result if isinstance(result, str) else None


def get_events(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Best-effort mapping for traces. No tracing is invented if response omits it."""
    for field in ("events", "agent_trace", "trace"):
        val = data.get(field)
        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict)]
        if isinstance(val, dict):
            inner = val.get("events")
            if isinstance(inner, list):
                return [x for x in inner if isinstance(x, dict)]
    result = data.get("result")
    if isinstance(result, dict):
        return get_events(result)
    return []
