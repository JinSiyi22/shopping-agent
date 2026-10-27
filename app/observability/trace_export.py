"""Opt-in export of already collected AgentTrace events (no new model calls).

Set SHOPPING_AGENT_TRACE_DIR=demo/fixtures to record runs.
Set SHOPPING_AGENT_TRACE_INCLUDE_TEXT=1 only for safe, non-private demo prompts.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re

logger = logging.getLogger(__name__)


def export_agent_trace_if_enabled(*, trace, user_query: str, final_answer: str, steps=None):
    directory = os.environ.get("SHOPPING_AGENT_TRACE_DIR", "").strip()
    if not directory:
        return None

    include_text = os.environ.get("SHOPPING_AGENT_TRACE_INCLUDE_TEXT", "0") == "1"
    output_dir = Path(directory).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    events = []
    for e in trace.events:
        kind = str(e.event_type)
        label = str(e.name)
        events.append({
            "step": e.step,
            "node": kind,
            "event_type": kind,
            "name": label,
            "tool": label if kind == "research" and label not in {"tool_selection", "no_tool"} else "—",
            "latency_ms": e.latency_ms,
            "success": e.success,
            "status": "success" if e.success else "failed",
        })

    record = {
        "source": "recorded_real_run",
        "trace_id": str(trace.trace_id),
        "scenario": "Captured Multi-Agent run",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "query": user_query if include_text else "[query hidden: enable only for safe demo prompts]",
        "answer": final_answer if include_text else "[answer hidden: enable only for safe demo prompts]",
        "steps": steps,
        "total_latency_ms": round(trace.total_latency_ms(), 2),
        "events": events,
        "limitations": "Event order is append/completion order. Parallel durations can overlap; sum of event latencies is not total latency. Only recorded events are shown.",
    }

    trace_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(trace.trace_id))
    target = output_dir / f"real_trace_{trace_id}.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(target, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    logger.info("multi_agent_trace_exported path=%s events=%s", target, len(events))
    return target
