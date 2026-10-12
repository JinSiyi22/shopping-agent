from dataclasses import (
    dataclass,
    field,
)
from time import perf_counter
from uuid import uuid4


@dataclass
class TraceEvent:
    step: int
    event_type: str
    name: str
    latency_ms: float | None = None
    success: bool = True


@dataclass
class AgentTrace:
    trace_id: str = field(
        default_factory=lambda: str(
            uuid4()
        )
    )

    started_at: float = field(
        default_factory=perf_counter
    )

    events: list[TraceEvent] = field(
        default_factory=list
    )

    def add_event(
        self,
        event: TraceEvent,
    ):
        self.events.append(
            event
        )

    def total_latency_ms(
        self,
    ) -> float:

        return (
            perf_counter()
            - self.started_at
        ) * 1000