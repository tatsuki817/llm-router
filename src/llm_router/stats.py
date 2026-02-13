"""
llm-router: Request metrics and statistics tracking.

Thread-safe statistics collection for monitoring router performance,
model utilization, cache efficiency, and error rates.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_router.models import LLMResponse, Priority


@dataclass
class RouterStats:
    """Snapshot of router performance metrics.

    Attributes:
        total_requests: Total requests submitted to the router.
        successes: Requests that completed successfully.
        failures: Requests that failed (backend error, timeout, etc.).
        cache_hits: Requests served from the semantic cache.
        rejected_by_backpressure: Requests dropped due to queue depth limits.
        circuit_breaker_rejections: Requests skipped because the model's circuit was open.
        requests_by_model: Per-model request counts.
        requests_by_priority: Per-priority-level request counts.
        failures_by_model: Per-model failure counts.
        avg_latency_ms: Running average response latency.
        avg_queue_wait_ms: Running average queue wait time.
    """

    total_requests: int = 0
    successes: int = 0
    failures: int = 0
    cache_hits: int = 0
    rejected_by_backpressure: int = 0
    circuit_breaker_rejections: int = 0
    requests_by_model: dict[str, int] = field(default_factory=dict)
    requests_by_priority: dict[str, int] = field(default_factory=dict)
    failures_by_model: dict[str, int] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    avg_queue_wait_ms: float = 0.0


class StatsTracker:
    """Thread-safe statistics tracker for the router.

    All methods are safe to call from any thread. Stats are collected
    in real-time and can be retrieved as a snapshot via get_stats().
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats = RouterStats()
        self._latency_count = 0
        self._queue_wait_count = 0

    def record_request(self, model: str, priority: Priority) -> None:
        """Record that a request was submitted to a model."""
        with self._lock:
            self._stats.total_requests += 1
            self._stats.requests_by_model[model] = (
                self._stats.requests_by_model.get(model, 0) + 1
            )
            pname = priority.name
            self._stats.requests_by_priority[pname] = (
                self._stats.requests_by_priority.get(pname, 0) + 1
            )

    def record_response(
        self, model: str, response: LLMResponse, queue_wait_ms: float
    ) -> None:
        """Record a completed response from a model."""
        with self._lock:
            if response.success:
                self._stats.successes += 1
            else:
                self._stats.failures += 1
                self._stats.failures_by_model[model] = (
                    self._stats.failures_by_model.get(model, 0) + 1
                )

            # Running average latency
            self._latency_count += 1
            self._stats.avg_latency_ms += (
                response.latency_ms - self._stats.avg_latency_ms
            ) / self._latency_count

            # Running average queue wait
            self._queue_wait_count += 1
            self._stats.avg_queue_wait_ms += (
                queue_wait_ms - self._stats.avg_queue_wait_ms
            ) / self._queue_wait_count

    def record_cache_hit(self) -> None:
        """Record a semantic cache hit."""
        with self._lock:
            self._stats.cache_hits += 1

    def record_rejection(self, priority: Priority) -> None:
        """Record a request rejected by backpressure."""
        with self._lock:
            self._stats.rejected_by_backpressure += 1

    def record_circuit_breaker_skip(self) -> None:
        """Record a request skipped because the circuit breaker was open."""
        with self._lock:
            self._stats.circuit_breaker_rejections += 1

    def get_stats(self) -> dict[str, Any]:
        """Get a snapshot of current statistics as a dictionary."""
        with self._lock:
            return {
                "total_requests": self._stats.total_requests,
                "successes": self._stats.successes,
                "failures": self._stats.failures,
                "success_rate": (
                    self._stats.successes
                    / max(self._stats.successes + self._stats.failures, 1)
                ),
                "cache_hits": self._stats.cache_hits,
                "rejected_by_backpressure": self._stats.rejected_by_backpressure,
                "circuit_breaker_rejections": self._stats.circuit_breaker_rejections,
                "requests_by_model": dict(self._stats.requests_by_model),
                "requests_by_priority": dict(self._stats.requests_by_priority),
                "failures_by_model": dict(self._stats.failures_by_model),
                "avg_latency_ms": round(self._stats.avg_latency_ms, 1),
                "avg_queue_wait_ms": round(self._stats.avg_queue_wait_ms, 1),
            }

    def reset(self) -> None:
        """Reset all statistics to zero."""
        with self._lock:
            self._stats = RouterStats()
            self._latency_count = 0
            self._queue_wait_count = 0
