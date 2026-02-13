"""
llm-router: Circuit breaker for per-model fault tolerance.

Prevents cascading failures by temporarily stopping requests to a failing
model. After a recovery timeout, the circuit enters a half-open state to
test if the model has recovered.

State machine:
    CLOSED ──(failures >= threshold)──> OPEN
    OPEN ──(recovery_timeout elapsed)──> HALF_OPEN
    HALF_OPEN ──(successes >= threshold)──> CLOSED
    HALF_OPEN ──(any failure)──> OPEN
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation — requests flow through
    OPEN = "open"  # Circuit tripped — requests are blocked
    HALF_OPEN = "half_open"  # Recovery test — limited requests allowed


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker.

    Attributes:
        failure_threshold: Consecutive failures before the circuit opens.
        recovery_timeout: Seconds to wait in OPEN state before trying HALF_OPEN.
        success_threshold: Consecutive successes in HALF_OPEN to close the circuit.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3


class CircuitBreaker:
    """Per-model circuit breaker.

    Tracks failures for a single model and transitions between states
    to prevent sending requests to a consistently failing backend.

    Thread-safe: all state transitions are protected by a reentrant lock.

    Example::

        cb = CircuitBreaker("ollama-llama3", CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2,
        ))

        if cb.is_available():
            response = await backend.generate(request, model)
            if response.success:
                cb.record_success()
            else:
                cb.record_failure()
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._last_state_change: float = time.time()
        self._total_trips = 0
        self._recent_errors: list[str] = []
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state, accounting for automatic OPEN → HALF_OPEN transition."""
        with self._lock:
            if self._state == CircuitState.OPEN and self._should_half_open():
                self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    def is_available(self) -> bool:
        """Check if this model can accept requests.

        Returns True if CLOSED or HALF_OPEN (testing recovery).
        Returns False if OPEN (circuit is tripped).
        """
        return self.state != CircuitState.OPEN

    def record_success(self) -> None:
        """Record a successful request to this model."""
        with self._lock:
            # Auto-transition OPEN → HALF_OPEN if recovery timeout elapsed
            if self._state == CircuitState.OPEN and self._should_half_open():
                self._transition_to(CircuitState.HALF_OPEN)

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    self._failure_count = 0
                    self._success_count = 0
                    self._recent_errors.clear()
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success during normal operation
                self._failure_count = 0

    def record_failure(self, error: str | None = None) -> None:
        """Record a failed request to this model.

        Args:
            error: Optional error description for diagnostics.
        """
        with self._lock:
            # Auto-transition OPEN → HALF_OPEN if recovery timeout elapsed
            if self._state == CircuitState.OPEN and self._should_half_open():
                self._transition_to(CircuitState.HALF_OPEN)

            self._failure_count += 1
            self._last_failure_time = time.time()

            if error:
                self._recent_errors.append(error)
                # Keep only last 10 errors
                if len(self._recent_errors) > 10:
                    self._recent_errors = self._recent_errors[-10:]

            if self._state == CircuitState.HALF_OPEN:
                # Any failure during recovery test → reopen
                self._transition_to(CircuitState.OPEN)
                self._success_count = 0
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self.config.failure_threshold
            ):
                self._transition_to(CircuitState.OPEN)
                self._total_trips += 1

    def reset(self) -> None:
        """Manually reset the circuit to CLOSED state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._recent_errors.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker diagnostics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_trips": self._total_trips,
                "last_failure": self._last_failure_time,
                "last_state_change": self._last_state_change,
                "recent_errors": list(self._recent_errors),
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "success_threshold": self.config.success_threshold,
                },
            }

    def _should_half_open(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return False
        elapsed = time.time() - self._last_failure_time
        return elapsed >= self.config.recovery_timeout

    def _transition_to(self, new_state: CircuitState) -> None:
        """Internal state transition (caller must hold lock)."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()
        if old_state != new_state:
            import logging

            logging.getLogger(__name__).info(
                f"Circuit breaker '{self.name}': {old_state.value} → {new_state.value}"
            )
