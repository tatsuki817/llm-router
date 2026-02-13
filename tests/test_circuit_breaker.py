"""Tests for the circuit breaker state machine."""

import time

from llm_router.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState


class TestCircuitBreakerStates:
    """State machine transitions: CLOSED → OPEN → HALF_OPEN → CLOSED."""

    def test_initial_state_is_closed(self) -> None:
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available() is True

    def test_opens_after_failure_threshold(self) -> None:
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))

        cb.record_failure("err1")
        cb.record_failure("err2")
        assert cb.state == CircuitState.CLOSED

        cb.record_failure("err3")
        assert cb.state == CircuitState.OPEN
        assert cb.is_available() is False

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))

        cb.record_failure("err1")
        cb.record_failure("err2")
        cb.record_success()  # Reset
        cb.record_failure("err1")
        cb.record_failure("err2")
        # Still closed — success reset the counter
        assert cb.state == CircuitState.CLOSED

    def test_transitions_to_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(
            "test",
            CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1),
        )

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.is_available() is True

    def test_half_open_closes_after_success_threshold(self) -> None:
        cb = CircuitBreaker(
            "test",
            CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=0.1,
                success_threshold=2,
            ),
        )

        # Trip the breaker
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN  # Need 2

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self) -> None:
        cb = CircuitBreaker(
            "test",
            CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1),
        )

        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure("still broken")
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerStats:
    """Diagnostics and stats reporting."""

    def test_stats_report(self) -> None:
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))
        cb.record_failure("error A")
        cb.record_failure("error B")

        stats = cb.get_stats()
        assert stats["name"] == "test"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 2
        assert stats["total_trips"] == 0
        assert "error A" in stats["recent_errors"]
        assert "error B" in stats["recent_errors"]

    def test_total_trips_counted(self) -> None:
        cb = CircuitBreaker(
            "test",
            CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.05),
        )

        # Trip 1
        cb.record_failure()
        cb.record_failure()
        assert cb.get_stats()["total_trips"] == 1

        # Recover
        time.sleep(0.1)
        cb.record_success()
        cb.record_success()
        cb.record_success()

        # Trip 2
        cb.record_failure()
        cb.record_failure()
        assert cb.get_stats()["total_trips"] == 2

    def test_manual_reset(self) -> None:
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2))
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available() is True

    def test_recent_errors_limited_to_10(self) -> None:
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=20))
        for i in range(15):
            cb.record_failure(f"error_{i}")

        stats = cb.get_stats()
        assert len(stats["recent_errors"]) == 10
        assert stats["recent_errors"][-1] == "error_14"
