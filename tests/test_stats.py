"""Tests for the statistics tracker."""


from llm_router.models import LLMResponse, Priority
from llm_router.stats import StatsTracker


class TestStatsTracker:
    """Statistics recording and retrieval."""

    def test_initial_stats(self) -> None:
        tracker = StatsTracker()
        stats = tracker.get_stats()
        assert stats["total_requests"] == 0
        assert stats["successes"] == 0
        assert stats["failures"] == 0
        assert stats["success_rate"] == 0.0

    def test_record_request(self) -> None:
        tracker = StatsTracker()
        tracker.record_request("model-a", Priority.HIGH)
        tracker.record_request("model-a", Priority.NORMAL)
        tracker.record_request("model-b", Priority.LOW)

        stats = tracker.get_stats()
        assert stats["total_requests"] == 3
        assert stats["requests_by_model"]["model-a"] == 2
        assert stats["requests_by_model"]["model-b"] == 1
        assert stats["requests_by_priority"]["HIGH"] == 1
        assert stats["requests_by_priority"]["NORMAL"] == 1

    def test_record_success(self) -> None:
        tracker = StatsTracker()
        response = LLMResponse(
            content="ok", model_used="test", success=True, latency_ms=100.0
        )
        tracker.record_response("model-a", response, queue_wait_ms=5.0)

        stats = tracker.get_stats()
        assert stats["successes"] == 1
        assert stats["failures"] == 0
        assert stats["avg_latency_ms"] == 100.0
        assert stats["avg_queue_wait_ms"] == 5.0

    def test_record_failure(self) -> None:
        tracker = StatsTracker()
        response = LLMResponse(
            content="", model_used="test", success=False,
            latency_ms=50.0, error="timeout"
        )
        tracker.record_response("model-a", response, queue_wait_ms=0.0)

        stats = tracker.get_stats()
        assert stats["failures"] == 1
        assert stats["failures_by_model"]["model-a"] == 1

    def test_running_average_latency(self) -> None:
        tracker = StatsTracker()
        for latency in [100.0, 200.0, 300.0]:
            response = LLMResponse(
                content="ok", model_used="test", success=True, latency_ms=latency
            )
            tracker.record_response("model-a", response, queue_wait_ms=0.0)

        stats = tracker.get_stats()
        assert abs(stats["avg_latency_ms"] - 200.0) < 0.1

    def test_success_rate(self) -> None:
        tracker = StatsTracker()
        for success in [True, True, True, False]:
            response = LLMResponse(
                content="ok" if success else "",
                model_used="test",
                success=success,
                latency_ms=10.0,
            )
            tracker.record_response("m", response, queue_wait_ms=0.0)

        stats = tracker.get_stats()
        assert stats["success_rate"] == 0.75

    def test_cache_hit_tracking(self) -> None:
        tracker = StatsTracker()
        tracker.record_cache_hit()
        tracker.record_cache_hit()

        assert tracker.get_stats()["cache_hits"] == 2

    def test_rejection_tracking(self) -> None:
        tracker = StatsTracker()
        tracker.record_rejection(Priority.LOW)
        tracker.record_rejection(Priority.NORMAL)

        assert tracker.get_stats()["rejected_by_backpressure"] == 2

    def test_reset(self) -> None:
        tracker = StatsTracker()
        tracker.record_request("m", Priority.HIGH)
        tracker.record_cache_hit()
        tracker.reset()

        stats = tracker.get_stats()
        assert stats["total_requests"] == 0
        assert stats["cache_hits"] == 0
