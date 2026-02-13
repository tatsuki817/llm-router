"""Tests for the priority queue with backpressure protection."""


from llm_router.models import LLMRequest, Priority
from llm_router.queue import RequestQueue


def make_request(priority: Priority = Priority.NORMAL, prompt: str = "test") -> LLMRequest:
    """Helper to create a test request."""
    return LLMRequest(prompt=prompt, priority=priority)


class TestRequestQueue:
    """Priority queue basic functionality."""

    def test_empty_queue(self) -> None:
        q = RequestQueue()
        assert q.depth == 0
        assert q.is_empty
        assert q.get_nowait() is None

    def test_submit_and_get(self) -> None:
        q = RequestQueue()
        req = make_request()
        assert q.submit(req) is True
        assert q.depth == 1
        assert not q.is_empty

        got = q.get_nowait()
        assert got is req
        assert q.depth == 0

    def test_priority_ordering(self) -> None:
        """Higher priority (lower value) comes out first."""
        q = RequestQueue()
        low = make_request(Priority.LOW, "low")
        critical = make_request(Priority.CRITICAL, "critical")
        normal = make_request(Priority.NORMAL, "normal")

        q.submit(low)
        q.submit(critical)
        q.submit(normal)

        assert q.get_nowait().prompt == "critical"
        assert q.get_nowait().prompt == "normal"
        assert q.get_nowait().prompt == "low"

    def test_fifo_within_same_priority(self) -> None:
        """Same priority uses FIFO ordering."""
        q = RequestQueue()
        first = make_request(Priority.NORMAL, "first")
        second = make_request(Priority.NORMAL, "second")
        third = make_request(Priority.NORMAL, "third")

        q.submit(first)
        q.submit(second)
        q.submit(third)

        assert q.get_nowait().prompt == "first"
        assert q.get_nowait().prompt == "second"
        assert q.get_nowait().prompt == "third"


class TestBackpressure:
    """Queue depth limits and backpressure rejection."""

    def test_hard_limit(self) -> None:
        """Rejects everything when at max_depth."""
        q = RequestQueue(max_depth=2)
        assert q.submit(make_request(Priority.CRITICAL)) is True
        assert q.submit(make_request(Priority.CRITICAL)) is True
        assert q.submit(make_request(Priority.CRITICAL)) is False
        assert q.depth == 2

    def test_per_priority_depth_limit(self) -> None:
        """LOW requests rejected when depth >= limit, but CRITICAL still accepted."""
        q = RequestQueue(max_depth=10)
        q.set_depth_limit(Priority.LOW, 2)

        assert q.submit(make_request(Priority.LOW)) is True
        assert q.submit(make_request(Priority.LOW)) is True  # depth = 2
        assert q.submit(make_request(Priority.LOW)) is False  # rejected: depth >= 2

        # CRITICAL still accepted at same depth
        assert q.submit(make_request(Priority.CRITICAL)) is True

    def test_multiple_depth_limits(self) -> None:
        """Different limits for different priorities."""
        q = RequestQueue(max_depth=10)
        q.set_depth_limit(Priority.LOW, 2)
        q.set_depth_limit(Priority.NORMAL, 4)

        # Fill to depth 2
        q.submit(make_request(Priority.HIGH))
        q.submit(make_request(Priority.HIGH))

        # LOW rejected at depth 2
        assert q.submit(make_request(Priority.LOW)) is False
        # NORMAL still OK at depth 2
        assert q.submit(make_request(Priority.NORMAL)) is True
        assert q.submit(make_request(Priority.NORMAL)) is True  # depth = 4
        # NORMAL rejected at depth 4
        assert q.submit(make_request(Priority.NORMAL)) is False
        # HIGH still OK
        assert q.submit(make_request(Priority.HIGH)) is True

    def test_is_accepting(self) -> None:
        """is_accepting reflects current depth vs limits."""
        q = RequestQueue(max_depth=5)
        q.set_depth_limit(Priority.LOW, 2)

        assert q.is_accepting(Priority.LOW) is True
        q.submit(make_request(Priority.HIGH))
        q.submit(make_request(Priority.HIGH))

        assert q.is_accepting(Priority.LOW) is False
        assert q.is_accepting(Priority.HIGH) is True

    def test_stats_tracking(self) -> None:
        """Stats count submissions and rejections."""
        q = RequestQueue(max_depth=2)
        q.submit(make_request())
        q.submit(make_request())
        q.submit(make_request())  # rejected

        stats = q.get_stats()
        assert stats["total_submitted"] == 2
        assert stats["total_rejected"] == 1
        assert stats["depth"] == 2
