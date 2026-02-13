"""
llm-router: Thread-safe priority queue with per-priority backpressure protection.

Requests are ordered by priority (CRITICAL first) with FIFO ordering within
the same priority level. Backpressure limits prevent low-priority requests
from overwhelming the queue when the system is under load.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_router.models import LLMRequest, Priority

logger = logging.getLogger(__name__)


class RequestQueue:
    """Thread-safe priority queue with configurable backpressure.

    Backpressure works by setting per-priority depth limits. When the queue
    depth exceeds a limit, requests at or below that priority are rejected
    (submit returns False). Higher-priority requests are always accepted.

    Example::

        q = RequestQueue(max_depth=10)
        q.set_depth_limit(Priority.LOW, 3)      # Reject LOW when depth >= 3
        q.set_depth_limit(Priority.NORMAL, 5)    # Reject NORMAL when depth >= 5
        # CRITICAL and HIGH are always accepted (up to max_depth)

        accepted = q.submit(request)
        if not accepted:
            # Request was rejected by backpressure
            ...
    """

    def __init__(self, max_depth: int = 10) -> None:
        self._queue: queue.PriorityQueue[LLMRequest] = queue.PriorityQueue()
        self._max_depth = max_depth
        self._depth_limits: dict[Priority, int] = {}
        self._lock = threading.Lock()
        self._total_submitted = 0
        self._total_rejected = 0

    def set_depth_limit(self, priority: Priority, max_depth: int) -> None:
        """Set the queue depth at which requests at this priority are rejected.

        When queue depth >= max_depth, requests with priority >= the given level
        (i.e., equal or lower urgency) will be rejected by submit().

        Args:
            priority: The priority level threshold.
            max_depth: Reject at this queue depth.
        """
        self._depth_limits[priority] = max_depth

    def submit(self, request: LLMRequest) -> bool:
        """Submit a request to the queue.

        Returns True if accepted, False if rejected by backpressure.
        Rejection means the queue is too deep for this request's priority.
        """
        with self._lock:
            current_depth = self._queue.qsize()

            # Hard limit: reject everything beyond max_depth
            if current_depth >= self._max_depth:
                self._total_rejected += 1
                logger.debug(
                    f"Queue at hard limit ({current_depth}/{self._max_depth}), "
                    f"rejected {request.priority.name} request"
                )
                return False

            # Per-priority backpressure limits
            for limit_priority, limit_depth in sorted(
                self._depth_limits.items(), key=lambda x: x[0]
            ):
                if current_depth >= limit_depth and request.priority >= limit_priority:
                    self._total_rejected += 1
                    logger.debug(
                        f"Queue depth {current_depth} >= {limit_depth}, "
                        f"rejected {request.priority.name} request (limit for "
                        f"{limit_priority.name}+)"
                    )
                    return False

            self._queue.put(request)
            self._total_submitted += 1
            return True

    def get_nowait(self) -> LLMRequest | None:
        """Get the next highest-priority request without blocking.

        Returns None if the queue is empty.
        """
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    @property
    def depth(self) -> int:
        """Current number of pending requests."""
        return self._queue.qsize()

    @property
    def is_empty(self) -> bool:
        """Whether the queue has no pending requests."""
        return self._queue.empty()

    def is_accepting(self, priority: Priority) -> bool:
        """Check if the queue would accept a request at this priority.

        Useful for routing decisions without actually submitting.
        """
        current_depth = self._queue.qsize()

        if current_depth >= self._max_depth:
            return False

        for limit_priority, limit_depth in sorted(
            self._depth_limits.items(), key=lambda x: x[0]
        ):
            if current_depth >= limit_depth and priority >= limit_priority:
                return False

        return True

    def get_stats(self) -> dict[str, int]:
        """Queue statistics."""
        return {
            "depth": self.depth,
            "max_depth": self._max_depth,
            "total_submitted": self._total_submitted,
            "total_rejected": self._total_rejected,
            "depth_limits": {p.name: d for p, d in self._depth_limits.items()},
        }
