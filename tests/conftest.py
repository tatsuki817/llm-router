"""Shared test fixtures and mock backends for llm-router tests."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_router.models import LLMRequest, LLMResponse


class MockBackend:
    """Configurable mock backend for testing.

    By default, returns successful responses. Can be configured to fail,
    delay, or return custom content.
    """

    def __init__(
        self,
        response_content: str = "Mock response",
        should_fail: bool = False,
        error_message: str = "Mock error",
        latency_ms: float = 10.0,
        delay_seconds: float = 0.0,
    ) -> None:
        self.response_content = response_content
        self.should_fail = should_fail
        self.error_message = error_message
        self.latency_ms = latency_ms
        self.delay_seconds = delay_seconds
        self.call_count = 0
        self.last_request: LLMRequest | None = None
        self.last_model: str | None = None
        self._closed = False

    async def generate(
        self, request: LLMRequest, model: str, **kwargs: Any
    ) -> LLMResponse:
        self.call_count += 1
        self.last_request = request
        self.last_model = model

        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        if self.should_fail:
            return LLMResponse(
                content="",
                model_used=model,
                success=False,
                latency_ms=self.latency_ms,
                error=self.error_message,
            )

        return LLMResponse(
            content=self.response_content,
            model_used=model,
            success=True,
            latency_ms=self.latency_ms,
            tokens_used=len(request.prompt.split()) * 2,
        )

    async def health_check(self) -> bool:
        return not self.should_fail

    async def close(self) -> None:
        self._closed = True


class FailNTimesBackend(MockBackend):
    """Backend that fails the first N times, then succeeds."""

    def __init__(self, fail_count: int = 3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fail_count = fail_count

    async def generate(
        self, request: LLMRequest, model: str, **kwargs: Any
    ) -> LLMResponse:
        self.call_count += 1
        self.last_request = request
        self.last_model = model

        if self.call_count <= self.fail_count:
            return LLMResponse(
                content="",
                model_used=model,
                success=False,
                latency_ms=self.latency_ms,
                error=f"Failure {self.call_count}/{self.fail_count}",
            )

        return LLMResponse(
            content=self.response_content,
            model_used=model,
            success=True,
            latency_ms=self.latency_ms,
        )


@pytest.fixture
def mock_backend() -> MockBackend:
    """A successful mock backend."""
    return MockBackend()


@pytest.fixture
def failing_backend() -> MockBackend:
    """A consistently failing mock backend."""
    return MockBackend(should_fail=True)
