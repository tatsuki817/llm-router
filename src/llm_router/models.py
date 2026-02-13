"""
llm-router: Data models for requests, responses, routing configuration.

All public types used throughout the library are defined here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class Priority(IntEnum):
    """Request priority levels. Lower value = higher priority.

    Used by routing rules to determine which model handles a request.
    CRITICAL requests always go to the best available model.

    Example::

        response = await router.generate("urgent task", priority=Priority.CRITICAL)
    """

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class LLMRequest:
    """A request to be routed to an LLM model.

    Attributes:
        prompt: The user prompt text.
        priority: Routing priority (CRITICAL > HIGH > NORMAL > LOW).
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
        system_prompt: Optional system/instruction prompt.
        timeout: Per-request timeout in seconds.
        metadata: Arbitrary key-value pairs for tracking, logging, or cache keying.
    """

    prompt: str
    priority: Priority = Priority.NORMAL
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: str | None = None
    timeout: float = 120.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Internal fields — set by the router, not by the user
    _submitted_at: float = field(default_factory=time.time, repr=False)
    _target_model: str | None = field(default=None, repr=False)

    def __lt__(self, other: object) -> bool:
        """Priority queue ordering: lower priority value = higher urgency."""
        if not isinstance(other, LLMRequest):
            return NotImplemented
        if self.priority != other.priority:
            return self.priority < other.priority
        return self._submitted_at < other._submitted_at  # FIFO within same priority


@dataclass
class LLMResponse:
    """Response from an LLM model.

    Attributes:
        content: The generated text.
        model_used: Name of the model that produced this response.
        success: Whether the generation succeeded.
        latency_ms: End-to-end latency in milliseconds.
        tokens_used: Token count (prompt + completion), if reported by the backend.
        error: Error message if success is False.
        queue_wait_ms: Time spent waiting in the priority queue.
        cached: True if this response came from the semantic cache.
    """

    content: str
    model_used: str
    success: bool
    latency_ms: float
    tokens_used: int | None = None
    error: str | None = None
    queue_wait_ms: float = 0.0
    cached: bool = False


@dataclass
class ModelConfig:
    """Configuration for a registered LLM model.

    Attributes:
        name: Unique identifier for this model (e.g., "local-gpu", "cloud-gpt4").
        provider: Backend provider name ("ollama", "openai").
        model: Model identifier (e.g., "llama3:8b", "gpt-4o").
        base_url: Backend API base URL. Defaults depend on the provider.
        api_key: API key for authenticated providers (OpenAI, Azure, etc.).
        tags: Arbitrary tags for organizing models (e.g., ["fast", "gpu"]).
        max_queue_depth: Maximum pending requests before backpressure kicks in.
        timeout: Default timeout in seconds for requests to this model.
        extra_options: Provider-specific options passed through to the backend.
    """

    name: str
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    tags: list[str] = field(default_factory=list)
    max_queue_depth: int = 10
    timeout: float = 120.0
    extra_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingRule:
    """Defines how requests at a given priority are routed to models.

    Attributes:
        priority: The priority level this rule applies to.
        targets: Ordered list of model names to try first.
        fallback: Ordered list of fallback model names if all targets are unavailable.
        max_queue_depth: Optional queue depth override for this rule's priority.
    """

    priority: Priority
    targets: list[str]
    fallback: list[str] = field(default_factory=list)
    max_queue_depth: int | None = None
