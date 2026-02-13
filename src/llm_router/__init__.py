"""
llm-router: Intelligent LLM request router.

Priority-based routing, multi-model support, circuit breakers,
and semantic caching for production LLM applications.

Quickstart::

    from llm_router import LLMRouter, Priority

    router = LLMRouter()
    router.add_model("fast", provider="ollama", model="qwen2.5:3b")
    router.add_model("smart", provider="ollama", model="llama3:8b")
    router.add_route(Priority.CRITICAL, targets=["smart"])
    router.add_route(Priority.NORMAL, targets=["fast"], fallback=["smart"])

    async with router:
        response = await router.generate("Hello!", priority=Priority.NORMAL)
        print(response.content)
"""

from llm_router.backends.base import Backend
from llm_router.backends.ollama import OllamaBackend
from llm_router.backends.openai import OpenAIBackend
from llm_router.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from llm_router.models import LLMRequest, LLMResponse, ModelConfig, Priority, RoutingRule
from llm_router.queue import RequestQueue
from llm_router.router import LLMRouter, register_backend
from llm_router.stats import StatsTracker

__version__ = "0.1.0"

__all__ = [
    # Core
    "LLMRouter",
    "Priority",
    "LLMRequest",
    "LLMResponse",
    "ModelConfig",
    "RoutingRule",
    # Backends
    "Backend",
    "OllamaBackend",
    "OpenAIBackend",
    "register_backend",
    # Features
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "SemanticCache",
    "RequestQueue",
    "StatsTracker",
]


def __getattr__(name: str) -> object:
    """Lazy import for optional dependencies."""
    if name == "SemanticCache":
        from llm_router.cache import SemanticCache

        return SemanticCache
    raise AttributeError(f"module 'llm_router' has no attribute {name!r}")
