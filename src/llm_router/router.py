"""
llm-router: Main LLMRouter class — the orchestrator.

Wires together models, backends, priority queues, circuit breakers,
and semantic caching into a single, easy-to-use interface.

Routing algorithm:
1. Check semantic cache → return if hit
2. Find routing rule for the request's priority
3. Try each target model in order:
   a. Check circuit breaker → skip if OPEN
   b. Check queue depth → skip if backpressure limit hit
   c. Send request to model's backend
4. If all targets fail, try fallback models (same checks)
5. If everything fails, return an error response
6. On success: store in cache, update stats, record circuit breaker success
7. On failure: update stats, record circuit breaker failure
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any

from llm_router.backends.ollama import OllamaBackend
from llm_router.backends.openai import OpenAIBackend
from llm_router.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from llm_router.models import LLMRequest, LLMResponse, ModelConfig, Priority, RoutingRule
from llm_router.queue import RequestQueue
from llm_router.stats import StatsTracker

if TYPE_CHECKING:
    from llm_router.backends.base import Backend

logger = logging.getLogger(__name__)


# Backend factory: provider name → backend class
_BACKEND_REGISTRY: dict[str, type] = {
    "ollama": OllamaBackend,
    "openai": OpenAIBackend,
}


class LLMRouter:
    """Intelligent LLM request router.

    Routes requests to the best available model based on priority,
    queue depth, and circuit breaker state. Optionally caches responses
    using semantic similarity matching.

    Quickstart::

        router = LLMRouter()
        router.add_model("fast", provider="ollama", model="qwen2.5:3b")
        router.add_model("smart", provider="ollama", model="llama3:8b")

        router.add_route(Priority.CRITICAL, targets=["smart"])
        router.add_route(Priority.NORMAL, targets=["fast"], fallback=["smart"])

        async with router:
            response = await router.generate("Hello!", priority=Priority.NORMAL)
            print(response.content)

    Full configuration::

        router = LLMRouter()
        router.add_model("local-gpu", provider="ollama", model="llama3:8b",
                         max_queue_depth=5)
        router.add_model("cloud", provider="openai", model="gpt-4o",
                         api_key="sk-...")

        router.add_route(Priority.CRITICAL, targets=["local-gpu"], fallback=["cloud"])
        router.add_route(Priority.NORMAL, targets=["local-gpu"])

        router.enable_cache(similarity_threshold=0.92)
        router.enable_circuit_breaker(failure_threshold=5, recovery_timeout=60)

        async with router:
            response = await router.generate("Explain quantum computing")
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelConfig] = {}
        self._backends: dict[str, Backend] = {}
        self._queues: dict[str, RequestQueue] = {}
        self._routes: dict[Priority, RoutingRule] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._cb_config: CircuitBreakerConfig | None = None
        self._cache: Any | None = None  # SemanticCache, lazy import
        self._stats = StatsTracker()
        self._started = False

    # ──────────────────────────────────────────────────────────────────────
    # Model Management
    # ──────────────────────────────────────────────────────────────────────

    def add_model(
        self,
        name: str,
        provider: str,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        tags: list[str] | None = None,
        max_queue_depth: int = 10,
        timeout: float = 120.0,
        backend: Backend | None = None,
        **extra_options: Any,
    ) -> LLMRouter:
        """Register an LLM model with the router.

        Args:
            name: Unique identifier for this model.
            provider: Backend provider ("ollama", "openai"), or ignored if
                backend is provided directly.
            model: Model identifier (e.g., "llama3:8b", "gpt-4o").
            base_url: Override the default backend URL.
            api_key: API key for authenticated providers.
            tags: Arbitrary tags for organizing models.
            max_queue_depth: Maximum queue depth before backpressure.
            timeout: Default request timeout in seconds.
            backend: Provide a custom Backend instance instead of using
                the built-in provider factory.
            **extra_options: Provider-specific options passed to the backend.

        Returns:
            self (for method chaining).

        Raises:
            ValueError: If the model name is already registered or provider
                is unknown and no backend is provided.
        """
        if name in self._models:
            raise ValueError(f"Model '{name}' is already registered")

        config = ModelConfig(
            name=name,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            tags=tags or [],
            max_queue_depth=max_queue_depth,
            timeout=timeout,
            extra_options=extra_options,
        )
        self._models[name] = config

        # Create or use provided backend
        if backend is not None:
            self._backends[name] = backend
        else:
            self._backends[name] = self._create_backend(config)

        # Create a queue for this model
        self._queues[name] = RequestQueue(max_depth=max_queue_depth)

        # Create circuit breaker if enabled
        if self._cb_config:
            self._circuit_breakers[name] = CircuitBreaker(name, self._cb_config)

        logger.info(f"Registered model '{name}': {provider}/{model}")
        return self

    def remove_model(self, name: str) -> None:
        """Remove a registered model.

        Args:
            name: The model name to remove.

        Raises:
            KeyError: If the model is not registered.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' is not registered")

        del self._models[name]
        del self._backends[name]
        del self._queues[name]
        self._circuit_breakers.pop(name, None)
        logger.info(f"Removed model '{name}'")

    # ──────────────────────────────────────────────────────────────────────
    # Routing Rules
    # ──────────────────────────────────────────────────────────────────────

    def add_route(
        self,
        priority: Priority,
        targets: list[str],
        fallback: list[str] | None = None,
        max_queue_depth: int | None = None,
    ) -> LLMRouter:
        """Define a routing rule for a priority level.

        Args:
            priority: The priority level this rule applies to.
            targets: Ordered list of model names to try first.
            fallback: Ordered list of fallback models if all targets fail.
            max_queue_depth: Override queue depth limit for this rule.

        Returns:
            self (for method chaining).
        """
        self._routes[priority] = RoutingRule(
            priority=priority,
            targets=targets,
            fallback=fallback or [],
            max_queue_depth=max_queue_depth,
        )
        return self

    # ──────────────────────────────────────────────────────────────────────
    # Optional Features
    # ──────────────────────────────────────────────────────────────────────

    def enable_cache(
        self,
        similarity_threshold: float = 0.92,
        db_path: str = ":memory:",
        ttl_hours: int = 24,
        max_entries: int = 10000,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> LLMRouter:
        """Enable semantic response caching.

        Requires: pip install llm-router[cache]

        Args:
            similarity_threshold: Minimum cosine similarity for cache hits.
            db_path: SQLite database path (":memory:" for in-memory).
            ttl_hours: Cache entry time-to-live in hours.
            max_entries: Maximum cache entries before LRU eviction.
            embedding_model: Sentence-transformer model name.

        Returns:
            self (for method chaining).
        """
        from llm_router.cache import SemanticCache

        self._cache = SemanticCache(
            db_path=db_path,
            similarity_threshold=similarity_threshold,
            ttl_hours=ttl_hours,
            max_entries=max_entries,
            embedding_model=embedding_model,
        )
        logger.info(
            f"Semantic cache enabled (threshold={similarity_threshold}, "
            f"ttl={ttl_hours}h, max={max_entries})"
        )
        return self

    def enable_circuit_breaker(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
    ) -> LLMRouter:
        """Enable per-model circuit breakers.

        Args:
            failure_threshold: Consecutive failures to open the circuit.
            recovery_timeout: Seconds before testing recovery (HALF_OPEN).
            success_threshold: Consecutive successes to close the circuit.

        Returns:
            self (for method chaining).
        """
        self._cb_config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
        )
        # Create circuit breakers for already-registered models
        for name in self._models:
            if name not in self._circuit_breakers:
                self._circuit_breakers[name] = CircuitBreaker(name, self._cb_config)
        logger.info(
            f"Circuit breakers enabled (threshold={failure_threshold}, "
            f"recovery={recovery_timeout}s)"
        )
        return self

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the router: warm up models, open connections.

        Call this before generating requests, or use the async context manager.
        """
        if self._started:
            return

        # Warm up Ollama models
        for name, backend in self._backends.items():
            if isinstance(backend, OllamaBackend):
                config = self._models[name]
                await backend.warmup(config.model)

        self._started = True
        logger.info(
            f"LLMRouter started with {len(self._models)} model(s): "
            f"{', '.join(self._models.keys())}"
        )

    async def stop(self) -> None:
        """Stop the router: close all backend connections."""
        for backend in self._backends.values():
            try:
                await backend.close()
            except Exception as e:
                logger.warning(f"Error closing backend: {e}")

        if self._cache is not None:
            with contextlib.suppress(Exception):
                self._cache.close()

        self._started = False
        logger.info("LLMRouter stopped")

    async def __aenter__(self) -> LLMRouter:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.stop()

    # ──────────────────────────────────────────────────────────────────────
    # Core API
    # ──────────────────────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        priority: Priority = Priority.NORMAL,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a response by routing to the best available model.

        Args:
            prompt: The user prompt text.
            priority: Request priority for routing.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            system_prompt: Optional system prompt.
            timeout: Per-request timeout (overrides model default).
            metadata: Arbitrary key-value pairs for tracking.

        Returns:
            LLMResponse with content and metadata.
        """
        request = LLMRequest(
            prompt=prompt,
            priority=priority,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            timeout=timeout or 120.0,
            metadata=metadata or {},
        )

        # 1. Check semantic cache
        if self._cache is not None and temperature <= 0.3:
            hit = self._cache.lookup(prompt, metadata)
            if hit is not None:
                response, similarity = hit
                response.queue_wait_ms = 0.0
                self._stats.record_cache_hit()
                logger.debug(f"Cache hit (similarity={similarity:.3f})")
                return response

        # 2. Find routing rule
        rule = self._routes.get(priority)
        if rule is None:
            # No explicit rule — try all models in registration order
            rule = RoutingRule(
                priority=priority,
                targets=list(self._models.keys()),
            )

        # 3. Try targets, then fallbacks
        all_candidates = list(rule.targets) + list(rule.fallback)
        errors: list[str] = []

        for model_name in all_candidates:
            if model_name not in self._models:
                errors.append(f"{model_name}: not registered")
                continue

            # 3a. Check circuit breaker
            cb = self._circuit_breakers.get(model_name)
            if cb and not cb.is_available():
                self._stats.record_circuit_breaker_skip()
                errors.append(f"{model_name}: circuit breaker OPEN")
                continue

            # 3b. Check queue depth / backpressure
            queue = self._queues[model_name]
            effective_depth = (
                rule.max_queue_depth
                if rule.max_queue_depth is not None
                else self._models[model_name].max_queue_depth
            )

            if queue.depth >= effective_depth:
                self._stats.record_rejection(priority)
                errors.append(
                    f"{model_name}: queue full ({queue.depth}/{effective_depth})"
                )
                continue

            # 3c. Submit request to this model
            config = self._models[model_name]
            backend = self._backends[model_name]

            if timeout is not None:
                request.timeout = timeout
            elif config.timeout:
                request.timeout = config.timeout

            self._stats.record_request(model_name, priority)

            submitted_at = time.time()
            response = await backend.generate(
                request, config.model, **config.extra_options
            )
            queue_wait_ms = (time.time() - submitted_at) * 1000 - response.latency_ms
            response.queue_wait_ms = max(0.0, queue_wait_ms)
            response.model_used = f"{model_name}:{config.model}"

            self._stats.record_response(model_name, response, response.queue_wait_ms)

            if response.success:
                # Record circuit breaker success
                if cb:
                    cb.record_success()

                # Store in cache
                if self._cache is not None and temperature <= 0.3:
                    self._cache.store(prompt, response, metadata)

                return response
            else:
                # Record circuit breaker failure
                if cb:
                    cb.record_failure(response.error)
                errors.append(f"{model_name}: {response.error}")
                continue

        # 4. All models failed
        return LLMResponse(
            content="",
            model_used="none",
            success=False,
            latency_ms=0,
            error=f"All models failed: {'; '.join(errors)}",
        )

    # ──────────────────────────────────────────────────────────────────────
    # Observability
    # ──────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get router performance statistics."""
        stats = self._stats.get_stats()
        if self._cache is not None:
            stats["cache"] = self._cache.get_stats()
        return stats

    def get_model_status(self) -> dict[str, dict[str, Any]]:
        """Get per-model health and queue status."""
        result: dict[str, dict[str, Any]] = {}
        for name, config in self._models.items():
            status: dict[str, Any] = {
                "provider": config.provider,
                "model": config.model,
                "queue_depth": self._queues[name].depth,
                "max_queue_depth": config.max_queue_depth,
            }
            cb = self._circuit_breakers.get(name)
            if cb:
                status["circuit_breaker"] = cb.get_stats()
            result[name] = status
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _create_backend(config: ModelConfig) -> Backend:
        """Create a backend instance from model config."""
        provider = config.provider.lower()

        if provider == "ollama":
            return OllamaBackend(
                base_url=config.base_url or "http://localhost:11434",
                default_options=config.extra_options,
            )
        elif provider == "openai":
            return OpenAIBackend(
                api_key=config.api_key or "",
                base_url=config.base_url or "https://api.openai.com/v1",
            )
        elif provider in _BACKEND_REGISTRY:
            return _BACKEND_REGISTRY[provider]()  # type: ignore[call-arg]
        else:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Available: {list(_BACKEND_REGISTRY.keys())}. "
                f"Or pass a custom `backend=` instance."
            )


def register_backend(name: str, backend_class: type) -> None:
    """Register a custom backend provider.

    Example::

        from llm_router import register_backend

        class MyBackend:
            async def generate(self, request, model, **kwargs):
                ...
            async def health_check(self):
                ...
            async def close(self):
                ...

        register_backend("my_provider", MyBackend)
        router.add_model("custom", provider="my_provider", model="my-model")
    """
    _BACKEND_REGISTRY[name.lower()] = backend_class
