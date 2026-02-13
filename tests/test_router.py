"""Integration tests for the main LLMRouter class."""

import pytest

from conftest import MockBackend
from llm_router import LLMRouter, Priority


class TestRouterBasics:
    """Basic router setup and generation."""

    @pytest.mark.asyncio
    async def test_single_model_generate(self) -> None:
        backend = MockBackend(response_content="Hello!")
        router = LLMRouter()
        router.add_model("test", provider="ollama", model="test-model", backend=backend)

        response = await router.generate("Hi", priority=Priority.NORMAL)

        assert response.success is True
        assert response.content == "Hello!"
        assert "test" in response.model_used
        assert backend.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self) -> None:
        backend = MockBackend()
        router = LLMRouter()
        router.add_model("test", provider="ollama", model="m", backend=backend)

        await router.generate("Hi", system_prompt="Be concise")

        assert backend.last_request is not None
        assert backend.last_request.system_prompt == "Be concise"

    @pytest.mark.asyncio
    async def test_no_models_returns_error(self) -> None:
        router = LLMRouter()
        response = await router.generate("Hi")

        assert response.success is False
        assert "not registered" in response.error or "All models failed" in response.error

    @pytest.mark.asyncio
    async def test_add_remove_model(self) -> None:
        router = LLMRouter()
        backend = MockBackend()
        router.add_model("test", provider="ollama", model="m", backend=backend)

        assert "test" in router.get_model_status()

        router.remove_model("test")
        assert "test" not in router.get_model_status()

    @pytest.mark.asyncio
    async def test_duplicate_model_raises(self) -> None:
        router = LLMRouter()
        router.add_model("test", provider="ollama", model="m", backend=MockBackend())

        with pytest.raises(ValueError, match="already registered"):
            router.add_model("test", provider="ollama", model="m2", backend=MockBackend())

    @pytest.mark.asyncio
    async def test_method_chaining(self) -> None:
        router = (
            LLMRouter()
            .add_model("a", provider="ollama", model="m1", backend=MockBackend())
            .add_model("b", provider="ollama", model="m2", backend=MockBackend())
            .add_route(Priority.CRITICAL, targets=["a"])
            .add_route(Priority.NORMAL, targets=["b"], fallback=["a"])
        )
        assert len(router.get_model_status()) == 2


class TestRouting:
    """Priority-based routing to different models."""

    @pytest.mark.asyncio
    async def test_routes_critical_to_target(self) -> None:
        smart_backend = MockBackend(response_content="smart answer")
        fast_backend = MockBackend(response_content="fast answer")

        router = LLMRouter()
        router.add_model("smart", provider="ollama", model="big", backend=smart_backend)
        router.add_model("fast", provider="ollama", model="small", backend=fast_backend)
        router.add_route(Priority.CRITICAL, targets=["smart"])
        router.add_route(Priority.LOW, targets=["fast"])

        response = await router.generate("urgent!", priority=Priority.CRITICAL)
        assert response.content == "smart answer"
        assert smart_backend.call_count == 1
        assert fast_backend.call_count == 0

    @pytest.mark.asyncio
    async def test_routes_low_to_fast_model(self) -> None:
        smart_backend = MockBackend(response_content="smart answer")
        fast_backend = MockBackend(response_content="fast answer")

        router = LLMRouter()
        router.add_model("smart", provider="ollama", model="big", backend=smart_backend)
        router.add_model("fast", provider="ollama", model="small", backend=fast_backend)
        router.add_route(Priority.CRITICAL, targets=["smart"])
        router.add_route(Priority.LOW, targets=["fast"])

        response = await router.generate("casual", priority=Priority.LOW)
        assert response.content == "fast answer"

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self) -> None:
        primary = MockBackend(should_fail=True, error_message="primary down")
        fallback = MockBackend(response_content="fallback works")

        router = LLMRouter()
        router.add_model("primary", provider="ollama", model="p", backend=primary)
        router.add_model("fallback", provider="ollama", model="f", backend=fallback)
        router.add_route(Priority.NORMAL, targets=["primary"], fallback=["fallback"])

        response = await router.generate("test")
        assert response.success is True
        assert response.content == "fallback works"
        assert primary.call_count == 1
        assert fallback.call_count == 1

    @pytest.mark.asyncio
    async def test_no_route_uses_all_models(self) -> None:
        """Without explicit routes, tries all models in registration order."""
        backend = MockBackend(response_content="ok")
        router = LLMRouter()
        router.add_model("only", provider="ollama", model="m", backend=backend)

        response = await router.generate("test")
        assert response.success is True
        assert backend.call_count == 1


class TestCircuitBreakerIntegration:
    """Circuit breaker integration with routing."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_skips_open_model(self) -> None:
        failing = MockBackend(should_fail=True)
        healthy = MockBackend(response_content="healthy")

        router = LLMRouter()
        router.enable_circuit_breaker(failure_threshold=2, recovery_timeout=60)
        router.add_model("failing", provider="ollama", model="f", backend=failing)
        router.add_model("healthy", provider="ollama", model="h", backend=healthy)
        router.add_route(Priority.NORMAL, targets=["failing"], fallback=["healthy"])

        # First 2 requests: failing model tried, then fallback
        await router.generate("test1")
        await router.generate("test2")

        # Circuit should now be open for "failing"
        status = router.get_model_status()
        assert status["failing"]["circuit_breaker"]["state"] == "open"

        # Third request: should skip "failing" and go directly to "healthy"
        failing.call_count = 0
        response = await router.generate("test3")
        assert response.success is True
        assert failing.call_count == 0  # Skipped!
        assert healthy.call_count >= 3


class TestBackpressureIntegration:
    """Queue depth protection in routing."""

    @pytest.mark.asyncio
    async def test_skips_model_at_queue_limit(self) -> None:
        """When a model's queue is full, routes to fallback."""
        slow = MockBackend(response_content="slow")
        fast = MockBackend(response_content="fast")

        router = LLMRouter()
        router.add_model("slow", provider="ollama", model="s", backend=slow,
                         max_queue_depth=0)  # Always "full"
        router.add_model("fast", provider="ollama", model="f", backend=fast)
        router.add_route(Priority.NORMAL, targets=["slow"], fallback=["fast"])

        response = await router.generate("test")
        assert response.success is True
        assert slow.call_count == 0  # Skipped due to queue limit
        assert fast.call_count == 1


class TestContextManager:
    """Async context manager lifecycle."""

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        backend = MockBackend()

        async with LLMRouter() as router:
            router.add_model("test", provider="ollama", model="m", backend=backend)
            response = await router.generate("Hi")
            assert response.success is True

        assert backend._closed is True


class TestObservability:
    """Stats and model status reporting."""

    @pytest.mark.asyncio
    async def test_stats_after_requests(self) -> None:
        backend = MockBackend()
        router = LLMRouter()
        router.add_model("test", provider="ollama", model="m", backend=backend)

        await router.generate("test1", priority=Priority.HIGH)
        await router.generate("test2", priority=Priority.LOW)

        stats = router.get_stats()
        assert stats["total_requests"] == 2
        assert stats["successes"] == 2
        assert stats["requests_by_priority"]["HIGH"] == 1
        assert stats["requests_by_priority"]["LOW"] == 1

    @pytest.mark.asyncio
    async def test_model_status(self) -> None:
        router = LLMRouter()
        router.enable_circuit_breaker()
        router.add_model("m1", provider="ollama", model="a", backend=MockBackend())
        router.add_model("m2", provider="openai", model="b", backend=MockBackend())

        status = router.get_model_status()
        assert "m1" in status
        assert "m2" in status
        assert status["m1"]["provider"] == "ollama"
        assert status["m2"]["provider"] == "openai"
        assert "circuit_breaker" in status["m1"]
