"""
llm-router: Abstract backend protocol.

Any LLM provider can be integrated by implementing this protocol.
Built-in implementations exist for Ollama and OpenAI-compatible APIs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from llm_router.models import LLMRequest, LLMResponse


@runtime_checkable
class Backend(Protocol):
    """Protocol defining the interface for LLM backends.

    Implement this to add support for a custom LLM provider.

    Example::

        class MyBackend:
            async def generate(self, request: LLMRequest, model: str, **kwargs) -> LLMResponse:
                # Call your custom API
                result = await my_api.complete(model, request.prompt)
                return LLMResponse(
                    content=result.text,
                    model_used=model,
                    success=True,
                    latency_ms=result.duration_ms,
                )

            async def health_check(self) -> bool:
                return await my_api.ping()

            async def close(self) -> None:
                await my_api.disconnect()
    """

    async def generate(
        self, request: LLMRequest, model: str, **kwargs: Any
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            request: The incoming request with prompt, temperature, etc.
            model: The model identifier to use (e.g., "llama3:8b", "gpt-4o").
            **kwargs: Provider-specific options.

        Returns:
            LLMResponse with the generated content or error details.
        """
        ...

    async def health_check(self) -> bool:
        """Check if the backend is reachable and healthy.

        Returns:
            True if the backend is ready to accept requests.
        """
        ...

    async def close(self) -> None:
        """Clean up resources (HTTP sessions, connections, etc.)."""
        ...
