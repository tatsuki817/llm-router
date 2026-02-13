"""
llm-router: OpenAI-compatible API backend.

Works with any provider that exposes a /v1/chat/completions endpoint:
- OpenAI (GPT-4o, GPT-4, etc.)
- Azure OpenAI Service
- vLLM serving
- LM Studio
- Ollama's OpenAI-compatible endpoint
- Together AI, Groq, Fireworks, etc.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import aiohttp

from llm_router.models import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIBackend:
    """OpenAI-compatible API backend.

    Sends requests to any /v1/chat/completions-compatible endpoint.
    Handles authentication, error responses, and token counting.

    Example::

        # OpenAI
        backend = OpenAIBackend(api_key="sk-...")

        # Local vLLM server
        backend = OpenAIBackend(
            base_url="http://localhost:8000/v1",
            api_key="not-needed",
        )

        response = await backend.generate(request, model="gpt-4o")
        await backend.close()
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        default_headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the OpenAI-compatible backend.

        Args:
            api_key: API key for authentication.
            base_url: API base URL (must include /v1).
            default_headers: Additional headers for every request.
        """
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._default_headers = default_headers or {}
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazily create and return the aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {
                "Content-Type": "application/json",
                **self._default_headers,
            }
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def generate(
        self, request: LLMRequest, model: str, **kwargs: Any
    ) -> LLMResponse:
        """Generate a response via the /v1/chat/completions endpoint.

        Args:
            request: The LLM request.
            model: Model identifier (e.g., "gpt-4o", "gpt-3.5-turbo").
            **kwargs: Additional parameters passed to the API.

        Returns:
            LLMResponse with content, latency, and token counts.
        """
        url = f"{self.base_url}/chat/completions"

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            **kwargs,
        }

        start_time = time.time()
        session = await self._get_session()

        try:
            timeout = aiohttp.ClientTimeout(total=request.timeout)
            async with session.post(url, json=payload, timeout=timeout) as resp:
                latency_ms = (time.time() - start_time) * 1000

                if resp.status == 200:
                    data = await resp.json()
                    choices = data.get("choices", [])
                    content = (
                        choices[0].get("message", {}).get("content", "")
                        if choices
                        else ""
                    )
                    usage = data.get("usage", {})
                    tokens = usage.get("total_tokens") if usage else None

                    return LLMResponse(
                        content=content,
                        model_used=model,
                        success=True,
                        latency_ms=latency_ms,
                        tokens_used=tokens,
                    )
                else:
                    error_text = await resp.text()
                    return LLMResponse(
                        content="",
                        model_used=model,
                        success=False,
                        latency_ms=latency_ms,
                        error=f"HTTP {resp.status}: {error_text[:200]}",
                    )

        except asyncio.TimeoutError:
            return LLMResponse(
                content="",
                model_used=model,
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"Timeout after {request.timeout}s",
            )
        except aiohttp.ClientError as e:
            return LLMResponse(
                content="",
                model_used=model,
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"Connection error: {e}",
            )

    async def health_check(self) -> bool:
        """Check if the OpenAI-compatible endpoint is reachable.

        Attempts to list models via GET /v1/models.
        """
        session = await self._get_session()
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with session.get(
                f"{self.base_url}/models", timeout=timeout
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
