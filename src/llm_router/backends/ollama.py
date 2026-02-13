"""
llm-router: Ollama backend using the native /api/chat endpoint.

Uses the native Ollama API instead of the OpenAI-compatible endpoint because
the native API reliably supports critical features:
- keep_alive: persistent model residency (avoid reload latency)
- options.num_gpu: control GPU layer offloading
- options.num_thread: CPU thread isolation for multi-model setups

The OpenAI-compatible endpoint (/v1/chat/completions) does NOT reliably
pass these options through to the Ollama engine.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import aiohttp

from llm_router.models import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class OllamaBackend:
    """Native Ollama API backend.

    Connects to a local or remote Ollama instance via the /api/chat endpoint.
    Supports model warmup with persistent residency (keep_alive=-1) and
    fine-grained GPU/CPU control via Ollama options.

    Example::

        backend = OllamaBackend(base_url="http://localhost:11434")
        await backend.warmup("llama3:8b")
        response = await backend.generate(request, model="llama3:8b")
        await backend.close()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_options: dict[str, Any] | None = None,
        keep_alive: int = -1,
    ) -> None:
        """Initialize the Ollama backend.

        Args:
            base_url: Ollama server URL (without /api path).
            default_options: Default Ollama options applied to every request
                (e.g., {"num_gpu": 0, "num_thread": 4} for CPU-only mode).
            keep_alive: Model residency time in seconds. -1 = permanent.
        """
        self.base_url = base_url.rstrip("/")
        self.default_options = default_options or {}
        self.keep_alive = keep_alive
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazily create and return the aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def generate(
        self, request: LLMRequest, model: str, **kwargs: Any
    ) -> LLMResponse:
        """Generate a response via Ollama's native /api/chat endpoint.

        Args:
            request: The LLM request.
            model: Ollama model identifier (e.g., "llama3:8b", "qwen2.5:3b").
            **kwargs: Additional Ollama options merged into the request.

        Returns:
            LLMResponse with content, latency, and token counts.
        """
        url = f"{self.base_url}/api/chat"

        # Build options: defaults < per-model extras < per-request kwargs
        options: dict[str, Any] = {}
        options.update(self.default_options)
        options["temperature"] = request.temperature
        options["num_predict"] = request.max_tokens
        options.update(kwargs)

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": model,
            "messages": messages,
            "options": options,
            "keep_alive": self.keep_alive,
            "stream": False,
        }

        start_time = time.time()
        session = await self._get_session()

        try:
            timeout = aiohttp.ClientTimeout(total=request.timeout)
            async with session.post(url, json=payload, timeout=timeout) as resp:
                latency_ms = (time.time() - start_time) * 1000

                if resp.status == 200:
                    data = await resp.json()
                    content = data.get("message", {}).get("content", "")
                    eval_count = data.get("eval_count", 0)
                    prompt_eval_count = data.get("prompt_eval_count", 0)
                    tokens = (eval_count + prompt_eval_count) or None

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

    async def warmup(self, model: str, keep_alive: int | None = None) -> bool:
        """Pre-load a model into Ollama's memory.

        Sends a minimal generation request (1 token) to trigger model loading
        and sets keep_alive to keep the model resident in memory.

        Args:
            model: Model to warm up.
            keep_alive: Override the default keep_alive for this warmup.

        Returns:
            True if warmup succeeded.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": "hi",
            "keep_alive": keep_alive if keep_alive is not None else self.keep_alive,
            "options": {"num_predict": 1, **self.default_options},
            "stream": False,
        }

        session = await self._get_session()
        try:
            timeout = aiohttp.ClientTimeout(total=180)
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status == 200:
                    logger.info(f"Ollama model warmed up: {model}")
                    return True
                else:
                    text = await resp.text()
                    logger.warning(f"Ollama warmup HTTP {resp.status}: {text[:100]}")
                    return False
        except Exception as e:
            logger.warning(f"Ollama warmup failed for {model}: {e}")
            return False

    async def health_check(self) -> bool:
        """Check if the Ollama server is reachable."""
        session = await self._get_session()
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with session.get(f"{self.base_url}/api/tags", timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
