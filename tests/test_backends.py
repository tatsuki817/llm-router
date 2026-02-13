"""Tests for backend implementations using mocked HTTP responses."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_router.backends.ollama import OllamaBackend
from llm_router.backends.openai import OpenAIBackend
from llm_router.models import LLMRequest


def make_request(prompt: str = "Hello", **kwargs) -> LLMRequest:
    return LLMRequest(prompt=prompt, **kwargs)


class TestOllamaBackend:
    """Ollama backend with mocked aiohttp."""

    @pytest.mark.asyncio
    async def test_successful_generation(self) -> None:
        backend = OllamaBackend()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "message": {"content": "Hello world!"},
            "eval_count": 10,
            "prompt_eval_count": 5,
        })

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))
        mock_session.closed = False

        backend._session = mock_session

        request = make_request("Say hello")
        response = await backend.generate(request, model="llama3:8b")

        assert response.success is True
        assert response.content == "Hello world!"
        assert response.tokens_used == 15
        assert response.model_used == "llama3:8b"

    @pytest.mark.asyncio
    async def test_http_error(self) -> None:
        backend = OllamaBackend()

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))
        mock_session.closed = False

        backend._session = mock_session

        response = await backend.generate(make_request(), model="llama3:8b")

        assert response.success is False
        assert "HTTP 500" in response.error

    @pytest.mark.asyncio
    async def test_default_options_applied(self) -> None:
        """Default options (e.g., num_gpu=0) should be sent in every request."""
        backend = OllamaBackend(
            default_options={"num_gpu": 0, "num_thread": 4}
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "message": {"content": "ok"},
        })

        captured_payload = {}

        def capture_post(url, json=None, **kwargs):
            captured_payload.update(json or {})
            return AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=False),
            )

        mock_session = AsyncMock()
        mock_session.post = capture_post
        mock_session.closed = False
        backend._session = mock_session

        await backend.generate(make_request(), model="test")

        assert captured_payload["options"]["num_gpu"] == 0
        assert captured_payload["options"]["num_thread"] == 4

    @pytest.mark.asyncio
    async def test_system_prompt_included(self) -> None:
        backend = OllamaBackend()

        captured_payload = {}

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"message": {"content": "ok"}})

        def capture_post(url, json=None, **kwargs):
            captured_payload.update(json or {})
            return AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=False),
            )

        mock_session = AsyncMock()
        mock_session.post = capture_post
        mock_session.closed = False
        backend._session = mock_session

        request = make_request(system_prompt="You are a helpful assistant")
        await backend.generate(request, model="test")

        messages = captured_payload["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant"
        assert messages[1]["role"] == "user"


class TestOpenAIBackend:
    """OpenAI backend with mocked aiohttp."""

    @pytest.mark.asyncio
    async def test_successful_generation(self) -> None:
        backend = OpenAIBackend(api_key="test-key")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "Hello from GPT!"}}],
            "usage": {"total_tokens": 25},
        })

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))
        mock_session.closed = False
        mock_session.headers = {}

        backend._session = mock_session

        response = await backend.generate(make_request(), model="gpt-4o")

        assert response.success is True
        assert response.content == "Hello from GPT!"
        assert response.tokens_used == 25

    @pytest.mark.asyncio
    async def test_rate_limit_error(self) -> None:
        backend = OpenAIBackend(api_key="test-key")

        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.text = AsyncMock(return_value="Rate limit exceeded")

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))
        mock_session.closed = False
        mock_session.headers = {}

        backend._session = mock_session

        response = await backend.generate(make_request(), model="gpt-4o")

        assert response.success is False
        assert "429" in response.error

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        backend = OpenAIBackend()
        mock_session = AsyncMock()
        mock_session.closed = False
        backend._session = mock_session

        await backend.close()
        mock_session.close.assert_called_once()
