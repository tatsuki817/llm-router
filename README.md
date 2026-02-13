# llm-router

**Intelligent LLM request router with priority queues, multi-model routing, circuit breakers, and semantic caching.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Why?

Running multiple LLM models (local Ollama, cloud OpenAI, etc.) in production means dealing with:

- **Priority management** — Critical requests shouldn't wait behind low-priority background tasks
- **Multi-model routing** — Send quality-critical tasks to your best model, volume tasks to a cheaper one
- **Fault tolerance** — When a model goes down, automatically failover instead of crashing
- **Redundant calls** — Similar prompts shouldn't hit the LLM twice if the answer is already cached
- **Backpressure** — A slow model shouldn't accept more requests than it can handle

**llm-router** solves all of these in a single library with a clean async Python API.

---

## Installation

```bash
# Core (Ollama + OpenAI backends)
pip install llm-router

# With semantic caching
pip install llm-router[cache]

# Everything
pip install llm-router[all]
```

---

## Quickstart

```python
import asyncio
from llm_router import LLMRouter, Priority

async def main():
    router = LLMRouter()
    router.add_model("local", provider="ollama", model="qwen2.5:3b")

    async with router:
        response = await router.generate(
            "What is the capital of France?",
            priority=Priority.NORMAL,
        )
        print(response.content)

asyncio.run(main())
```

---

## Multi-Model Routing

Route requests to different models based on priority:

```python
router = LLMRouter()

# Register models
router.add_model("gpu-fast", provider="ollama", model="llama3:8b",
                 max_queue_depth=5)
router.add_model("cpu-cheap", provider="ollama", model="qwen2.5:3b",
                 max_queue_depth=3, num_gpu=0, num_thread=4)
router.add_model("cloud", provider="openai", model="gpt-4o",
                 api_key="sk-...")

# Define routing rules
router.add_route(Priority.CRITICAL, targets=["gpu-fast"], fallback=["cloud"])
router.add_route(Priority.HIGH,     targets=["gpu-fast"], fallback=["cpu-cheap"])
router.add_route(Priority.NORMAL,   targets=["cpu-cheap"], fallback=["gpu-fast"])
router.add_route(Priority.LOW,      targets=["cpu-cheap"])
```

**How routing works:**

1. Find the routing rule for the request's priority
2. Try each target model in order
3. Skip models with open circuit breakers or full queues
4. Fall back to fallback models if all targets fail
5. Return error only if every model is unavailable

---

## Circuit Breakers

Automatically stop sending requests to failing models:

```python
router.enable_circuit_breaker(
    failure_threshold=5,     # Open after 5 consecutive failures
    recovery_timeout=60.0,   # Wait 60s before testing recovery
    success_threshold=3,     # Close after 3 consecutive successes
)
```

**State machine:** `CLOSED` → (5 failures) → `OPEN` → (60s) → `HALF_OPEN` → (3 successes) → `CLOSED`

When a model's circuit is OPEN, the router skips it and tries the next model in the routing rule.

---

## Semantic Caching

Cache responses and return them for semantically similar prompts:

```bash
pip install llm-router[cache]
```

```python
router.enable_cache(
    similarity_threshold=0.92,  # Cosine similarity for cache hit
    ttl_hours=24,               # Cache entries expire after 24h
    max_entries=10000,           # LRU eviction beyond this limit
)

# These two prompts are semantically similar enough to share a cached response:
await router.generate("What is Python?", temperature=0.1)
await router.generate("Tell me about the Python programming language", temperature=0.1)
```

> **Note:** Only responses with `temperature <= 0.3` are cached, since higher temperatures produce non-deterministic outputs.

Uses [sentence-transformers](https://www.sbert.net/) (all-MiniLM-L6-v2) for embedding-based similarity matching with SQLite persistence.

---

## Backpressure Protection

Prevent queue overload by rejecting low-priority requests when the system is busy:

```python
router.add_model("local", provider="ollama", model="llama3:8b",
                 max_queue_depth=5)  # Reject when 5+ pending
```

The router automatically skips models at their queue depth limit and tries the next candidate. Critical and high-priority requests are always preferred over normal and low.

---

## Custom Backends

Integrate any LLM provider by implementing the `Backend` protocol:

```python
from llm_router import LLMRouter, Backend, LLMRequest, LLMResponse, register_backend

class MyCustomBackend:
    async def generate(self, request: LLMRequest, model: str, **kwargs) -> LLMResponse:
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

# Register globally
register_backend("my_provider", MyCustomBackend)
router.add_model("custom", provider="my_provider", model="my-model")

# Or pass directly
router.add_model("custom", provider="custom", model="my-model",
                 backend=MyCustomBackend())
```

---

## Observability

```python
# Router-wide stats
router.get_stats()
# {
#     "total_requests": 150,
#     "successes": 145,
#     "failures": 5,
#     "success_rate": 0.967,
#     "cache_hits": 23,
#     "avg_latency_ms": 340.5,
#     "requests_by_model": {"gpu-fast": 80, "cpu-cheap": 70},
#     "requests_by_priority": {"CRITICAL": 10, "HIGH": 40, "NORMAL": 80, "LOW": 20},
# }

# Per-model health
router.get_model_status()
# {
#     "gpu-fast": {
#         "provider": "ollama",
#         "model": "llama3:8b",
#         "queue_depth": 2,
#         "circuit_breaker": {"state": "closed", "failure_count": 0},
#     },
# }
```

---

## FastAPI Proxy Server

See [examples/fastapi_server.py](examples/fastapi_server.py) for a full REST API proxy that routes LLM requests through the router:

```bash
pip install llm-router fastapi uvicorn
uvicorn examples.fastapi_server:app

curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello!", "priority": "high"}'
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    LLMRouter                         │
│                                                      │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────┐ │
│  │ Semantic  │   │   Routing    │   │   Stats     │ │
│  │  Cache    │──>│   Engine     │──>│  Tracker    │ │
│  └──────────┘   └──────┬───────┘   └─────────────┘ │
│                         │                            │
│         ┌───────────────┼───────────────┐            │
│         ▼               ▼               ▼            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │  Model A    │ │  Model B    │ │  Model C    │   │
│  │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │   │
│  │ │ Circuit │ │ │ │ Circuit │ │ │ │ Circuit │ │   │
│  │ │ Breaker │ │ │ │ Breaker │ │ │ │ Breaker │ │   │
│  │ ├─────────┤ │ │ ├─────────┤ │ │ ├─────────┤ │   │
│  │ │Priority │ │ │ │Priority │ │ │ │Priority │ │   │
│  │ │ Queue   │ │ │ │ Queue   │ │ │ │ Queue   │ │   │
│  │ ├─────────┤ │ │ ├─────────┤ │ │ ├─────────┤ │   │
│  │ │ Backend │ │ │ │ Backend │ │ │ │ Backend │ │   │
│  │ │(Ollama) │ │ │ │(OpenAI) │ │ │ │(Custom) │ │   │
│  │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │   │
│  └─────────────┘ └─────────────┘ └─────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
