"""
llm-router FastAPI proxy server example.

Exposes a REST API that routes LLM requests to the best available model.
Useful as a drop-in proxy between your app and your LLM fleet.

Prerequisites:
    pip install llm-router fastapi uvicorn
    ollama pull llama3:8b
    ollama pull qwen2.5:3b

Run:
    uvicorn examples.fastapi_server:app --reload

Usage:
    curl -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Hello!", "priority": "normal"}'
"""

from contextlib import asynccontextmanager
from typing import Optional

from llm_router import LLMRouter, Priority

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "FastAPI is required for this example. "
        "Install with: pip install fastapi uvicorn"
    )

# ── Router Setup ──

router = LLMRouter()
router.add_model("smart", provider="ollama", model="llama3:8b", max_queue_depth=5)
router.add_model("fast", provider="ollama", model="qwen2.5:3b", max_queue_depth=3)
router.add_route(Priority.CRITICAL, targets=["smart"])
router.add_route(Priority.HIGH, targets=["smart"], fallback=["fast"])
router.add_route(Priority.NORMAL, targets=["fast"], fallback=["smart"])
router.add_route(Priority.LOW, targets=["fast"])
router.enable_circuit_breaker(failure_threshold=5, recovery_timeout=60)


# ── FastAPI App ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    await router.start()
    yield
    await router.stop()


app = FastAPI(title="LLM Router", lifespan=lifespan)

PRIORITY_MAP = {
    "critical": Priority.CRITICAL,
    "high": Priority.HIGH,
    "normal": Priority.NORMAL,
    "low": Priority.LOW,
}


class GenerateRequest(BaseModel):
    prompt: str
    priority: str = "normal"
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: Optional[str] = None


class GenerateResponse(BaseModel):
    content: str
    model_used: str
    success: bool
    latency_ms: float
    tokens_used: Optional[int] = None
    cached: bool = False
    error: Optional[str] = None


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    priority = PRIORITY_MAP.get(req.priority.lower(), Priority.NORMAL)
    response = await router.generate(
        prompt=req.prompt,
        priority=priority,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        system_prompt=req.system_prompt,
    )
    return GenerateResponse(
        content=response.content,
        model_used=response.model_used,
        success=response.success,
        latency_ms=response.latency_ms,
        tokens_used=response.tokens_used,
        cached=response.cached,
        error=response.error,
    )


@app.get("/stats")
async def stats():
    return router.get_stats()


@app.get("/models")
async def models():
    return router.get_model_status()


@app.get("/health")
async def health():
    return {"status": "ok", "models": len(router.get_model_status())}
