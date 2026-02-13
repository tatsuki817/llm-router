"""
llm-router multi-model routing — GPU + CPU model setup.

Routes critical requests to the powerful GPU model and low-priority
background tasks to the lightweight CPU model.

Prerequisites:
    pip install llm-router
    ollama pull llama3:8b     # GPU model
    ollama pull qwen2.5:3b    # CPU model (run with OLLAMA_NUM_GPU=0)
"""

import asyncio
from llm_router import LLMRouter, Priority


async def main():
    router = LLMRouter()

    # Powerful model for important tasks
    router.add_model(
        "gpu-model",
        provider="ollama",
        model="llama3:8b",
        max_queue_depth=5,
    )

    # Lightweight model for background tasks
    router.add_model(
        "cpu-model",
        provider="ollama",
        model="qwen2.5:3b",
        max_queue_depth=3,
        # Force CPU-only execution in Ollama
        num_gpu=0,
        num_thread=4,
    )

    # Route critical/high to GPU, normal/low to CPU with GPU fallback
    router.add_route(Priority.CRITICAL, targets=["gpu-model"])
    router.add_route(Priority.HIGH, targets=["gpu-model"], fallback=["cpu-model"])
    router.add_route(Priority.NORMAL, targets=["cpu-model"], fallback=["gpu-model"])
    router.add_route(Priority.LOW, targets=["cpu-model"])

    # Enable fault tolerance
    router.enable_circuit_breaker(failure_threshold=3, recovery_timeout=30)

    async with router:
        # Critical task → GPU model
        critical = await router.generate(
            "Analyze this financial data and make a trading recommendation",
            priority=Priority.CRITICAL,
            temperature=0.1,
        )
        print(f"[CRITICAL] Model: {critical.model_used}, Latency: {critical.latency_ms:.0f}ms")

        # Low-priority task → CPU model
        low = await router.generate(
            "Write a haiku about Python programming",
            priority=Priority.LOW,
            temperature=0.9,
        )
        print(f"[LOW] Model: {low.model_used}, Latency: {low.latency_ms:.0f}ms")

        # Check stats
        print(f"\nRouter stats: {router.get_stats()}")
        print(f"\nModel status: {router.get_model_status()}")


if __name__ == "__main__":
    asyncio.run(main())
