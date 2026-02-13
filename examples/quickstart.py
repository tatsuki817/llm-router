"""
llm-router quickstart — Minimal example.

Prerequisites:
    pip install llm-router
    ollama pull qwen2.5:3b   # or any model you have
"""

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

        if response.success:
            print(f"Response: {response.content}")
            print(f"Model: {response.model_used}")
            print(f"Latency: {response.latency_ms:.0f}ms")
        else:
            print(f"Error: {response.error}")


if __name__ == "__main__":
    asyncio.run(main())
