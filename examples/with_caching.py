"""
llm-router semantic caching example.

Demonstrates how semantic caching avoids redundant LLM calls for
similar prompts. The cache uses sentence-transformer embeddings to
match semantically equivalent queries.

Prerequisites:
    pip install llm-router[cache]
    ollama pull qwen2.5:3b
"""

import asyncio
from llm_router import LLMRouter, Priority


async def main():
    router = LLMRouter()
    router.add_model("local", provider="ollama", model="qwen2.5:3b")

    # Enable semantic caching
    # - Responses with temperature <= 0.3 are cached
    # - Cache hit if cosine similarity >= 0.92
    # - Entries expire after 24 hours
    router.enable_cache(
        similarity_threshold=0.92,
        ttl_hours=24,
        max_entries=5000,
    )

    async with router:
        # First call — cache miss, calls the LLM
        r1 = await router.generate(
            "What is the capital of France?",
            temperature=0.1,  # Low temperature = cacheable
        )
        print(f"First call:  {r1.content[:60]}...")
        print(f"  Cached: {r1.cached}, Latency: {r1.latency_ms:.0f}ms")

        # Second call — exact same prompt, cache hit
        r2 = await router.generate(
            "What is the capital of France?",
            temperature=0.1,
        )
        print(f"\nSecond call: {r2.content[:60]}...")
        print(f"  Cached: {r2.cached}, Latency: {r2.latency_ms:.1f}ms")

        # Third call — semantically similar, should also hit cache
        r3 = await router.generate(
            "Tell me the capital city of France",
            temperature=0.1,
        )
        print(f"\nSimilar call: {r3.content[:60]}...")
        print(f"  Cached: {r3.cached}, Latency: {r3.latency_ms:.1f}ms")

        # High temperature = not cached
        r4 = await router.generate(
            "What is the capital of France?",
            temperature=0.9,
        )
        print(f"\nHigh temp:   {r4.content[:60]}...")
        print(f"  Cached: {r4.cached}, Latency: {r4.latency_ms:.0f}ms")

        # Cache stats
        stats = router.get_stats()
        print(f"\nCache stats: {stats.get('cache', {})}")


if __name__ == "__main__":
    asyncio.run(main())
