"""
llm-router: Semantic response cache using sentence-transformer embeddings.

Caches LLM responses and returns them for semantically similar future prompts.
Uses cosine similarity between prompt embeddings to determine cache hits.
Persists to SQLite for durability across restarts.

Requires the 'cache' extra: pip install llm-router[cache]

Features:
- Exact hash match (fastest path)
- Semantic similarity matching via sentence-transformers
- Configurable similarity threshold (default: 0.92)
- TTL-based expiration
- LRU eviction when max entries exceeded
- SQLite persistence (or in-memory for testing)
- Comprehensive hit/miss statistics
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from llm_router.models import LLMResponse

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_numpy = None
_sentence_transformers = None


def _import_numpy():  # type: ignore[return]
    global _numpy
    if _numpy is None:
        try:
            import numpy

            _numpy = numpy
        except ImportError as err:
            raise ImportError(
                "numpy is required for semantic caching. "
                "Install with: pip install llm-router[cache]"
            ) from err
    return _numpy


def _import_sentence_transformers():  # type: ignore[return]
    global _sentence_transformers
    if _sentence_transformers is None:
        try:
            import sentence_transformers

            _sentence_transformers = sentence_transformers
        except ImportError as err:
            raise ImportError(
                "sentence-transformers is required for semantic caching. "
                "Install with: pip install llm-router[cache]"
            ) from err
    return _sentence_transformers


@dataclass
class CacheStats:
    """Cache performance statistics."""

    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    total_entries: int = 0
    tokens_saved: int = 0
    avg_similarity: float = 0.0
    _similarity_count: int = field(default=0, repr=False)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction (0.0 to 1.0)."""
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries


class SemanticCache:
    """Semantic LLM response cache.

    Embeds prompts using a sentence-transformer model and uses cosine similarity
    to find cached responses for semantically similar queries.

    Lookup strategy:
    1. Exact hash match (fast path — SHA-256 of prompt text)
    2. Semantic similarity search across all cached embeddings
    3. Return cached response if similarity >= threshold

    Example::

        cache = SemanticCache(
            similarity_threshold=0.92,
            ttl_hours=24,
            max_entries=5000,
        )

        # Check cache
        hit = cache.lookup("What is Python?")
        if hit:
            response, similarity = hit
            print(f"Cache hit ({similarity:.2f}): {response.content}")
        else:
            # Cache miss — call LLM and store result
            response = await backend.generate(request, model)
            cache.store("What is Python?", response)
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        similarity_threshold: float = 0.92,
        ttl_hours: int = 24,
        max_entries: int = 10000,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Initialize the semantic cache.

        Args:
            db_path: SQLite database path. Use ":memory:" for in-memory cache.
            similarity_threshold: Minimum cosine similarity for a cache hit (0.0-1.0).
            ttl_hours: Time-to-live for cache entries in hours.
            max_entries: Maximum cache entries before LRU eviction.
            embedding_model: Sentence-transformer model for prompt embeddings.
        """
        self.similarity_threshold = similarity_threshold
        self.ttl_hours = ttl_hours
        self.max_entries = max_entries
        self._embedding_model_name = embedding_model
        self._model = None  # Lazy-loaded
        self._lock = threading.Lock()
        self.stats = CacheStats()

        # SQLite setup
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def _init_db(self) -> None:
        """Create the cache table if it doesn't exist."""
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_hash TEXT UNIQUE NOT NULL,
                prompt_text TEXT NOT NULL,
                response_content TEXT NOT NULL,
                model_used TEXT NOT NULL,
                embedding BLOB,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            )
        """)
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_hash ON cache(prompt_hash)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_accessed ON cache(last_accessed)"
        )
        self._db.commit()

    def _get_embedding_model(self) -> Any:
        """Lazily load the sentence-transformer model."""
        if self._model is None:
            st = _import_sentence_transformers()
            self._model = st.SentenceTransformer(self._embedding_model_name)
            logger.info(f"Loaded embedding model: {self._embedding_model_name}")
        return self._model

    def _embed(self, text: str) -> Any:
        """Compute the embedding vector for a text string."""
        np = _import_numpy()
        model = self._get_embedding_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return np.array(embedding, dtype=np.float32)

    @staticmethod
    def _cosine_similarity(a: Any, b: Any) -> float:
        """Compute cosine similarity between two vectors."""
        np = _import_numpy()
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    @staticmethod
    def _hash_prompt(prompt: str) -> str:
        """SHA-256 hash of the prompt text."""
        return hashlib.sha256(prompt.encode()).hexdigest()

    def lookup(
        self, prompt: str, metadata: dict[str, Any] | None = None
    ) -> tuple[LLMResponse, float] | None:
        """Look up a cached response for a prompt.

        Args:
            prompt: The prompt text to search for.
            metadata: Optional metadata for cache keying (not yet used, reserved).

        Returns:
            Tuple of (LLMResponse, similarity_score) if found, else None.
        """
        with self._lock:
            self.stats.total_queries += 1
            now = time.time()

            # Fast path: exact hash match
            prompt_hash = self._hash_prompt(prompt)
            row = self._db.execute(
                "SELECT response_content, model_used, created_at "
                "FROM cache WHERE prompt_hash = ?",
                (prompt_hash,),
            ).fetchone()

            if row:
                response_content, model_used, created_at = row
                age_hours = (now - created_at) / 3600
                if age_hours <= self.ttl_hours:
                    self._db.execute(
                        "UPDATE cache SET last_accessed = ?, access_count = access_count + 1 "
                        "WHERE prompt_hash = ?",
                        (now, prompt_hash),
                    )
                    self._db.commit()
                    self.stats.cache_hits += 1
                    self.stats.exact_hits += 1
                    return (
                        LLMResponse(
                            content=response_content,
                            model_used=f"cache:{model_used}",
                            success=True,
                            latency_ms=0.1,
                            cached=True,
                        ),
                        1.0,
                    )

            # Semantic similarity search
            try:
                query_embedding = self._embed(prompt)
            except ImportError:
                # sentence-transformers not installed — skip semantic search
                self.stats.cache_misses += 1
                return None

            rows = self._db.execute(
                "SELECT id, prompt_text, response_content, model_used, embedding, created_at "
                "FROM cache WHERE embedding IS NOT NULL"
            ).fetchall()

            np = _import_numpy()
            best_similarity = 0.0
            best_row = None

            for row in rows:
                row_id, _, _, _, embedding_blob, created_at = row
                age_hours = (now - created_at) / 3600
                if age_hours > self.ttl_hours:
                    continue

                cached_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                similarity = self._cosine_similarity(query_embedding, cached_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_row = row

            if best_row and best_similarity >= self.similarity_threshold:
                row_id, _, response_content, model_used, _, _ = best_row
                self._db.execute(
                    "UPDATE cache SET last_accessed = ?, access_count = access_count + 1 "
                    "WHERE id = ?",
                    (now, row_id),
                )
                self._db.commit()
                self.stats.cache_hits += 1
                self.stats.semantic_hits += 1
                self.stats._similarity_count += 1
                self.stats.avg_similarity += (
                    best_similarity - self.stats.avg_similarity
                ) / self.stats._similarity_count

                return (
                    LLMResponse(
                        content=response_content,
                        model_used=f"cache:{model_used}",
                        success=True,
                        latency_ms=0.1,
                        cached=True,
                    ),
                    best_similarity,
                )

            self.stats.cache_misses += 1
            return None

    def store(
        self,
        prompt: str,
        response: LLMResponse,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a response in the cache.

        Args:
            prompt: The original prompt text.
            response: The LLM response to cache.
            metadata: Optional metadata stored alongside the entry.
        """
        if not response.success or not response.content:
            return

        with self._lock:
            prompt_hash = self._hash_prompt(prompt)
            now = time.time()

            # Compute embedding
            embedding_blob = None
            try:
                embedding = self._embed(prompt)
                embedding_blob = embedding.tobytes()
            except ImportError:
                pass  # No sentence-transformers — store without embedding

            import json

            metadata_json = json.dumps(metadata or {})

            self._db.execute(
                "INSERT OR REPLACE INTO cache "
                "(prompt_hash, prompt_text, response_content, model_used, "
                "embedding, created_at, last_accessed, access_count, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)",
                (
                    prompt_hash,
                    prompt,
                    response.content,
                    response.model_used,
                    embedding_blob,
                    now,
                    now,
                    metadata_json,
                ),
            )
            self._db.commit()

            self.stats.total_entries = self._db.execute(
                "SELECT COUNT(*) FROM cache"
            ).fetchone()[0]

            # LRU eviction if over limit
            if self.stats.total_entries > self.max_entries:
                self._evict()

    def record_tokens_saved(self, tokens: int) -> None:
        """Record tokens saved by a cache hit."""
        with self._lock:
            self.stats.tokens_saved += tokens

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._db.execute("DELETE FROM cache")
            self._db.commit()
            self.stats.total_entries = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            return {
                "total_queries": self.stats.total_queries,
                "cache_hits": self.stats.cache_hits,
                "cache_misses": self.stats.cache_misses,
                "hit_rate": round(self.stats.hit_rate, 3),
                "exact_hits": self.stats.exact_hits,
                "semantic_hits": self.stats.semantic_hits,
                "total_entries": self.stats.total_entries,
                "tokens_saved": self.stats.tokens_saved,
                "avg_similarity": round(self.stats.avg_similarity, 3),
                "similarity_threshold": self.similarity_threshold,
            }

    def _evict(self) -> None:
        """Evict least-recently-accessed entries to stay under max_entries."""
        overflow = self.stats.total_entries - self.max_entries
        if overflow <= 0:
            return

        # Delete the oldest accessed entries
        self._db.execute(
            "DELETE FROM cache WHERE id IN ("
            "  SELECT id FROM cache ORDER BY last_accessed ASC LIMIT ?"
            ")",
            (overflow,),
        )
        self._db.commit()
        self.stats.total_entries = self._db.execute(
            "SELECT COUNT(*) FROM cache"
        ).fetchone()[0]
        logger.debug(f"Cache evicted {overflow} entries (LRU)")

    def close(self) -> None:
        """Close the SQLite connection."""
        self._db.close()
