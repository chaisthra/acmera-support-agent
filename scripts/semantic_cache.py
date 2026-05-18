"""
Semantic Cache for Project B — guardrail and pipeline response caching.

Two backends:
  SemanticCache       — in-memory (single process, lost on restart)
  RedisSemanticCache  — Redis-backed (shared across tasks, survives restarts)

get_semantic_cache()  — returns Redis version if REDIS_HOST is set, else in-memory.
"""
import os
import json
import uuid
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class SemanticCache:
    """In-memory semantic cache using cosine similarity on embeddings."""

    def __init__(self, threshold=0.92):
        self.threshold = threshold
        self.entries   = []  # list of {embedding, query, answer}

    def _cosine(self, a, b):
        a, b = np.array(a), np.array(b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(np.dot(a, b) / norm)

    def get(self, query_embedding):
        best_sim, best = 0.0, None
        for entry in self.entries:
            sim = self._cosine(query_embedding, entry["embedding"])
            if sim > best_sim:
                best_sim, best = sim, entry
        if best_sim >= self.threshold and best:
            return {
                "answer":           best["answer"],
                "query":            best["query"],
                "cache_similarity": round(best_sim, 4),
            }
        return None

    def set(self, query, embedding, answer):
        arr = np.array(embedding)
        arr = arr / np.linalg.norm(arr)
        self.entries.append({
            "query":     query,
            "embedding": arr.tolist(),
            "answer":    answer,
        })

    def size(self):
        return len(self.entries)


class RedisSemanticCache(SemanticCache):
    """
    Redis-backed semantic cache.

    Each entry is stored as a JSON blob under key:
        semantic_cache:<namespace>:<uuid>
    with a configurable TTL (default 3600s).

    On get(): SCAN all keys in the namespace, deserialize, compute cosine
    similarity in Python, return best hit above threshold.

    On set(): write new entry to Redis + keep local in-memory copy for
    fast repeated access within the same process.

    Falls back to in-memory SemanticCache if Redis is unreachable.
    """

    def __init__(self, threshold=0.92, namespace="default", ttl=3600):
        super().__init__(threshold=threshold)
        self.namespace = namespace
        self.ttl       = ttl
        self._redis    = self._connect()
        if self._redis:
            self._load_from_redis()

    def _connect(self):
        host = os.getenv("REDIS_HOST")
        if not host:
            return None
        try:
            import redis
            client = redis.Redis(
                host=host,
                port=int(os.getenv("REDIS_PORT", 6379)),
                decode_responses=True,
                socket_connect_timeout=1,
            )
            client.ping()
            return client
        except Exception:
            return None

    def _prefix(self):
        return f"semantic_cache:{self.namespace}:"

    def _load_from_redis(self):
        """Warm the in-memory store from Redis on startup."""
        try:
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match=f"{self._prefix()}*", count=100)
                for key in keys:
                    raw = self._redis.get(key)
                    if raw:
                        entry = json.loads(raw)
                        # avoid duplicates if already in memory
                        if not any(e["query"] == entry["query"] for e in self.entries):
                            self.entries.append(entry)
                if cursor == 0:
                    break
        except Exception:
            pass

    def set(self, query, embedding, answer):
        arr = np.array(embedding)
        arr = arr / np.linalg.norm(arr)
        entry = {
            "query":     query,
            "embedding": arr.tolist(),
            "answer":    answer,
        }
        self.entries.append(entry)
        if self._redis:
            try:
                key = f"{self._prefix()}{uuid.uuid4().hex}"
                self._redis.setex(key, self.ttl, json.dumps(entry))
            except Exception:
                pass


def get_semantic_cache(threshold=0.92, namespace="default", ttl=3600) -> SemanticCache:
    """
    Returns a RedisSemanticCache if REDIS_HOST is configured,
    otherwise falls back to in-memory SemanticCache.
    """
    if os.getenv("REDIS_HOST"):
        return RedisSemanticCache(threshold=threshold, namespace=namespace, ttl=ttl)
    return SemanticCache(threshold=threshold)
