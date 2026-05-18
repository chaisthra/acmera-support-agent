"""
Self-contained retrieval layer for Project B.

Duplicated from Project A's rag.py so Project B is a standalone repo.
In Week 3, this gets replaced by the LangGraph agent's tool-based retrieval.

Not meant to be run directly — imported by support_pipeline.py and eval_harness.py.
"""
import os
import json
from openai import OpenAI
from langfuse.decorators import observe, langfuse_context
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

TOP_K = 5

INTENT_DOC_FILTERS = {
    "return_or_refund":   ["01_return_policy.md", "07_promotional_events.md",
                           "12_corporate_gifting.md", "04_warranty_policy.md"],
    "order_status":       ["03_shipping_policy.md", "06_support_faq.md"],
    "billing_or_payment": ["05_payment_methods.md", "13_acmera_wallet.md"],
    "product_info":       ["09_electronics_catalog.md", "04_warranty_policy.md",
                           "17_smart_home_ecosystem.md"],
    "membership":         ["02_premium_membership.md", "06_support_faq.md"],
    "general":            None,  # no filter
}


def get_connection():
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5432")),
        user=os.getenv("PG_USER", "workshop"),
        password=os.getenv("PG_PASSWORD"),
        dbname=os.getenv("PG_DATABASE", "acmera_kb"),
    )
    register_vector(conn)
    return conn


@observe(name="query_embedding")
def embed_query(query):
    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    return response.data[0].embedding


@observe(name="retrieval")
def retrieve(query_embedding, top_k=TOP_K):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, doc_name, chunk_index, content, metadata,
                  1 - (embedding <=> %s::vector) AS similarity
           FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s""",
        (query_embedding, query_embedding, top_k),
    )
    results = []
    for row in cur.fetchall():
        results.append({
            "id": row[0], "doc_name": row[1], "chunk_index": row[2],
            "content": row[3],
            "metadata": row[4] if isinstance(row[4], dict) else json.loads(row[4]),
            "similarity": round(float(row[5]), 4),
        })
    cur.close()
    conn.close()

    langfuse_context.update_current_observation(metadata={
        "top_k": top_k,
        "results": [{"doc_name": r["doc_name"], "chunk_index": r["chunk_index"],
                     "similarity": r["similarity"]} for r in results],
    })
    return results


@observe(name="retrieval_filtered")
def retrieve_filtered(query_embedding, intent: str, top_k=TOP_K):
    """Dense retrieval scoped to intent-relevant documents."""
    doc_filter = INTENT_DOC_FILTERS.get(intent)
    conn = get_connection()
    cur = conn.cursor()

    if doc_filter:
        cur.execute(
            """SELECT id, doc_name, chunk_index, content, metadata,
                      1 - (embedding <=> %s::vector) AS similarity
               FROM chunks
               WHERE doc_name = ANY(%s)
               ORDER BY embedding <=> %s::vector LIMIT %s""",
            (query_embedding, doc_filter, query_embedding, top_k),
        )
    else:
        cur.execute(
            """SELECT id, doc_name, chunk_index, content, metadata,
                      1 - (embedding <=> %s::vector) AS similarity
               FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s""",
            (query_embedding, query_embedding, top_k),
        )

    results = []
    for row in cur.fetchall():
        results.append({
            "id": row[0], "doc_name": row[1], "chunk_index": row[2],
            "content": row[3],
            "metadata": row[4] if isinstance(row[4], dict) else json.loads(row[4]),
            "similarity": round(float(row[5]), 4),
        })
    cur.close()
    conn.close()

    langfuse_context.update_current_observation(metadata={
        "intent": intent, "doc_filter": doc_filter, "top_k": top_k,
        "results": [{"doc_name": r["doc_name"], "similarity": r["similarity"]} for r in results],
    })
    return results


def deduplicate_chunks(chunks: list, threshold: float = 0.75) -> tuple[list, list]:
    """
    Remove near-duplicate chunks using word-level Jaccard similarity.
    Returns (unique_chunks, removed_log) where removed_log has Jaccard scores.
    """
    seen_words, seen_chunks, unique, removed_log = [], [], [], []
    for chunk in chunks:
        words = set(chunk["content"].lower().split())
        dup_score, dup_against = None, None
        for seen, seen_chunk in zip(seen_words, seen_chunks):
            if not words or not seen:
                continue
            jaccard = len(words & seen) / max(len(words | seen), 1)
            if jaccard >= threshold:
                dup_score, dup_against = jaccard, seen_chunk
                break
        if dup_score is not None:
            removed_log.append({
                "chunk": chunk,
                "jaccard": round(dup_score, 4),
                "duplicate_of": dup_against,
            })
        else:
            unique.append(chunk)
            seen_words.append(words)
            seen_chunks.append(chunk)
    return unique, removed_log


@observe(name="context_assembly")
def assemble_context(retrieved_chunks):
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Source: {chunk['doc_name']}, Chunk {chunk['chunk_index']}]\n{chunk['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    langfuse_context.update_current_observation(metadata={
        "num_chunks": len(retrieved_chunks),
        "total_context_chars": len(context),
    })
    return context


# ── Advanced retrieval stack ──────────────────────────────────────────────────

def _jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _deduplicate(chunks, threshold=0.75):
    kept = []
    for chunk in chunks:
        if not any(_jaccard(chunk["content"], k["content"]) >= threshold for k in kept):
            kept.append(chunk)
    return kept


def _expand_context(chunk, all_chunks, window=1):
    same_doc = sorted(
        [c for c in all_chunks if c["doc_name"] == chunk["doc_name"]],
        key=lambda c: c["chunk_index"],
    )
    indices = [c["chunk_index"] for c in same_doc]
    try:
        pos = indices.index(chunk["chunk_index"])
    except ValueError:
        return [chunk]
    return same_doc[max(0, pos - window): pos + window + 1]


def _compress(chunks, max_tokens=2000):
    def _score(c):
        return c.get("cohere_score") or c.get("similarity") or 0
    kept, total = [], 0
    for chunk in sorted(chunks, key=_score, reverse=True):
        tokens = int(len(chunk["content"].split()) * 1.3)
        if total + tokens > max_tokens:
            break
        kept.append(chunk)
        total += tokens
    return kept


def _order_by_source(chunks):
    from collections import defaultdict
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_name"]].append(c)
    ordered = []
    for doc in sorted(by_doc):
        ordered.extend(sorted(by_doc[doc], key=lambda c: c["chunk_index"]))
    return ordered


def _load_all_chunks():
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT id, doc_name, chunk_index, content, metadata FROM chunks")
    rows = cur.fetchall()
    cur.close(); conn.close()
    return [
        {"id": r[0], "doc_name": r[1], "chunk_index": r[2], "content": r[3],
         "metadata": r[4] if isinstance(r[4], dict) else json.loads(r[4])}
        for r in rows
    ]


class CohereReranker:
    MODEL = "rerank-v4.0-pro"

    def __init__(self):
        import cohere
        self.co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

    def rerank(self, query, chunks, top_k=5):
        import time
        if not chunks:
            return []
        time.sleep(6)  # trial key: 10 calls/min
        response = self.co.rerank(
            model=self.MODEL,
            query=query,
            documents=[c["content"] for c in chunks],
            top_n=top_k,
        )
        reranked = []
        for result in response.results:
            chunk = chunks[result.index].copy()
            chunk["cohere_score"] = round(result.relevance_score, 6)
            reranked.append(chunk)
        return reranked


@observe(name="retrieve_advanced")
def retrieve_advanced(query: str, intent: str, top_k: int = TOP_K) -> tuple[str, list]:
    """
    Full advanced retrieval pipeline:
    embed → filtered dense (top_k×2) → Cohere rerank → expand → deduplicate
    → compress → order_by_source → format context string.

    Falls back to basic filtered retrieval if COHERE_API_KEY is not set.
    Returns (context_string, reranked_chunks).
    """
    query_embedding = embed_query(query)
    candidates      = retrieve_filtered(query_embedding, intent, top_k=top_k * 2)

    cohere_key = os.getenv("COHERE_API_KEY")
    if cohere_key:
        try:
            reranked = CohereReranker().rerank(query, candidates, top_k=top_k)
        except Exception:
            reranked = candidates[:top_k]
    else:
        reranked = candidates[:top_k]

    all_chunks = _load_all_chunks()

    seen_ids, expanded = set(), []
    for chunk in _deduplicate(reranked):
        for neighbour in _expand_context(chunk, all_chunks, window=1):
            if neighbour["id"] not in seen_ids:
                n = neighbour.copy()
                if "cohere_score" in chunk and "cohere_score" not in n:
                    n["cohere_score"] = chunk["cohere_score"] * 0.9
                expanded.append(n)
                seen_ids.add(neighbour["id"])

    expanded   = _deduplicate(expanded)
    compressed = _compress(expanded, max_tokens=2000)
    ordered    = _order_by_source(compressed)

    parts = [
        f"[Source: {c['doc_name']}, Chunk {c['chunk_index']}]\n{c['content']}"
        for c in ordered
    ]
    context = "\n\n---\n\n".join(parts)

    langfuse_context.update_current_observation(metadata={
        "intent": intent, "candidates": len(candidates),
        "after_rerank": len(reranked), "after_expand": len(expanded),
        "after_compress": len(compressed), "cohere_used": bool(cohere_key),
    })
    return context, reranked
