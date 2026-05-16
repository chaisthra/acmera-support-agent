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
        port=os.getenv("PG_PORT", "5434"),
        user=os.getenv("PG_USER", "workshop"),
        password=os.getenv("PG_PASSWORD", "workshop123"),
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
