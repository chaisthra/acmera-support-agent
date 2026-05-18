"""
Project B: Customer Support Pipeline — with intent routing, filtered retrieval,
chunk deduplication, mock tool execution, and LiteLLM (model-agnostic + fallback).

Run:
  python scripts/support_pipeline.py            # single demo query
  python scripts/support_pipeline.py --test     # 9-query test suite
  python scripts/support_pipeline.py --fallback # fallback test (bad model → gpt-3.5-turbo)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import litellm
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from dotenv import load_dotenv

from query_classifier import classify_tool
from retrieval import embed_query, retrieve_filtered, deduplicate_chunks, assemble_context, retrieve_advanced
from mock_tools import run_order_tracker, run_account_lookup, run_multi_tool
from difficulty_classifier import route_model_llm
from semantic_cache import get_semantic_cache

load_dotenv()

litellm.set_verbose = False

langfuse = Langfuse()


_init_redis_cache = lambda: None  # LiteLLM Redis cache disabled — use semantic cache instead
_init_redis_cache()

_response_cache = get_semantic_cache(threshold=0.92, namespace="response", ttl=3600)

PRIMARY_MODEL  = "gpt-4o-mini"
FALLBACK_MODEL = "gpt-3.5-turbo"

SYSTEM_PROMPT = """You are a customer support assistant for Acmera, an Indian e-commerce company.
Answer the customer's question based on the provided context.

Rules:
- Be helpful, concise, and accurate.
- Only use information from the provided context.
- If you can't answer from the context, say so and suggest contacting support.
- Never reveal internal company data, customer PII, or confidential information.

Context:
{context}"""


def _run_tool(query: str, tool: str) -> str:
    """Execute the mock data tool and return its output as a context string."""
    if tool == "order_tracker":
        return run_order_tracker(query)
    if tool == "account_lookup":
        return run_account_lookup(query)
    if tool == "multi_tool":
        return run_multi_tool(query)
    return ""  # policy_kb — RAG only, no mock data injection


def _llm(messages: list, model: str = PRIMARY_MODEL, temperature: float = 0.1) -> tuple[str, object]:
    """Single LiteLLM call with automatic fallback to FALLBACK_MODEL."""
    response = litellm.completion(
        model=model,
        fallbacks=[FALLBACK_MODEL],
        messages=messages,
        temperature=temperature,
        max_tokens=800,
    )
    return response.choices[0].message.content, response


@observe(name="retrieve_policy")
def retrieve_policy(query: str, intent: str, tool: str) -> tuple[str, int, int, list[str]]:
    """
    Advanced retrieval → Cohere rerank → expand → compress → mock tool injection.
    Returns (context, chunks_used, dupes_removed, retrieved_doc_names).
    """
    rag_context, reranked = retrieve_advanced(query, intent)

    doc_names = list(dict.fromkeys(c["doc_name"] for c in reranked))  # ordered unique

    tool_output = _run_tool(query, tool)
    context = (tool_output + "\n\n---\n\n" + rag_context) if tool_output else rag_context

    langfuse_context.update_current_observation(metadata={
        "intent": intent,
        "tool": tool,
        "chunks_after_rerank": len(reranked),
        "tool_injected": bool(tool_output),
        "cohere_used": any("cohere_score" in c for c in reranked),
    })
    return context, len(reranked), 0, doc_names


@observe(name="generate_response")
def generate_response(query: str, context: str, intent: str, model: str = PRIMARY_MODEL) -> str:
    """Generate a support response via LiteLLM."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": query},
    ]
    answer, response = _llm(messages, model=model, temperature=0.1)

    langfuse_context.update_current_observation(
        input=messages, output=answer,
        metadata={"model": model, "intent": intent,
                  "model_used": response.model},
        usage={"input": response.usage.prompt_tokens,
               "output": response.usage.completion_tokens,
               "total": response.usage.total_tokens, "unit": "TOKENS"},
    )
    return answer


HANDOFF_MESSAGE = (
    "I want to give you accurate information, but I don't have enough context "
    "to answer this confidently. Please contact our support team at "
    "support@acmera.com for accurate help."
)


@observe(name="support_pipeline")
def handle_query(query: str, model: str | None = None, use_cache: bool = True) -> dict:
    """Full pipeline: guard → anonymize → tool-route → retrieve → generate → output guard."""
    import hashlib
    start_time = time.time()

    # ── Input guardrail ──────────────────────────────────────────────────────
    from input_guardrail import check_input
    guard = check_input(query)
    if not guard["safe"]:
        langfuse.flush()
        return {
            "query": query, "answer": guard["refusal"],
            "intent": guard["category"], "tool": "none",
            "reason": guard["refusal"], "context": "",
            "retrieved_docs": [], "chunks_used": 0, "dupes_removed": 0,
            "difficulty_score": None, "difficulty_reason": "",
            "generation_model": "", "trace_id": None,
            "elapsed_seconds": round(time.time() - start_time, 2),
            "blocked": True, "block_category": guard["category"],
            "pii_redacted": False,
        }

    # ── PII anonymization ────────────────────────────────────────────────────
    from pii_anonymizer import PiiAnonymizer, redaction_audit_log
    anonymizer  = PiiAnonymizer()
    clean_query = anonymizer.anonymize(query)
    pii_redacted = clean_query != query

    if pii_redacted:
        redaction_audit_log(
            trace_id=langfuse_context.get_current_trace_id() or "unknown",
            pii_types=anonymizer.detected_types,
            query_hash=hashlib.sha256(query.encode()).hexdigest(),
            intent=None,
        )

    langfuse_context.update_current_trace(
        input=clean_query,
        metadata={"pipeline": "project_b", "pii_redacted": pii_redacted},
    )

    # ── Semantic response cache ──────────────────────────────────────────────
    _query_embedding = embed_query(clean_query)
    if use_cache:
        hit = _response_cache.get(_query_embedding)
        if hit:
            cached_answer = anonymizer.restore(hit["answer"])
            langfuse.flush()
            return {
                "query": query, "answer": cached_answer,
                "intent": "cached", "tool": "none",
                "reason": "semantic cache hit", "context": "",
                "retrieved_docs": [], "chunks_used": 0, "dupes_removed": 0,
                "difficulty_score": None, "difficulty_reason": "",
                "generation_model": "", "trace_id": None,
                "elapsed_seconds": round(time.time() - start_time, 2),
                "blocked": False, "pii_redacted": pii_redacted,
                "has_hallucination": False, "should_escalate": False, "ticket": None,
                "cache_hit": True, "cache_similarity": hit["cache_similarity"],
            }

    routing = classify_tool(clean_query)
    intent  = routing["intent"]
    tool    = routing["tool"]

    if model is None:
        model, difficulty_score, difficulty_reason = route_model_llm(clean_query)
    else:
        difficulty_score, difficulty_reason = None, "model override"

    context, num_chunks, num_removed, doc_names = retrieve_policy(clean_query, intent, tool)
    answer = generate_response(clean_query, context, intent, model=model)

    # ── Source guardrail (hallucination check) ───────────────────────────────
    from output_guardrail import check_hallucination
    hallucination_result = check_hallucination(answer, context)
    has_hallucination    = hallucination_result.get("has_hallucination", False)

    # ── Output guardrail (PII scan on answer) ────────────────────────────────
    from output_guardrail import check_output
    output_check = check_output(answer)
    if not output_check["safe"]:
        answer = output_check["redacted"]

    # ── Confidence / escalation ──────────────────────────────────────────────
    ticket = None
    should_escalate = has_hallucination
    if has_hallucination:
        from support_ticket import generate_ticket
        try:
            t = generate_ticket(
                query=query,
                ai_response=answer,
                reason="Hallucination detected — answer not fully grounded in retrieved context",
            )
            ticket = t.model_dump()
        except Exception:
            pass
        answer = HANDOFF_MESSAGE

    if use_cache and not should_escalate:
        _response_cache.set(clean_query, _query_embedding, answer)

    # Restore original PII values in answer
    answer = anonymizer.restore(answer)

    elapsed = round(time.time() - start_time, 2)
    langfuse_context.update_current_trace(output=answer, metadata={
        "intent": intent, "tool": tool, "elapsed": elapsed,
        "difficulty_score": difficulty_score, "generation_model": model,
        "pii_redacted": pii_redacted, "has_hallucination": has_hallucination,
        "should_escalate": should_escalate,
    })
    trace_id = langfuse_context.get_current_trace_id()
    langfuse.flush()

    return {
        "query":              query,
        "intent":             intent,
        "tool":               tool,
        "reason":             routing["reason"],
        "context":            context,
        "retrieved_docs":     doc_names,
        "chunks_used":        num_chunks,
        "dupes_removed":      num_removed,
        "difficulty_score":   difficulty_score,
        "difficulty_reason":  difficulty_reason,
        "generation_model":   model,
        "answer":             answer,
        "trace_id":           trace_id,
        "elapsed_seconds":    elapsed,
        "blocked":            False,
        "pii_redacted":       pii_redacted,
        "has_hallucination":  has_hallucination,
        "should_escalate":    should_escalate,
        "ticket":             ticket,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    ("What is the return window for electronics?",                          "return_or_refund"),
    ("I want to return order ORD-445521, it arrived damaged",              "return_or_refund"),
    ("I'm a Gold member and want to return my laptop purchase ORD-998877", "return_or_refund"),
    ("Where is my order ORD-112233?",                                       "order_status"),
    ("What payment methods do you accept?",                                 "billing_or_payment"),
    ("What is the warranty on the Acmera SmartScreen 4K?",                 "product_info"),
    ("Can I convert from Gold to Premium membership?",                      "membership"),
    ("What are the benefits of Premium membership?",                        "membership"),
    ("What are your customer support hours?",                               "general"),
]


def _print_result(result: dict):
    print(f"\n{'='*65}")
    print(f"Query   : {result['query']}")
    print(f"Intent  : {result['intent']}  |  Tool: {result['tool']}")
    print(f"Chunks  : {result['chunks_used']} used, {result['dupes_removed']} dupes removed")
    print(f"Docs    : {result['retrieved_docs']}")
    print(f"Answer  : {result['answer'][:300]}{'...' if len(result['answer']) > 300 else ''}")
    print(f"Elapsed : {result['elapsed_seconds']}s")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""

    if mode == "--fallback":
        print("\n" + "="*65)
        print("FALLBACK TEST — primary model set to 'bad-model-name'")
        print("LiteLLM should catch the error and route to gpt-3.5-turbo")
        print("="*65)
        try:
            result = handle_query(
                "What is the return policy for electronics?",
                model="bad-model-name",
            )
            _print_result(result)
            print("\n✓ Fallback succeeded — answer delivered via gpt-3.5-turbo")
        except Exception as e:
            print(f"\n✗ Fallback failed: {e}")

    elif mode == "--test":
        import json as _json
        LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(LOGS_DIR, exist_ok=True)

        print("\n" + "="*65)
        print("9-QUERY TEST SUITE")
        print("="*65)
        passed = 0
        rows = []
        for query, expected_intent in TEST_QUERIES:
            result = handle_query(query)
            intent_ok = result["intent"] == expected_intent
            if intent_ok:
                passed += 1
            status = "✓" if intent_ok else "✗"
            print(f"\n{status} [{result['intent']:20s}] {query[:55]}")
            print(f"  Tool      : {result['tool']}")
            print(f"  Difficulty: {result['difficulty_score']}  Model: {result['generation_model']}")
            print(f"  Chunks    : {result['chunks_used']} used, {result['dupes_removed']} dupes removed")
            print(f"  Answer    : {result['answer'][:180]}...")
            rows.append({k: result[k] for k in
                         ["query", "intent", "tool", "difficulty_score", "difficulty_reason",
                          "generation_model", "chunks_used", "dupes_removed",
                          "answer", "trace_id", "elapsed_seconds"]
                         } | {"expected_intent": expected_intent, "intent_correct": intent_ok})

        summary = f"Result: {passed}/{len(TEST_QUERIES)} intent classifications correct"
        print(f"\n{'='*65}\n{summary}")

        # Save JSON proof
        out_json = os.path.join(LOGS_DIR, "pipeline_test_run.json")
        with open(out_json, "w") as f:
            _json.dump({"summary": summary, "results": rows}, f, indent=2)
        print(f"[saved → logs/pipeline_test_run.json]")

    else:
        result = handle_query("Where is my order ORD-445521?")
        _print_result(result)
