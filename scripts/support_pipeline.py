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

load_dotenv()

litellm.set_verbose = False

langfuse = Langfuse()


def _init_redis_cache() -> None:
    host = os.getenv("REDIS_HOST")
    if not host:
        return
    try:
        litellm.cache = litellm.Cache(
            type="redis",
            host=host,
            port=int(os.getenv("REDIS_PORT", 6379)),
        )
    except Exception:
        pass


_init_redis_cache()

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


@observe(name="support_pipeline")
def handle_query(query: str, model: str | None = None) -> dict:
    """Full pipeline: tool-route → difficulty-route → filtered retrieval → mock tool → respond."""
    start_time = time.time()
    langfuse_context.update_current_trace(input=query, metadata={"pipeline": "project_b"})

    routing = classify_tool(query)
    intent  = routing["intent"]
    tool    = routing["tool"]

    # Difficulty routing — LLM picks gpt-4o or gpt-4o-mini for generation
    if model is None:
        model, difficulty_score, difficulty_reason = route_model_llm(query)
    else:
        difficulty_score, difficulty_reason = None, "model override"

    context, num_chunks, num_removed, doc_names = retrieve_policy(query, intent, tool)
    answer = generate_response(query, context, intent, model=model)

    elapsed = round(time.time() - start_time, 2)
    langfuse_context.update_current_trace(output=answer, metadata={
        "intent": intent, "tool": tool, "elapsed": elapsed,
        "difficulty_score": difficulty_score, "generation_model": model,
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
