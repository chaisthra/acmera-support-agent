"""
Project B — Week 3: 5-Node LangGraph Agent

Nodes:
  classify_node  — intent + tool selection
  tool_node      — executes policy_kb / order_tracker / account_lookup
  evaluate_node  — decides: respond | tool_call (need more data) | escalate
  respond_node   — generates final answer via LiteLLM with difficulty routing
  escalate_node  — returns structured escalation + ticket reference

Run:
  python scripts/agent.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
import litellm
from dotenv import load_dotenv

from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from query_classifier import classify_tool
from retrieval import embed_query, retrieve_filtered, deduplicate_chunks, assemble_context, retrieve_advanced
from mock_tools import run_order_tracker, run_account_lookup
from difficulty_classifier import route_model_llm
from semantic_cache import get_semantic_cache

load_dotenv()
litellm.set_verbose = False
langfuse = Langfuse()

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


_init_redis_cache = lambda: None  # LiteLLM Redis cache disabled — use semantic cache instead
_init_redis_cache()

_response_cache = get_semantic_cache(threshold=0.92, namespace="response", ttl=3600)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query:            str
    intent:           str
    tool:             str                            # routed tool name
    tool_results:     Annotated[list, operator.add]  # accumulates across calls
    tools_called:     Annotated[list, operator.add]  # tracks which tools ran
    final_answer:     str
    should_escalate:  bool
    steps_taken:      int
    next_action:      str                            # internal routing field
    difficulty_score: int                            # 1-5 from difficulty classifier
    generation_model: str                            # gpt-4o or gpt-4o-mini
    force_escalate:   bool
    use_cache:        bool                           # False → bypass LiteLLM Redis cache


# ---------------------------------------------------------------------------
# Node 1 — classify
# ---------------------------------------------------------------------------

@observe(name="classify_node")
def classify_node(state: AgentState) -> dict:
    routing = classify_tool(state["query"])
    model, diff_score, _ = route_model_llm(state["query"])
    out = {
        "intent":           routing["intent"],
        "tool":             routing["tool"],
        "difficulty_score": diff_score,
        "generation_model": model,
    }
    langfuse_context.update_current_observation(
        input=state["query"],
        output=out,
        metadata={"intent": routing["intent"], "tool": routing["tool"],
                  "difficulty_score": diff_score, "generation_model": model},
    )
    return out


# ---------------------------------------------------------------------------
# Node 2 — tool_call
# ---------------------------------------------------------------------------

def _policy_kb(query: str, intent: str) -> str:
    """Filtered dense → Cohere rerank → expand → compress → assemble."""
    context, _ = retrieve_advanced(query, intent)
    return context


ESCALATION_SIGNALS = [
    "someone else placed", "unauthorized access", "hacked", "compromised",
    "not me", "billing fraud", "wrong person", "logged into my account",
    "account deleted", "wallet disappeared", "balance gone",
    "i never ordered", "didn't place this order",
]

def _check_escalation_needed(query: str) -> bool:
    q = query.lower()
    return any(signal in q for signal in ESCALATION_SIGNALS)


@observe(name="tool_node")
def tool_node(state: AgentState) -> dict:
    query          = state["query"]
    tool           = state.get("tool", "policy_kb")
    already_called = set(state.get("tools_called", []))
    intent         = state.get("intent", "general")

    if tool == "multi_tool":
        if "order_tracker" not in already_called:
            result    = run_order_tracker(query)
            tool_name = "order_tracker"
        elif "account_lookup" not in already_called:
            result    = run_account_lookup(query)
            tool_name = "account_lookup"
        else:
            result    = _policy_kb(query, intent)
            tool_name = "policy_kb"
    elif tool == "order_tracker":
        result    = run_order_tracker(query)
        tool_name = "order_tracker"
    elif tool == "account_lookup":
        result    = run_account_lookup(query)
        tool_name = "account_lookup"
    else:
        result    = _policy_kb(query, intent)
        tool_name = "policy_kb"

    force_escalate = _check_escalation_needed(query)

    out = {
        "tool_results":   [f"[{tool_name}]\n{result}"],
        "tools_called":   [tool_name],
        "steps_taken":    state.get("steps_taken", 0) + 1,
        "force_escalate": force_escalate,
    }
    langfuse_context.update_current_observation(
        input={"query": query, "tool": tool_name, "step": state.get("steps_taken", 0) + 1},
        output={"result_preview": result[:300]},
        metadata={"tool_name": tool_name, "result_length": len(result)},
    )
    return out

# ---------------------------------------------------------------------------
# Node 3 — evaluate
# ---------------------------------------------------------------------------

EVALUATE_PROMPT = """You are evaluating a customer support agent's progress.

Query   : {query}
Intent  : {intent}
Tools called: {tools_called}
Steps   : {steps_taken}

Tool results so far:
{tool_results}

Decide the next action:
- "respond"   : Results are sufficient to answer the query fully
- "tool_call" : Need more data (e.g. multi-tool query only has 1 result so far)
- "escalate"  : Requires human intervention

Rules:
- steps_taken >= 3 → always "respond" (loop prevention)
- Escalate ONLY when the query or tool results EXPLICITLY indicate:
  * Account compromise or unauthorized access (customer explicitly states it)
  * Refund overdue beyond the stated timeline AND customer has already followed up
  * Customer uses threatening or abusive language
  * Legal action or regulatory complaint explicitly mentioned
  * Double charge or billing fraud confirmed in tool results
  * Customer explicitly states package is lost or stolen (not just undelivered or in transit)
- DO NOT escalate for: orders in "shipped" or "processing" status, missing delivery dates, standard policy questions, or any routine order status inquiry
- multi_tool intent with only 1 tool called so far → "tool_call"
- Otherwise, if results are substantive → "respond"

Respond ONLY with JSON: {{"action": "respond"|"tool_call"|"escalate", "reason": "one line"}}"""

@observe(name="evaluate_node")
def evaluate_node(state: AgentState) -> dict:
    steps        = state.get("steps_taken", 0)
    tools_called = state.get("tools_called", [])
    tool_results = state.get("tool_results", [])
    
    if state.get("force_escalate"):
        return {"next_action": "escalate", "should_escalate": True}

    if steps >= 3:
        langfuse_context.update_current_observation(
            input={"steps_taken": steps},
            output={"action": "respond", "reason": "step limit reached"},
            metadata={"short_circuit": True},
        )
        return {"next_action": "respond"}

    results_text = "\n\n".join(tool_results) if tool_results else "None"

    cache_opts = {} if state.get("use_cache", True) else {"cache": {"no-cache": True}}
    response = litellm.completion(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=80,
        messages=[{"role": "user", "content": EVALUATE_PROMPT.format(
            query=state["query"],
            intent=state.get("intent", "unknown"),
            tools_called=", ".join(tools_called),
            steps_taken=steps,
            tool_results=results_text[:2000],
        )}],
        **cache_opts,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    result = json.loads(raw)
    action = result.get("action", "respond")
    reason = result.get("reason", "")

    langfuse_context.update_current_observation(
        input={"query": state["query"], "tools_called": tools_called, "steps_taken": steps},
        output={"action": action, "reason": reason},
        metadata={"tools_called": tools_called, "steps_taken": steps},
    )
    return {
        "next_action":    action,
        "should_escalate": action == "escalate",
    }


def _route(state: AgentState) -> str:
    return state.get("next_action", "respond")


# ---------------------------------------------------------------------------
# Node 4 — respond
# ---------------------------------------------------------------------------

RESPOND_SYSTEM = """You are a customer support assistant for Acmera, an Indian e-commerce company.
Answer the customer's question using the tool results below. Be helpful, concise, and accurate.
Only use information from the provided results. If something is unclear, say so.

{tool_results}"""


@observe(name="respond_node")
def respond_node(state: AgentState) -> dict:
    model        = state.get("generation_model") or "gpt-4o-mini"
    tool_results = "\n\n---\n\n".join(state.get("tool_results", []))
    messages = [
        {"role": "system", "content": RESPOND_SYSTEM.format(tool_results=tool_results)},
        {"role": "user",   "content": state["query"]},
    ]

    cache_opts = {} if state.get("use_cache", True) else {"cache": {"no-cache": True}}
    response = litellm.completion(
        model=model,
        fallbacks=["gpt-3.5-turbo"],
        temperature=0.1,
        max_tokens=600,
        messages=messages,
        **cache_opts,
    )
    answer = response.choices[0].message.content
    langfuse_context.update_current_observation(
        input=messages,
        output=answer,
        metadata={"model": model, "model_used": response.model},
        usage={"input": response.usage.prompt_tokens,
               "output": response.usage.completion_tokens,
               "total": response.usage.total_tokens, "unit": "TOKENS"},
    )
    return {"final_answer": answer}


# ---------------------------------------------------------------------------
# Node 5 — escalate
# ---------------------------------------------------------------------------

@observe(name="escalate_node")
def escalate_node(state: AgentState) -> dict:
    ref = f"ESC-{abs(hash(state['query'])) % 100000:05d}"
    answer = (
        f"I'm escalating your query to our specialist team for human review.\n\n"
        f"Situation: {state['query']}\n\n"
        f"A support agent will contact you within 2 hours via your registered email.\n"
        f"Reference number: {ref}"
    )
    langfuse_context.update_current_observation(
        input={"query": state["query"], "intent": state.get("intent", "")},
        output={"escalation_ref": ref, "answer_preview": answer[:200]},
        metadata={"escalation_ref": ref},
    )
    return {"final_answer": answer, "should_escalate": True}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

graph = StateGraph(AgentState)
graph.add_node("classify",  classify_node)
graph.add_node("tool_call", tool_node)
graph.add_node("evaluate",  evaluate_node)
graph.add_node("respond",   respond_node)
graph.add_node("escalate",  escalate_node)

graph.set_entry_point("classify")
graph.add_edge("classify",  "tool_call")
graph.add_edge("tool_call", "evaluate")
graph.add_conditional_edges("evaluate", _route,
    {"respond": "respond", "tool_call": "tool_call", "escalate": "escalate"})
graph.add_edge("respond",  END)
graph.add_edge("escalate", END)

agent = graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@observe(name="langgraph_agent")
def run_agent(query: str, use_cache: bool = True) -> dict:
    """Run the agent and return state + trajectory. Traces to Langfuse."""
    import hashlib
    start = time.time()

    # ── Input guardrail ──────────────────────────────────────────────────────
    from input_guardrail import check_input
    guard = check_input(query)
    if not guard["safe"]:
        langfuse.flush()
        return {
            "query": query, "final_answer": guard["refusal"],
            "trajectory": ["blocked"], "intent": guard["category"],
            "tools_called": [], "steps_taken": 0, "difficulty_score": 0,
            "generation_model": "", "should_escalate": False,
            "context": "", "elapsed_seconds": round(time.time() - start, 2),
            "trace_id": None, "blocked": True,
            "block_category": guard["category"], "pii_redacted": False,
        }

    # ── PII anonymization ────────────────────────────────────────────────────
    from pii_anonymizer import PiiAnonymizer, redaction_audit_log
    anonymizer   = PiiAnonymizer()
    clean_query  = anonymizer.anonymize(query)
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
        metadata={"pipeline": "project_b_agent", "pii_redacted": pii_redacted},
    )

    # ── Semantic response cache ──────────────────────────────────────────────
    _query_embedding = embed_query(clean_query)
    if use_cache:
        hit = _response_cache.get(_query_embedding)
        if hit:
            cached_answer = anonymizer.restore(hit["answer"])
            langfuse.flush()
            return {
                "query": query, "final_answer": cached_answer,
                "trajectory": ["cache_hit"], "intent": "cached",
                "tools_called": [], "steps_taken": 0, "difficulty_score": 0,
                "generation_model": "", "should_escalate": False,
                "context": "", "elapsed_seconds": round(time.time() - start, 2),
                "trace_id": None, "blocked": False, "pii_redacted": pii_redacted,
                "ticket": None, "cache_hit": True, "cache_similarity": hit["cache_similarity"],
            }

    initial = {
        "query":            clean_query,
        "intent":           "",
        "tool":             "",
        "tool_results":     [],
        "tools_called":     [],
        "final_answer":     "",
        "should_escalate":  False,
        "steps_taken":      0,
        "next_action":      "",
        "difficulty_score": 0,
        "generation_model": "",
        "use_cache":        use_cache,
        "force_escalate":   False,
    }

    trajectory  = []
    final_state = {k: v for k, v in initial.items()}

    for step in agent.stream(initial):
        for node_name, node_output in step.items():
            trajectory.append(node_name)
            if isinstance(node_output, dict):
                for k, v in node_output.items():
                    if isinstance(v, list) and isinstance(final_state.get(k), list):
                        final_state[k] = final_state[k] + v
                    else:
                        final_state[k] = v

    final_answer     = final_state.get("final_answer", "")
    should_escalate  = final_state.get("should_escalate", False)
    context          = "\n\n---\n\n".join(final_state.get("tool_results", []))

    # ── Source guardrail (hallucination check) ───────────────────────────────
    from output_guardrail import check_hallucination, check_output
    if context and final_answer:
        hall = check_hallucination(final_answer, context)
        if hall.get("has_hallucination"):
            should_escalate = True

    # ── Output guardrail (PII scan on answer) ────────────────────────────────
    output_check = check_output(final_answer)
    if not output_check["safe"]:
        final_answer = output_check["redacted"]

    # ── Escalation ticket ────────────────────────────────────────────────────
    ticket = None
    if should_escalate:
        from support_ticket import generate_ticket
        try:
            t = generate_ticket(
                query=query,
                ai_response=final_answer,
                reason="Agent escalation — low confidence or hallucination detected",
            )
            ticket = t.model_dump()
        except Exception:
            pass

    if use_cache and not should_escalate:
        _response_cache.set(clean_query, _query_embedding, final_answer)

    # Restore original PII values in answer
    final_answer = anonymizer.restore(final_answer)

    elapsed = round(time.time() - start, 2)
    langfuse_context.update_current_trace(
        output=final_answer,
        metadata={
            "trajectory":       trajectory,
            "tools_called":     final_state.get("tools_called", []),
            "intent":           final_state.get("intent", ""),
            "difficulty_score": final_state.get("difficulty_score", 0),
            "generation_model": final_state.get("generation_model", ""),
            "should_escalate":  should_escalate,
            "pii_redacted":     pii_redacted,
            "elapsed_seconds":  elapsed,
        },
    )
    trace_id = langfuse_context.get_current_trace_id()
    langfuse.flush()

    return {
        "query":            query,
        "trajectory":       trajectory,
        "intent":           final_state.get("intent", ""),
        "tools_called":     final_state.get("tools_called", []),
        "steps_taken":      final_state.get("steps_taken", 0),
        "difficulty_score": final_state.get("difficulty_score", 0),
        "generation_model": final_state.get("generation_model", ""),
        "should_escalate":  should_escalate,
        "final_answer":     final_answer,
        "context":          context,
        "elapsed_seconds":  elapsed,
        "trace_id":         trace_id,
        "blocked":          False,
        "pii_redacted":     pii_redacted,
        "ticket":           ticket,
    }


# ---------------------------------------------------------------------------
# CLI — 3 test queries showing trajectory
# ---------------------------------------------------------------------------

TEST_CASES = [
    # (label, query)
    ("SIMPLE  (1 tool)",   "What is the return window for electronics?"),
    ("COMPLEX (2 tools)",  "I'm a Gold member and want to return my laptop from order ORD-445521 — what are my options?"),
    ("ESCALATION",         "Someone logged into my account without my permission and placed orders — I need urgent help."),
]


def _fmt_result(label: str, result: dict) -> list[str]:
    lines = [
        "",
        "=" * 70,
        f"  {label}",
        "=" * 70,
        f"  Query          : {result['query']}",
        f"  Trajectory     : {' → '.join(result['trajectory'])}",
        f"  Intent         : {result['intent']}",
        f"  Tools called   : {result['tools_called']}",
        f"  Steps taken    : {result['steps_taken']}",
        f"  Difficulty     : {result['difficulty_score']} → {result['generation_model']}",
        f"  Escalated      : {result['should_escalate']}",
        f"  Elapsed        : {result['elapsed_seconds']}s",
        f"  Langfuse trace : {result.get('trace_id', 'n/a')}",
        f"  Answer         : {result['final_answer'][:300]}{'...' if len(result['final_answer']) > 300 else ''}",
    ]
    return lines


if __name__ == "__main__":
    all_lines = ["# Agent Trajectories — 3 Test Queries", ""]
    all_results = []

    for label, query in TEST_CASES:
        print(f"\nRunning: {label}...")
        result = run_agent(query)
        lines  = _fmt_result(label, result)
        for l in lines:
            print(l)
        all_lines.extend(lines)
        all_results.append(result)

    print("\n" + "=" * 70)
    print("  TRAJECTORY SUMMARY")
    print("=" * 70)
    summary_lines = ["", "=" * 70, "TRAJECTORY SUMMARY", "=" * 70]
    for result in all_results:
        line = f"  {result['trajectory']}  →  tools={result['tools_called']}  steps={result['steps_taken']}"
        print(line)
        summary_lines.append(line)
    all_lines.extend(summary_lines)

    # Save proof
    os.makedirs(LOGS_DIR, exist_ok=True)

    md_path = os.path.join(LOGS_DIR, "agent_trajectories.md")
    with open(md_path, "w") as f:
        f.write("\n".join(all_lines) + "\n")
    print(f"\n[saved → logs/agent_trajectories.md]")

    json_path = os.path.join(LOGS_DIR, "agent_trajectories.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[saved → logs/agent_trajectories.json]")
