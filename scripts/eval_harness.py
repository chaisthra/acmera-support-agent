"""
Project B Evaluation Harness — Session 2 + Week 3 Agent Eval

4-dimensional eval:
  1. check_classification()    — did it identify the right intent?
  2. check_retrieval_hit()     — did expected source doc appear in retrieved chunks?
  3. judge_faithfulness()      — is the answer grounded in context?
  4. judge_correctness()       — does it match the expected answer?
  5. check_routing()           — should this have been escalated?

Session 2 additions:
  6. run_stratified_eval()     — breakdown by intent and difficulty
  7. attach_langfuse_scores()  — attach all dimensions to Langfuse traces
  8. save_baseline()           — lock current scores as regression anchor

Week 3 additions:
  9. run_agent_eval()          — same 5 dimensions against LangGraph agent
                                 records steps_taken and trajectory per query

Flags:
  --agent            run eval against LangGraph agent instead of naive pipeline
  --save-baseline    save scores to baseline_scores.json after eval
  --no-langfuse      skip Langfuse score attachment (faster, no network)
  --category <name>  run eval only on entries matching that category

Run:
  python scripts/eval_harness.py              # naive pipeline
  python scripts/eval_harness.py --agent      # LangGraph agent + comparison
"""
import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import litellm
from dotenv import load_dotenv

load_dotenv()

litellm.set_verbose = False

SCRIPT_DIR = os.path.dirname(__file__)


# =========================================================================
# GOLDEN DATASET
# =========================================================================

def load_golden_dataset(category_filter: str | None = None) -> list:
    path = os.path.join(SCRIPT_DIR, "golden_dataset.json")
    if not os.path.exists(path):
        print("No golden_dataset.json found for Project B.")
        return []
    with open(path) as f:
        data = json.load(f)
    if category_filter:
        data = [e for e in data if e.get("category") == category_filter]
        print(f"Category filter '{category_filter}': {len(data)} entries")
    return data


# =========================================================================
# DIMENSION 1: CLASSIFICATION
# =========================================================================

def check_classification(predicted_intent: str, expected_intent: str) -> bool:
    return predicted_intent == expected_intent


# =========================================================================
# DIMENSION 2: RETRIEVAL HIT
# =========================================================================

def check_retrieval_hit(retrieved_docs: list[str], expected_source: str) -> bool:
    """Did the expected source document appear anywhere in the retrieved set?"""
    return expected_source in retrieved_docs


# =========================================================================
# DIMENSION 3: ROUTING
# =========================================================================

def check_routing(predicted_escalation: bool, expected_escalation: bool) -> bool:
    return predicted_escalation == expected_escalation


# =========================================================================
# DIMENSION 4 & 5: GENERATION QUALITY
# =========================================================================

def judge_faithfulness(query: str, answer: str, context: str) -> dict:
    """LLM-as-judge: Is the answer grounded in context? Returns {score, reason}."""
    prompt = f"""You are an evaluation judge. Score whether the answer is grounded in the provided context.

Rubric:
- Score 5: Every claim in the answer is explicitly supported by the context.
- Score 4: Almost all claims supported; minor unsupported details.
- Score 3: Some claims are supported but others are not in the context.
- Score 2: Most claims are not supported by the context.
- Score 1: Answer contains fabricated information not present in the context.

Question: {query}

Context:
{context}

Answer:
{answer}

Respond with JSON only, no markdown fences:
{{"score": <1-5>, "reason": "<one sentence explanation>"}}"""

    response = litellm.completion(
        model="gpt-4o-mini",
        fallbacks=["gpt-3.5-turbo"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def judge_correctness(query: str, answer: str, expected_answer: str) -> dict:
    """LLM-as-judge: Does the answer match the expected answer? Returns {score, reason}."""
    prompt = f"""You are an evaluation judge. Score whether the generated answer correctly addresses the question compared to the expected answer.

Rubric:
- Score 5: Generated answer is fully correct and covers all key points of the expected answer.
- Score 4: Mostly correct with minor omissions or imprecise details.
- Score 3: Partially correct — captures some key points but misses others.
- Score 2: Mostly incorrect or significantly incomplete.
- Score 1: Wrong or completely unrelated to the expected answer.

Question: {query}

Expected answer: {expected_answer}

Generated answer: {answer}

Respond with JSON only, no markdown fences:
{{"score": <1-5>, "reason": "<one sentence explanation>"}}"""

    response = litellm.completion(
        model="gpt-4o-mini",
        fallbacks=["gpt-3.5-turbo"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


# =========================================================================
# EVAL RUNNER
# =========================================================================

def run_eval(category_filter: str | None = None) -> list:
    """
    Run 5-dimensional eval:
    1. Classification accuracy
    2. Retrieval hit rate
    3. Faithfulness
    4. Correctness
    5. Routing (escalation detection) — naive pipeline = 0% on should-escalate cases

    Naive pipeline: predicted_escalation is always False.
    Routing on should-escalate cases = 0% (Week 4 target to improve).
    """
    from support_pipeline import handle_query
    # _init_redis_cache() fires on import; disable it — eval is synchronous,
    # there is no running event loop, and async_set_cache raises RuntimeError.
    litellm.cache = None

    dataset = load_golden_dataset(category_filter)
    if not dataset:
        return []

    print(f"\nRunning Project B eval on {len(dataset)} queries...\n")

    results = []
    for entry in dataset:
        print(f"  [{entry['id']}] {entry['query'][:60]}...")
        try:
            pipeline_result = handle_query(entry["query"])

            predicted_escalation = False  # naive pipeline never escalates
            context       = pipeline_result["context"]
            retrieved_docs = pipeline_result.get("retrieved_docs", [])

            classification_correct = check_classification(pipeline_result["intent"], entry["expected_intent"])
            retrieval_hit          = check_retrieval_hit(retrieved_docs, entry.get("expected_source", ""))
            routing_correct        = check_routing(predicted_escalation, entry["expected_escalation"])
            faith                  = judge_faithfulness(entry["query"], pipeline_result["answer"], context)
            correct                = judge_correctness(entry["query"], pipeline_result["answer"], entry["expected_answer"])

            print(f"         class={classification_correct}  hit={retrieval_hit}  "
                  f"routing={routing_correct}  faith={faith['score']}  correct={correct['score']}")

            results.append({
                "id":                    entry["id"],
                "query":                 entry["query"],
                "category":              entry.get("category", "unknown"),
                "difficulty":            entry.get("difficulty", "easy"),
                "expected_intent":       entry["expected_intent"],
                "predicted_intent":      pipeline_result["intent"],
                "expected_escalation":   entry["expected_escalation"],
                "predicted_escalation":  predicted_escalation,
                "expected_source":       entry.get("expected_source", ""),
                "retrieved_docs":        retrieved_docs,
                "expected_answer":       entry["expected_answer"],
                "answer":                pipeline_result["answer"],
                "trace_id":              pipeline_result["trace_id"],
                "elapsed_seconds":       pipeline_result["elapsed_seconds"],
                "classification_correct": classification_correct,
                "retrieval_hit":         retrieval_hit,
                "routing_correct":       routing_correct,
                "faithfulness":          faith,
                "correctness":           correct,
            })
        except Exception as e:
            print(f"         ERROR — skipping: {e}")
            continue

    _print_scorecard(results)

    out_path = os.path.join(SCRIPT_DIR, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def _print_scorecard(results: list, label: str = "PROJECT B SCORECARD"):
    n = len(results)
    if n == 0:
        return

    classification_acc = sum(1 for r in results if r["classification_correct"]) / n

    hit_results = [r for r in results if r.get("retrieval_hit") is not None]
    retrieval_hit_rate = (
        sum(1 for r in hit_results if r["retrieval_hit"]) / len(hit_results)
        if hit_results else None
    )

    should_escalate = [r for r in results if r["expected_escalation"]]
    should_handle   = [r for r in results if not r["expected_escalation"]]

    missed_esc_caught = (
        sum(1 for r in should_escalate if r["routing_correct"]) / len(should_escalate)
        if should_escalate else float("nan")
    )
    correct_handle = (
        sum(1 for r in should_handle if r["routing_correct"]) / len(should_handle)
        if should_handle else float("nan")
    )
    overall_routing = sum(1 for r in results if r["routing_correct"]) / n

    avg_faith   = sum(r["faithfulness"]["score"] for r in results) / n
    avg_correct = sum(r["correctness"]["score"] for r in results) / n

    print()
    print("=" * 55)
    print(f"  {label}")
    print("=" * 55)
    print(f"  Queries evaluated       : {n}")
    print()
    print(f"  [1] Classification acc  : {classification_acc:.0%}")
    if retrieval_hit_rate is not None:
        print(f"  [2] Retrieval hit rate  : {retrieval_hit_rate:.0%}")
    else:
        print(f"  [2] Retrieval hit rate  : N/A (not tracked at agent level)")
    print()
    print(f"  [3] Routing accuracy")
    print(f"      overall             : {overall_routing:.0%}")
    print(f"      correct-handle      : {correct_handle:.0%}"
          f"  ({len(should_handle)} queries expected_escalation=False)")
    print(f"      missed-esc caught   : {missed_esc_caught:.0%}"
          f"  ({len(should_escalate)} queries expected_escalation=True)")
    print()
    print(f"  [4] Avg faithfulness    : {avg_faith:.2f} / 5")
    print(f"  [5] Avg correctness     : {avg_correct:.2f} / 5")
    print("=" * 55)

    return {
        "n": n,
        "classification_acc": classification_acc,
        "retrieval_hit_rate": retrieval_hit_rate,
        "routing_overall": overall_routing,
        "routing_correct_handle": correct_handle,
        "routing_missed_esc_caught": missed_esc_caught,
        "avg_faithfulness": avg_faith,
        "avg_correctness": avg_correct,
    }


# =========================================================================
# AGENT EVAL
# =========================================================================

def run_agent_eval(category_filter: str | None = None) -> list:
    """
    Run 5-dimensional eval against the LangGraph agent.
    Key differences vs naive pipeline:
    - predicted_escalation comes from agent's should_escalate (not hardcoded False)
    - steps_taken and trajectory recorded per query
    - retrieval_hit is None (doc names not surfaced at agent level)
    """
    from agent import run_agent
    # same as run_eval — kill the async Redis cache before any completions fire
    litellm.cache = None

    dataset = load_golden_dataset(category_filter)
    if not dataset:
        return []

    print(f"\nRunning AGENT eval on {len(dataset)} queries...\n")

    results = []
    for entry in dataset:
        print(f"  [{entry['id']}] {entry['query'][:60]}...")
        try:
            agent_result = run_agent(entry["query"])

            predicted_escalation = agent_result["should_escalate"]
            answer  = agent_result["final_answer"]
            context = agent_result.get("context") or answer

            classification_correct = check_classification(agent_result["intent"], entry["expected_intent"])
            routing_correct        = check_routing(predicted_escalation, entry["expected_escalation"])
            faith                  = judge_faithfulness(entry["query"], answer, context)
            correct                = judge_correctness(entry["query"], answer, entry["expected_answer"])

            print(f"         class={classification_correct}  routing={routing_correct}"
                  f"  esc={predicted_escalation}  steps={agent_result['steps_taken']}"
                  f"  faith={faith['score']}  correct={correct['score']}"
                  f"  traj={agent_result['trajectory']}")

            results.append({
                "id":                     entry["id"],
                "query":                  entry["query"],
                "category":               entry.get("category", "unknown"),
                "difficulty":             entry.get("difficulty", "easy"),
                "expected_intent":        entry["expected_intent"],
                "predicted_intent":       agent_result["intent"],
                "expected_escalation":    entry["expected_escalation"],
                "predicted_escalation":   predicted_escalation,
                "expected_source":        entry.get("expected_source", ""),
                "retrieved_docs":         [],          # not tracked at agent level
                "expected_answer":        entry["expected_answer"],
                "answer":                 answer,
                "trace_id":               agent_result.get("trace_id", ""),
                "elapsed_seconds":        agent_result["elapsed_seconds"],
                "steps_taken":            agent_result["steps_taken"],
                "tools_called":           agent_result["tools_called"],
                "trajectory":             agent_result["trajectory"],
                "difficulty_score":       agent_result.get("difficulty_score", 0),
                "generation_model":       agent_result.get("generation_model", ""),
                "classification_correct": classification_correct,
                "retrieval_hit":          None,        # N/A for agent
                "routing_correct":        routing_correct,
                "faithfulness":           faith,
                "correctness":            correct,
            })
        except Exception as e:
            import traceback
            print(f"         ERROR — skipping: {e}")
            traceback.print_exc()
            continue

    scores = _print_scorecard(results, label="LANGGRAPH AGENT SCORECARD")
    _print_steps_distribution(results)

    baseline = _load_baseline()
    if baseline and scores:
        _print_comparison(baseline, scores)

    LOGS_DIR = os.path.join(SCRIPT_DIR, "..", "logs")
    os.makedirs(LOGS_DIR, exist_ok=True)
    out_path = os.path.join(LOGS_DIR, "agent_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAgent eval results saved to {out_path}")

    return results


def _print_steps_distribution(results: list):
    """Print how many queries needed 1 / 2 / 3 tool-call steps."""
    n = len(results)
    if n == 0:
        return

    from collections import Counter
    dist = Counter(r["steps_taken"] for r in results)

    print()
    print("=" * 55)
    print("  STEPS TAKEN DISTRIBUTION")
    print("=" * 55)
    for steps in sorted(dist):
        count = dist[steps]
        bar   = "█" * count
        label = {1: "single tool call", 2: "multi-tool / 2nd eval", 3: "step cap hit"}.get(steps, f"{steps} steps")
        print(f"  {steps} step{'s' if steps != 1 else ''} : {count:>3} queries ({count/n:.0%})  {label}")
        # show which queries hit each level
        qs = [r["id"] for r in results if r["steps_taken"] == steps]
        print(f"           {', '.join(qs)}")

    avg_steps = sum(r["steps_taken"] for r in results) / n
    print(f"\n  Avg steps per query: {avg_steps:.2f}")
    print("=" * 55)


def _load_baseline() -> dict | None:
    path = os.path.join(SCRIPT_DIR, "baseline_scores.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _print_comparison(baseline: dict, agent: dict):
    """Side-by-side comparison table: naive pipeline baseline vs LangGraph agent."""

    def pct(v):
        if v is None or (isinstance(v, float) and v != v):
            return "  N/A "
        return f"{v:.0%}"

    def delta(a, b):
        if a is None or b is None:
            return "  —  "
        if isinstance(a, float) and a != a:
            return "  —  "
        if isinstance(b, float) and b != b:
            return "  —  "
        d = b - a
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.0%}"

    def score5(v):
        if v is None:
            return "  N/A "
        return f"{v:.2f}/5"

    def delta5(a, b):
        if a is None or b is None:
            return "  —  "
        d = b - a
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.2f}"

    b_esc  = baseline.get("routing_missed_esc_caught")
    b_hdl  = baseline.get("routing_correct_handle")
    a_esc  = agent.get("routing_missed_esc_caught")
    a_hdl  = agent.get("routing_correct_handle")

    print()
    print("=" * 70)
    print("  COMPARISON: NAIVE PIPELINE vs LANGGRAPH AGENT")
    print("=" * 70)
    print(f"  {'Dimension':<30} {'Naive':>8}  {'Agent':>8}  {'Delta':>8}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Classification acc':<30} {pct(baseline.get('classification_acc')):>8}  "
          f"{pct(agent.get('classification_acc')):>8}  "
          f"{delta(baseline.get('classification_acc'), agent.get('classification_acc')):>8}")
    print(f"  {'Retrieval hit rate':<30} {pct(baseline.get('retrieval_hit_rate')):>8}  "
          f"{'N/A':>8}  {'—':>8}")
    print(f"  {'Routing overall':<30} {pct(baseline.get('routing_overall')):>8}  "
          f"{pct(agent.get('routing_overall')):>8}  "
          f"{delta(baseline.get('routing_overall'), agent.get('routing_overall')):>8}")
    print(f"  {'  → missed-esc caught':<30} {pct(b_esc):>8}  "
          f"{pct(a_esc):>8}  "
          f"{delta(b_esc, a_esc):>8}   ← KEY METRIC")
    print(f"  {'  → correct-handle':<30} {pct(b_hdl):>8}  "
          f"{pct(a_hdl):>8}  "
          f"{delta(b_hdl, a_hdl):>8}")
    print(f"  {'Avg faithfulness':<30} {score5(baseline.get('avg_faithfulness')):>8}  "
          f"{score5(agent.get('avg_faithfulness')):>8}  "
          f"{delta5(baseline.get('avg_faithfulness'), agent.get('avg_faithfulness')):>8}")
    print(f"  {'Avg correctness':<30} {score5(baseline.get('avg_correctness')):>8}  "
          f"{score5(agent.get('avg_correctness')):>8}  "
          f"{delta5(baseline.get('avg_correctness'), agent.get('avg_correctness')):>8}")
    print("=" * 70)
    if a_esc is not None and b_esc is not None and a_esc > b_esc:
        pct_gain = a_esc - b_esc
        print(f"\n  Routing improvement: missed-esc caught went from {pct(b_esc)} → {pct(a_esc)} (+{pct_gain:.0%})")


# =========================================================================
# STRATIFIED EVAL
# =========================================================================

def run_stratified_eval(results: list):
    """Break down scores by intent (classification accuracy) and difficulty (correctness)."""
    from collections import defaultdict

    intent_buckets = defaultdict(list)
    for r in results:
        intent_buckets[r["expected_intent"]].append(r)

    print()
    print("=" * 65)
    print("  STRATIFIED — Classification accuracy by intent")
    print("=" * 65)
    print(f"  {'Intent':<22} {'Total':>5}  {'Correct':>7}  {'Acc':>5}  Misclassified as")
    print(f"  {'-'*22}  {'-'*5}  {'-'*7}  {'-'*5}  ----------------")

    intent_rows = []
    for intent, rows in sorted(intent_buckets.items()):
        total   = len(rows)
        correct = sum(1 for r in rows if r["classification_correct"])
        acc     = correct / total
        wrong   = [r["predicted_intent"] for r in rows if not r["classification_correct"]]
        wrong_str = ", ".join(wrong) if wrong else "—"
        intent_rows.append((acc, intent, total, correct, wrong_str))

    min_acc = min(a for a, *_ in intent_rows) if intent_rows else 0
    for acc, intent, total, correct, wrong_str in sorted(intent_rows):
        flag = " ◄ worst" if acc == min_acc else ""
        print(f"  {intent:<22} {total:>5}  {correct:>7}  {acc:>4.0%}  {wrong_str}{flag}")

    print()
    if intent_rows:
        worst = min(intent_rows, key=lambda x: x[0])
        print(f"  Worst intent: '{worst[1]}' at {worst[0]:.0%} accuracy")

    diff_buckets = defaultdict(list)
    for r in results:
        diff_buckets[r["difficulty"]].append(r)

    print()
    print("=" * 65)
    print("  STRATIFIED — Avg correctness by difficulty")
    print("=" * 65)
    print(f"  {'Difficulty':<12} {'Total':>5}  {'Avg Correct':>11}  {'Retrieval Hit':>13}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*11}  {'-'*13}")
    for diff in ["easy", "medium", "hard"]:
        if diff not in diff_buckets:
            continue
        rows    = diff_buckets[diff]
        avg_c   = sum(r["correctness"]["score"] for r in rows) / len(rows)
        hit_pct = sum(1 for r in rows if r["retrieval_hit"]) / len(rows)
        print(f"  {diff:<12} {len(rows):>5}  {avg_c:>10.2f}/5  {hit_pct:>12.0%}")

    print("=" * 65)

    return intent_rows


# =========================================================================
# LANGFUSE SCORE ATTACHMENT
# =========================================================================

def attach_langfuse_scores(results: list):
    """Attach all 5 eval dimensions to Langfuse traces."""
    try:
        from langfuse import Langfuse
        lf = Langfuse()
    except Exception as e:
        print(f"  Langfuse unavailable — skipping score attachment: {e}")
        return

    attached = 0
    for r in results:
        trace_id = r.get("trace_id")
        if not trace_id:
            continue
        try:
            lf.score(trace_id=trace_id, name="classification_correct",
                     value=1.0 if r["classification_correct"] else 0.0)
            if r.get("retrieval_hit") is not None:
                lf.score(trace_id=trace_id, name="retrieval_hit",
                         value=1.0 if r["retrieval_hit"] else 0.0)
            lf.score(trace_id=trace_id, name="routing_correct",
                     value=1.0 if r["routing_correct"] else 0.0)
            lf.score(trace_id=trace_id, name="faithfulness",
                     value=r["faithfulness"]["score"] / 5.0,
                     comment=r["faithfulness"].get("reason", ""))
            lf.score(trace_id=trace_id, name="correctness",
                     value=r["correctness"]["score"] / 5.0,
                     comment=r["correctness"].get("reason", ""))
            attached += 1
        except Exception as e:
            print(f"  Score attach failed for {trace_id}: {e}")

    lf.flush()
    print(f"  Attached scores to {attached}/{len(results)} traces in Langfuse")


# =========================================================================
# SAVE BASELINE
# =========================================================================

def save_baseline(results: list):
    """Save current scores as baseline_scores.json — the Week 4 regression anchor."""
    if not results:
        return

    n = len(results)
    should_escalate = [r for r in results if r["expected_escalation"]]
    should_handle   = [r for r in results if not r["expected_escalation"]]

    hit_results = [r for r in results if r.get("retrieval_hit") is not None]
    retrieval_hit_rate = (
        round(sum(1 for r in hit_results if r["retrieval_hit"]) / len(hit_results), 4)
        if hit_results else None
    )

    baseline = {
        "saved_at":              datetime.utcnow().isoformat() + "Z",
        "n":                     n,
        "classification_acc":    round(sum(1 for r in results if r["classification_correct"]) / n, 4),
        "retrieval_hit_rate":    retrieval_hit_rate,
        "routing_overall":       round(sum(1 for r in results if r["routing_correct"]) / n, 4),
        "routing_correct_handle": round(
            sum(1 for r in should_handle if r["routing_correct"]) / len(should_handle), 4
        ) if should_handle else None,
        "routing_missed_esc_caught": round(
            sum(1 for r in should_escalate if r["routing_correct"]) / len(should_escalate), 4
        ) if should_escalate else None,
        "avg_faithfulness":      round(sum(r["faithfulness"]["score"] for r in results) / n, 4),
        "avg_correctness":       round(sum(r["correctness"]["score"] for r in results) / n, 4),
        "note":                  "Naive pipeline baseline — Week 4 target: improve routing_missed_esc_caught to >0%",
    }

    path = os.path.join(SCRIPT_DIR, "baseline_scores.json")
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"\nBaseline saved to {path}")
    print(f"  Classification : {baseline['classification_acc']:.0%}")
    print(f"  Retrieval hit  : {baseline['retrieval_hit_rate']:.0%}")
    print(f"  Routing overall: {baseline['routing_overall']:.0%}  "
          f"(missed-esc caught: {baseline['routing_missed_esc_caught']:.0%})")
    print(f"  Faithfulness   : {baseline['avg_faithfulness']:.2f}/5")
    print(f"  Correctness    : {baseline['avg_correctness']:.2f}/5")


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="store_true",
                        help="Run eval against LangGraph agent instead of naive pipeline")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save scores to baseline_scores.json after eval")
    parser.add_argument("--no-langfuse", action="store_true",
                        help="Skip attaching scores to Langfuse traces")
    parser.add_argument("--category", type=str, default=None,
                        help="Run eval only on entries matching this category")
    args = parser.parse_args()

    if args.agent:
        results = run_agent_eval(category_filter=args.category)
    else:
        results = run_eval(category_filter=args.category)

    if not results:
        sys.exit(0)

    run_stratified_eval(results)

    if not args.no_langfuse:
        print("\nAttaching scores to Langfuse...")
        attach_langfuse_scores(results)

    if args.save_baseline:
        save_baseline(results)
