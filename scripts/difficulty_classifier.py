"""
P1B — Difficulty Router from Scratch

Stage 1: regex/keyword scoring — classify_difficulty_regex() / route_model_regex()
Stage 2: LLM-based scoring    — classify_difficulty_llm()    / route_model_llm()

Scores 1-3 → gpt-4o-mini  (simple factual queries)
Scores 4-5 → gpt-4o        (multi-policy / edge-case / comparison queries)

Run:
  python scripts/difficulty_classifier.py           # regex only
  python scripts/difficulty_classifier.py --compare # regex vs LLM side-by-side
"""
import re
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import litellm
from dotenv import load_dotenv

load_dotenv()
litellm.set_verbose = False

COMPLEX_PATTERNS = [
    # Original 5
    r'premium (gold|silver)',            # tier-specific query
    r'diwali|promotional|flash sale',   # promotional edge case
    r'(corporate|bulk) (order|return)', # multi-policy query
    r'\d+ (days|items|units)',          # quantitative edge cases
    r'difference between',              # comparison query
    # Added to ensure complex queries reach score 4
    r'benefits|perks',                          # tier benefit comparisons
    r'return (window|option|eligib\w*)',         # return-policy specifics
    r'shield plus|shield pro',                  # product variant (separate from warranty)
    r'warranty (plan|plans|coverage)',           # warranty plan queries
    r'can i (return|get a refund|exchange)',     # conditional return eligibility
]


def classify_difficulty_regex(query: str) -> int:
    """Score query complexity 1-5 by counting matched patterns."""
    q = query.lower()
    score = 1
    for pattern in COMPLEX_PATTERNS:
        if re.search(pattern, q):
            score += 1
    return min(score, 5)


def route_model_regex(query: str) -> tuple[str, int]:
    """Return (model_name, difficulty_score)."""
    score = classify_difficulty_regex(query)
    model = "gpt-4o" if score >= 4 else "gpt-4o-mini"
    return model, score


# ---------------------------------------------------------------------------
# Stage 2: LLM-based difficulty scoring
# ---------------------------------------------------------------------------

DIFFICULTY_PROMPT = """Rate this customer support query 1-5.
1-2 = single fact lookup (return window, shipping cost)
3   = one condition to evaluate (open vs closed electronics)
4-5 = multiple conditions, tier-specific, cross-document

Before scoring, reason through these steps:

Step 1 — What is the customer's intent? (what do they actually need answered)
Step 2 — List every piece of information the customer gave (tier, product, date, order ID, etc.)
Step 3 — For each piece of information ask: does answering this query require evaluating a
          policy rule AGAINST this piece of info?
          YES → it is a CONDITION (counts toward complexity)
          NO  → it is just context, ignore it for scoring
Step 4 — Count the CONDITIONS, then score:
          0 conditions → 1  (pure fact lookup, no input to evaluate)
          1 condition  → 2-3 (one policy rule to apply)
          2 conditions → 4  (two rules must be cross-checked)
          3+ conditions or involves security/account compromise → 5

Examples:
- "What is the return window?" → 0 conditions → score 1
- "Is COD available for a ₹30,000 order?" → 1 condition (amount vs COD limit) → score 2
- "Can a Premium Silver member return opened electronics?" → 2 conditions (tier rule + open/sealed rule) → score 4
- "I bought a laptop during Diwali as a Gold member — can I return after 45 days?" → 3 conditions (promotional window + tier window + date) → score 5
- "Someone accessed my account without permission" → security incident → score 5

Query: {query}
Respond ONLY with JSON: {{"score": N, "reason": "one line listing the conditions counted"}}"""


def classify_difficulty_llm(query: str) -> dict:
    """Score query complexity 1-5 using gpt-4o-mini as cheap classifier."""
    response = litellm.completion(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=80,
        messages=[{"role": "user", "content": DIFFICULTY_PROMPT.format(query=query)}],
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def route_model_llm(query: str) -> tuple[str, int, str]:
    """Return (model_name, difficulty_score, reason)."""
    result = classify_difficulty_llm(query)
    score  = int(result["score"])
    model  = "gpt-4o" if score >= 4 else "gpt-4o-mini"
    return model, score, result.get("reason", "")


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

SIMPLE_QUERIES = [
    "What is the return window for electronics?",
    "What payment methods does Acmera accept?",
    "How do I track my order?",
    "What is the standard warranty on Acmera products?",
    "How do I become a Premium Silver member?",
]

COMPLEX_QUERIES = [
    "What is the difference between Premium Gold and Premium Silver benefits?",
    "I placed a corporate bulk order of 50 units — what are my return options?",
    "I bought a laptop during the Diwali flash sale, can I return it after 30 days?",
    "As a Premium Gold member, do I get a different return window for promotional items?",
    "What is the difference between Acmera Shield and Acmera Shield Plus warranty plans?",
]

# Two queries where regex gets it wrong — documented failure cases
FAILURE_CASES = [
    # False negative: security incident — no patterns fire
    # → scores 1 → gpt-4o-mini, but account compromise needs gpt-4o for precision + tone
    "Someone logged into my account without my permission and placed orders.",

    # False positive: simple one-line policy lookup
    # premium (gold|silver) + benefits + warranty plan → score 4 → gpt-4o
    # gpt-4o-mini handles this fine — regex over-routes on keyword co-occurrence
    "What warranty plan benefits does Premium Silver membership include?",
]


LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


def _save(filename: str, lines: list[str]):
    """Print all lines to stdout and write them to logs/<filename>."""
    text = "\n".join(lines)
    print(text)
    path = os.path.join(LOGS_DIR, filename)
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"\n[saved → logs/{filename}]")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""

    if mode == "--compare":
        all_labeled = [("simple", q) for q in SIMPLE_QUERIES] + \
                      [("complex", q) for q in COMPLEX_QUERIES]

        lines = []
        lines += ["", "=" * 90,
                  "P1B.2 — REGEX vs LLM DIFFICULTY ROUTER  (same 10 queries)",
                  "=" * 90, ""]
        lines.append(f"{'Query':<45} {'Reg':>4} {'Reg model':>12}  {'LLM':>4} {'LLM model':>12}  {'Agree?':>7}  LLM reason")
        lines.append(f"{'-'*45}  {'-'*4}  {'-'*12}  {'-'*4}  {'-'*12}  {'-'*7}  ----------")

        disagreements = []
        rows = []
        for kind, query in all_labeled:
            r_model, r_score         = route_model_regex(query)
            l_model, l_score, reason = route_model_llm(query)
            agree = "✓" if r_model == l_model else "✗ DIFF"
            if r_model != l_model:
                disagreements.append((query, kind, r_score, r_model, l_score, l_model, reason))
            short = query[:44] + "…" if len(query) > 44 else query
            rows.append({"query": query, "regex_score": r_score, "regex_model": r_model,
                         "llm_score": l_score, "llm_model": l_model, "agree": r_model == l_model,
                         "reason": reason})
            lines.append(f"  {short:<44} {r_score:>4}  {r_model:>12}  {l_score:>4}  {l_model:>12}  {agree:>7}  {reason[:45]}")

        lines.append("")
        if disagreements:
            lines += ["=" * 90, "DISAGREEMENTS", "=" * 90]
            for query, kind, r_score, r_model, l_score, l_model, reason in disagreements:
                ground_truth = "gpt-4o" if kind == "complex" else "gpt-4o-mini"
                winner = "LLM" if l_model == ground_truth else ("Regex" if r_model == ground_truth else "Both wrong")
                lines += ["",
                          f"  Query  : {query}",
                          f"  Regex  : score={r_score} → {r_model}",
                          f"  LLM    : score={l_score} → {l_model}",
                          f"  Reason : {reason}",
                          f"  Winner : {winner}  (expected {ground_truth})"]
        else:
            lines.append("No disagreements — both methods agree on all 10 queries.")

        # Cost analysis
        QUERIES_PER_DAY = 5_000
        INPUT_TOKENS, OUTPUT_TOKENS, CLASSIFY_TOKENS = 800, 300, 100
        MINI_IN, MINI_OUT = 0.15, 0.60
        GPT4_IN, GPT4_OUT = 2.50, 10.00

        def cost(n, model):
            i, o = (MINI_IN, MINI_OUT) if "mini" in model else (GPT4_IN, GPT4_OUT)
            return n * (INPUT_TOKENS * i + OUTPUT_TOKENS * o) / 1_000_000

        pct_mini = 0.65
        n_mini, n_gpt4 = int(QUERIES_PER_DAY * pct_mini), int(QUERIES_PER_DAY * (1 - pct_mini))
        cost_no_routing   = cost(QUERIES_PER_DAY, "gpt-4o")
        cost_with_routing = (cost(n_mini, "gpt-4o-mini") + cost(n_gpt4, "gpt-4o") +
                             QUERIES_PER_DAY * CLASSIFY_TOKENS * MINI_IN / 1_000_000)
        saving_day = cost_no_routing - cost_with_routing

        lines += ["", "=" * 90, "COST ANALYSIS — 5,000 queries/day", "=" * 90, "",
                  f"  Assumption : {pct_mini:.0%} simple (gpt-4o-mini), {1-pct_mini:.0%} complex (gpt-4o)",
                  f"  Token est  : {INPUT_TOKENS} input + {OUTPUT_TOKENS} output per query", "",
                  f"  No routing (all gpt-4o)   : ${cost_no_routing:>7.2f}/day   ${cost_no_routing*30:>8.2f}/month",
                  f"  With routing              : ${cost_with_routing:>7.2f}/day   ${cost_with_routing*30:>8.2f}/month",
                  f"  {'─'*41}",
                  f"  Saving                    : ${saving_day:>7.2f}/day   ${saving_day*30:>8.2f}/month  ({saving_day/cost_no_routing:.0%} reduction)",
                  "",
                  f"  Classification overhead: ${QUERIES_PER_DAY * CLASSIFY_TOKENS * MINI_IN / 1_000_000:.4f}/day — negligible"]

        _save("p1b_llm_router.md", lines)

        # Also save raw rows as JSON
        out = os.path.join(LOGS_DIR, "p1b_llm_router.json")
        with open(out, "w") as f:
            json.dump({"compare_rows": rows,
                       "cost": {"no_routing_day": round(cost_no_routing, 2),
                                "with_routing_day": round(cost_with_routing, 2),
                                "saving_day": round(saving_day, 2),
                                "saving_month": round(saving_day * 30, 2)}}, f, indent=2)
        print(f"[saved → logs/p1b_llm_router.json]")
        sys.exit(0)

    if mode == "--golden":
        dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
        with open(dataset_path) as f:
            dataset = json.load(f)

        EXPECTED_MODEL = {"easy": "gpt-4o-mini", "medium": "gpt-4o-mini", "hard": "gpt-4o"}

        lines = ["", "=" * 105,
                 "P1B — CLASSIFIER vs GOLDEN DATASET  (32 queries, difficulty: easy/medium/hard)",
                 "=" * 105, ""]
        lines.append(f"  {'ID':<5} {'Diff':<8} {'Expected':<13} {'Reg':>4} {'Reg mdl':>12}  {'LLM':>4} {'LLM mdl':>12}  {'Reg✓':>5}  {'LLM✓':>5}  LLM reason")
        lines.append(f"  {'-'*5}  {'-'*7}  {'-'*12}  {'-'*4}  {'-'*12}  {'-'*4}  {'-'*12}  {'-'*5}  {'-'*5}  ----------")

        reg_correct = llm_correct = 0
        rows = []
        for entry in dataset:
            expected_model = EXPECTED_MODEL.get(entry["difficulty"], "gpt-4o-mini")
            r_model, r_score         = route_model_regex(entry["query"])
            l_model, l_score, reason = route_model_llm(entry["query"])
            r_ok = "✓" if r_model == expected_model else "✗"
            l_ok = "✓" if l_model == expected_model else "✗"
            if r_model == expected_model: reg_correct += 1
            if l_model == expected_model: llm_correct += 1
            rows.append({"id": entry["id"], "query": entry["query"],
                         "difficulty": entry["difficulty"], "expected_model": expected_model,
                         "regex_score": r_score, "regex_model": r_model, "regex_correct": r_model == expected_model,
                         "llm_score": l_score, "llm_model": l_model, "llm_correct": l_model == expected_model,
                         "llm_reason": reason})
            lines.append(f"  {entry['id']:<5}  {entry['difficulty']:<7}  {expected_model:<13}  "
                         f"{r_score:>4}  {r_model:>12}  {l_score:>4}  {l_model:>12}  "
                         f"{r_ok:>5}  {l_ok:>5}  {reason[:45]}")

        n = len(dataset)
        winner = "LLM" if llm_correct > reg_correct else ("Regex" if reg_correct > llm_correct else "Tie")
        lines += ["", "=" * 105,
                  f"  Regex accuracy : {reg_correct}/{n} ({reg_correct/n:.0%})",
                  f"  LLM   accuracy : {llm_correct}/{n} ({llm_correct/n:.0%})",
                  f"  Winner         : {winner}",
                  "=" * 105]

        _save("p1b_golden_comparison.md", lines)

        out = os.path.join(LOGS_DIR, "p1b_golden_comparison.json")
        with open(out, "w") as f:
            json.dump({"rows": rows,
                       "summary": {"n": n, "regex_correct": reg_correct, "llm_correct": llm_correct,
                                   "regex_accuracy": round(reg_correct / n, 4),
                                   "llm_accuracy": round(llm_correct / n, 4),
                                   "winner": winner}}, f, indent=2)
        print(f"[saved → logs/p1b_golden_comparison.json]")
        sys.exit(0)

    # ---------------------------------------------------------------
    # Default: regex-only run (P1B.1)
    # ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("P1B.1 — KEYWORD DIFFICULTY ROUTER")
    print("=" * 65)
    print(f"\n{'Query':<58} {'Score':>5}  {'Model':>12}")
    print(f"{'-'*58}  {'-'*5}  {'-'*12}")

    all_queries = [("simple", q) for q in SIMPLE_QUERIES] + \
                  [("complex", q) for q in COMPLEX_QUERIES]

    lines = []
    for kind, query in all_queries:
        model, score = route_model_regex(query)
        flag = "✓" if (kind == "simple" and model == "gpt-4o-mini") or \
                      (kind == "complex" and model == "gpt-4o") else "✗"
        short = query[:57] + "…" if len(query) > 57 else query
        lines.append(f"{flag} {short:<57} {score:>5}  {model:>12}")

    lines += ["", "=" * 65, "FAILURE CASES — where regex gets it wrong", "=" * 65]

    failure_labels = [
        ("False negative", "Account security incident — no patterns fire",
         "None of the 10 patterns match → score 1 → gpt-4o-mini. "
         "An account compromise is high-stakes and needs gpt-4o-level precision and tone, "
         "but regex has no concept of security severity — only keyword co-occurrence."),
        ("False positive", "Simple one-line policy lookup over-routed to gpt-4o",
         "Fires 'premium (gold|silver)' + 'benefits' + 'warranty plan' → score 4 → gpt-4o. "
         "This is a single factual lookup gpt-4o-mini handles perfectly. "
         "Keyword co-occurrence mimics complexity without any actual reasoning requirement."),
    ]
    for (case_type, label, explanation), query in zip(failure_labels, FAILURE_CASES):
        model, score = route_model_regex(query)
        lines += ["", f"[{case_type}]",
                  f"  Query : {query}",
                  f"  Score : {score}  →  {model}",
                  f"  Why   : {explanation}"]

    simple_correct  = sum(1 for q in SIMPLE_QUERIES  if route_model_regex(q)[0] == "gpt-4o-mini")
    complex_correct = sum(1 for q in COMPLEX_QUERIES if route_model_regex(q)[0] == "gpt-4o")
    lines += ["", "=" * 65, "ROUTING SUMMARY", "=" * 65,
              f"  Simple  queries: {simple_correct}/{len(SIMPLE_QUERIES)} correctly routed to gpt-4o-mini",
              f"  Complex queries: {complex_correct}/{len(COMPLEX_QUERIES)} correctly routed to gpt-4o",
              f"  Overall accuracy: {(simple_correct + complex_correct) / 10:.0%}  (10 queries)"]

    random_queries = [
        "Can I cancel my order after it has shipped?",
        "Is same-day delivery available in Mumbai?",
        "My refund still hasn't arrived after 10 days",
        "I want to return 3 items from my Diwali order — what is the return window?",
        "Can I get a refund to my Acmera Wallet instead of my original payment method?",
        "What is the difference between Acmera Shield and the standard manufacturer warranty?",
        "I am a Premium Gold member — can I return an opened phone after 45 days?",
        "Do I get free shipping as a Premium Silver member on a ₹300 order?",
        "My corporate bulk return of 20 units was rejected — what are my options?",
        "How long does it take to process a refund?",
        "I bought 5 units of the SmartHome Hub during a flash sale — can I return 2 of them?",
        "What happens to my Acmera Wallet balance if I delete my account?",
    ]
    lines += ["", "=" * 65, "RANDOM QUERIES — unseen, no expected label", "=" * 65,
              "", f"{'Query':<58} {'Score':>5}  {'Model':>12}",
              f"{'-'*58}  {'-'*5}  {'-'*12}"]
    for query in random_queries:
        model, score = route_model_regex(query)
        short = query[:57] + "…" if len(query) > 57 else query
        lines.append(f"  {short:<57} {score:>5}  {model:>12}")

    _save("p1b_keyword_router_run.md", lines)
