"""
Rule-based vs LLM intent classifier comparison.

Run:
  python scripts/classifier_scratch.py
"""
import os
import json
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

SCRIPT_DIR = os.path.dirname(__file__)

KEYWORDS = {
    "return_or_refund":   ["return", "refund", "send back", "money back"],
    "order_status":       ["order", "where is", "tracking", "delivered"],
    "billing_or_payment": ["charged", "payment", "invoice", "wallet"],
    "product_info":       ["specs", "battery", "compatible", "model"],
    "membership":         ["premium", "gold", "silver", "tier", "membership"],
    "general":            [],  # fallback
}


def classify_keyword(query: str) -> str:
    q = query.lower()
    for intent, keywords in KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return intent
    return "general"


def classify_llm(query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": (
                "Classify into one of: return_or_refund, order_status, "
                "billing_or_payment, product_info, membership, general. "
                "Respond with ONLY the category name."
            )},
            {"role": "user", "content": query},
        ],
    )
    result = response.choices[0].message.content.strip().lower().replace(" ", "_")
    VALID = {"return_or_refund", "order_status", "billing_or_payment",
             "product_info", "membership", "general"}
    return result if result in VALID else "general"


if __name__ == "__main__":
    with open(os.path.join(SCRIPT_DIR, "golden_dataset.json")) as f:
        dataset = json.load(f)

    kw_by_intent  = defaultdict(list)
    llm_by_intent = defaultdict(list)
    llm_failures  = []

    print(f"\nRunning LLM classifier on {len(dataset)} queries...")
    for entry in dataset:
        kw_pred  = classify_keyword(entry["query"])
        llm_pred = classify_llm(entry["query"])
        expected = entry["expected_intent"]

        kw_correct  = kw_pred  == expected
        llm_correct = llm_pred == expected

        kw_by_intent[expected].append(kw_correct)
        llm_by_intent[expected].append(llm_correct)

        if not llm_correct:
            llm_failures.append({
                "query":    entry["query"],
                "expected": expected,
                "got":      llm_pred,
            })

    print("\nKeyword vs LLM Classifier — Per-Intent Accuracy")
    print(f"{'Intent':<22} {'N':>3}  {'Keyword':>8}  {'LLM':>6}  {'Delta':>7}")
    print("-" * 55)
    kw_total = llm_total = n_total = 0
    for intent in KEYWORDS:
        kw_rows  = kw_by_intent[intent]
        llm_rows = llm_by_intent[intent]
        n = len(kw_rows)
        if n == 0:
            continue
        kw_acc  = sum(kw_rows)  / n
        llm_acc = sum(llm_rows) / n
        delta   = llm_acc - kw_acc
        arrow   = f"+{delta:.0%}" if delta > 0 else f"{delta:.0%}" if delta < 0 else "—"
        kw_total  += sum(kw_rows)
        llm_total += sum(llm_rows)
        n_total   += n
        print(f"  {intent:<20} {n:>3}  {kw_acc:>7.0%}  {llm_acc:>6.0%}  {arrow:>7}")

    print("-" * 55)
    print(f"  {'OVERALL':<20} {n_total:>3}  {kw_total/n_total:>7.0%}  {llm_total/n_total:>6.0%}")

    print(f"\nLLM failure examples ({len(llm_failures)} total):")
    for f in llm_failures[:3]:
        print(f"  Query   : {f['query'][:70]}")
        print(f"  Expected: {f['expected']}  →  Got: {f['got']}")
        print()
