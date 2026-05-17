"""
Query classifier — maps customer queries to tools.

Tools:
  policy_kb      — RAG retrieval over Acmera policy corpus
  order_tracker  — order status + shipping lookup (mock_data/orders.json)
  account_lookup — customer record + membership tier (mock_data/customers.json)
  multi_tool     — query needs more than one tool (agent decides order)

Run:
  python scripts/query_classifier.py
"""
import os
import litellm
from dotenv import load_dotenv

load_dotenv()
litellm.set_verbose = False

INTENTS = [
    "return_or_refund",
    "order_status",
    "billing_or_payment",
    "product_info",
    "membership",
    "general",
]

INTENT_TO_TOOL = {
    "order_status":       "order_tracker",
    "product_info":       "policy_kb",
    "membership":         "account_lookup",
    "billing_or_payment": "policy_kb",
    "general":            "policy_kb",
    # return_or_refund → needs LLM disambiguation (see below)
}


def _classify_intent(query: str) -> str:
    """LLM intent classifier via LiteLLM with fallback."""
    response = litellm.completion(
        model="gpt-4o-mini",
        fallbacks=["gpt-3.5-turbo"],
        temperature=0,
        messages=[
    {"role": "system", "content": (
        "Classify this customer query into exactly one category.\n"
        "Respond with ONLY the category name, nothing else.\n\n"
        "Categories:\n"
        "- return_or_refund: customer wants to return an item or get a refund\n"
        "- order_status: asking about order tracking, shipping, delivery, or a specific order ID\n"
        "- billing_or_payment: payment issues, charges, invoices, double billing, payment methods\n"
        "- product_info: questions about product features, specifications, availability, or compatibility\n"
        "- membership: membership tiers, rewards points, upgrade eligibility, membership benefits\n"
        "- general: greetings, complaints, feedback, anything that doesn't fit the above categories\n\n"
        "When in doubt between two categories, pick the more specific one."
    )},
    {"role": "user", "content": query},
    ],
    )
    intent = response.choices[0].message.content.strip().lower().replace(" ", "_")
    return intent if intent in INTENTS else "general"


def _disambiguate_return(query: str) -> str:
    """
    return_or_refund can be:
      - policy_kb    : general question about return policy (no specific order)
      - order_tracker: return/refund on a specific order (needs order lookup)
      - multi_tool   : mentions membership tier + specific order (both tools needed)
    """
    response = litellm.completion(
        model="gpt-4o-mini",
        fallbacks=["gpt-3.5-turbo"],
        temperature=0,
        messages=[
            {"role": "system", "content": (
                "A customer query is about returns or refunds. "
                "Decide which tool(s) are needed:\n"
                "- policy_kb: general policy question, no specific order mentioned\n"
                "- order_tracker: mentions a specific order ID or item being returned\n"
                "- multi_tool: mentions both a specific order AND membership tier/account details\n"
                "Respond with ONLY one of: policy_kb, order_tracker, multi_tool"
            )},
            {"role": "user", "content": query},
        ],
    )
    result = response.choices[0].message.content.strip().lower().replace(" ", "_")
    valid = {"policy_kb", "order_tracker", "multi_tool"}
    return result if result in valid else "policy_kb"


def classify_tool(query: str) -> dict:
    """
    Classify query → tool selection.

    Returns:
      {
        "intent": str,
        "tool":   str,   # policy_kb | order_tracker | account_lookup | multi_tool
        "reason": str,
      }
    """
    intent = _classify_intent(query)

    if intent in INTENT_TO_TOOL:
        tool = INTENT_TO_TOOL[intent]
        reason = f"Intent '{intent}' maps directly to {tool}"
    else:
        # return_or_refund — LLM disambiguation
        tool = _disambiguate_return(query)
        reason = f"Intent 'return_or_refund' disambiguated to {tool}"

    return {"intent": intent, "tool": tool, "reason": reason}


if __name__ == "__main__":
    test_queries = [
        "What is the return window for electronics?",
        "Where is my order ORD-445521?",
        "Am I eligible for Premium Gold membership?",
        "I'm a Gold member and want to return my Diwali laptop purchase ORD-998877",
        "I was charged twice for my last payment",
    ]

    print("\nTool Router — classify_tool() output")
    print("=" * 70)
    for q in test_queries:
        result = classify_tool(q)
        print(f"Query  : {q}")
        print(f"Intent : {result['intent']}")
        print(f"Tool   : {result['tool']}")
        print(f"Reason : {result['reason']}")
        print("-" * 70)
