# B3 — Tool Router: classify_tool()

## Implementation

Two-stage routing:
1. **Fast path** — `INTENT_TO_TOOL` lookup table for 5 unambiguous intents (one LLM call for intent, zero for tool)
2. **LLM disambiguation** — `return_or_refund` needs a second LLM call to decide between `policy_kb`, `order_tracker`, or `multi_tool`

## Test Output — All 4 Tool Paths

| Query | Intent | Tool |
|---|---|---|
| "What is the return window for electronics?" | return_or_refund | **policy_kb** |
| "Where is my order ORD-445521?" | order_status | **order_tracker** |
| "Am I eligible for Premium Gold membership?" | membership | **account_lookup** |
| "I'm a Gold member and want to return my Diwali laptop purchase ORD-998877" | return_or_refund | **multi_tool** |
| "I was charged twice for my last payment" | billing_or_payment | **policy_kb** |

All 4 tool paths covered correctly:
- `policy_kb` — general return policy question (no order ID)
- `order_tracker` — specific order lookup
- `account_lookup` — membership/tier question
- `multi_tool` — Gold member + specific order = needs both account_lookup and order_tracker

## Why return_or_refund Needs Disambiguation

"What is the return window?" and "I want to return ORD-998877" are both `return_or_refund` intent but need completely different tools. The first is answered by the policy KB (no customer data needed). The second requires the order tracker to check order status and potentially account_lookup to verify tier-based return windows. A single lookup table cannot distinguish these — a second LLM call is cheaper than routing to the wrong tool and getting an unusable answer.
