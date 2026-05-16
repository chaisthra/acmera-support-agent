# B2 — Keyword vs LLM Classifier Comparison

## Per-Intent Accuracy

| Intent | N | Keyword | LLM | Delta |
|---|---|---|---|---|
| return_or_refund | 6 | 100% | 100% | — |
| order_status | 7 | 57% | 57% | — |
| billing_or_payment | 8 | 50% | 50% | — |
| product_info | 4 | 0% | 75% | **+75%** |
| membership | 5 | 100% | 100% | — |
| general | 2 | 50% | 50% | — |
| **OVERALL** | **32** | **62%** | **72%** | **+10%** |

## LLM Failure Examples

**"My package arrived and the box was completely crushed"**
- Expected: `order_status` → Got: `product_info`
- The LLM sees "box was crushed" and thinks product condition/quality, not delivery status. Without context about what Acmera's support categories mean, "crushed box" reads as a product issue.

**"I received the wrong item in my order"**
- Expected: `order_status` → Got: `return_or_refund`
- Reasonable confusion — wrong item delivery naturally leads to a return/refund action. The LLM is anticipating the next step rather than classifying the current query type.

## Reflection

The LLM dramatically fixed `product_info` (+75%) by understanding semantic meaning — "warranty on electronics" maps to product_info without needing the exact word "specs" or "battery". This is the core advantage: semantic understanding over surface matching.

However, `order_status`, `billing_or_payment`, and `general` showed no improvement. These intents have genuine boundary ambiguity — a crushed package is both an order and a product issue; a wrong item is both an order and a return issue. The LLM picks one interpretation, sometimes the wrong one. No classifier, rule-based or LLM, handles inherently ambiguous queries well without more context. That's what the agent's multi-tool reasoning layer solves in Week 3.
