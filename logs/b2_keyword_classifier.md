# B2 — Rule-Based Keyword Classifier

## Per-Intent Accuracy

| Intent | Total | Correct | Accuracy |
|---|---|---|---|
| return_or_refund | 6 | 6 | 100% |
| order_status | 7 | 4 | 57% |
| billing_or_payment | 8 | 4 | 50% |
| product_info | 4 | 0 | 0% |
| membership | 5 | 5 | 100% |
| general | 2 | 1 | 50% |
| **OVERALL** | **32** | **20** | **62%** |

## Two Failure Examples

**Example 1: "I was charged twice for the same order"**
- Expected: `billing_or_payment` → Got: `order_status`
- Why: The word "order" appears in the query and `order_status` is checked before `billing_or_payment` in the keyword loop. The classifier hits "order" first and stops — it never reaches the billing keywords. This is a keyword ordering problem, not a missing keyword problem.

**Example 2: "What is the standard warranty on electronics?"**
- Expected: `product_info` → Got: `general`
- Why: The keyword list for `product_info` only covers very specific technical terms (`specs`, `battery`, `compatible`, `model`). "Warranty" and "electronics" are not in the list. The query falls through all intents and defaults to `general`. This is a vocabulary coverage problem — the keyword list is too narrow for the range of ways customers ask about products.

## Why Keyword Matching Fails

`product_info` scored 0% because customers never used the exact words `specs`, `battery`, `compatible`, or `model`. They used `warranty`, `electronics`, `features` — related but unlisted synonyms. A keyword classifier is only as good as its vocabulary, and customer language is unpredictably varied.

`order_status` at 57% failed on queries like "track my shipment", "package arrived crushed", "same-day delivery" — all legitimate order queries that don't contain the word "order" or "tracking". The concept is right but the surface form is different.

**Conclusion:** 62% overall accuracy with 0% on `product_info` makes keyword matching unusable in production. An LLM classifier understands intent semantically — it doesn't need "warranty" to be in a list to know the query is about a product.
