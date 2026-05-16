# B3 — Chunk Deduplication (Jaccard similarity, threshold=0.75)

## Test Query: "Can I convert from gold to premium member?" (intent: membership)

### Before deduplication (10 chunks)
| cosine | doc | chunk |
|---|---|---|
| 0.5833 | 06_support_faq.md | 5 |
| 0.5618 | 02_premium_membership.md | 1 |
| 0.5339 | 02_premium_membership.md | 6 |
| 0.5325 | 02_premium_membership.md | 4 |
| 0.5028 | 02_premium_membership.md | 3 |
| 0.4823 | 02_premium_membership.md | 0 |
| 0.4729 | 02_premium_membership.md | 5 |
| 0.4684 | 06_support_faq.md | 6 |
| 0.4555 | 02_premium_membership.md | 10 |
| 0.4489 | 02_premium_membership.md | 7 |

### After deduplication: 10 chunks — 0 removed

### Why no duplicates were found

All 10 chunks are from `02_premium_membership.md` or `06_support_faq.md` but each covers a **different section** of the policy — tier thresholds, point earn rates, benefits per tier, upgrade process, FAQ answers. Even though they come from the same document, their word-level Jaccard similarity is below 0.75 because each chunk contains distinct policy details.

This is correct behaviour. The deduplicator is working as intended — it targets truly redundant content (same text repeated across multiple chunks), not "same document, different section." A threshold of 0.75 is tight enough that different policy sections don't trigger it.

**When deduplication would fire:** If the corpus had multiple FAQ documents that repeated the same return policy text verbatim (e.g., `06_support_faq.md` and `16_customer_faq.md` both containing identical sentences about the 30-day return window), those would score Jaccard ≥ 0.75 and the lower-ranked copy would be removed.

---

## Filtered Retrieval Summary (from previous test)

| Query | Without filter | With filter | Chunk removed |
|---|---|---|---|
| Warranty on electronics | 02_premium_membership.md chunk 9 in slot 4 | Replaced by 04_warranty_policy.md chunk 3 | ✓ noise removed |
| Convert gold to premium | 08_support_tickets.md chunk 8 in slot 3 | Replaced by 02_premium_membership.md chunk 3 | ✓ noise removed |
