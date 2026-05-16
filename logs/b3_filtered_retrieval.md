# B3 — Metadata Filtering: retrieve_filtered()

## Query 1: "What is the standard warranty on electronics?" (intent: product_info)

| Mode | Doc | Chunk | Similarity |
|---|---|---|---|
| Without filter | 04_warranty_policy.md | 0 | 0.5813 |
| Without filter | 04_warranty_policy.md | 1 | 0.4974 |
| Without filter | 04_warranty_policy.md | 4 | 0.4934 |
| Without filter | **02_premium_membership.md** | **9** | **0.4881** |
| Without filter | 04_warranty_policy.md | 2 | 0.4840 |
| | | | |
| With filter | 04_warranty_policy.md | 0 | 0.5813 |
| With filter | 04_warranty_policy.md | 1 | 0.4974 |
| With filter | 04_warranty_policy.md | 4 | 0.4934 |
| With filter | 04_warranty_policy.md | 2 | 0.4840 |
| With filter | 04_warranty_policy.md | 3 | 0.4541 |

**Difference:** Without filter, slot 4 was taken by `02_premium_membership.md` (chunk 9) — irrelevant to a warranty question. With filter, that slot is replaced by `04_warranty_policy.md` (chunk 3), keeping all 5 results on-topic.

---

## Query 2: "How do I reach Premium Gold membership?" (intent: membership)

| Mode | Doc | Chunk | Similarity |
|---|---|---|---|
| Without filter | 06_support_faq.md | 5 | 0.6595 |
| Without filter | 02_premium_membership.md | 1 | 0.6327 |
| Without filter | **08_support_tickets.md** | **8** | **0.6273** |
| Without filter | 02_premium_membership.md | 4 | 0.5913 |
| Without filter | 02_premium_membership.md | 3 | 0.5464 |
| | | | |
| With filter | 06_support_faq.md | 5 | 0.6595 |
| With filter | 02_premium_membership.md | 1 | 0.6327 |
| With filter | 02_premium_membership.md | 4 | 0.5913 |
| With filter | 02_premium_membership.md | 3 | 0.5464 |
| With filter | 02_premium_membership.md | 0 | 0.5447 |

**Difference:** Without filter, slot 3 was taken by `08_support_tickets.md` (chunk 8) — a support process doc, not membership policy. With filter, that slot is replaced by `02_premium_membership.md` (chunk 0), which directly covers Gold tier requirements.

---

## Summary

Metadata filtering removes off-topic chunks that score well on embedding similarity but belong to irrelevant documents. In both cases, filtering replaced one noise chunk with an additional on-topic chunk from the correct document — effectively using all 5 retrieval slots for relevant content rather than 4.
