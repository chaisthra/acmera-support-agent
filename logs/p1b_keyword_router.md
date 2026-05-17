# P1B.1 — Keyword Difficulty Router

## Routing Rules
- Score 1–3 → `gpt-4o-mini` (simple factual queries)
- Score 4–5 → `gpt-4o` (multi-policy / edge-case / comparison queries)

## Patterns (10 total)

| Pattern | Targets |
|---|---|
| `premium (gold\|silver)` | Tier-specific queries |
| `diwali\|promotional\|flash sale` | Promotional edge cases |
| `(corporate\|bulk) (order\|return)` | Multi-policy bulk queries |
| `\d+ (days\|items\|units)` | Quantitative conditions |
| `difference between` | Comparison queries |
| `benefits\|perks` | Tier benefit comparisons |
| `return (window\|option\|eligib\w*)` | Return policy specifics |
| `shield plus\|shield pro` | Product variant queries |
| `warranty (plan\|plans\|coverage)` | Warranty plan queries |
| `can i (return\|get a refund\|exchange)` | Conditional return eligibility |

---

## Labeled Test Results (10 queries)

| Query | Score | Model | Correct? |
|---|---|---|---|
| What is the return window for electronics? | 2 | gpt-4o-mini | ✓ |
| What payment methods does Acmera accept? | 1 | gpt-4o-mini | ✓ |
| How do I track my order? | 1 | gpt-4o-mini | ✓ |
| What is the standard warranty on Acmera products? | 1 | gpt-4o-mini | ✓ |
| How do I become a Premium Silver member? | 2 | gpt-4o-mini | ✓ |
| What is the difference between Premium Gold and Premium Silver benefits? | 4 | gpt-4o | ✓ |
| I placed a corporate bulk order of 50 units — what are my return options? | 4 | gpt-4o | ✓ |
| I bought a laptop during the Diwali flash sale, can I return it after 30 days? | 4 | gpt-4o | ✓ |
| As a Premium Gold member, do I get a different return window for promotional items? | 4 | gpt-4o | ✓ |
| What is the difference between Acmera Shield and Acmera Shield Plus warranty plans? | 4 | gpt-4o | ✓ |

**Labeled accuracy: 10/10 (100%)**

---

## Random Query Results (12 unseen queries)

| Query | Score | Model |
|---|---|---|
| Can I cancel my order after it has shipped? | 1 | gpt-4o-mini |
| Is same-day delivery available in Mumbai? | 1 | gpt-4o-mini |
| My refund still hasn't arrived after 10 days | 2 | gpt-4o-mini |
| I want to return 3 items from my Diwali order — what is the return window? | 4 | gpt-4o |
| Can I get a refund to my Acmera Wallet instead of my original payment method? | 2 | gpt-4o-mini |
| What is the difference between Acmera Shield and the standard manufacturer warranty? | 2 | gpt-4o-mini ⚠️ |
| I am a Premium Gold member — can I return an opened phone after 45 days? | 4 | gpt-4o |
| Do I get free shipping as a Premium Silver member on a ₹300 order? | 2 | gpt-4o-mini |
| My corporate bulk return of 20 units was rejected — what are my options? | 3 | gpt-4o-mini ⚠️ |
| How long does it take to process a refund? | 1 | gpt-4o-mini |
| I bought 5 units of the SmartHome Hub during a flash sale, can I return 2 of them? | 4 | gpt-4o |
| What happens to my Acmera Wallet balance if I delete my account? | 1 | gpt-4o-mini |

---

## Documented Failures (from labeled set)

### False Negative
**Query:** "Someone logged into my account without my permission and placed orders."  
**Score:** 1 → `gpt-4o-mini`  
**Problem:** No patterns fire. Account compromise is high-stakes and needs gpt-4o precision and tone, but regex has no concept of security severity — only keyword co-occurrence.

### False Positive
**Query:** "What warranty plan benefits does Premium Silver membership include?"  
**Score:** 4 → `gpt-4o`  
**Problem:** Fires `premium (gold|silver)` + `benefits` + `warranty plan` → score 4. This is a single factual lookup that gpt-4o-mini handles perfectly. Keyword co-occurrence mimics complexity without any actual reasoning requirement.

---

## Flagged Misses in Random Queries

### Miss 1 — Broadening needed on `return options`
**Query:** "My corporate bulk return of 20 units was rejected — what are my options?"  
**Score:** 3 → `gpt-4o-mini` (should be gpt-4o)  
**Why:** Fires `(corporate|bulk) (order|return)` + `\d+ units` = 2 patterns. The pattern `return (window|option|eligib\w*)` doesn't fire because the query says `"my options"` not `"return options"` — the word `return` and `options` are separated.  
**Fix:** Broaden to `r'(return )?(window|options?|eligib\w*)'` to catch standalone `"options"` in a return context.

### Miss 2 — `warranty` without `plan`
**Query:** "What is the difference between Acmera Shield and the standard manufacturer warranty?"  
**Score:** 2 → `gpt-4o-mini` (should be gpt-4o)  
**Why:** Only `difference between` fires (+1). `warranty (plan|plans|coverage)` doesn't match `"manufacturer warranty"` because the word `plan` isn't present — the pattern requires `warranty` to be immediately followed by `plan/plans/coverage`.  
**Fix:** Add `r'manufacturer warranty'` or broaden to `r'warranty\b'` (with care not to over-fire on simple warranty questions).

---

## Conclusion

Regex routing achieves **100% accuracy on labeled queries** by targeting specific Acmera domain patterns. It breaks on:
1. **Semantic complexity with no keywords** — security incidents, emotional urgency, account edge cases
2. **Surface word variation** — `"my options"` vs `"return options"`, `"manufacturer warranty"` vs `"warranty plan"`

These are the exact gaps the LLM router (P1B.2) addresses.

---

## Why Regex Classification is Naive — and Why LLM Wins

### The fundamental problem with regex

Regex counts keyword *co-occurrence*, not *reasoning complexity*. A query scores high if it happens to contain certain words — not because it genuinely requires multiple policy lookups. This creates two failure modes that no amount of pattern-tuning fully fixes:

**1. Semantic blindness**
"Someone logged into my account without my permission" scores 1 — no patterns fire. But this is a security incident: it needs gpt-4o for precise guidance, careful tone, and multi-step remediation advice. Regex has no concept of severity, urgency, or risk.

**2. Vocabulary dependency**
"What are my options after a rejected bulk return?" scores 3 instead of 4 because it says `"my options"` rather than `"return options"` — two words apart breaks the pattern. The query is identical in complexity; the word order is not.

Both failures require the same fix: something that understands *what the query is asking*, not just *what words it contains*.

### What the LLM does differently

The LLM difficulty classifier (P1B.2) reasons about **conditions** — actual policy rules that must be applied against customer-provided information:

- It identifies the customer's intent
- It lists every piece of information given ("I am Premium Gold", "bought during Diwali", "45 days ago")
- For each piece, it asks: *does answering this require evaluating a policy rule against this?* If yes → condition
- It counts conditions and maps to a score

This means "I'm a Premium Gold member — what's my return window?" correctly scores low (the tier is just a lookup key, one condition) while "I'm a Premium Gold member who bought during the Diwali flash sale 45 days ago — can I return?" correctly scores high (tier window + promotional window + date — three rules cross-checked).

Regex cannot make this distinction. The LLM can.

### When regex is still the right tool

Regex has a real advantage for **speed and cost**: zero LLM calls, sub-millisecond latency, fully deterministic. For a high-throughput pipeline where the query distribution is well-understood and stable, a well-tuned regex router can match LLM accuracy on the labeled set at a fraction of the cost.

The production answer is often a **hybrid**: regex as a fast pre-filter (block obvious spam, catch clearly-simple queries) + LLM for the ambiguous middle. P1B.2 implements the LLM half.
