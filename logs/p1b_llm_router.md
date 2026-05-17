
==========================================================================================
P1B.2 — REGEX vs LLM DIFFICULTY ROUTER  (same 10 queries)
==========================================================================================

Query                                          Reg    Reg model   LLM    LLM model   Agree?  LLM reason
---------------------------------------------  ----  ------------  ----  ------------  -------  ----------
  What is the return window for electronics?      2   gpt-4o-mini     1   gpt-4o-mini        ✓  pure fact lookup, no input to evaluate
  What payment methods does Acmera accept?        1   gpt-4o-mini     1   gpt-4o-mini        ✓  pure fact lookup, no input to evaluate
  How do I track my order?                        1   gpt-4o-mini     1   gpt-4o-mini        ✓  pure fact lookup, no input to evaluate
  What is the standard warranty on Acmera prod…    1   gpt-4o-mini     1   gpt-4o-mini        ✓  pure fact lookup, no input to evaluate
  How do I become a Premium Silver member?        2   gpt-4o-mini     1   gpt-4o-mini        ✓  pure fact lookup, no input to evaluate
  What is the difference between Premium Gold …    4        gpt-4o     1   gpt-4o-mini   ✗ DIFF  pure fact lookup, no input to evaluate
  I placed a corporate bulk order of 50 units …    4        gpt-4o     3   gpt-4o-mini   ✗ DIFF  tier-specific return policy for bulk orders a
  I bought a laptop during the Diwali flash sa…    4        gpt-4o     4        gpt-4o        ✓  promotional window + tier window + date
  As a Premium Gold member, do I get a differe…    4        gpt-4o     2   gpt-4o-mini   ✗ DIFF  1 condition (tier rule vs promotional item re
  What is the difference between Acmera Shield…    4        gpt-4o     1   gpt-4o-mini   ✗ DIFF  pure fact lookup, no input to evaluate

==========================================================================================
DISAGREEMENTS
==========================================================================================

  Query  : What is the difference between Premium Gold and Premium Silver benefits?
  Regex  : score=4 → gpt-4o
  LLM    : score=1 → gpt-4o-mini
  Reason : pure fact lookup, no input to evaluate
  Winner : Regex  (expected gpt-4o)

  Query  : I placed a corporate bulk order of 50 units — what are my return options?
  Regex  : score=4 → gpt-4o
  LLM    : score=3 → gpt-4o-mini
  Reason : tier-specific return policy for bulk orders and quantity of units
  Winner : Regex  (expected gpt-4o)

  Query  : As a Premium Gold member, do I get a different return window for promotional items?
  Regex  : score=4 → gpt-4o
  LLM    : score=2 → gpt-4o-mini
  Reason : 1 condition (tier rule vs promotional item return policy)
  Winner : Regex  (expected gpt-4o)

  Query  : What is the difference between Acmera Shield and Acmera Shield Plus warranty plans?
  Regex  : score=4 → gpt-4o
  LLM    : score=1 → gpt-4o-mini
  Reason : pure fact lookup, no input to evaluate
  Winner : Regex  (expected gpt-4o)

==========================================================================================
COST ANALYSIS — 5,000 queries/day
==========================================================================================

  Assumption : 65% simple (gpt-4o-mini), 35% complex (gpt-4o)
  Token est  : 800 input + 300 output per query

  No routing (all gpt-4o)   : $  25.00/day   $  750.00/month
  With routing              : $   9.80/day   $  294.00/month
  ─────────────────────────────────────────
  Saving                    : $  15.20/day   $  456.00/month  (61% reduction)

  Classification overhead: $0.0750/day — negligible
