# Agent Trajectories — 3 Test Queries


======================================================================
  SIMPLE  (1 tool)
======================================================================
  Query          : What is the return window for electronics?
  Trajectory     : classify → tool_call → evaluate → respond
  Intent         : return_or_refund
  Tools called   : ['policy_kb']
  Steps taken    : 1
  Difficulty     : 1 → gpt-4o-mini
  Escalated      : False
  Elapsed        : 11.91s
  Langfuse trace : 32f4fde9-04e1-4825-830a-afc83b609bdb
  Answer         : All products purchased from Acmera, including electronics, are eligible for return within 30 calendar days of the delivery date. If the item was purchased between November 15 and December 31, it has an extended return window until January 31 of the following year.

======================================================================
  COMPLEX (2 tools)
======================================================================
  Query          : I'm a Gold member and want to return my laptop from order ORD-445521 — what are my options?
  Trajectory     : classify → tool_call → evaluate → respond
  Intent         : return_or_refund
  Tools called   : ['order_tracker']
  Steps taken    : 1
  Difficulty     : 4 → gpt-4o
  Escalated      : False
  Elapsed        : 6.43s
  Langfuse trace : 6ea4a37c-c1f7-46b6-a89c-485d42f1d499
  Answer         : As a Gold member, you have the option to return your laptop from order ORD-445521. Since the order was delivered on 2025-09-05, please ensure that you are within the return window specified by Acmera's return policy. Typically, Acmera allows returns within a certain number of days from the delivery ...

======================================================================
  ESCALATION
======================================================================
  Query          : Someone logged into my account without my permission and placed orders — I need urgent help.
  Trajectory     : classify → tool_call → evaluate → escalate
  Intent         : general
  Tools called   : ['policy_kb']
  Steps taken    : 1
  Difficulty     : 5 → gpt-4o
  Escalated      : True
  Elapsed        : 5.48s
  Langfuse trace : 667c29da-9f3f-487b-bf6a-3f20ad93c25b
  Answer         : I'm escalating your query to our specialist team for human review.

Situation: Someone logged into my account without my permission and placed orders — I need urgent help.

A support agent will contact you within 2 hours via your registered email.
Reference number: ESC-42084

======================================================================
TRAJECTORY SUMMARY
======================================================================
  ['classify', 'tool_call', 'evaluate', 'respond']  →  tools=['policy_kb']  steps=1
  ['classify', 'tool_call', 'evaluate', 'respond']  →  tools=['order_tracker']  steps=1
  ['classify', 'tool_call', 'evaluate', 'escalate']  →  tools=['policy_kb']  steps=1
