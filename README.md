# Acmera Support Agent — LangGraph E-commerce Support Agent

A 5-node LangGraph agent for e-commerce customer support. Routes queries across intent classification, tool execution, escalation detection, and structured ticket generation.

Built on top of the same corpus and infrastructure as the RAG system but designed as a stateful agentic pipeline rather than a single-pass retrieval system.


---

## What it does

Takes a customer query, classifies intent, routes to the right tool, evaluates whether enough information was gathered, and either responds, loops back for more data, or escalates with a structured ticket.

---

## Agent Architecture

```
User Query
  → PII Anonymization (Presidio)
  → Input Guardrail (topic restriction + prompt injection check)
  → Semantic Cache (Redis, shared namespace with RAG system)
      ↓ cache miss
  → Node 1: Classify
      Intent detection across 6 categories
      Tool selection: policy_kb / order_tracker / account_lookup / multi_tool
      Disambiguation LLM call for return_or_refund ambiguity
      GPT-4o-mini, fallback GPT-3.5-turbo
  → Node 2: Tool Call
      policy_kb → RAG retrieval pipeline (filtered dense + Cohere rerank)
      order_tracker → order lookup from mock data
      account_lookup → customer and membership tier lookup
      Source guardrail filters restricted chunks before context assembly
  → Node 3: Evaluate
      Decides: respond / tool_call (need more data) / escalate
      Max 3 attempts before forcing respond
      Keyword-based force escalation for security-sensitive queries
  → Node 4: Respond
      Difficulty classifier scores query 1-5
      Score > 3 → GPT-4o
      Score ≤ 3 → GPT-4o-mini
      LiteLLM with fallback
  → Node 5: Escalate
      Structured ticket with priority, team routing, customer sentiment
      Returns ticket reference instead of generated answer
  → Output Guardrail (hallucination detection + PII restore)
  → Response
```

---

## Eval Results

Evaluated against a 32-query golden dataset across 6 intent categories and multiple difficulty levels.

| Metric | Naive Baseline | Agent | Delta |
|---|---|---|---|
| Classification Accuracy | 72% | 81% | +9% |
| Routing Accuracy | 66% | 78% | +12% |
| Escalation Detection | 0% | 36% | +36% |
| Correct Handle | 100% | 100% | — |
| Avg Faithfulness | 4.75/5 | 4.16/5 | -0.59 |
| Avg Correctness | 3.88/5 | 3.44/5 | -0.44 |

Escalation detection going from 0% to 36% is the key metric. The naive pipeline had no escalation mechanism at all.

Faithfulness and correctness dropped slightly because the agent handles harder queries than the naive baseline.

All 5 nodes traced in LangFuse per query.

---

## Known Gaps and Planned Fixes

| Gap | Fix |
|---|---|
| No HITL breakpoint before escalation executes | LangGraph interrupt before execute action node, reviewer approve or reject with comment |
| No automatic retry on LLM unavailability | Exponential backoff with max retries and supervisor escalation path |
| Escalation detection at 36%, 64% of escalation cases still missed | Expand keyword signals, add LLM-based escalation scoring alongside keyword check |
| No document lifecycle management | Automated ingestion and index refresh pipeline |
| No role-based access control | Data boundaries per customer account |
| Mock data in JSON files baked into image | Move orders and customers tables to RDS, update on SQL insert |

---

## AWS Deployment

- ECS Fargate (1GB RAM, separate ECS service on shared infrastructure)
- ALB with health check on /health
- RDS PostgreSQL (shared with RAG system)
- Redis (shared semantic cache namespace)
- Auto-scaling: 1 to 3 tasks above 60% CPU
- LangFuse for full trace observability across all 5 nodes

---

## Setup

### Prerequisites

- Python 3.11+
- Docker Desktop
- OpenAI API key
- LangFuse account (cloud.langfuse.com free tier)
- Cohere API key (optional, falls back to dense retrieval)

### Local

```bash
cp .env.example .env
# Fill in API keys
docker-compose up -d
pip install -r requirements.txt
python scripts/setup_db.py
python scripts/ingest.py
python scripts/demo.py
```

### Deploy to AWS

```bash
docker build --platform linux/amd64 -t project-b .
docker push $AWS_ACCOUNT_ID.dkr.ecr.ap-south-1.amazonaws.com/acmera-support-dev:latest
sam deploy --stack-name acmera-support-dev --region ap-south-1 --capabilities CAPABILITY_IAM --guided
```

---

## Repo Structure

```
project-b/
├── corpus/                      # 19 Acmera policy documents (shared with RAG)
├── mock_data/
│   ├── orders.json              # mock order data
│   └── customers.json           # mock customer and membership data
├── scripts/
│   ├── setup_db.py              # pgvector table + HNSW index
│   ├── ingest.py                # chunk + embed + store
│   ├── agent.py                 # LangGraph 5-node agent
│   ├── query_classifier.py      # intent classification + tool selection
│   ├── mock_tools.py            # order tracker + account lookup
│   ├── retrieval.py             # advanced RAG retrieval pipeline
│   ├── semantic_cache.py        # Redis-backed semantic cache
│   ├── input_guardrail.py       # topic + safety classifier
│   ├── output_guardrail.py      # hallucination detection
│   ├── pii_anonymizer.py        # Presidio anonymize + restore
│   ├── difficulty_classifier.py # query routing 1-5
│   ├── eval_harness.py          # classification, routing, faithfulness, correctness
│   └── demo.py                  # interactive CLI
├── api.py                       # FastAPI wrapper
├── Dockerfile
├── template.yaml                # AWS SAM template
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Corpus

19 markdown documents covering returns, payments, membership tiers, warranty, shipping, promotions, electronics catalog, sustainability, corporate gifting, and support FAQs. Same corpus as the RAG system.
