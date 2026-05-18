"""
Project B — FastAPI Web Server

Run (local):
  uvicorn app:app --host 0.0.0.0 --port 8080 --reload

Endpoints:
  GET  /health        liveness probe
  POST /query         core agent endpoint (curl-testable)
  POST /api/ask       web UI endpoint (agent + pipeline modes)
  POST /ingest        setup pgvector schema + embed corpus
  GET  /              serves web/index.html
"""
import os
import sys
import json
import hashlib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR      = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR   = os.path.join(ROOT_DIR, "scripts")
WEB_DIR       = os.path.join(ROOT_DIR, "web")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
REDIS_TTL     = 3600  # response cache TTL — 1 hour

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

app = FastAPI(title="Acmera Support Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Redis cache ──────────────────────────────────────────────────────────────
# Two layers:
#   1. LiteLLM Redis cache  — per-call cache for all LLM completions (temp=0)
#   2. Response-level cache — skips the entire agent graph on repeated queries

def _setup_litellm_cache() -> None:
    """Wire LiteLLM to Redis so all classify/evaluate/respond calls are cached."""
    host = os.getenv("REDIS_HOST")
    if not host:
        print("[cache] REDIS_HOST not set — running without LiteLLM cache")
        return
    try:
        import litellm
        litellm.cache = litellm.Cache(
            type="redis",
            host=host,
            port=int(os.getenv("REDIS_PORT", 6379)),
        )
        print(f"[cache] LiteLLM → Redis {host}:{os.getenv('REDIS_PORT', 6379)}")
    except Exception as e:
        print(f"[cache] LiteLLM Redis setup skipped: {e}")


def _get_redis():
    """Lazy Redis client for response-level caching. Returns None if unconfigured."""
    host = os.getenv("REDIS_HOST")
    if not host:
        return None
    try:
        import redis as redis_lib
        client = redis_lib.Redis(
            host=host,
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True,
            socket_connect_timeout=1,
        )
        client.ping()
        return client
    except Exception:
        return None


def _cache_key(query: str, mode: str) -> str:
    digest = hashlib.sha256(f"{mode}:{query.strip().lower()}".encode()).hexdigest()[:20]
    return f"agent:v1:{digest}"


_setup_litellm_cache()


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


class QueryRequest(BaseModel):
    query:     str
    use_cache: bool = True


@app.post("/query")
def query_endpoint(req: QueryRequest):
    """
    Core agent endpoint — programmatic access and curl testing.
    Returns should_escalate, steps_taken, trajectory so the deliverable
    curl command can verify routing behaviour.
    """
    r   = _get_redis() if req.use_cache else None
    key = _cache_key(req.query, "agent")

    if r:
        cached = r.get(key)
        if cached:
            result = json.loads(cached)
            result["cache_hit"] = True
            return result

    try:
        from agent import run_agent
        result = run_agent(req.query, use_cache=req.use_cache)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    trace_id = result.get("trace_id")
    response = {
        "query":            req.query,
        "answer":           result["final_answer"],
        "intent":           result.get("intent", ""),
        "should_escalate":  result.get("should_escalate", False),
        "steps_taken":      result.get("steps_taken", 0),
        "tools_called":     result.get("tools_called", []),
        "trajectory":       result.get("trajectory", []),
        "difficulty_score": result.get("difficulty_score", 0),
        "generation_model": result.get("generation_model", ""),
        "elapsed_seconds":  result.get("elapsed_seconds", 0),
        "trace_id":         trace_id,
        "trace_url":        f"{LANGFUSE_HOST}/trace/{trace_id}" if trace_id else None,
        "cache_hit":        False,
    }

    if r:
        r.setex(key, REDIS_TTL, json.dumps(response))
    return response


class AskRequest(BaseModel):
    query:           str
    mode:            str = "agent"       # "agent" | "pipeline"
    model_override:  str | None = None   # None = auto | "gpt-4o-mini" | "gpt-4o"
    show_trajectory: bool = True
    use_cache:       bool = True


@app.post("/api/ask")
def ask_endpoint(req: AskRequest):
    """Web UI endpoint — supports both LangGraph agent and naive pipeline modes."""
    r   = _get_redis() if req.use_cache else None
    key = _cache_key(req.query, req.mode)

    if r:
        cached = r.get(key)
        if cached:
            result = json.loads(cached)
            result["cache_hit"] = True
            return result

    try:
        if req.mode == "agent":
            from agent import run_agent
            result   = run_agent(req.query, use_cache=req.use_cache)
            trace_id = result.get("trace_id")
            response = {
                "query":            req.query,
                "answer":           result["final_answer"],
                "intent":           result.get("intent", ""),
                "tools_called":     result.get("tools_called", []),
                "trajectory":       result.get("trajectory", []),
                "steps_taken":      result.get("steps_taken", 0),
                "difficulty_score": result.get("difficulty_score", 0),
                "generation_model": result.get("generation_model", ""),
                "should_escalate":  result.get("should_escalate", False),
                "elapsed_seconds":  result.get("elapsed_seconds", 0),
                "trace_id":         trace_id,
                "trace_url":        f"{LANGFUSE_HOST}/trace/{trace_id}" if trace_id else None,
                "mode":             "agent",
                "cache_hit":        False,
            }
        else:
            from support_pipeline import handle_query
            result   = handle_query(req.query, model=req.model_override)
            trace_id = result.get("trace_id")
            tool     = result.get("tool", "policy_kb")
            tools_called = ["order_tracker", "account_lookup"] if tool == "multi_tool" else [tool]
            response = {
                "query":            req.query,
                "answer":           result["answer"],
                "intent":           result.get("intent", ""),
                "tools_called":     tools_called,
                "trajectory":       ["retrieve", "generate"],
                "steps_taken":      1,
                "difficulty_score": result.get("difficulty_score", 0),
                "generation_model": result.get("generation_model", ""),
                "should_escalate":  False,
                "elapsed_seconds":  result.get("elapsed_seconds", 0),
                "trace_id":         trace_id,
                "trace_url":        f"{LANGFUSE_HOST}/trace/{trace_id}" if trace_id else None,
                "mode":             "pipeline",
                "cache_hit":        False,
            }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if r:
        r.setex(key, REDIS_TTL, json.dumps(response))
    return response


@app.post("/ingest")
def ingest_endpoint():
    """Create pgvector schema then embed and store all corpus documents."""
    try:
        # Only creates orders + customers tables — never touches chunks (Project A's corpus)
        from setup_db import setup_mock_tables
        setup_mock_tables()
        from seed_mock_data import seed
        seed()
        return {"status": "done", "note": "mock tables seeded; corpus lives in Project A RDS"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))
