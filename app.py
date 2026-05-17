"""
Project B — FastAPI Web Server

Run:
  uvicorn app:app --host 0.0.0.0 --port 8081 --reload
Open: http://localhost:8081
"""
import os
import sys

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

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

app = FastAPI(title="Acmera Support Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


class AskRequest(BaseModel):
    query:              str
    mode:               str = "agent"      # "agent" | "pipeline"
    model_override:     str | None = None  # None = auto | "gpt-4o-mini" | "gpt-4o"
    show_trajectory:    bool = True


@app.post("/api/ask")
def ask_endpoint(req: AskRequest):
    try:
        if req.mode == "agent":
            from agent import run_agent
            result   = run_agent(req.query)
            trace_id = result.get("trace_id")
            return {
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
            }
        else:
            from support_pipeline import handle_query
            result   = handle_query(req.query, model=req.model_override)
            trace_id = result.get("trace_id")
            return {
                "query":            req.query,
                "answer":           result["answer"],
                "intent":           result.get("intent", ""),
                "tools_called":     [result.get("tool", "policy_kb")],
                "trajectory":       ["retrieve", "generate"],
                "steps_taken":      1,
                "difficulty_score": result.get("difficulty_score", 0),
                "generation_model": result.get("generation_model", ""),
                "should_escalate":  False,
                "elapsed_seconds":  result.get("elapsed_seconds", 0),
                "trace_id":         trace_id,
                "trace_url":        f"{LANGFUSE_HOST}/trace/{trace_id}" if trace_id else None,
                "mode":             "pipeline",
            }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))
