"""
api/server.py — FastAPI server for SEC-Rag.

Exposes a single POST /query endpoint that streams results as Server-Sent Events.

Events (in order):
  {"type": "plan",    "data": {tickers, years, filing_types, reasoning}}
  {"type": "nodes",   "data": [{node_id, ticker, filing_type, fiscal_year, fiscal_quarter, section_label, score, text_preview}]}
  {"type": "token",   "data": "<token>"}   (one per streamed token)
  {"type": "sources", "data": [{ticker, filing_type, fiscal_year, section_label, node_id}]}
  {"type": "done"}
  {"type": "error",   "data": "<message>"}  (on exception)

Usage:
    uvicorn api.server:app --reload --port 8000
"""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Make src/ importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from index_loader import load_index
from sec_query import SecQueryEngine

MANIFEST_PATH = Path(__file__).parent.parent / "files" / "edgar_corpus" / "manifest.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    Settings.embed_model = OpenAIEmbedding()
    Settings.llm = OpenAI(model="gpt-4o")

    index = load_index()
    app.state.engine = SecQueryEngine(
        index=index,
        manifest_path=str(MANIFEST_PATH),
        verbose=False,
    )
    print("[server] SecQueryEngine ready.")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


def _event(payload: dict) -> dict:
    return {"data": json.dumps(payload)}


@app.post("/query")
async def query_endpoint(request: Request, body: QueryRequest):
    engine: SecQueryEngine = request.app.state.engine
    query = body.query

    async def event_generator():
        try:
            # Stage 1: Plan
            plan = await engine.aplan_query(query)
            yield _event({"type": "plan", "data": plan})

            # Stage 2: Retrieve nodes
            nodes = await engine._retrieve_nodes(query, plan)

            nodes_payload = []
            for n in nodes:
                m = n.node.metadata
                nodes_payload.append({
                    "node_id":       n.node_id,
                    "ticker":        m.get("ticker"),
                    "filing_type":   m.get("filing_type"),
                    "fiscal_year":   m.get("fiscal_year"),
                    "fiscal_quarter": m.get("fiscal_quarter"),
                    "section_label": m.get("section_label"),
                    "score":         round(n.score, 4) if n.score is not None else None,
                    "text_preview":  n.node.get_content()[:300],
                })
            yield _event({"type": "nodes", "data": nodes_payload})

            # Stage 3: Stream synthesis tokens
            async for token in engine._synthesize_stream(query, nodes):
                yield _event({"type": "token", "data": token})

            # Stage 4: Sources (deduplicated)
            seen_sources = set()
            sources = []
            for n in nodes:
                m = n.node.metadata
                key = (m.get("ticker"), m.get("filing_type"), m.get("fiscal_year"), m.get("section_label"))
                if key not in seen_sources:
                    seen_sources.add(key)
                    sources.append({
                        "ticker":        m.get("ticker"),
                        "filing_type":   m.get("filing_type"),
                        "fiscal_year":   m.get("fiscal_year"),
                        "section_label": m.get("section_label"),
                        "node_id":       n.node_id,
                    })
            yield _event({"type": "sources", "data": sources})

            yield _event({"type": "done"})

        except Exception as e:
            yield _event({"type": "error", "data": str(e)})

    return EventSourceResponse(event_generator())
