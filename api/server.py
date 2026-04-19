"""
api/server.py — FastAPI server for SEC-Rag.

Exposes a single POST /query endpoint that streams results as Server-Sent Events.

Events (in order):
  {"type": "plan",    "data": {tickers, years, filing_types, reasoning}}
  {"type": "nodes",   "data": [{node_id, ticker, filing_type, fiscal_year, fiscal_quarter, section_label, score, text_preview}]}
  {"type": "token",   "data": "<answer>"}  (one event containing the complete answer)
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(
        model="gpt-4o",
        additional_kwargs={
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "synthesis_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer":    {"type": "string"},
                            "citations": {"type": "array", "items": {"type": "integer"}},
                        },
                        "required": ["answer", "citations"],
                        "additionalProperties": False,
                    },
                },
            }
        },
    )

    index = load_index()
    app.state.engine = SecQueryEngine(index=index, verbose=True)
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
            # Stage 1: Retrieve nodes (no LLM, pure vector search)
            nodes = await engine._retrieve_nodes(query)

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
                    "score":         round(float(n.score), 4) if n.score is not None else None,
                    "text_preview":  n.node.get_content()[:300],
                })
            yield _event({"type": "nodes", "data": nodes_payload})

            # Stage 3: Synthesize answer (single LLM call; returns cited node IDs)
            answer, cited_node_ids = engine._synthesize(query, nodes)
            yield _event({"type": "token", "data": answer})

            # Stage 4: Sources — only nodes actually cited in the answer
            cited_set = set(cited_node_ids)
            seen_sources = set()
            sources = []
            for n in nodes:
                if n.node_id not in cited_set:
                    continue
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
