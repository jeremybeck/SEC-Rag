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
import os
import sys
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
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
    Settings.llm = OpenAI(model="gpt-4o")

    index = load_index()
    app.state.engine = SecQueryEngine(index=index, verbose=True)
    print("[server] SecQueryEngine ready.")

    # Feedback DB pool — asyncpg accepts the standard postgresql:// URL directly
    feedback_pool = await asyncpg.create_pool(
        os.environ["DATABASE_URL"], min_size=1, max_size=5
    )
    async with feedback_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS query_feedback (
                id            SERIAL PRIMARY KEY,
                query_text    TEXT        NOT NULL,
                answer_text   TEXT        NOT NULL,
                cited_sources JSONB,
                data_quality  TEXT,
                rating        TEXT,
                feedback_text TEXT,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
    app.state.feedback_pool = feedback_pool
    print("[server] Feedback table ready.")

    yield

    await feedback_pool.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class FeedbackRequest(BaseModel):
    query_text:    str
    answer_text:   str
    cited_sources: list[dict] = []
    data_quality:  str | None = None
    rating:        str | None = None   # "up" | "down" | None
    feedback_text: str | None = None


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

            # Stage 3: Synthesize answer (single LLM call)
            result = engine._synthesize(query, nodes)
            yield _event({"type": "token", "data": result.answer})

            # Stage 3b: Data quality assessment
            if result.data_quality is not None:
                yield _event({"type": "quality", "data": {
                    "rating":           result.data_quality.rating,
                    "summary":          result.data_quality.summary,
                    "missing_coverage": result.data_quality.missing_coverage,
                }})

            # Stage 4: Sources — deduplicated by section, collecting all citation indices per section
            cited_set = set(result.cited_node_ids)
            node_id_to_citation_index = {nid: i + 1 for i, nid in enumerate(result.cited_node_ids)}
            node_id_to_quote = dict(zip(result.cited_node_ids, result.cited_quotes))

            # First pass: group all citation indices and quotes by section key
            section_indices: dict = defaultdict(list)
            section_quotes:  dict = defaultdict(list)
            section_node_id: dict = {}
            section_meta:    dict = {}
            for n in nodes:
                if n.node_id not in cited_set:
                    continue
                m = n.node.metadata
                key = (m.get("ticker"), m.get("filing_type"), m.get("fiscal_year"), m.get("section_label"))
                section_indices[key].append(node_id_to_citation_index[n.node_id])
                q = node_id_to_quote.get(n.node_id, "")
                if q:
                    section_quotes[key].append(q)
                if key not in section_node_id:
                    section_node_id[key] = n.node_id
                    section_meta[key] = m

            # Second pass: build sources in citation-index order (sort by first index in each group)
            sources = []
            for key in sorted(section_indices, key=lambda k: section_indices[k][0]):
                m = section_meta[key]
                indices = sorted(section_indices[key])
                quotes  = section_quotes[key]
                sources.append({
                    "ticker":           m.get("ticker"),
                    "filing_type":      m.get("filing_type"),
                    "fiscal_year":      m.get("fiscal_year"),
                    "section_label":    m.get("section_label"),
                    "node_id":          section_node_id[key],
                    "citation_indices": indices,
                    "quote":            quotes[0] if quotes else "",
                })
            yield _event({"type": "sources", "data": sources})

            yield _event({"type": "done"})

        except Exception as e:
            yield _event({"type": "error", "data": str(e)})

    return EventSourceResponse(event_generator())


@app.post("/feedback", status_code=204)
async def submit_feedback(request: Request, body: FeedbackRequest):
    pool = request.app.state.feedback_pool
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO query_feedback
                (query_text, answer_text, cited_sources, data_quality, rating, feedback_text)
            VALUES ($1, $2, $3::jsonb, $4, $5, $6)
            """,
            body.query_text,
            body.answer_text,
            json.dumps(body.cited_sources),
            body.data_quality,
            body.rating,
            body.feedback_text or None,
        )
