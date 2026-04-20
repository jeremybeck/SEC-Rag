"""
api/server.py — FastAPI server for SEC-Rag.

Exposes POST /query (SSE) and POST /feedback endpoints.

SSE event sequence for /query:
  {"type": "filters", "data": {tickers, years, industries}}          — immediate, pre-retrieval
  {"type": "nodes",   "data": [{node_id, ticker, filing_type, ...}]} — after retrieval pipeline
  {"type": "token",   "data": "<chunk>"}                             — streamed answer characters (multiple events, raw indices)
  {"type": "answer",  "data": "<answer>"}                            — final answer with remapped citation indices (replaces streamed text)
  {"type": "quality", "data": {rating, summary, missing_coverage}}   — NLI faithfulness result
  {"type": "sources", "data": [{ticker, ..., citation_index, quote}]}— one entry per citation
  {"type": "done"}
  {"type": "error",   "data": "<message>"}                           — on exception

Usage:
    uvicorn api.server:app --reload --port 8000
"""

import asyncio
import json
import os
import sys
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
    """
    FastAPI lifespan context manager.

    On startup: initialises LlamaIndex settings, loads the pgvector index,
    creates the SecQueryEngine (which pre-loads both the cross-encoder reranker
    and the DeBERTa NLI model), and sets up the asyncpg feedback pool.
    On shutdown: closes the feedback pool.
    """
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
    """Wrap a dict as an SSE data frame for sse-starlette."""
    return {"data": json.dumps(payload)}


@app.post("/query")
async def query_endpoint(request: Request, body: QueryRequest):
    engine: SecQueryEngine = request.app.state.engine
    query = body.query

    async def event_generator():
        try:
            # Stage 0: Emit metadata filters immediately — fast spaCy + regex, no I/O
            filters = engine.parse_query_filters(query)
            yield _event({"type": "filters", "data": filters})

            # Stage 1: Retrieve nodes (no LLM, pure vector search)
            retrieval = await engine._retrieve_nodes(query)
            nodes = retrieval.nodes

            nodes_payload = []
            for n in nodes:
                m = n.node.metadata
                nodes_payload.append({
                    "node_id":        n.node_id,
                    "ticker":         m.get("ticker"),
                    "filing_type":    m.get("filing_type"),
                    "fiscal_year":    m.get("fiscal_year"),
                    "fiscal_quarter": m.get("fiscal_quarter"),
                    "section_label":  m.get("section_label"),
                    "score":          round(float(n.score), 4) if n.score is not None else None,
                    "text_preview":   n.node.get_content()[:300],
                })
            yield _event({"type": "nodes", "data": nodes_payload})

            # Stage 3: Stream synthesis answer token by token, collect final result
            result = None
            async for item in engine._synthesize_streaming(query, nodes):
                if isinstance(item, str):
                    yield _event({"type": "token", "data": item})
                else:
                    result = item

            if result is None:
                # Streaming yielded nothing — fall back to blocking synthesis
                result = await asyncio.get_event_loop().run_in_executor(
                    None, engine._synthesize, query, nodes
                )

            # Replace the streamed raw-index answer with the remapped display answer
            yield _event({"type": "answer", "data": result.answer})

            # Stage 3b: Data quality assessment
            if result.data_quality is not None:
                yield _event({"type": "quality", "data": {
                    "rating":           result.data_quality.rating,
                    "summary":          result.data_quality.summary,
                    "missing_coverage": result.data_quality.missing_coverage,
                }})

            # Stage 4: Sources — one entry per citation index, each with its own quote
            node_lookup = {n.node_id: n for n in nodes}
            sources = []
            for i, nid in enumerate(result.cited_node_ids):
                citation_index = i + 1
                quote = result.cited_quotes[i] if i < len(result.cited_quotes) else ""
                node = node_lookup.get(nid)
                if node is None:
                    continue
                m = node.node.metadata
                sources.append({
                    "ticker":         m.get("ticker"),
                    "filing_type":    m.get("filing_type"),
                    "fiscal_year":    m.get("fiscal_year"),
                    "section_label":  m.get("section_label"),
                    "node_id":        nid,
                    "citation_index": citation_index,
                    "quote":          quote,
                })
            yield _event({"type": "sources", "data": sources})

            yield _event({"type": "done"})

        except Exception as e:
            yield _event({"type": "error", "data": str(e)})

    return EventSourceResponse(event_generator())


@app.post("/feedback", status_code=204)
async def submit_feedback(request: Request, body: FeedbackRequest):
    """Persist a thumbs-up/down rating and optional free-text to query_feedback."""
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
