"""
sec_query.py — Single-LLM-call RAG engine for SEC EDGAR filings.

Pipeline (1 LLM call total):
  1. _retrieve_nodes — dense vector retrieval via pgvector ANN (no LLM)
  2. _synthesize_stream — single LLM call to answer the question from retrieved context

Chunk enrichment (not metadata filters) drives retrieval precision.
Each indexed chunk starts with a self-describing header:
    "Apple (AAPL) | 10-K Annual Report | FY2024 | RISK FACTORS (ITEM 1A)"
so cosine similarity alone routes queries to the right documents.

Basic usage:
    engine = SecQueryEngine(index=my_index)
    response = await engine.query("What risks did Apple cite in 2024?")
"""

import asyncio
import math
from typing import AsyncGenerator

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.postprocessor import SentenceTransformerRerank, SimilarityPostprocessor
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.tools import QueryEngineTool, ToolMetadata


class SecQueryEngine:
    """
    RAG engine for SEC filings using a single LLM call.

    Retrieval is driven by chunk enrichment: each chunk's text starts with
    a structured header (company, filing type, fiscal year, section) so that
    semantic similarity alone finds the right passages — no metadata filters,
    no planning LLM call required.

    Parameters
    ----------
    index : VectorStoreIndex
        Your pre-built LlamaIndex vector store (from pgvector).
    top_k : int
        Candidates to fetch from ANN search before reranking (default 50).
    verbose : bool
        Print retrieval summary (default True).
    debug : bool
        Attach LlamaDebugHandler for detailed tracing (default False).
    max_synthesis_nodes : int
        Maximum chunks passed to the synthesis LLM (default 20).
    """

    def __init__(
        self,
        index:               VectorStoreIndex,
        top_k:               int  = 5000,
        verbose:             bool = True,
        debug:               bool = False,
        rerank_pool:         int  = 200,
        max_synthesis_nodes: int  = 50,
    ):
        self.index               = index
        self.top_k               = top_k
        self.verbose             = verbose
        self.max_synthesis_nodes = max_synthesis_nodes
        self._rerank_pool        = rerank_pool

        if debug:
            self._debug_handler = LlamaDebugHandler(print_trace_on_end=True)
            Settings.callback_manager = CallbackManager([self._debug_handler])
        else:
            self._debug_handler = None

        # Cross-encoder reranker (~22 MB download on first use)
        # top_n matches max_synthesis_nodes so the reranker produces exactly the final set
        self._reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=max_synthesis_nodes,
        )
        # Drop nodes below cosine similarity 0.72 before reranking
        self._sim_filter = SimilarityPostprocessor(similarity_cutoff=0.40)

        if self.verbose:
            print("[SecQueryEngine] Ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def query(self, query: str) -> str:
        """Run a natural-language query against the SEC corpus."""
        nodes = await self._retrieve_nodes(query)
        if self.verbose:
            print(f"[SecQueryEngine] {len(nodes)} nodes after retrieval pipeline")
        return self._synthesize(query, nodes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _retrieve_nodes(self, query: str) -> list[NodeWithScore]:
        """
        Dense vector retrieval — no metadata filters, no LLM calls.

        Relies on chunk enrichment (company/year/section prefix in chunk text)
        to surface relevant passages via cosine similarity.
        """
        retriever = self.index.as_retriever(similarity_top_k=self.top_k)
        nodes = await retriever.aretrieve(query)

        # Deduplicate by node_id
        seen, unique = set(), []
        for n in nodes:
            if n.node_id not in seen:
                seen.add(n.node_id)
                unique.append(n)
        nodes = unique

        if self.verbose:
            scores = [n.score for n in nodes if n.score is not None]
            if scores:
                print(
                    f"[Retrieval] {len(nodes)} candidates  "
                    f"score range: {min(scores):.4f}–{max(scores):.4f}  "
                    f"mean: {sum(scores)/len(scores):.4f}"
                )
            else:
                print(f"[Retrieval] {len(nodes)} candidates (no scores)")

        # Drop off-topic nodes below similarity floor — skip if it would wipe everything
        filtered = self._sim_filter.postprocess_nodes(nodes)
        if filtered:
            if self.verbose:
                print(f"[SimFilter] {len(filtered)} nodes passed cutoff {self._sim_filter.similarity_cutoff}")
            nodes = filtered
        else:
            if self.verbose:
                print(f"[SimFilter] cutoff {self._sim_filter.similarity_cutoff} removed all nodes — skipping filter")

        # Pre-cap by cosine score before reranking — rerankers are slow on 500+ candidates
        nodes = sorted(nodes, key=lambda n: n.score or 0, reverse=True)[:self._rerank_pool]

        # Cross-encoder reranking for precision
        if len(nodes) > 1:
            nodes = self._reranker.postprocess_nodes(
                nodes, query_bundle=QueryBundle(query_str=query)
            )
            # Normalize raw logits → probabilities via sigmoid so scores are interpretable
            for n in nodes:
                if n.score is not None:
                    n.score = 1.0 / (1.0 + math.exp(-float(n.score)))
            if self.verbose:
                scores = [n.score for n in nodes if n.score is not None]
                if scores:
                    print(
                        f"[Reranker] {len(nodes)} nodes  "
                        f"score range: {min(scores):.4f}–{max(scores):.4f}"
                    )

        # Cap nodes sent to synthesis LLM to avoid context-window overflow
        nodes = sorted(nodes, key=lambda n: n.score or 0, reverse=True)[: self.max_synthesis_nodes]

        if self.verbose:
            print(f"[SecQueryEngine] {len(nodes)} nodes passed to synthesis")

        return nodes

    def _build_synthesis_prompt(self, query: str, nodes: list[NodeWithScore]) -> str:
        """Build the single synthesis prompt from retrieved chunks."""
        context_parts = []
        for i, n in enumerate(nodes, 1):
            context_parts.append(f"[{i}]\n{n.node.get_content()}")
        context = "\n\n---\n\n".join(context_parts)

        return f"""You are a financial analyst. Answer the question below using only the provided SEC filing excerpts.
Be specific, cite the company and fiscal year where relevant, and be concise.

Question: {query}

SEC Filing Excerpts:
{context}

Answer:"""

    def _synthesize(self, query: str, nodes: list[NodeWithScore]) -> str:
        """Single LLM call to produce a final answer from retrieved nodes."""
        prompt = self._build_synthesis_prompt(query, nodes)
        response = Settings.llm.complete(prompt)
        return response.text.strip()

    async def _synthesize_stream(
        self, query: str, nodes: list[NodeWithScore]
    ) -> AsyncGenerator[str, None]:
        """Streaming variant of _synthesize — yields tokens as they arrive."""
        prompt = self._build_synthesis_prompt(query, nodes)
        async for chunk in await Settings.llm.astream_complete(prompt):
            if chunk.delta:
                yield chunk.delta

    # ------------------------------------------------------------------
    # LlamaIndex-compatible interface
    # ------------------------------------------------------------------

    def as_query_engine(self) -> "BaseQueryEngine":
        """Return a standard LlamaIndex BaseQueryEngine wrapping this engine."""
        return _DynamicSecQueryEngine(self)

    def as_tool(
        self,
        name:        str = "sec_filings",
        description: str = (
            "Query SEC 10-K and 10-Q filings for major US public companies. "
            "Understands sector names (e.g. 'pharma', 'big tech'), relative time "
            "('last 2 years', 'most recent'), and filing type ('annual', 'quarterly'). "
            "Use for revenue, margins, risk factors, and management commentary."
        ),
    ) -> "QueryEngineTool":
        """Return a QueryEngineTool wrapping this engine."""
        return QueryEngineTool(
            query_engine=self.as_query_engine(),
            metadata=ToolMetadata(name=name, description=description),
        )


# ---------------------------------------------------------------------------
# Thin BaseQueryEngine wrapper — delegates to SecQueryEngine
# ---------------------------------------------------------------------------

class _DynamicSecQueryEngine(BaseQueryEngine):
    """Internal LlamaIndex-compatible wrapper. Use SecQueryEngine.as_query_engine()."""

    def __init__(self, sec_engine: "SecQueryEngine"):
        self._sec = sec_engine
        super().__init__(callback_manager=Settings.callback_manager or CallbackManager())

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            return loop.run_until_complete(self._aquery(query_bundle))
        except RuntimeError:
            return asyncio.run(self._aquery(query_bundle))

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        query_str = (
            query_bundle.query_str
            if isinstance(query_bundle, QueryBundle)
            else str(query_bundle)
        )
        nodes = await self._sec._retrieve_nodes(query_str)
        if self._sec.verbose:
            print(f"[Nodes retrieved] {len(nodes)}")
        answer = self._sec._synthesize(query_str, nodes)
        return Response(response=answer)
