"""
sec_query.py — Single-LLM-call RAG engine for SEC EDGAR filings.

Pipeline (1 LLM call total):
  1. _retrieve_nodes — dense vector retrieval via pgvector ANN (no LLM)
  2. _synthesize — single LLM call; returns (answer_text, cited_node_ids)

Chunk enrichment (not metadata filters) drives retrieval precision.
Each indexed chunk starts with a self-describing header:
    "Apple (AAPL) | 10-K Annual Report | FY2024 | RISK FACTORS (ITEM 1A)"
so cosine similarity alone routes queries to the right documents.

Basic usage:
    engine = SecQueryEngine(index=my_index)
    response = await engine.query("What risks did Apple cite in 2024?")
"""

import asyncio
import json
import math
import re
from collections import defaultdict
from pathlib import Path
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

    _PROMPTS_DIR = Path(__file__).parent / "prompts"

    def __init__(
        self,
        index:               VectorStoreIndex,
        top_k:               int  = 2500,
        verbose:             bool = True,
        debug:               bool = False,
        rerank_pool:         int   = 200,
        rerank_per_ticker:   int   = 15,
        max_synthesis_nodes: int   = 50,
        mmr_lambda:          float = 0.5,
        mmr_soft_cap:        int   = 5,
        prompt_file:         str   = "synthesis_v2.txt",
    ):
        self.index                = index
        self.top_k                = top_k
        self.verbose              = verbose
        self.max_synthesis_nodes  = max_synthesis_nodes
        self._rerank_pool         = rerank_pool
        self._rerank_per_ticker   = rerank_per_ticker
        self._mmr_lambda          = mmr_lambda
        self._mmr_soft_cap        = mmr_soft_cap
        self._prompt_template     = (self._PROMPTS_DIR / prompt_file).read_text()

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

        # Pre-rerank pool: guarantee each ticker gets at least rerank_per_ticker candidates,
        # then fill remaining slots with the highest-cosine nodes from any ticker.
        # Without this, a dominant company fills all slots and the cross-encoder never
        # sees the other company's chunks.
        nodes = sorted(nodes, key=lambda n: n.score or 0, reverse=True)
        ticker_counts: dict[str, int] = defaultdict(int)
        pool, remainder = [], []
        for n in nodes:
            ticker = n.node.metadata.get("ticker", "")
            if ticker_counts[ticker] < self._rerank_per_ticker:
                pool.append(n)
                ticker_counts[ticker] += 1
            else:
                remainder.append(n)
        for n in remainder:
            if len(pool) >= self._rerank_pool:
                break
            pool.append(n)
        nodes = pool

        if self.verbose:
            print(f"[PreRerank] {len(nodes)} candidates across {len(ticker_counts)} tickers")

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

        # Ticker-aware MMR selection — balances relevance with company diversity
        nodes = self._mmr_diversity_select(nodes)

        if self.verbose:
            print(f"[SecQueryEngine] {len(nodes)} nodes passed to synthesis")

        return nodes

    def _mmr_diversity_select(self, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """
        Greedy MMR that balances reranker relevance with ticker diversity.

        Each step selects the candidate that maximises:
            mmr_lambda * reranker_score  -  (1 - mmr_lambda) * overrepresentation_penalty

        overrepresentation_penalty = min(chunks_already_selected_for_ticker / mmr_soft_cap, 1.0)

        With mmr_lambda=0.5 and mmr_soft_cap=5, the first 5 chunks from any
        company are chosen on pure relevance; beyond that, penalty ramps to 1.0
        and other companies' chunks are strongly preferred.
        """
        from collections import Counter, defaultdict

        selected: list[NodeWithScore] = []
        candidates = list(nodes)
        ticker_counts: dict[str, int] = defaultdict(int)

        while len(selected) < self.max_synthesis_nodes and candidates:
            best_score = -float("inf")
            best_idx   = 0
            for i, n in enumerate(candidates):
                relevance = float(n.score or 0.0)
                ticker    = n.node.metadata.get("ticker", "")
                penalty   = min(ticker_counts[ticker] / self._mmr_soft_cap, 1.0)
                combined  = self._mmr_lambda * relevance - (1 - self._mmr_lambda) * penalty
                if combined > best_score:
                    best_score = combined
                    best_idx   = i

            chosen = candidates.pop(best_idx)
            selected.append(chosen)
            ticker_counts[chosen.node.metadata.get("ticker", "")] += 1

        if self.verbose:
            dist = dict(Counter(n.node.metadata.get("ticker") for n in selected))
            print(f"[MMR] {len(selected)} nodes — ticker distribution: {dist}")

        return selected

    def _build_synthesis_prompt(self, query: str, nodes: list[NodeWithScore]) -> str:
        """Build the single synthesis prompt from retrieved chunks."""
        context_parts = []
        for i, n in enumerate(nodes, 1):
            context_parts.append(f"[{i}]\n{n.node.get_content()}")
        context = "\n\n---\n\n".join(context_parts)
        return self._prompt_template.format(query=query, context=context)

    def _synthesize(self, query: str, nodes: list[NodeWithScore]) -> tuple[str, list[str]]:
        """Single LLM call; returns (answer_text, cited_node_ids)."""
        prompt = self._build_synthesis_prompt(query, nodes)
        response = Settings.llm.complete(prompt)
        return self._parse_synthesis_response(response.text.strip(), nodes)

    def _parse_synthesis_response(
        self, text: str, nodes: list[NodeWithScore]
    ) -> tuple[str, list[str]]:
        """
        Parse the JSON synthesis response into (answer_text, cited_node_ids).

        Tries strict JSON parse first; falls back to regex extraction of the
        outermost {...} block (handles markdown fences or stray preamble).
        If both fail, returns (raw_text, []) so the caller always gets a string.
        """
        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if parsed is None or not isinstance(parsed, dict):
            print("[SecQueryEngine] WARNING: could not parse JSON response — returning raw text")
            return text, []

        answer = parsed.get("answer", text)
        raw_citations = parsed.get("citations", [])
        cited_node_ids = [
            nodes[i - 1].node_id
            for i in raw_citations
            if isinstance(i, int) and 1 <= i <= len(nodes)
        ]
        return answer, cited_node_ids

    # NOTE: no longer called by the server; retained for potential direct use.
    async def _synthesize_stream(
        self, query: str, nodes: list[NodeWithScore]
    ) -> AsyncGenerator[str, None]:
        """Streaming variant — yields raw tokens. Does not parse citations."""
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
        answer, _ = self._sec._synthesize(query_str, nodes)
        return Response(response=answer)
