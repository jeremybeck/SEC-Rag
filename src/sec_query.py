"""
sec_query.py — Single-LLM-call RAG engine for SEC EDGAR filings.

Pipeline (1 LLM call total):
  1. _retrieve_nodes — dense vector retrieval via pgvector ANN (no LLM)
  2. _synthesize — single LLM call; returns SynthesisResult with answer, citations, and confidence

Chunk enrichment (not metadata filters) drives retrieval precision.
Each indexed chunk starts with a self-describing header:
    "Apple (AAPL) | 10-K Annual Report | FY2024 | RISK FACTORS (ITEM 1A)"
so cosine similarity alone routes queries to the right documents.

Basic usage:
    engine = SecQueryEngine(index=my_index)
    response = await engine.query("What risks did Apple cite in 2024?")
"""

import asyncio
import datetime
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Literal

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.postprocessor import SentenceTransformerRerank, SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    FilterCondition,
)
from pydantic import BaseModel

CURRENT_YEAR: int = datetime.date.today().year


class Citation(BaseModel):
    """A single citation: the chunk index and the verbatim quote from that chunk."""
    index: int
    quote: str


class DataQualityAssessment(BaseModel):
    """LLM-generated assessment of how well the retrieved context covers the question."""
    rating:           Literal["HIGH", "MEDIUM", "LOW"]
    summary:          str
    missing_coverage: list[str]


class SynthesisResponse(BaseModel):
    """Structured output schema for the synthesis LLM call."""
    answer:       str
    citations:    list[Citation]
    data_quality: DataQualityAssessment


@dataclass
class SynthesisResult:
    """Return value from _synthesize — named fields instead of a positional tuple."""
    answer:         str
    cited_node_ids: list[str]
    cited_quotes:   list[str]
    data_quality:   DataQualityAssessment | None


# ─────────────────────────────────────────────────────────────────────────────
# Query-to-metadata filter helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_year_filters(query: str) -> list[int] | None:
    """
    Extract fiscal years mentioned in a query.

    Handles:
    - Explicit years: "2024", "fiscal 2025"
    - Relative spans: "last 3 years", "past two years"
    - Recency signals: "recent", "latest", "current" → last 2 years

    Returns a list of ints, or None if no year signal is found.
    """
    years: set[int] = set()

    for y in re.findall(r'\b(202[3-9]|20[3-9]\d)\b', query):
        years.add(int(y))

    n_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    m = re.search(r'(?:last|past)\s+(\w+)\s+years?', query, re.IGNORECASE)
    if m:
        n_str = m.group(1).lower()
        n = int(n_str) if n_str.isdigit() else n_map.get(n_str, 2)
        years.update(CURRENT_YEAR - i for i in range(n))

    if re.search(r'\b(recent|latest|current)\b', query, re.IGNORECASE):
        years.update([CURRENT_YEAR - 1, CURRENT_YEAR])

    return list(years) if years else None


def extract_company_filters(query: str) -> list[str] | None:
    """
    Extract ticker symbols from a query (handles both symbols and full names).

    Ticker symbols: matched as uppercase sequences (e.g. "AAPL", "JPM").
    Company names: whole-word matched against words >= 5 chars from TICKER_NAMES
    (e.g. "Apple", "Google", "Johnson") to avoid false positives on short fragments.

    Returns a sorted list of uppercase tickers, or None if no company found.
    """
    try:
        from filing_parser import TICKER_NAMES
    except ImportError:
        return None

    # Map from unambiguous company name words (>= 5 chars) to ticker.
    # Short tickers (< 3 chars) like "T", "V", "MA" are too ambiguous for
    # word-based matching — they are caught by the uppercase regex below.
    name_map: dict[str, str] = {}
    for ticker, name in TICKER_NAMES.items():
        for word in name.lower().split():
            # Strip punctuation like "&" from "AT&T", skip short words
            clean = re.sub(r'[^a-z]', '', word)
            if len(clean) >= 5:
                name_map[clean] = ticker

    found: set[str] = set()

    # Match known ticker symbols appearing as uppercase tokens in the query
    for m in re.finditer(r'\b([A-Z]{2,5})\b', query):
        candidate = m.group(1)
        if candidate in TICKER_NAMES:
            found.add(candidate)

    # Match company name words as whole words (not substrings)
    for word, ticker in name_map.items():
        if re.search(r'\b' + word + r'\b', query, re.IGNORECASE):
            found.add(ticker)

    return sorted(found) if found else None


def build_metadata_filters(query: str) -> MetadataFilters | None:
    """
    Build LlamaIndex MetadataFilters from fiscal year signals in a query.

    Year filters use fiscal_year (int), ORed together.
    Returns None if no year signal is found (no pre-filtering applied).
    """
    years = extract_year_filters(query)
    if not years:
        return None

    return MetadataFilters(
        filters=[
            MetadataFilter(key="fiscal_year", value=y, operator=FilterOperator.EQ)
            for y in years
        ],
        condition=FilterCondition.OR,
    )


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
        prompt_file:         str   = "synthesis_v3.txt",
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
        metadata_filters = build_metadata_filters(query)
        retriever = self.index.as_retriever(
            similarity_top_k=self.top_k,
            filters=metadata_filters,
        )
        if self.verbose and metadata_filters is not None:
            print(f"[Retrieval] metadata filter — years={extract_year_filters(query)}")
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
        return (
            self._prompt_template
            .replace("{query}", query)
            .replace("{context}", context)
        )

    # word/ticker fragment → canonical ticker, built once at class level
    _NAME_TO_TICKER: dict[str, str] = {}
    _ALL_TICKERS:    set[str]        = set()

    @classmethod
    def _build_name_map(cls) -> None:
        """Populate _NAME_TO_TICKER and _ALL_TICKERS from filing_parser.TICKER_NAMES."""
        if cls._NAME_TO_TICKER:
            return
        try:
            from filing_parser import TICKER_NAMES
            for ticker, name in TICKER_NAMES.items():
                cls._ALL_TICKERS.add(ticker)
                cls._NAME_TO_TICKER[ticker.lower()] = ticker
                for word in name.lower().split():
                    if len(word) >= 4:
                        cls._NAME_TO_TICKER[word] = ticker
        except ImportError:
            pass

    def _check_claimed_numbers(
        self,
        answer: str,
        citations: list[Citation],
        nodes: list[NodeWithScore],
    ) -> list[int]:
        """
        For each [N] citation in the answer, find numbers claimed in nearby text
        and verify they appear in node N's full text.

        Returns original citation indices where claimed numbers cannot be found
        in the source node — indicating the model drew on training knowledge
        rather than the provided excerpt.

        Number patterns matched: dollar amounts, comma-separated figures,
        and numbers followed by billion/million/thousand/%.
        """
        number_pat = re.compile(
            r'\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|thousand|B|M))?'
            r'|\b\d+(?:,\d{3})+(?:\.\d+)?\b'
            r'|\b\d+(?:\.\d+)?\s*(?:billion|million|thousand|%)',
            re.IGNORECASE,
        )
        citation_pat = re.compile(r'\[(\d+)\]')

        valid_orig = {c.index for c in citations}
        cit_positions = [
            (m.start(), int(m.group(1)))
            for m in citation_pat.finditer(answer)
            if int(m.group(1)) in valid_orig
        ]
        if not cit_positions:
            return []

        unverified: set[int] = set()

        for nm in number_pat.finditer(answer):
            num_str = nm.group(0)
            pos     = nm.start()

            # Find nearest valid [N] within 300 chars before or 100 chars after
            nearest_idx, best_dist = None, 9999
            for cpos, cidx in cit_positions:
                if cpos <= pos:
                    d = pos - cpos
                    if d <= 300 and d < best_dist:
                        best_dist, nearest_idx = d, cidx
                else:
                    d = cpos - pos
                    if d <= 100 and d < best_dist:
                        best_dist, nearest_idx = d, cidx

            if nearest_idx is None or not (1 <= nearest_idx <= len(nodes)):
                continue

            node_text = nodes[nearest_idx - 1].node.get_content()
            norm_text = node_text.lower().replace(',', '').replace('$', '')
            norm_num  = num_str.lower().replace(',', '').replace('$', '').strip()

            if norm_num and len(norm_num) >= 3 and norm_num not in norm_text:
                unverified.add(nearest_idx)

        return list(unverified)

    def _synthesize(self, query: str, nodes: list[NodeWithScore]) -> SynthesisResult:
        """
        Single LLM call via structured_predict.

        Citation indices in the answer are renumbered 1-based by first appearance.
        Returns a SynthesisResult with answer, cited_node_ids, cited_quotes, and data_quality.
        """
        prompt = self._build_synthesis_prompt(query, nodes)
        try:
            result: SynthesisResponse = Settings.llm.structured_predict(
                SynthesisResponse,
                PromptTemplate("{prompt_str}"),
                prompt_str=prompt,
            )
            # Build quote lookup keyed by original chunk index
            quote_map: dict[int, str] = {c.index: c.quote for c in result.citations}

            # Collect original citation indices in order of first appearance in answer text
            seen: list[int] = []
            for m in re.finditer(r'\[(\d+)\]', result.answer):
                idx = int(m.group(1))
                if idx not in seen and 1 <= idx <= len(nodes):
                    seen.append(idx)
            # Append any cited but not mentioned inline
            for c in result.citations:
                if isinstance(c.index, int) and c.index not in seen and 1 <= c.index <= len(nodes):
                    seen.append(c.index)

            # Renumber [N] → [1], [2], ... by first-appearance order
            remap = {orig: disp for disp, orig in enumerate(seen, 1)}
            answer = re.sub(
                r'\[(\d+)\]',
                lambda m: f"[{remap[int(m.group(1))]}]" if int(m.group(1)) in remap else m.group(0),
                result.answer,
            )
            cited_node_ids = [nodes[orig - 1].node_id for orig in seen]
            cited_quotes   = [quote_map.get(orig, "") for orig in seen]

            # --- Programmatic calibration overrides ---
            # Work on original (pre-remap) answer so citation indices match nodes directly.
            data_quality = result.data_quality

            # 1. Number verification: check that specific numbers claimed near each [N]
            #    can actually be found in node N's full text. This catches the main
            #    confabulation pattern: model recalls a specific figure from training
            #    knowledge, cites a related-but-different excerpt, and rates itself HIGH.
            if data_quality is not None and result.citations:
                unverified = self._check_claimed_numbers(
                    result.answer, result.citations, nodes
                )
                n_total = len(result.citations)
                n_bad   = len(unverified)
                if n_bad > 0 and data_quality.rating == "HIGH":
                    frac_bad   = n_bad / n_total
                    new_rating = "LOW" if frac_bad > 0.5 else "MEDIUM"
                    caveats = [
                        f"Citation [{i}]: specific numbers in answer not found in source chunk"
                        for i in sorted(unverified)
                    ]
                    data_quality = DataQualityAssessment(
                        rating=new_rating,
                        summary=(
                            f"Number verification override: {n_bad}/{n_total} cited sources "
                            f"do not contain the specific figures claimed in the answer. "
                            f"Answer may draw on training knowledge rather than provided excerpts."
                        ),
                        missing_coverage=caveats,
                    )
                    print(
                        f"[SecQueryEngine] Rating overridden HIGH→{new_rating}: "
                        f"{n_bad}/{n_total} citations have unverified numbers"
                    )

            # 2. Ticker / company mismatch: if query names a company and all cited nodes
            #    are from a different company, the answer is almost certainly from
            #    training knowledge. Handles both ticker symbols ("JNJ") and full names
            #    ("Johnson & Johnson").
            if data_quality is not None and seen:
                self._build_name_map()
                query_lower = query.lower()
                asked_ticker = None
                # Try ticker symbol first (e.g. "JNJ", "GOOG") — O(1) lookup
                tm = re.search(r'\b([A-Z]{2,5})\b', query)
                if tm and tm.group(1) in self._ALL_TICKERS:
                    asked_ticker = tm.group(1)
                # Fall back to company-name word match
                if asked_ticker is None:
                    for word, ticker in self._NAME_TO_TICKER.items():
                        if word in query_lower:
                            asked_ticker = ticker
                            break

                if asked_ticker:
                    cited_tickers_set = {
                        nodes[orig - 1].node.metadata.get("ticker", "")
                        for orig in seen
                    }
                    if cited_tickers_set and asked_ticker not in cited_tickers_set:
                        data_quality = DataQualityAssessment(
                            rating="LOW",
                            summary=(
                                f"Source mismatch: question asks about {asked_ticker} but "
                                f"all cited sources are from "
                                f"{', '.join(sorted(cited_tickers_set))}. "
                                f"Answer likely drawn from training knowledge."
                            ),
                            missing_coverage=[
                                f"No {asked_ticker} excerpts were cited; "
                                f"cited sources: {', '.join(sorted(cited_tickers_set))}"
                            ],
                        )
                        print(
                            f"[SecQueryEngine] Rating overridden→LOW: company mismatch "
                            f"(asked={asked_ticker}, cited={cited_tickers_set})"
                        )

            return SynthesisResult(
                answer=answer,
                cited_node_ids=cited_node_ids,
                cited_quotes=cited_quotes,
                data_quality=data_quality,
            )
        except Exception as e:
            print(f"[SecQueryEngine] WARNING: structured_predict failed ({e}) — falling back to plain complete")
            response = Settings.llm.complete(prompt)
            return SynthesisResult(
                answer=response.text.strip(),
                cited_node_ids=[],
                cited_quotes=[],
                data_quality=None,
            )

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
        r = self._sec._synthesize(query_str, nodes)
        return Response(response=r.answer)
