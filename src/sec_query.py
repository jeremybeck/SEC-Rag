"""
sec_query.py
------------
Dynamic query planning and retrieval for SEC EDGAR filings
using LlamaIndex + OpenAI.

Pipeline (2 LLM calls total):
  1. _plan_query  — one LLM call before retrieval to resolve tickers/years/types
  2. _retrieve_nodes — pure vector retrieval, no LLM
  3. _synthesize  — one LLM call after retrieval to produce the final answer

Basic usage:
    from sec_query import SecQueryEngine

    engine = SecQueryEngine(
        index=my_existing_index,
        manifest_path="manifest.json",
    )

    # Option 1: call directly (async)
    response = await engine.query("What did pharma companies say about drug pricing in 2024?")
    print(response)

    # Option 2: use as a standard LlamaIndex query engine
    #   — works with agents, pipelines, RouterQueryEngine, etc.
    query_engine = engine.as_query_engine()
    response = await query_engine.aquery("Compare AAPL and MSFT margins in 2024")
    print(response)

    # Option 3: wrap as a tool for an agent or RouterQueryEngine
    tool = engine.as_tool(
        name="sec_filings",
        description="Query SEC 10-K and 10-Q filings across all companies and years."
    )
"""

import asyncio
import json
from datetime import date
from typing import AsyncGenerator, Optional

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from pydantic import BaseModel, Field

from filing_parser import FilingInfo, parse_filing_filename


# ---------------------------------------------------------------------------
# Structured output schema for query planning
# ---------------------------------------------------------------------------

class QueryPlan(BaseModel):
    tickers: list[str] = Field(description="Ticker symbols to query, e.g. ['AAPL', 'MSFT']")
    years: list[int] = Field(description="Fiscal years to query, e.g. [2023, 2024]")
    filing_types: list[str] = Field(description="Filing types to query: '10-K', '10-Q', or both")
    reasoning: str = Field(description="One sentence explanation of why these tickers/years/types were chosen")


# ---------------------------------------------------------------------------
# Ticker metadata — company name, sector, industry
# Extend this dict with any additional tickers in your corpus
# ---------------------------------------------------------------------------
TICKER_METADATA: dict[str, dict] = {
    "AAPL":  {"name": "Apple",             "sector": "Technology",       "industry": "Consumer Electronics"},
    "MSFT":  {"name": "Microsoft",         "sector": "Technology",       "industry": "Software"},
    "NVDA":  {"name": "NVIDIA",            "sector": "Technology",       "industry": "Semiconductors"},
    "AMZN":  {"name": "Amazon",            "sector": "Technology",       "industry": "E-Commerce / Cloud"},
    "GOOG":  {"name": "Alphabet",          "sector": "Technology",       "industry": "Internet / Advertising"},
    "META":  {"name": "Meta",              "sector": "Technology",       "industry": "Social Media"},
    "AMD":   {"name": "AMD",               "sector": "Technology",       "industry": "Semiconductors"},
    "INTC":  {"name": "Intel",             "sector": "Technology",       "industry": "Semiconductors"},
    "CRM":   {"name": "Salesforce",        "sector": "Technology",       "industry": "Software"},
    "ADBE":  {"name": "Adobe",             "sector": "Technology",       "industry": "Software"},
    "ORCL":  {"name": "Oracle",            "sector": "Technology",       "industry": "Software"},
    "CSCO":  {"name": "Cisco",             "sector": "Technology",       "industry": "Networking"},
    "IBM":   {"name": "IBM",               "sector": "Technology",       "industry": "IT Services"},
    "TSLA":  {"name": "Tesla",             "sector": "Consumer",         "industry": "Electric Vehicles"},
    "JPM":   {"name": "JPMorgan Chase",    "sector": "Financials",       "industry": "Banking"},
    "BAC":   {"name": "Bank of America",   "sector": "Financials",       "industry": "Banking"},
    "GS":    {"name": "Goldman Sachs",     "sector": "Financials",       "industry": "Investment Banking"},
    "MS":    {"name": "Morgan Stanley",    "sector": "Financials",       "industry": "Investment Banking"},
    "AXP":   {"name": "American Express",  "sector": "Financials",       "industry": "Payments"},
    "BLK":   {"name": "BlackRock",         "sector": "Financials",       "industry": "Asset Management"},
    "BRK":   {"name": "Berkshire Hathaway","sector": "Financials",       "industry": "Conglomerate"},
    "V":     {"name": "Visa",              "sector": "Financials",       "industry": "Payments"},
    "MA":    {"name": "Mastercard",        "sector": "Financials",       "industry": "Payments"},
    "JNJ":   {"name": "Johnson & Johnson", "sector": "Healthcare",       "industry": "Pharmaceuticals"},
    "PFE":   {"name": "Pfizer",            "sector": "Healthcare",       "industry": "Pharmaceuticals"},
    "MRK":   {"name": "Merck",             "sector": "Healthcare",       "industry": "Pharmaceuticals"},
    "LLY":   {"name": "Eli Lilly",         "sector": "Healthcare",       "industry": "Pharmaceuticals"},
    "ABBV":  {"name": "AbbVie",            "sector": "Healthcare",       "industry": "Pharmaceuticals"},
    "UNH":   {"name": "UnitedHealth",      "sector": "Healthcare",       "industry": "Health Insurance"},
    "TMO":   {"name": "Thermo Fisher",     "sector": "Healthcare",       "industry": "Life Sciences"},
    "XOM":   {"name": "ExxonMobil",        "sector": "Energy",           "industry": "Oil & Gas"},
    "CVX":   {"name": "Chevron",           "sector": "Energy",           "industry": "Oil & Gas"},
    "KO":    {"name": "Coca-Cola",         "sector": "Consumer Staples", "industry": "Beverages"},
    "PEP":   {"name": "PepsiCo",           "sector": "Consumer Staples", "industry": "Beverages"},
    "WMT":   {"name": "Walmart",           "sector": "Consumer Staples", "industry": "Retail"},
    "COST":  {"name": "Costco",            "sector": "Consumer Staples", "industry": "Retail"},
    "TGT":   {"name": "Target",            "sector": "Consumer Staples", "industry": "Retail"},
    "PG":    {"name": "Procter & Gamble",  "sector": "Consumer Staples", "industry": "Household Products"},
    "NKE":   {"name": "Nike",              "sector": "Consumer",         "industry": "Apparel & Footwear"},
    "MCD":   {"name": "McDonald's",        "sector": "Consumer",         "industry": "Restaurants"},
    "SBUX":  {"name": "Starbucks",         "sector": "Consumer",         "industry": "Restaurants"},
    "DIS":   {"name": "Disney",            "sector": "Communication",    "industry": "Entertainment"},
    "NFLX":  {"name": "Netflix",           "sector": "Communication",    "industry": "Streaming"},
    "CMCSA": {"name": "Comcast",           "sector": "Communication",    "industry": "Telecom / Media"},
    "T":     {"name": "AT&T",              "sector": "Communication",    "industry": "Telecom"},
    "VZ":    {"name": "Verizon",           "sector": "Communication",    "industry": "Telecom"},
    "BA":    {"name": "Boeing",            "sector": "Industrials",      "industry": "Aerospace & Defense"},
    "CAT":   {"name": "Caterpillar",       "sector": "Industrials",      "industry": "Heavy Equipment"},
    "LMT":   {"name": "Lockheed Martin",   "sector": "Industrials",      "industry": "Aerospace & Defense"},
    "RTX":   {"name": "RTX",               "sector": "Industrials",      "industry": "Aerospace & Defense"},
    "DE":    {"name": "Deere & Company",   "sector": "Industrials",      "industry": "Heavy Equipment"},
    "UPS":   {"name": "UPS",               "sector": "Industrials",      "industry": "Logistics"},
    "HD":    {"name": "Home Depot",        "sector": "Consumer",         "industry": "Home Improvement Retail"},
    "GE":    {"name": "GE",                "sector": "Industrials",      "industry": "Conglomerate"},
}


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------
class SecQueryEngine:
    """
    Dynamic SEC filings query engine.

    Parses a manifest.json at startup to understand corpus coverage,
    then for each query:
      1. Plans which tickers / years / filing types are needed (one LLM call)
      2. Retrieves nodes from the vector store using metadata filters (no LLM)
      3. Synthesizes a final answer from all retrieved nodes (one LLM call)

    Parameters
    ----------
    index : VectorStoreIndex
        Your pre-built LlamaIndex vector store.
    manifest_path : str
        Path to manifest.json describing the corpus.
    top_k : int
        Number of chunks to retrieve per (ticker, year, filing_type) combination (default 6).
    verbose : bool
        Print query plan and retrieval summary (default True).
    debug : bool
        Attach LlamaDebugHandler for detailed tracing (default False).
    """

    def __init__(
        self,
        index:         VectorStoreIndex,
        manifest_path: str,
        top_k:         int  = 6,
        verbose:       bool = True,
        debug:         bool = False,
    ):
        self.index   = index
        self.top_k   = top_k
        self.verbose = verbose

        if debug:
            self._debug_handler = LlamaDebugHandler(print_trace_on_end=True)
            Settings.callback_manager = CallbackManager([self._debug_handler])
        else:
            self._debug_handler = None

        self._manifest      = self._load_manifest(manifest_path)
        self.all_tickers    = sorted(set(m["ticker"]      for m in self._manifest))
        self.all_years      = sorted(set(m["fiscal_year"] for m in self._manifest))
        self._coverage      = self._build_coverage()

        if self.verbose:
            print(
                f"[SecQueryEngine] Loaded manifest: "
                f"{len(self.all_tickers)} tickers, "
                f"years {min(self.all_years)}–{max(self.all_years)}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def query(self, query: str) -> str:
        """Run a natural-language query against the SEC corpus."""
        plan  = self._plan_query(query)                    # LLM call #1 (before retrieval)
        nodes = await self._retrieve_nodes(query, plan)    # pure vector retrieval

        if self.verbose:
            print(f"\n[Plan]\n{json.dumps(plan, indent=2)}")
            print(f"[Nodes retrieved] {len(nodes)}\n")

        return self._synthesize(query, nodes)              # LLM call #2 (after retrieval)

    def get_coverage(self, ticker: str) -> dict:
        """Return years and filing types available for a given ticker."""
        return self._coverage.get(ticker.upper(), {})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_manifest(self, manifest_path: str) -> list[dict]:
        with open(manifest_path) as f:
            files = json.load(f)["files"]
        manifest = []
        skipped  = 0
        for fname in files:
            try:
                info = parse_filing_filename(fname)
                manifest.append({
                    "ticker":         info.ticker,
                    "filing_type":    info.filing_type,
                    "fiscal_year":    info.fiscal_year,
                    "fiscal_quarter": info.fiscal_quarter,
                    "filename":       fname,
                })
            except ValueError:
                skipped += 1
        if skipped and self.verbose:
            print(f"[SecQueryEngine] Warning: skipped {skipped} unparseable filenames")
        return manifest

    def _build_coverage(self) -> dict:
        coverage = {}
        for m in self._manifest:
            t = m["ticker"]
            if t not in coverage:
                coverage[t] = {"years": set(), "types": set()}
            coverage[t]["years"].add(m["fiscal_year"])
            coverage[t]["types"].add(m["filing_type"])
        return coverage

    def _build_plan_prompt(self) -> PromptTemplate:
        """Build the planning PromptTemplate, grounded against current manifest coverage."""
        coverage_summary = {
            ticker: {
                "years": sorted(info["years"]),
                "types": sorted(info["types"]),
                **TICKER_METADATA.get(ticker, {}),
            }
            for ticker, info in self._coverage.items()
        }
        return PromptTemplate(
            f"""Today's date is {date.today().isoformat()}.
You are a query planner for a SEC EDGAR filings database.

# Available data (ONLY use tickers/years from this list):
{json.dumps(coverage_summary, indent=2)}

# Query
"{{query}}"

# Instructions
Think step by step before producing the plan:

1. TICKERS: Identify which companies are relevant.
   - Named directly ("Apple", "AAPL") → include that ticker
   - Sector/industry terms → map to all matching tickers in the available data
     e.g. "pharma" → any ticker where industry contains "Pharmaceutical"
     e.g. "big tech" → AAPL, MSFT, GOOG, META, AMZN, NVDA
     e.g. "banks" → JPM, BAC, GS, MS if present
   - "all companies" or no company specified → return all available tickers
   - Only include a ticker if it has data for the resolved years

2. YEARS: Identify the fiscal year(s) needed.
   - Explicit year ("2024", "fiscal 2023") → use that year
   - "last N years" / "past N years" → the N most recent years in the data
   - "most recent" / "latest" → single most recent year available per ticker
   - "since YYYY" → all years from YYYY to most recent
   - No year specified → all available years for the matched tickers

3. FILING TYPES:
   - "annual report", "10-K", "yearly" → ["10-K"]
   - "quarterly", "10-Q", "quarter" → ["10-Q"]
   - Specific quarter mentioned ("Q3 2024") → ["10-Q"]
   - Not specified → ["10-K", "10-Q"]
"""
        )

    def _clamp_plan(self, plan_obj: QueryPlan) -> dict:
        """Remove tickers/years that don't actually exist in the manifest."""
        tickers = [
            t for t in plan_obj.tickers
            if t in self._coverage
            and any(y in self._coverage[t]["years"] for y in plan_obj.years)
        ]
        years = [
            y for y in plan_obj.years
            if any(y in self._coverage[t]["years"] for t in tickers)
        ]
        return {
            "tickers": tickers,
            "years": years,
            "filing_types": plan_obj.filing_types,
            "reasoning": plan_obj.reasoning,
        }

    def _plan_query(self, query: str) -> dict:
        """
        LLM call #1 (before retrieval): resolve tickers, years, and filing types.
        Uses OpenAI structured output (JSON schema enforcement) — no regex or
        json.loads needed. Grounded against manifest coverage.
        """
        plan_obj: QueryPlan = Settings.llm.structured_predict(
            QueryPlan,
            self._build_plan_prompt(),
            query=query,
        )
        return self._clamp_plan(plan_obj)

    async def _retrieve_nodes(self, query: str, plan: dict) -> list[NodeWithScore]:
        """
        Pure vector retrieval — no LLM calls.

        Runs one filtered retriever per (ticker, filing_type, fiscal_year) combination
        concurrently, then deduplicates results by node_id.
        """
        async def _retrieve_one(ticker, filing_type, fiscal_year):
            filters = [
                MetadataFilter(key="ticker", value=ticker, operator=FilterOperator.EQ),
                MetadataFilter(key="filing_type", value=filing_type, operator=FilterOperator.EQ),
                MetadataFilter(key="fiscal_year", value=fiscal_year, operator=FilterOperator.EQ),
            ]
            retriever = self.index.as_retriever(
                similarity_top_k=self.top_k,
                filters=MetadataFilters(filters=filters),
            )
            return await retriever.aretrieve(query)

        tasks = []
        for ticker in plan["tickers"]:
            coverage = self._coverage[ticker]
            for year in plan["years"]:
                if year not in coverage["years"]:
                    continue
                for filing_type in plan["filing_types"]:
                    if filing_type not in coverage["types"]:
                        continue
                    tasks.append(_retrieve_one(ticker, filing_type, year))

        results = await asyncio.gather(*tasks)

        # Flatten and deduplicate by node_id
        seen, nodes = set(), []
        for batch in results:
            for n in batch:
                if n.node_id not in seen:
                    seen.add(n.node_id)
                    nodes.append(n)
        return nodes

    def _build_synthesis_prompt(self, query: str, nodes: list[NodeWithScore]) -> str:
        """Build the synthesis prompt shared by _synthesize and _synthesize_stream."""
        context_parts = []
        for i, n in enumerate(nodes, 1):
            m = n.node.metadata
            quarter = m.get("fiscal_quarter")
            period = f"FY{m.get('fiscal_year', '')} Q{quarter}" if quarter else f"FY{m.get('fiscal_year', '')}"
            header = (
                f"[{i}] {m.get('ticker', '')} {m.get('filing_type', '')} "
                f"{period} — {m.get('section_label', '')}"
            )
            context_parts.append(f"{header}\n{n.node.get_content()}")
        context = "\n\n---\n\n".join(context_parts)

        return f"""You are a financial analyst. Answer the question below using only the provided SEC filing excerpts.
Be specific, cite the company and fiscal year where relevant, and be concise.

Question: {query}

SEC Filing Excerpts:
{context}

Answer:"""

    def _synthesize(self, query: str, nodes: list[NodeWithScore]) -> str:
        """
        LLM call #2 (after retrieval): produce a final answer from retrieved nodes.
        This is the only generative call made after the retrieval pipeline.
        """
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

    async def aplan_query(self, query: str) -> dict:
        """Async version of _plan_query — uses astructured_predict (non-blocking)."""
        plan_obj: QueryPlan = await Settings.llm.astructured_predict(
            QueryPlan,
            self._build_plan_prompt(),
            query=query,
        )
        return self._clamp_plan(plan_obj)

    # ------------------------------------------------------------------
    # LlamaIndex-compatible interface
    # ------------------------------------------------------------------

    def as_query_engine(self) -> "BaseQueryEngine":
        """
        Return a standard LlamaIndex BaseQueryEngine wrapping this engine.

        The returned object can be used anywhere LlamaIndex expects a
        query engine: RouterQueryEngine, agent tools, QueryPipeline, etc.

        Example
        -------
        qe = sec_engine.as_query_engine()

        # sync
        response = qe.query("What did pharma companies say about pricing in 2024?")

        # async
        response = await qe.aquery("Compare AAPL and MSFT margins in 2024")
        """
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
        """
        Return a QueryEngineTool wrapping this engine — ready to hand to an
        agent, RouterQueryEngine, or SubQuestionQueryEngine as a single tool
        representing the entire SEC corpus.

        Example
        -------
        from llama_index.core.query_engine import RouterQueryEngine
        from llama_index.core.selectors import LLMSingleSelector

        router = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[
                sec_engine.as_tool(),
                other_tool,
            ],
        )
        """
        return QueryEngineTool(
            query_engine=self.as_query_engine(),
            metadata=ToolMetadata(name=name, description=description),
        )


# ---------------------------------------------------------------------------
# Thin BaseQueryEngine wrapper — delegates to SecQueryEngine
# ---------------------------------------------------------------------------

class _DynamicSecQueryEngine(BaseQueryEngine):
    """
    Internal LlamaIndex-compatible wrapper around SecQueryEngine.
    Instantiate via SecQueryEngine.as_query_engine(), not directly.
    """

    def __init__(self, sec_engine: "SecQueryEngine"):
        self._sec = sec_engine
        super().__init__(callback_manager=Settings.callback_manager or CallbackManager())

    # -- sync path --
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            return loop.run_until_complete(self._aquery(query_bundle))
        except RuntimeError:
            return asyncio.run(self._aquery(query_bundle))

    # -- async path (preferred) --
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        query_str = (
            query_bundle.query_str
            if isinstance(query_bundle, QueryBundle)
            else str(query_bundle)
        )
        plan  = self._sec._plan_query(query_str)
        nodes = await self._sec._retrieve_nodes(query_str, plan)

        if self._sec.verbose:
            print(f"\n[Plan]\n{json.dumps(plan, indent=2)}")
            print(f"[Nodes retrieved] {len(nodes)}\n")

        answer = self._sec._synthesize(query_str, nodes)
        return Response(response=answer)
