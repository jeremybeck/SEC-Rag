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

# Lazy-loaded DeBERTa NLI model for post-hoc faithfulness scoring
_nli_model = None

def _get_nli_model():
    """
    Lazy-load and cache the DeBERTa NLI cross-encoder.

    Downloads the model from HuggingFace on first call (~22 MB); subsequent
    calls return the cached instance. Thread-safe for read access after initial load.
    Label order: [contradiction=0, neutral=1, entailment=2].
    """
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _nli_model


class Citation(BaseModel):
    """A single citation: the chunk index and the verbatim quote from that chunk."""
    index: int
    quote: str


class DataQualityAssessment(BaseModel):
    """Post-hoc NLI faithfulness assessment of the synthesized answer."""
    rating:           Literal["HIGH", "MEDIUM", "LOW"]
    summary:          str
    missing_coverage: list[str]
    nli_score:        float | None = None


class SynthesisResponse(BaseModel):
    """Structured output schema for the synthesis LLM call."""
    answer:    str
    citations: list[Citation]


@dataclass
class SynthesisResult:
    """Return value from _synthesize — named fields instead of a positional tuple."""
    answer:         str
    cited_node_ids: list[str]
    cited_quotes:   list[str]
    data_quality:   DataQualityAssessment | None


@dataclass
class RetrievalResult:
    """Return value from _retrieve_nodes — nodes plus the metadata filters that were applied."""
    nodes:      list  # list[NodeWithScore]
    tickers:    list[str]
    years:      list[int]
    industries: list[str]


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


def build_metadata_filters(query: str) -> MetadataFilters | None:
    """
    Build LlamaIndex MetadataFilters from fiscal year and company signals in a query.

    Year filters  — fiscal_year (int), ORed together.
    Ticker filters — ticker (str), ORed together, via spaCy PhraseMatcher.

    When both are present, the two OR-groups are ANDed:
        (year=2024 OR year=2025) AND (ticker=AAPL OR ticker=MSFT)

    Returns None if neither signal is found (no pre-filtering applied).
    """
    from company_matcher import match_query

    years   = extract_year_filters(query)
    matches = match_query(query)
    tickers = matches["tickers"]

    if not years and not tickers:
        return None

    if years and tickers:
        # Nested OR groups ANDed together
        return MetadataFilters(
            filters=[
                MetadataFilters(
                    filters=[
                        MetadataFilter(key="fiscal_year", value=y, operator=FilterOperator.EQ)
                        for y in years
                    ],
                    condition=FilterCondition.OR,
                ),
                MetadataFilters(
                    filters=[
                        MetadataFilter(key="ticker", value=t, operator=FilterOperator.EQ)
                        for t in tickers
                    ],
                    condition=FilterCondition.OR,
                ),
            ],
            condition=FilterCondition.AND,
        )

    if years:
        return MetadataFilters(
            filters=[
                MetadataFilter(key="fiscal_year", value=y, operator=FilterOperator.EQ)
                for y in years
            ],
            condition=FilterCondition.OR,
        )

    # tickers only
    return MetadataFilters(
        filters=[
            MetadataFilter(key="ticker", value=t, operator=FilterOperator.EQ)
            for t in tickers
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
        Pre-built LlamaIndex vector store (from pgvector).
    top_k : int
        ANN candidates fetched from pgvector before reranking (default 2500).
    rerank_pool : int
        Cross-encoder rerank pool size (default 200). Pre-rerank step guarantees
        rerank_per_ticker nodes per company in this pool.
    rerank_per_ticker : int
        Minimum candidates per ticker in the rerank pool (default 15).
    max_synthesis_nodes : int
        Final chunks passed to the LLM, allocated evenly across tickers (default 50).
    verbose : bool
        Print per-stage retrieval stats (default True).
    debug : bool
        Attach LlamaDebugHandler for detailed tracing (default False).
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
        prompt_file:         str   = "synthesis_v6.txt",
    ):
        self.index                = index
        self.top_k                = top_k
        self.verbose              = verbose
        self.max_synthesis_nodes  = max_synthesis_nodes
        self._rerank_pool         = rerank_pool
        self._rerank_per_ticker   = rerank_per_ticker
        self._prompt_template     = (self._PROMPTS_DIR / prompt_file).read_text()

        if debug:
            self._debug_handler = LlamaDebugHandler(print_trace_on_end=True)
            Settings.callback_manager = CallbackManager([self._debug_handler])
        else:
            self._debug_handler = None

        # Cross-encoder reranker (~22 MB download on first use)
        # top_n = rerank_pool so the reranker scores the full candidate pool;
        # the balanced ticker selection (_mmr_diversity_select) does the final cut.
        self._reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=rerank_pool,
        )
        # Drop nodes below cosine similarity 0.72 before reranking
        self._sim_filter = SimilarityPostprocessor(similarity_cutoff=0.40)

        # Pre-load NLI model at startup so the first query doesn't pay the download cost
        _get_nli_model()
        if self.verbose:
            print("[SecQueryEngine] NLI faithfulness model loaded.")

        if self.verbose:
            print("[SecQueryEngine] Ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def query(self, query: str) -> str:
        """Run a natural-language query against the SEC corpus."""
        retrieval = await self._retrieve_nodes(query)
        return self._synthesize(query, retrieval.nodes)

    def parse_query_filters(self, query: str) -> dict:
        """
        Fast, synchronous extraction of tickers/years/industries from a query.
        No I/O — just spaCy phrase matching and regex. Used to emit filter metadata
        to the client before the slower retrieval pipeline begins.
        """
        from company_matcher import match_query as _cm
        matches = _cm(query)
        years   = extract_year_filters(query) or []
        return {
            "tickers":    matches["tickers"],
            "years":      years,
            "industries": matches["industries"],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _retrieve_nodes(self, query: str) -> "RetrievalResult":
        """
        Full retrieval pipeline — ANN search → dedup → sim filter → reranking → MMR.

        No LLM calls. Returns a RetrievalResult containing the final ranked nodes
        plus the tickers/years/industries extracted from the query (used to emit
        filter metadata to the client before synthesis begins).

        Pipeline stages:
          1. Extract company/year metadata filters via spaCy + regex
          2. pgvector ANN search (top_k candidates with optional metadata pre-filter)
          3. Deduplicate by node_id
          4. Drop nodes below similarity floor (unless that would leave nothing)
          5. Per-ticker quota → cross-encoder rerank pool
          6. Cross-encoder reranking + sigmoid score normalisation
          7. Ticker-aware MMR diversity selection → final max_synthesis_nodes
        """
        from company_matcher import match_query as _cm
        _matches = _cm(query)
        _years   = extract_year_filters(query)
        metadata_filters = build_metadata_filters(query)
        if self.verbose:
            print(
                f"[Retrieval] entity extraction — "
                f"years={_years or 'none'}  "
                f"tickers={_matches['tickers'] or 'none'}  "
                f"industries={_matches['industries'] or 'none'}  "
                f"filter={'applied' if metadata_filters else 'none'}"
            )
        retriever = self.index.as_retriever(
            similarity_top_k=self.top_k,
            filters=metadata_filters,
        )
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

        return RetrievalResult(
            nodes=nodes,
            tickers=_matches["tickers"],
            years=_years or [],
            industries=_matches["industries"],
        )

    def _mmr_diversity_select(self, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """
        Balanced ticker selection: guarantees a fair share of synthesis context
        for every company present in the candidate pool.

        Algorithm:
          1. Group nodes by ticker, sorted by reranker score (best first).
          2. Process tickers from smallest available count to largest, so
             under-represented companies claim their slot before the dominant one.
          3. Each ticker's quota = min(available, ceil(remaining_budget / n_tickers_left)).
             Slack from exhausted tickers flows to the remaining ones.
          4. Take the top-quota nodes (by reranker score) from each ticker.
          5. Sort the final set by reranker score so synthesis sees the most
             relevant chunks first regardless of company.

        Example with 3 tickers and max_synthesis_nodes=50:
          TSLA(2 available), AAPL(15), JPM(170)
          → TSLA gets 2, AAPL gets 15, JPM gets 33  → 50 total
          Instead of the previous 2 + 3 + 45 output.
        """
        from collections import Counter, defaultdict

        # Group by ticker, each list sorted best-first
        by_ticker: dict[str, list[NodeWithScore]] = defaultdict(list)
        for n in sorted(nodes, key=lambda x: float(x.score or 0.0), reverse=True):
            by_ticker[n.node.metadata.get("ticker", "")].append(n)

        # Process smallest-available tickers first so they get their fair share
        ordered = sorted(by_ticker.keys(), key=lambda t: len(by_ticker[t]))
        remaining_budget = self.max_synthesis_nodes
        quota: dict[str, int] = {}
        for i, ticker in enumerate(ordered):
            n_left = len(ordered) - i
            fair   = math.ceil(remaining_budget / n_left)
            quota[ticker] = min(len(by_ticker[ticker]), fair)
            remaining_budget -= quota[ticker]

        selected: list[NodeWithScore] = []
        for ticker in ordered:
            selected.extend(by_ticker[ticker][: quota[ticker]])

        # Re-sort by reranker score so LLM sees highest-confidence chunks first
        selected.sort(key=lambda x: float(x.score or 0.0), reverse=True)

        if self.verbose:
            dist = dict(Counter(n.node.metadata.get("ticker") for n in selected))
            print(f"[Balanced] {len(selected)} nodes — ticker distribution: {dist}")

        return selected

    def _build_synthesis_prompt(self, query: str, nodes: list[NodeWithScore]) -> str:
        """
        Assemble the synthesis prompt by numbering each retrieved chunk [1], [2], …
        and substituting into the loaded prompt template.

        The numbered format is what the LLM uses when writing inline citations,
        and what _score_faithfulness uses to map [N] markers back to source nodes.
        """
        context_parts = []
        for i, n in enumerate(nodes, 1):
            context_parts.append(f"[{i}]\n{n.node.get_content()}")
        context = "\n\n---\n\n".join(context_parts)
        return (
            self._prompt_template
            .replace("{query}", query)
            .replace("{context}", context)
        )

    def _score_faithfulness(
        self,
        answer: str,
        citations: list[Citation],
        nodes: list[NodeWithScore],
    ) -> DataQualityAssessment:
        """
        Post-hoc faithfulness scoring via DeBERTa NLI.

        Premise  = the verbatim quote the LLM provided for each citation.
                   Short, targeted, and the right length for NLI models trained
                   on SNLI/MNLI. Also verified to actually appear in the raw chunk.
        Hypothesis = the answer sentence containing that citation marker.

        For each cited sentence:
          - Finds all [N] markers and their associated citation quotes.
          - Runs NLI: (quote, sentence) for each citation.
          - Takes MAX entailment across all citations for that sentence.
            (A sentence passes if supported by any of its cited quotes.)

        Additionally checks quote grounding: if a citation quote is not found
        in the raw chunk text, that citation is flagged as ungrounded.

        Final score = mean of per-sentence max scores → HIGH / MEDIUM / LOW.
        Model: cross-encoder/nli-deberta-v3-small.
        Output logit order: [contradiction, entailment, neutral].
        """
        import math

        citation_pat = re.compile(r'\[(\d+)\]')
        quote_map: dict[int, str] = {c.index: c.quote for c in citations}
        valid_indices = set(quote_map)

        # Split answer into sentences
        raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', answer.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        # For each sentence with citations, collect (quote, sentence) pairs.
        # A sentence may cite multiple chunks — it passes if ANY quote entails it.
        sentence_groups: list[tuple[str, list[str]]] = []  # (sentence, [quotes])
        ungrounded_flags: list[str] = []

        for sentence in sentences:
            cited_indices = [
                int(m.group(1))
                for m in citation_pat.finditer(sentence)
                if int(m.group(1)) in valid_indices and 1 <= int(m.group(1)) <= len(nodes)
            ]
            if not cited_indices:
                continue

            quotes = []
            for idx in cited_indices:
                quote = quote_map.get(idx, "").strip()
                if not quote:
                    continue
                # Grounding check: quote should appear verbatim in the raw chunk.
                # If not (LLM paraphrased), fall back to the raw chunk text so NLI
                # has a premise that actually contains the supporting information.
                node_text = nodes[idx - 1].node.get_content()
                if quote.lower() not in node_text.lower():
                    flag = f"[{idx}] LLM quote not verbatim — using raw chunk as NLI premise"
                    if flag not in ungrounded_flags:
                        ungrounded_flags.append(flag)
                    quote = node_text[:600]
                quotes.append(quote)

            if quotes:
                sentence_groups.append((sentence, quotes))

        if not sentence_groups:
            return DataQualityAssessment(
                rating="LOW",
                summary="No citation markers found in answer — faithfulness could not be assessed.",
                missing_coverage=ungrounded_flags,
            )

        # Build flat batch of (quote, sentence) NLI pairs.
        # Strip citation markers [N] from the hypothesis — they are out-of-distribution
        # for MNLI/SNLI-trained NLI models and systematically reduce entailment confidence.
        batch_pairs: list[tuple[str, str]] = []
        group_slice: list[tuple[int, int]] = []

        for sentence, quotes in sentence_groups:
            start = len(batch_pairs)
            clean_hyp = re.sub(r'\[\d+\]', '', sentence).strip()
            for quote in quotes:
                batch_pairs.append((quote, clean_hyp))
            group_slice.append((start, len(batch_pairs)))

        model = _get_nli_model()
        logits = model.predict(batch_pairs, apply_softmax=False)  # shape (N, 3)

        def _entailment(row) -> float:
            # Label order for cross-encoder/nli-deberta-v3-small: [contradiction=0, neutral=1, entailment=2]
            exps = [math.exp(x) for x in row]
            return exps[2] / sum(exps)

        # Numeric containment fallback: handles tabular data like "Total Net Sales| $ | 416.161 |"
        # which the NLI model can't parse but clearly supports the claim.
        # Extract meaningful numbers (3+ digits or decimals) from sentence, check all appear in quote.
        _num_pat = re.compile(r'\d[\d,]*\.?\d*')

        _year_pat = re.compile(r'^(19|20)\d{2}$')

        def _numeric_score(sentence: str, quote: str) -> float:
            # Normalize: strip formatting chars so "416.161" matches "| 416.161 |"
            norm_quote = re.sub(r'[|$,%\s]', '', quote).lower()
            # Extract financial figures — skip years (20xx/19xx) and single-digit citation indices
            nums = [
                re.sub(r',', '', n)
                for n in _num_pat.findall(sentence)
                if len(re.sub(r'[,.]', '', n)) >= 3 and not _year_pat.match(n)
            ]
            if not nums:
                return 0.0
            norm_nums = [re.sub(r'[,$]', '', n) for n in nums]
            return 0.80 if all(n in norm_quote for n in norm_nums) else 0.0

        all_probs = [_entailment(row) for row in logits]

        if self.verbose and batch_pairs:
            for i, (pair, row, prob) in enumerate(zip(batch_pairs[:3], logits[:3], all_probs[:3])):
                print(f"[NLI debug] pair {i}: logits={[round(float(x), 3) for x in row]}  ent={prob:.3f}")
                print(f"  premise:    {pair[0][:100]!r}")
                print(f"  hypothesis: {pair[1][:100]!r}")

        # Per sentence: take MAX across all cited quotes, combining NLI and numeric fallback
        sentence_scores: list[float] = []
        low_confidence_sentences: list[str] = []

        for (sentence, quotes), (start, end) in zip(sentence_groups, group_slice):
            nli_best = max(all_probs[start:end])
            num_best = max((_numeric_score(sentence, q) for q in quotes), default=0.0)
            best = max(nli_best, num_best)
            sentence_scores.append(best)
            if best < 0.45:
                truncated = sentence[:200] + ("…" if len(sentence) > 200 else "")
                low_confidence_sentences.append(truncated)

        mean_score = sum(sentence_scores) / len(sentence_scores)
        n_sentences = len(sentence_scores)

        if mean_score >= 0.75:
            rating = "HIGH"
        elif mean_score >= 0.45:
            rating = "MEDIUM"
        else:
            rating = "LOW"

        if self.verbose:
            print(
                f"[NLI] faithfulness score: {mean_score:.3f} "
                f"({n_sentences} cited sentences, {len(batch_pairs)} NLI pairs) → {rating}"
                + (f"  ungrounded={len(ungrounded_flags)}" if ungrounded_flags else "")
            )

        sentence_word = "sentence" if n_sentences == 1 else "sentences"
        return DataQualityAssessment(
            rating=rating,
            summary=f"Faithfulness checked across {n_sentences} cited {sentence_word}.",
            missing_coverage=low_confidence_sentences + ungrounded_flags,
            nli_score=round(mean_score, 3),
        )

    def _synthesize(self, query: str, nodes: list[NodeWithScore]) -> SynthesisResult:
        """
        Single structured LLM call that produces the final answer with citations.

        Steps:
          1. Build prompt from numbered chunks + synthesis_v5.txt template
          2. structured_predict → SynthesisResponse (answer + citation list)
          3. Renumber [N] markers 1-based by first appearance in the answer text
          4. Post-hoc NLI faithfulness scoring via _score_faithfulness
          5. Remap [N] indices in missing_coverage strings to display numbering

        Falls back to plain text completion if structured_predict fails (e.g., JSON
        parse error), returning no citations and no data_quality assessment.
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

            # Post-hoc faithfulness scoring via DeBERTa NLI.
            # Uses the original (pre-remap) answer so citation indices match nodes directly.
            data_quality = self._score_faithfulness(result.answer, result.citations, nodes)

            # Remap [N] indices in missing_coverage strings to match the display-order numbering.
            if data_quality and remap:
                data_quality.missing_coverage = [
                    re.sub(r'\[(\d+)\]', lambda m: f"[{remap.get(int(m.group(1)), m.group(1))}]", s)
                    for s in data_quality.missing_coverage
                ]

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

    async def _synthesize_streaming(
        self, query: str, nodes: list[NodeWithScore]
    ) -> AsyncGenerator["str | SynthesisResult", None]:
        """
        Stream the synthesis answer character by character, then yield a final SynthesisResult.

        Yields:
            str           — individual answer characters as they arrive from the LLM stream
            SynthesisResult — one final object after the full response is complete,
                              containing remapped citations, cited nodes, and NLI quality

        Streaming approach:
          1. Call Settings.llm.astream_complete() with the raw synthesis prompt
          2. Run a state machine (SCANNING → IN_ANSWER → DONE) over the token stream:
             - SCANNING: accumulate into a sliding window until '"answer":' is detected,
               then wait for the opening '"'
             - IN_ANSWER: yield each character, handling JSON escape sequences (\\n, \\", etc.)
             - DONE: discard remaining tokens (still buffer for JSON parse)
          3. After streaming, json.loads() the full buffered response to extract citations
          4. Apply the same renumbering, NLI scoring, and remap logic as _synthesize()

        Falls back to a plain-text SynthesisResult if JSON parsing fails.
        """
        import json

        prompt = self._build_synthesis_prompt(query, nodes)

        SCANNING, IN_ANSWER, DONE = 0, 1, 2
        state        = SCANNING
        escaped      = False          # previous char was backslash
        buffer_parts: list[str] = []  # accumulate delta tokens for post-stream parse
        scan_window  = ""             # sliding window for trigger detection
        seen_trigger = False          # have we seen '"answer":'
        TRIGGER      = '"answer":'

        try:
            async for chunk in await Settings.llm.astream_complete(prompt):
                token = chunk.delta
                if not token:
                    continue
                buffer_parts.append(token)

                if state == DONE:
                    continue

                for ch in token:
                    if state == SCANNING:
                        if not seen_trigger:
                            scan_window += ch
                            # Keep only enough history to match the trigger
                            if len(scan_window) > len(TRIGGER) + 2:
                                scan_window = scan_window[-len(TRIGGER):]
                            if TRIGGER in scan_window:
                                seen_trigger = True
                                scan_window = ""
                        else:
                            # Consuming optional whitespace before the opening quote
                            if ch == '"':
                                state = IN_ANSWER
                            elif ch not in ' \t\n\r':
                                # Unexpected non-whitespace — reset and keep scanning
                                seen_trigger = False
                                scan_window = ch

                    elif state == IN_ANSWER:
                        if escaped:
                            _escape_map = {
                                'n': '\n', 't': '\t', 'r': '\r',
                                '"': '"',  '\\': '\\', '/': '/',
                                'b': '\b', 'f': '\f',
                            }
                            yield _escape_map.get(ch, '\\' + ch)
                            escaped = False
                        elif ch == '\\':
                            escaped = True
                        elif ch == '"':
                            state = DONE
                        else:
                            yield ch

        except Exception as e:
            print(f"[SecQueryEngine] WARNING: astream_complete failed ({e!r})")

        # ----------------------------------------------------------------
        # Post-stream: parse the buffered JSON and build SynthesisResult
        # ----------------------------------------------------------------
        full_response = "".join(buffer_parts)
        try:
            data = json.loads(full_response)
            raw_answer = data.get("answer", "")
            raw_citations = [
                Citation(index=c["index"], quote=c.get("quote", ""))
                for c in data.get("citations", [])
                if isinstance(c.get("index"), int)
            ]
            # Apply same renumbering as _synthesize
            quote_map: dict[int, str] = {c.index: c.quote for c in raw_citations}
            seen: list[int] = []
            for m in re.finditer(r'\[(\d+)\]', raw_answer):
                idx = int(m.group(1))
                if idx not in seen and 1 <= idx <= len(nodes):
                    seen.append(idx)
            for c in raw_citations:
                if c.index not in seen and 1 <= c.index <= len(nodes):
                    seen.append(c.index)

            remap = {orig: disp for disp, orig in enumerate(seen, 1)}
            answer = re.sub(
                r'\[(\d+)\]',
                lambda m: f"[{remap[int(m.group(1))]}]" if int(m.group(1)) in remap else m.group(0),
                raw_answer,
            )
            cited_node_ids = [nodes[orig - 1].node_id for orig in seen]
            cited_quotes   = [quote_map.get(orig, "") for orig in seen]

            syn_response = SynthesisResponse(answer=raw_answer, citations=raw_citations)
            data_quality = self._score_faithfulness(syn_response.answer, syn_response.citations, nodes)
            if data_quality and remap:
                data_quality.missing_coverage = [
                    re.sub(r'\[(\d+)\]', lambda m: f"[{remap.get(int(m.group(1)), m.group(1))}]", s)
                    for s in data_quality.missing_coverage
                ]

            yield SynthesisResult(
                answer=answer,
                cited_node_ids=cited_node_ids,
                cited_quotes=cited_quotes,
                data_quality=data_quality,
            )
        except Exception as e:
            print(f"[SecQueryEngine] WARNING: streaming JSON parse failed ({e!r})")
            yield SynthesisResult(
                answer=full_response.strip(),
                cited_node_ids=[],
                cited_quotes=[],
                data_quality=None,
            )

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
        """
        Synchronous LlamaIndex query entry point.

        Runs _aquery on an event loop, using nest_asyncio when called from within
        an already-running loop (e.g., Jupyter notebooks).
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            return loop.run_until_complete(self._aquery(query_bundle))
        except RuntimeError:
            return asyncio.run(self._aquery(query_bundle))

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Async LlamaIndex query entry point — delegates to SecQueryEngine."""
        query_str = (
            query_bundle.query_str
            if isinstance(query_bundle, QueryBundle)
            else str(query_bundle)
        )
        retrieval = await self._sec._retrieve_nodes(query_str)
        if self._sec.verbose:
            print(f"[Nodes retrieved] {len(retrieval.nodes)}")
        r = self._sec._synthesize(query_str, retrieval.nodes)
        return Response(response=r.answer)
