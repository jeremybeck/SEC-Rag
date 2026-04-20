# Design Decisions

This document explains the reasoning behind key architectural choices in the SEC-Rag pipeline. The system is built around a hard constraint: **one LLM API call per user query**. Every decision in retrieval and synthesis flows from that requirement.

---

## Indexing

### Metadata generation

Each SEC filing is parsed from its raw EDGAR text file. The filename encodes the ticker, filing type (10-K or 10-Q), fiscal year, and fiscal quarter. These are extracted into structured metadata fields stored on every chunk:

| Field | Example |
|-------|---------|
| `ticker` | `AAPL` |
| `filing_type` | `10-K` |
| `fiscal_year` | `2024` |
| `fiscal_quarter` | `null` (annual) or `1`–`4` |
| `section_label` | `RISK FACTORS (ITEM 1A)` |

Section labels are normalized from the SEC's standard Item numbering (Item 1, Item 1A, Item 7, etc.) into human-readable labels. This makes them usable in both the UI (source chips) and metadata filters.

### Chunk enrichment — context in the embedding

Each chunk is prepended with a structured header before embedding:

```
Apple (AAPL) | 10-K Annual Report | FY2024 | RISK FACTORS (ITEM 1A)
<chunk text>
```

 Without it, a chunk about "revenue growth" is ambiguous — the embedding model doesn't know it belongs to Apple's FY2024 10-K. With the header, the embedding carries the company, period, and section context directly, so a query like "What were Apple's main risk factors in FY2024?" can find the right chunk by cosine similarity alone.


### Chunk size

Our first layer of chunking is structured parsing of the SEC filings by splitting at section headers. Then, each section is broken into chunks that are 800 tokens with 200-token overlap. This is larger than typical RAG deployments for two reasons:

1. **SEC filings read as analytical.** Risk factors, MD&A sections, and business descriptions develop arguments over multiple paragraphs. A smaller ~250-token chunk often cuts a point mid-thought. At 800 tokens, a single chunk typically covers a complete topic block (one risk factor, one revenue discussion segment).

2. **Single LLM call.** Because synthesis happens in one call, the model needs enough context per chunk to actually answer the question. Tiny chunks shift more inference burden onto the LLM and require more chunks to cover the same ground.

The 200-token overlap ensures that topic boundaries between adjacent chunks are not hard cuts — a sentence at the end of one chunk also appears at the start of the next.

---

## Retrieval

The retrieval pipeline runs **entirely without LLM calls**. Its job is to deliver a set of chunks to synthesis that is both relevant and diverse enough to support multi-company comparative questions. 

### Semantic retrieval via pgvector

The primary retrieval mechanism is approximate nearest-neighbor (ANN) search over the pgvector index using cosine similarity. A broad candidate pool (top-2500 by default) is retrieved first, then progressively narrowed by later stages.

Relying just on semantic similarity is not necessarily the ideal, but aligns with the single LLM call constraint, which precludes efficient decomposition and metadata generation.  To support semantic retrieval across companies, the company and period context is in the embedding itself.

### Metadata pre-filtering

Two lightweight heuristics extract structured signals from the query and apply them as pgvector metadata filters before the ANN search runs. This prevents unrelated filings from consuming candidate slots before the reranker and MMR stages can act.

**Year filtering** (`extract_year_filters` in `sec_query.py`):
- **Explicit years** (`"2024"`, `"fiscal year 2025"`) → filter to those exact years
- **Relative spans** (`"last 3 years"`, `"past two years"`) → expand to the N most recent years
- **Recency signals** (`"recent"`, `"latest"`, `"current"`) → last 2 years

When a year signal is detected, a `fiscal_year` metadata filter is applied at the pgvector query level. Synthesis evaluation of year-agnostic queries (Stratum D) showed the highest faithfulness degradation — wrong-year chunks consistently entered the context when no filter was applied. This motivated the year filter.

**Ticker filtering** (`company_matcher.py`, spaCy PhraseMatcher):
When a query names specific companies, a `ticker` metadata filter restricts the candidate pool to only those companies' filings. The matcher resolves both ticker symbols (`AAPL`) and company names (`Apple`, `Apple Inc.`, `Apple Computer`) to their canonical tickers, using spaCy's PhraseMatcher for fast substring detection.

When both year and ticker signals are present, the two OR-groups are ANDed: `(year=2024 OR year=2025) AND (ticker=AAPL OR ticker=TSLA)`.

If neither signal is found, no filter is applied and retrieval remains fully open.

### The diversity problem

Pure semantic retrieval has a structural weakness for comparative questions. A query like "Compare Apple and Microsoft's cloud strategies" produces embeddings that are most similar to whichever company has more or better-matched chunks. In practice, Apple (or whichever company's text is closest) fills the top 40 of 50 slots, leaving Microsoft with 10 — or none.

This is the central retrieval challenge under the single-LLM-call constraint. With multiple calls, you would decompose the query ("Apple cloud strategy", "Microsoft cloud strategy") and retrieve separately. With one call, you have to build diversity into the retrieval pipeline itself.

### Cross-encoder reranking

After ANN retrieval, a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores the candidate pool. Cross-encoders jointly encode the query and each chunk together — more accurate than cosine similarity but too slow for full-index search. Running it on the post-ANN pool of ~200 candidates is fast enough for real-time use (local inference, ~22 MB model, no API call).

Scores are sigmoid-normalized to [0, 1] for interpretability.

### Balanced quota diversity selection

The final selection stage enforces company diversity by assigning each represented ticker a guaranteed minimum chunk allocation before selecting on relevance.

The initial approach used a soft-penalty MMR variant that ramped down scores for over-represented companies. This failed in practice because the reranker's `top_n` pool was being cut to the final synthesis count *before* diversity selection ran. For a 3-company query where one company's filings dominated the corpus, the reranker eliminated the other companies' candidates entirely — leaving the diversity stage nothing to balance. Observed distribution: `{'JPM': 45, 'AAPL': 3, 'TSLA': 2}` for a 50-chunk synthesis window.

**Current approach — hard balanced quota**:

1. The reranker operates on a larger pool (`rerank_pool=200`) rather than the final synthesis count, preserving candidates from all companies through to the selection stage.

2. Tickers are sorted ascending by candidate count (fewest candidates first), then each ticker receives a fair share of the remaining budget:

```python
for i, ticker in enumerate(ordered_by_count):
    n_left = len(ordered) - i
    fair   = ceil(remaining_budget / n_left)
    quota[ticker] = min(len(candidates[ticker]), fair)
    remaining_budget -= quota[ticker]
```

Within each ticker's quota, chunks are selected in descending reranker score order.

This guarantees roughly equal representation for all companies named in the query regardless of how many chunks each company has in the reranker pool. A company with fewer high-scoring chunks is protected from being crowded out by a company with many.

The tradeoff is that a focused single-company query now receives a lower average chunk relevance than it would under pure relevance sorting — some slots go to lower-scoring chunks from a single company. In practice this is acceptable: single-company queries still saturate their quota with high-scoring material.

---

## Synthesis

### Single LLM call with structured output

The entire answer generation — reading 50 chunks, forming an answer, citing sources, and assessing confidence — happens in one `gpt-4o` call using OpenAI's function-calling API (via LlamaIndex `structured_predict`). This is enforced by the project constraint, and it places significant responsibility on a single call.

The output schema (enforced via Pydantic):

```json
{
  "answer": "prose with inline [N] citation markers",
  "citations": [
    {"index": N, "quote": "verbatim sentence from excerpt N"}
  ],
  "data_quality": {
    "rating": "HIGH | MEDIUM | LOW",
    "summary": "...",
    "missing_coverage": ["..."]
  }
}
```

### Citation design

Each citation carries the original chunk index and a verbatim quote — the specific sentence or phrase that supports the claim. This serves two purposes: it pins the cited source to a specific passage (not just "this document was used"), and it enables the UI to show the quoted text on hover over source chips.

Citation numbers in the answer are renumbered by first appearance, so the user sees `[1], [2], [3]...` regardless of which position in the 50-chunk context each source held.

### NLI faithfulness scoring (post-hoc)

After synthesis, a `cross-encoder/nli-deberta-v3-small` model scores each cited claim in the answer against its source excerpt independently of the LLM. This replaces the earlier approach of asking the LLM to self-rate its own confidence — which was systematically optimistic because the model cannot cleanly distinguish retrieved context from parametric memory.

**Scoring process** (`_score_faithfulness` in `sec_query.py`):

1. Sentence-split the answer. For each sentence containing `[N]` citation markers, find the corresponding quoted excerpts from the `citations` list.
2. **Grounding check**: if the LLM's verbatim quote cannot be found in the raw chunk text, fall back to the first 600 characters of the raw chunk as the NLI premise. LLMs frequently paraphrase rather than copy verbatim, so this fallback ensures the NLI model always has a valid premise.
3. Strip `[N]` markers from the hypothesis before building NLI pairs — they are out-of-distribution tokens for MNLI/SNLI-trained models and systematically reduce entailment confidence.
4. Run batch NLI inference. Label order for this model: `[contradiction=0, neutral=1, entailment=2]`. Entailment probability = `softmax(logits)[2]`.
5. **Numeric containment fallback**: for sentences with specific financial figures (dollar amounts, percentages, share counts), check whether those numbers appear in the source text even if NLI scores low. Tabular data like `"416.161"` in a pipe-delimited table scores near-zero on NLI semantically but is clearly grounded.
6. Per-sentence score = `max(NLI entailment, numeric containment)`. Mean over all cited sentences.
7. Map mean score to rating: HIGH (≥ 0.75), MEDIUM (≥ 0.45), LOW (< 0.45).

The `DataQualityAssessment` returned by `_score_faithfulness` carries `rating`, `summary`, `missing_coverage` (sentences scoring below 0.45), and `nli_score` (raw mean float).

**Why not LLM self-assessment?**

Synthesis evaluation showed ~30% overconfidence when the system self-rated: the model recalled financial figures from training memory, cited a related but vague excerpt, and rated itself HIGH. The judge found the specific number absent from the cited text. Post-hoc NLI is blind to the LLM's generation process — it only measures whether the cited source text actually entails the stated claim.

The NLI approach is not a complete solution: it measures sentence-level entailment of cited claims only, not correctness of uncited claims or whether the right document was retrieved in the first place.

### Prompt iterations (`src/prompts/`)

The synthesis prompt evolved across three versions as the pipeline matured and evaluation revealed specific failure modes.

**`synthesis_v1.txt` — baseline**
A minimal prompt: identify yourself as a financial analyst, use only the provided excerpts, cite sources inline, and be concise. The goal was to get the end-to-end pipeline (retrieval → LLM → frontend) working and validate that structured citations reached the UI correctly. No output format was enforced.

**`synthesis_v2.txt` — structured output + quality assessment**
Once the pipeline was stable, the prompt was extended with:
- A strict JSON output schema (answer, citations with per-citation verbatim quotes, data_quality block)
- Explicit guardrails (cite inline, prioritize correct filing type, flag year mismatches, do not fabricate figures)
- A `data_quality` self-assessment asking the model to rate context coverage as HIGH/MEDIUM/LOW and list specific gaps

The quality assessment was framed around *retrieval coverage* — did the right documents show up? This made the field useful for evaluating the retrieval pipeline but did not directly measure answer trustworthiness.

**`synthesis_v3.txt` — answer confidence + pre-answer check**
Synthesis evaluation revealed that the model was rating nearly every answer HIGH regardless of actual faithfulness. The judge LLM (used in `eval/run_synthesis_eval.py`) frequently found faithfulness scores of 0.0–0.4 for answers the model had self-rated HIGH. Two structural changes addressed this:

1. **Pre-answer check**: A new section inserted before answer generation explicitly instructs the model to verify that any specific number or financial figure it intends to cite is *visibly present* in the excerpt text — not recalled from training memory. It names the failure pattern ("confabulation") and instructs the model to write "the exact [metric] is not stated in the provided excerpts" rather than state a figure it cannot locate.

2. **Answer confidence** (replacing context coverage): The `data_quality` field was reframed from "how good was the retrieval?" to "how much should the user trust this answer?" The HIGH criteria was tightened to require that every specific figure was *read from the excerpt text*, with an explicit instruction to downgrade to MEDIUM when uncertain. The `missing_coverage` field was repurposed to list caveats that reduce answer confidence rather than retrieval gaps.

The re-framing reduced overconfidence in qualitative cases. For numerical questions, programmatic post-synthesis overrides (number verification and source mismatch detection, applied in `sec_query.py`) provide an additional check independent of the prompt.

**`synthesis_v4.txt` — structured reasoning chain**
Sending 50 heterogeneous chunks — potentially from 8–10 companies across multiple filing years — without explicit organization guidance left the model to implicitly decide how to process the context. For single-company queries this was fine, but for multi-company and comparative questions the model would blend or skip companies inconsistently.

A `### Structured Reasoning` section was added before `### Pre-Answer Check`. It instructs the model to work through four explicit steps before drafting the answer (this reasoning is preparation only — it does not appear in the JSON output):

1. **Organize by company**: group excerpts mentally by ticker, note what topics and fiscal periods each company's chunks cover. This builds an inventory before writing begins.
2. **Identify relevant companies**: filter to only the companies the question asks about; ignore excerpts from unrelated companies.
3. **Match filing type to question type**: annual questions → prioritize 10-K; quarterly questions → prioritize 10-Q. If a mismatch is unavoidable, flag it in `data_quality.missing_coverage`.
4. **Enumerate before comparing**: for comparison questions, list key facts per company side-by-side and identify meaningful differences before writing the summary. This ensures comparisons are grounded in the retrieved text rather than constructed from training memory.

**`synthesis_v5.txt` — enforced structured output**
Migrating synthesis from free-text to `structured_predict` (OpenAI function-calling API via LlamaIndex) required removing the `data_quality` self-assessment block from the output schema. The LLM self-confidence assessment was replaced by the post-hoc NLI scoring described above, freeing the JSON schema to focus purely on answer quality:

```json
{
  "answer": "prose with inline [N] citation markers",
  "citations": [{"index": N, "quote": "verbatim phrase from excerpt N"}]
}
```

The prompt was updated to match: the output format instructions describe only `answer` and `citations`, with `citations.quote` required to be a short verbatim extract (1–3 sentences) copied exactly from the source — not paraphrased. A `### Pre-Answer Check` section reinforces this with an explicit confabulation warning.

**`synthesis_v6.txt` — mandatory multi-company coverage + comparison**
Even with balanced retrieval delivering equal chunks per company, the LLM would write a thorough section for the company mentioned first and produce thin or absent coverage for the others. The model was "satisficing" — once it had written a complete-feeling answer for one company, it under-invested in the rest.

Two structural changes in v6 addressed this:

1. **Hard coverage requirement in the preamble**: "An answer that covers only a subset of the named companies is incomplete and unacceptable, even if the coverage of the ones you did address is excellent." This moves the requirement out of the reasoning steps (which the model can skip) into the instruction framing.

2. **Mandatory coverage audit (Step 6)**: After drafting the answer mentally, the model must check each named company off a list before writing JSON. If a company has no relevant excerpts, the answer must explicitly state this rather than silently omitting it.

3. **Comparison paragraph (Step 5)**: When the question contains comparison language ("compare", "contrast", "how do they differ", "versus", "similarities and differences"), the model is required to draft a dedicated `**Comparison**` paragraph after the per-company paragraphs. The paragraph must synthesize only facts already cited above — no new claims — and explicitly name the key similarities and differences. The Step 6 audit includes a checkbox for this paragraph.

### Streaming synthesis

`structured_predict` blocks until the LLM has generated the full JSON response, which means the user sees nothing for 5–15 seconds. To improve perceived latency, synthesis was converted to an SSE streaming architecture:

1. The server emits `filters` and `nodes` SSE events before the LLM call begins, so the retrieved documents sidebar populates immediately.
2. `_synthesize_streaming` calls `Settings.llm.astream_complete(prompt)` and runs a character-level state machine over the token stream:
   - **SCANNING**: accumulates a sliding window looking for `"answer":` in the raw JSON stream
   - **IN_ANSWER**: yields each character (handling JSON escape sequences: `\n`, `\"`, `\\`, etc.) until the closing unescaped `"`
   - **DONE**: discards remaining tokens while still buffering them
3. Each yielded character is wrapped as a `{"type": "token", "data": "<char>"}` SSE event. The frontend appends tokens to the answer as they arrive.
4. After streaming completes, the buffered full response is `json.loads()`-parsed to extract citations. The same renumbering and NLI scoring logic as the non-streaming path runs at this point.
5. A final `{"type": "answer", "data": "<remapped_answer>"}` SSE event replaces the streamed content with the citation-renumbered version (original chunk indices like `[15]` replaced with display-order `[1]`, `[2]`, `[3]`).

The tradeoff: the user sees the raw chunk indices during streaming, then sees a brief correction when the remapped answer arrives. This is preferable to waiting for the full response.

NLI scoring still happens after streaming on the blocking path (it cannot be streamed), so the quality assessment and source chips appear after a short delay following the end of the answer stream.

### Context window tradeoff

Sending 50 chunks to the model maximizes the chance that every company relevant to a comparative question is represented. The tradeoff is that the model must attend over a large, heterogeneous context — 50 chunks from potentially 8–10 different companies and filing years — in a single pass.

This is inherently harder than a focused single-company query. The prompt includes explicit instructions to only use the provided excerpts and to explicitly declare when information is missing, but enforcement is imperfect. The diversity controls in retrieval are what make 50-chunk synthesis viable; without them, the context would be dominated by a single company and comparative questions would fail silently.

---

## Summary of tradeoffs

| Design choice                     | Benefit | Cost |
|-----------------------------------|---|---|
| Chunk enrichment                  | No planning LLM call needed | Larger chunks, more tokens per embedding call |
| 800-token chunks                  | Coherent analytical context | Fewer chunks fit in synthesis window |
| Single LLM call                   | Low latency, simple architecture | Model must handle answer generation + citation in one pass |
| Metadata pre-filtering (year + ticker) | Eliminates wrong-period and wrong-company chunks before ANN search | Requires heuristic query parsing; false positives over-restrict retrieval |
| Cross-encoder reranking           | Precision improvement over cosine similarity | ~200ms local inference latency added |
| Balanced quota MMR                | Guaranteed per-company coverage for comparative queries | Single-company queries receive lower average chunk relevance |
| NLI post-hoc faithfulness         | Objective, reproducible faithfulness signal independent of LLM | Sentence-level only; doesn't catch uncited fabrications or wrong-document retrieval |
| Streaming synthesis               | Answer appears immediately; nodes sidebar updates before LLM call | Citation indices shown during streaming are raw; corrected after stream ends |
