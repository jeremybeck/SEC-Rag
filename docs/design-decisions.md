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

### Year-based metadata pre-filtering

A lightweight heuristic extracts fiscal year signals from the query before the ANN search:

- **Explicit years** (`"2024"`, `"fiscal year 2025"`) → filter to those years
- **Relative spans** (`"last 3 years"`, `"past two years"`) → expand to the N most recent years
- **Recency signals** (`"recent"`, `"latest"`, `"current"`) → last 2 years

When a year signal is detected, a `fiscal_year` metadata filter is applied at the pgvector query level. This prevents older filings from displacing current-year chunks in the ANN results — a common failure mode when multiple filing years exist for the same company.

If no year signal is found, no filter is applied and retrieval remains fully open.

### The diversity problem

Pure semantic retrieval has a structural weakness for comparative questions. A query like "Compare Apple and Microsoft's cloud strategies" produces embeddings that are most similar to whichever company has more or better-matched chunks. In practice, Apple (or whichever company's text is closest) fills the top 40 of 50 slots, leaving Microsoft with 10 — or none.

This is the central retrieval challenge under the single-LLM-call constraint. With multiple calls, you would decompose the query ("Apple cloud strategy", "Microsoft cloud strategy") and retrieve separately. With one call, you have to build diversity into the retrieval pipeline itself.

### Cross-encoder reranking

After ANN retrieval, a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores the candidate pool. Cross-encoders jointly encode the query and each chunk together — more accurate than cosine similarity but too slow for full-index search. Running it on the post-ANN pool of ~200 candidates is fast enough for real-time use (local inference, ~22 MB model, no API call).

Scores are sigmoid-normalized to [0, 1] for interpretability.

### Ticker-aware MMR diversity selection

The final selection stage is a greedy Maximal Marginal Relevance (MMR) variant that enforces company diversity. At each step it scores candidates by:

```
score = λ × reranker_score − (1 − λ) × overrepresentation_penalty
```

The overrepresentation penalty ramps from 0 to 1 as a ticker's chunk count in the selected set approaches a soft cap (default: 5). Once a company has 5 chunks, additional chunks from that company are strongly penalized in favor of other companies.

With `λ = 0.5` and `soft_cap = 5`, the first five chunks from any company are chosen on pure relevance, then diversity kicks in. This produces a 50-chunk synthesis context that covers all companies mentioned in the query while still prioritizing the most relevant chunks for each.

The limitation is that the soft cap is a heuristic. A four-company comparison with complex sub-questions may still under-serve some companies. The principled solution — query decomposition followed by per-company retrieval — would require additional LLM calls.

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

### Confidence self-assessment and its limitations

The model is asked to rate its own answer confidence (HIGH/MEDIUM/LOW) and explain the rating. The intent is to surface cases where the context was insufficient or the wrong documents were retrieved.

This is the weakest part of the synthesis design. The model has a strong tendency to rate itself HIGH because it cannot cleanly distinguish between knowledge from the provided excerpts and knowledge from its training data. When it recalls a specific financial figure from training memory, it still produces a plausible-sounding citation and rates the answer HIGH.

Two programmatic overrides partially mitigate this:

1. **Number verification**: For each cited chunk, numbers appearing in the answer near that citation are checked against the full text of the cited node. If specific figures (dollar amounts, share counts, percentages) cannot be found in the source text, the rating is downgraded from HIGH to MEDIUM or LOW.

2. **Source mismatch detection**: If the query names a company (by ticker or full name) but all cited nodes come from a different company, the rating is forced to LOW.

These are heuristics, not a solution. A more reliable approach would be a separate verification pass — but that requires a second LLM call.

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

### Context window tradeoff

Sending 50 chunks to the model maximizes the chance that every company relevant to a comparative question is represented. The tradeoff is that the model must attend over a large, heterogeneous context — 50 chunks from potentially 8–10 different companies and filing years — in a single pass.

This is inherently harder than a focused single-company query. The prompt includes explicit instructions to only use the provided excerpts and to explicitly declare when information is missing, but enforcement is imperfect. The diversity controls in retrieval are what make 50-chunk synthesis viable; without them, the context would be dominated by a single company and comparative questions would fail silently.

---

## Summary of tradeoffs

| Design choice                     | Benefit | Cost |
|-----------------------------------|---|---|
| Chunk enrichment                  | No planning LLM call needed | Larger chunks, more tokens per embedding call |
| 800-token chunks                  | Coherent analytical context | Fewer chunks fit in synthesis window |
| Single LLM call                   | Low latency, simple architecture | Model must do retrieval quality assessment + answer generation simultaneously |
| Ticker-aware chunk selection      | Comparative questions work | May under-serve any one company vs. a focused query |
| Cross-encoder reranking           | Precision improvement | ~200ms local inference latency added |
