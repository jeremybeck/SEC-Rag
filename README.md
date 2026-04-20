# SEC-Rag
SEC Filings RAG Application — query SEC 10-K/10-Q filings using LlamaIndex and pgvector.

![SEC-Rag screenshot](docs/assets/SEC_RAG.png)

## Architecture

```mermaid
flowchart TD
    Q([Query]) --> QP

    subgraph Retrieval["Retrieval — no LLM"]
        QP[Query Parser]
        ANN[pgvector ANN]
        RE[Cross-Encoder]
        MMR[Balanced Quota]
        QP -->|metadata filters| ANN
        ANN --> RE
        RE --> MMR
    end

    subgraph Synthesis["Synthesis — 1 LLM call"]
        LLM[GPT-4o]
        NLI[DeBERTa NLI]
        LLM --> NLI
    end

    MMR --> LLM
    NLI --> R([Answer + Citations + Quality])
```

## Design

See [Design Decisions](docs/design-decisions.md) for the reasoning behind key architectural choices in indexing, retrieval, and synthesis.

## Evaluation

See [Evaluation](docs/evaluation.md) for methodology, results, and how findings drove system changes.

## Prerequisites

- Docker and Docker Compose
- Python 3.14+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- Node.js 18+
- An OpenAI API key

## Setup

### 1. Install Python dependencies

```bash
uv sync
```

This installs all dependencies from `uv.lock` into a local `.venv`. All subsequent `python` commands should be run inside this environment — either prefix with `uv run` or activate it first:

```bash
source .venv/bin/activate
```

### 2. Download the spaCy language model

The query engine uses spaCy to extract company names and ticker symbols from queries. The model must be installed once after setup:

```bash
uv pip install en-core-web-sm
```

### 3. Prepare the corpus

The filing text files are distributed as a zip archive. Extract it before building the index:

```bash
unzip files/edgar_corpus.zip -d files/
```

This will populate `files/edgar_corpus/` with the `.txt` filings and `manifest.json`.

### 4. Start the pgvector database

```bash
docker compose up -d
```

This starts a PostgreSQL 16 instance with the pgvector extension on port `5432` with:
- **Database:** `secrag`
- **User:** `postgres`
- **Password:** `postgres`

Data is persisted in the `pgdata` Docker volume, so it survives container restarts.

### 5. Set environment variables

```bash
export OPENAI_API_KEY=sk-...
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/secrag
```

### 6. (First run) HuggingFace model downloads

Two local models are downloaded automatically from HuggingFace on first use — no manual step required, but an internet connection is needed:

- `cross-encoder/ms-marco-MiniLM-L-6-v2` (~22 MB) — cross-encoder reranker
- `cross-encoder/nli-deberta-v3-small` (~90 MB) — NLI faithfulness scoring

Both are cached after the first download.

### 7. Populate the vector index

Run the index builder to chunk the filings, generate embeddings, and store them in pgvector:

```bash
uv run python src/build_index.py
```

This will:
1. Read all filings listed in `files/edgar_corpus/manifest.json`
2. Chunk each document (800 tokens, 200 overlap)
3. Call the OpenAI embeddings API to generate vectors
4. Write all embeddings into the `sec_embeddings` table in pgvector

This only needs to be run once (or again when new filings are added to the corpus).

## Running the web app

### API server

```bash
uv run uvicorn api.server:app --reload --port 8000
```

The server starts on `http://localhost:8000`. On startup it loads the index from pgvector and initializes the query engine (takes a few seconds).

### Frontend (dev)

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The UI provides:
- A chat input at the bottom
- A streaming answer with markdown rendering
- A **Relevant Docs** sidebar on the right (grouped by ticker + filing)
- Source citation chips below the answer

### CLI (no frontend)

```bash
uv run python main.py
```

Edit `main.py` to change the query. The script loads the index from pgvector (no re-embedding) and runs it through the `SecQueryEngine`.

## Stopping the database

```bash
docker compose down
```

To also delete all stored data:

```bash
docker compose down -v
```
