# SEC-Rag
SEC Filings RAG Application — query SEC 10-K/10-Q filings using LlamaIndex and pgvector.

## Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Node.js 18+
- An OpenAI API key

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the corpus

The filing text files are distributed as a zip archive. Extract it before building the index:

```bash
unzip files/edgar_corpus.zip -d files/
```

This will populate `files/edgar_corpus/` with the `.txt` filings and `manifest.json`.

### 3. Start the pgvector database

```bash
docker compose up -d
```

This starts a PostgreSQL 16 instance with the pgvector extension on port `5432` with:
- **Database:** `secrag`
- **User:** `postgres`
- **Password:** `postgres`

Data is persisted in the `pgdata` Docker volume, so it survives container restarts.

### 4. Set environment variables

```bash
export OPENAI_API_KEY=sk-...
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/secrag
```

### 5. (First run) Cross-encoder model download

The retrieval pipeline uses a local cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~22 MB) for reranking. It is downloaded automatically from HuggingFace on first use via `sentence-transformers` — no manual step required, but an internet connection is needed the first time the query engine runs.

### 6. Populate the vector index

Run the index builder to chunk the filings, generate embeddings, and store them in pgvector:

```bash
python src/build_index.py
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
uvicorn api.server:app --reload --port 8000
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
- A **Retrieved Nodes** sidebar on the right (grouped by ticker + filing)
- Source citation chips below the answer

### CLI (no frontend)

```bash
python main.py
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

## Design

See [Design Decisions](docs/design-decisions.md) for the reasoning behind key architectural choices in indexing, retrieval, and synthesis.

## Evaluation

See [Evaluation](docs/evaluation.md) for methodology, results, and how findings drove system changes.