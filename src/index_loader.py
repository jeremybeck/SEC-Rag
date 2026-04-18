"""
index_loader.py — Connects to the pgvector-backed LlamaIndex vectorstore.

Call load_index() at app startup to get a ready-to-use VectorStoreIndex.
Requires the DATABASE_URL environment variable to be set.

If the vectorstore hasn't been populated yet, run:
    python src/migrate_to_pgvector.py   # migrate existing JSON embeddings (no re-embedding)
    -- or --
    python src/build_index.py           # rebuild from scratch (calls OpenAI embeddings API)
"""

import os
from urllib.parse import urlparse

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore

EMBED_DIM  = 1536
TABLE_NAME = "sec_embeddings"


def get_vector_store() -> PGVectorStore:
    """
    Build a PGVectorStore from the DATABASE_URL environment variable.

    Works with any standard PostgreSQL connection string, including those
    provided by Render's managed Postgres service.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set.\n"
            "Local: export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/secrag\n"
            "Render: set it in the environment variables dashboard."
        )

    parsed = urlparse(url)
    return PGVectorStore.from_params(
        host=parsed.hostname,
        port=str(parsed.port or 5432),
        database=parsed.path.lstrip("/"),
        user=parsed.username,
        password=parsed.password,
        table_name=TABLE_NAME,
        embed_dim=EMBED_DIM,
    )


def load_index() -> VectorStoreIndex:
    """
    Connect to the pgvector table and return a ready-to-query VectorStoreIndex.

    No file I/O and no embedding API calls — just a DB connection.
    """
    vector_store = get_vector_store()
    return VectorStoreIndex.from_vector_store(vector_store)
