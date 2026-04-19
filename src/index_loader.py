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
    # Build both sync and async connection strings explicitly so LlamaIndex
    # uses proper connection pools for both paths.
    db     = parsed.path.lstrip("/")
    host   = parsed.hostname
    port   = parsed.port or 5432
    user   = parsed.username
    passwd = parsed.password

    sync_url  = f"postgresql+psycopg2://{user}:{passwd}@{host}:{port}/{db}"
    async_url = f"postgresql+asyncpg://{user}:{passwd}@{host}:{port}/{db}"

    return PGVectorStore.from_params(
        connection_string=sync_url,
        async_connection_string=async_url,
        table_name=TABLE_NAME,
        embed_dim=EMBED_DIM,
        create_engine_kwargs={
            "pool_pre_ping": True,   # validate connections before use
            "pool_recycle": 300,     # recycle connections every 5 min
            "pool_size": 10,
            "max_overflow": 20,
        },
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )


def load_index() -> VectorStoreIndex:
    """
    Connect to the pgvector table and return a ready-to-query VectorStoreIndex.

    No file I/O and no embedding API calls — just a DB connection.
    """
    vector_store = get_vector_store()
    return VectorStoreIndex.from_vector_store(vector_store)
