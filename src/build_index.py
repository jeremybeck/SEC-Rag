"""
build_index.py — Run this script to build the vectorstore from scratch and
store it in pgvector. Use this when adding new filings to the corpus.

For first-time setup with existing JSON embeddings, prefer:
    python src/migrate_to_pgvector.py   (no OpenAI calls needed)

Usage:
    python src/build_index.py

Requires:
    OPENAI_API_KEY  — used for generating embeddings
    DATABASE_URL    — pgvector connection string
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding

from filing_parser import build_all_documents, load_manifest
from index_loader import get_vector_store

# =========================================================
# CONFIGURATION
# =========================================================

CORPUS_DIR    = Path(__file__).parent.parent / "files" / "edgar_corpus"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 200


def main() -> None:
    """
    Build the pgvector index from the edgar_corpus filing archive.

    Reads all filings listed in manifest.json, chunks each document
    (CHUNK_SIZE tokens, CHUNK_OVERLAP overlap), calls the OpenAI embeddings
    API to generate vectors, and writes everything to the sec_embeddings table.
    Only needs to run once, or again when new filings are added to the corpus.
    """
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    print(f"Corpus dir: {CORPUS_DIR}")

    file_list = load_manifest(CORPUS_DIR)
    print(f"Files: {len(file_list)}")

    documents = build_all_documents(
        file_list,
        corpus_dir=CORPUS_DIR,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    print(f"Total documents: {len(documents)}")

    vector_store     = get_vector_store()
    storage_context  = StorageContext.from_defaults(vector_store=vector_store)
    index            = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    print("Index built and stored in pgvector.")


if __name__ == "__main__":
    main()
