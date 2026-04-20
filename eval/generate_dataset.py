"""
eval/generate_dataset.py — Build a golden Q&A evaluation dataset from SEC filing chunks.

Parses filing text locally (no pgvector, no embedding API calls), selects a
representative sample of chunks from key companies and sections, then calls
GPT-4o to generate 2 questions per chunk.

Output:
  eval/dataset.json       — EmbeddingQAFinetuneDataset (used by run_retrieval_eval.py)
  eval/dataset_meta.json  — per-question metadata (ticker, section, year, etc.)

Usage:
  export OPENAI_API_KEY=sk-...
  python eval/generate_dataset.py
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llama_index.core import Settings
from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.core.llms.llm import LLM
from llama_index.llms.openai import OpenAI

from filing_parser import build_all_documents, load_manifest, TICKER_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CORPUS_DIR = Path(__file__).parent.parent / "files" / "edgar_corpus"
OUT_DIR    = Path(__file__).parent

# Companies with best multi-year 10-K + 10-Q coverage
TARGET_TICKERS = {
    "AAPL", "AMZN", "GOOG", "MSFT", "NVDA", "TSLA",
    "JPM",  "JNJ",  "PFE",  "XOM",  "KO",   "UNH",
}

# Sections most useful for business Q&A
TARGET_SECTION_KEYWORDS = {"RISK FACTORS", "MD&A", "BUSINESS"}

# Chunks per (ticker × section_keyword × filing_type) bucket
CHUNKS_PER_BUCKET = 3

# Prefer 10-K over 10-Q for richer narrative content
FILING_TYPE_PRIORITY = {"10-K": 0, "10-Q": 1}

QUESTIONS_PER_CHUNK = 2
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Node selection
# ─────────────────────────────────────────────────────────────────────────────

def _section_keyword(section_label: str) -> str | None:
    """Return the matching target keyword, or None if not a target section."""
    upper = section_label.upper()
    for kw in TARGET_SECTION_KEYWORDS:
        if kw in upper:
            return kw
    return None


def select_nodes(all_docs):
    """
    Filter and sample documents to a balanced, representative set.

    Strategy:
      - Keep only target tickers and target sections
      - Bucket by (ticker, section_keyword, filing_type)
      - Within each bucket prefer 10-K, then sort by fiscal_year desc (newest first)
      - Take up to CHUNKS_PER_BUCKET per bucket
    """
    # Group into buckets
    buckets: dict[tuple, list] = defaultdict(list)
    for doc in all_docs:
        m = doc.metadata
        ticker = m.get("ticker", "")
        if ticker not in TARGET_TICKERS:
            continue
        kw = _section_keyword(m.get("section_label", ""))
        if kw is None:
            continue
        filing_type = m.get("filing_type", "")
        key = (ticker, kw, filing_type)
        buckets[key].append(doc)

    selected = []
    for (ticker, kw, filing_type), docs in sorted(buckets.items()):
        # Sort newest first, then by chunk_id so we get the opening chunk of each section
        docs.sort(
            key=lambda d: (
                -d.metadata.get("fiscal_year", 0),
                d.metadata.get("chunk_id", 0),
            )
        )
        selected.extend(docs[:CHUNKS_PER_BUCKET])

    random.seed(RANDOM_SEED)
    random.shuffle(selected)
    print(f"[generate] Selected {len(selected)} chunks from {len(buckets)} buckets")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Sidecar metadata
# ─────────────────────────────────────────────────────────────────────────────

def build_sidecar(dataset, node_meta: dict[str, dict]) -> dict:
    """
    Map each question_id → filing metadata, looked up via its relevant node_id.
    """
    sidecar = {}
    for q_id, node_ids in dataset.relevant_docs.items():
        node_id = node_ids[0] if node_ids else None
        meta = node_meta.get(node_id, {})
        sidecar[q_id] = {
            "query": dataset.queries[q_id],
            "node_id": node_id,
            "ticker": meta.get("ticker"),
            "filing_type": meta.get("filing_type"),
            "fiscal_year": meta.get("fiscal_year"),
            "fiscal_quarter": meta.get("fiscal_quarter"),
            "section_label": meta.get("section_label"),
            "chunk_id": meta.get("chunk_id"),
        }
    return sidecar


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    Settings.llm = OpenAI(model="gpt-4o")

    print(f"[generate] Loading manifest from {CORPUS_DIR}")
    file_list = load_manifest(CORPUS_DIR)
    print(f"[generate] {len(file_list)} files in manifest")

    print("[generate] Parsing filing chunks (no embedding API calls)...")
    all_docs = build_all_documents(file_list, CORPUS_DIR)
    print(f"[generate] {len(all_docs)} total chunks parsed")

    selected = select_nodes(all_docs)

    # Build node_id → metadata lookup (LlamaIndex assigns node_id from doc_id or hash)
    node_meta = {doc.doc_id: doc.metadata for doc in selected}

    print(f"[generate] Generating {QUESTIONS_PER_CHUNK} questions per chunk via GPT-4o...")
    print(f"           (~{len(selected) * QUESTIONS_PER_CHUNK} questions total, "
          f"~{len(selected) * QUESTIONS_PER_CHUNK * 2} API calls)")

    dataset = generate_question_context_pairs(
        nodes=selected,
        llm=Settings.llm,
        num_questions_per_chunk=QUESTIONS_PER_CHUNK,
    )

    dataset_path = OUT_DIR / "dataset.json"
    dataset.save_json(str(dataset_path))
    print(f"[generate] Dataset saved → {dataset_path}  ({len(dataset.queries)} questions)")

    sidecar = build_sidecar(dataset, node_meta)
    meta_path = OUT_DIR / "dataset_meta.json"
    with open(meta_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[generate] Sidecar saved  → {meta_path}")

    # Quick summary
    tickers_covered = {v["ticker"] for v in sidecar.values() if v["ticker"]}
    sections_covered = {v["section_label"] for v in sidecar.values() if v["section_label"]}
    print(f"\n[generate] Summary")
    print(f"  Questions : {len(dataset.queries)}")
    print(f"  Tickers   : {sorted(tickers_covered)}")
    print(f"  Sections  : {sorted(sections_covered)}")


if __name__ == "__main__":
    main()
