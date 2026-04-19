"""
eval/run_eval.py — Evaluate SEC RAG retrieval quality against the golden dataset.

Matching strategy: metadata-based (ticker + filing_type + fiscal_year + section_label),
NOT node ID comparison. Node IDs differ between dataset generation and pgvector because
LlamaIndex assigns new UUIDs at each parse. Metadata is stable.

Metrics:
  HitRate@50   — % of queries where the golden section appears anywhere in top-50
  MRR          — mean reciprocal rank of the first golden-section chunk
  Multi-co Coverage — % of compound queries where BOTH companies appear in results

Requires:
  eval/dataset_meta.json  (from generate_dataset.py)
  DATABASE_URL + OPENAI_API_KEY

Usage:
  export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/secrag
  export OPENAI_API_KEY=sk-...
  python eval/run_eval.py
"""

import asyncio
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from index_loader import load_index
from sec_query import SecQueryEngine
from filing_parser import TICKER_NAMES

EVAL_DIR = Path(__file__).parent
OUT_CSV  = EVAL_DIR / "results.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Metadata-based hit detection
# ─────────────────────────────────────────────────────────────────────────────

def _is_golden_match(node_meta: dict, golden: dict) -> bool:
    """
    A retrieved node is a 'hit' if it came from the same company, filing type,
    fiscal year, and section as the chunk that generated the question.
    We intentionally don't require chunk_id to match — any chunk from the
    right section counts as a valid retrieval.
    """
    return (
        node_meta.get("ticker")       == golden["ticker"]
        and node_meta.get("filing_type")  == golden["filing_type"]
        and node_meta.get("fiscal_year")  == golden["fiscal_year"]
        and node_meta.get("section_label") == golden["section_label"]
    )


async def eval_one(engine: SecQueryEngine, query: str, golden: dict) -> dict:
    """Retrieve nodes for one query and return hit/rank info."""
    nodes = await engine._retrieve_nodes(query)
    rank = None
    for i, n in enumerate(nodes, 1):
        if _is_golden_match(n.node.metadata, golden):
            rank = i
            break
    return {
        "hit":  1.0 if rank is not None else 0.0,
        "mrr":  1.0 / rank if rank is not None else 0.0,
        "rank": rank,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-company evaluation
# ─────────────────────────────────────────────────────────────────────────────

async def run_single_company_eval(engine: SecQueryEngine, meta: dict) -> dict:
    ticker_hits: dict[str, list[float]] = defaultdict(list)
    ticker_mrrs: dict[str, list[float]] = defaultdict(list)
    all_hits, all_mrrs = [], []

    total = len(meta)
    print(f"[eval] Evaluating {total} single-company questions...")

    for idx, (q_id, info) in enumerate(meta.items(), 1):
        if idx % 20 == 0 or idx == total:
            print(f"  {idx}/{total}...")
        result = await eval_one(engine, info["query"], info)
        ticker = info.get("ticker", "UNKNOWN")
        all_hits.append(result["hit"])
        all_mrrs.append(result["mrr"])
        ticker_hits[ticker].append(result["hit"])
        ticker_mrrs[ticker].append(result["mrr"])

    overall_hit = _mean(all_hits)
    overall_mrr = _mean(all_mrrs)
    print(f"\n[eval] Overall — HitRate@50: {overall_hit:.3f}  MRR: {overall_mrr:.3f}")

    per_ticker = {}
    for ticker in sorted(ticker_hits):
        per_ticker[ticker] = {
            "hit_rate": _mean(ticker_hits[ticker]),
            "mrr":      _mean(ticker_mrrs[ticker]),
            "n":        len(ticker_hits[ticker]),
        }
        print(f"  {ticker:<6}  HitRate: {per_ticker[ticker]['hit_rate']:.3f}  "
              f"MRR: {per_ticker[ticker]['mrr']:.3f}  (n={per_ticker[ticker]['n']})")

    return {"overall_hit": overall_hit, "overall_mrr": overall_mrr, "per_ticker": per_ticker}


# ─────────────────────────────────────────────────────────────────────────────
# Multi-company evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _section_keyword(section_label: str) -> str | None:
    for kw in ("RISK FACTORS", "MD&A", "BUSINESS"):
        if kw in (section_label or "").upper():
            return kw
    return None


def build_compound_queries(meta: dict, max_pairs: int = 60) -> list[dict]:
    """
    Pair questions from different companies on the same section + year + filing type.
    Creates natural "compare X vs Y" compound queries.
    """
    groups: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for q_id, info in meta.items():
        if not info.get("ticker") or not info.get("fiscal_year"):
            continue
        kw = _section_keyword(info.get("section_label", ""))
        if kw is None:
            continue
        key = (kw, info["fiscal_year"], info.get("filing_type", "10-K"))
        groups[key][info["ticker"]].append(q_id)

    compound = []
    for (section, year, ftype), by_ticker in groups.items():
        tickers = [t for t, qs in by_ticker.items() if qs]
        for t1, t2 in combinations(tickers, 2):
            name1 = TICKER_NAMES.get(t1, t1)
            name2 = TICKER_NAMES.get(t2, t2)
            compound.append({
                "query": (
                    f"Compare the {section.lower()} disclosed by {name1} and {name2} "
                    f"in their {ftype} filings for fiscal year {year}."
                ),
                "ticker1": t1,
                "ticker2": t2,
                "section": section,
                "fiscal_year": year,
            })

    import random
    random.seed(42)
    random.shuffle(compound)
    return compound[:max_pairs]


async def run_multi_company_eval(engine: SecQueryEngine, compound_queries: list[dict]) -> float:
    print(f"\n[eval] Running {len(compound_queries)} multi-company compound queries...")
    covered = 0
    for item in compound_queries:
        nodes = await engine._retrieve_nodes(item["query"])
        retrieved_tickers = {n.node.metadata.get("ticker") for n in nodes}
        if item["ticker1"] in retrieved_tickers and item["ticker2"] in retrieved_tickers:
            covered += 1

    rate = covered / len(compound_queries) if compound_queries else 0.0
    print(f"[eval] Multi-company coverage: {covered}/{len(compound_queries)} = {rate:.3f}")
    return rate


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def print_table(per_ticker: dict, overall_hit: float, overall_mrr: float, multi_coverage: float):
    tickers = sorted(per_ticker.keys())
    col_w = 7
    header = f"{'Metric':<22} {'All':>{col_w}}" + "".join(f" {t:>{col_w}}" for t in tickers)
    sep    = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    print(f"{'HitRate@50':<22} {overall_hit:>{col_w}.3f}" +
          "".join(f" {per_ticker[t]['hit_rate']:>{col_w}.3f}" for t in tickers))
    print(f"{'MRR':<22} {overall_mrr:>{col_w}.3f}" +
          "".join(f" {per_ticker[t]['mrr']:>{col_w}.3f}" for t in tickers))
    print(f"{'Multi-co Coverage':<22} {multi_coverage:>{col_w}.3f}" +
          "".join(f" {'—':>{col_w}}" for _ in tickers))
    print(sep)


def write_csv(per_ticker: dict, overall_hit: float, overall_mrr: float, multi_coverage: float):
    tickers = sorted(per_ticker.keys())
    fieldnames = ["metric", "all"] + tickers
    rows = [
        {"metric": "HitRate@50",        "all": f"{overall_hit:.3f}",    **{t: f"{per_ticker[t]['hit_rate']:.3f}" for t in tickers}},
        {"metric": "MRR",               "all": f"{overall_mrr:.3f}",    **{t: f"{per_ticker[t]['mrr']:.3f}"      for t in tickers}},
        {"metric": "Multi-co Coverage", "all": f"{multi_coverage:.3f}", **{t: "—" for t in tickers}},
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] Results written → {OUT_CSV}")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    meta_path = EVAL_DIR / "dataset_meta.json"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found. Run generate_dataset.py first.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)
    print(f"[eval] {len(meta)} questions loaded from {meta_path}")

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm         = OpenAI(model="gpt-4o")

    print("[eval] Loading index from pgvector...")
    index  = load_index()
    engine = SecQueryEngine(index=index, verbose=False)
    print("[eval] Engine ready.\n")

    single = await run_single_company_eval(engine, meta)
    compound_queries = build_compound_queries(meta)
    multi_coverage   = await run_multi_company_eval(engine, compound_queries)

    print_table(single["per_ticker"], single["overall_hit"], single["overall_mrr"], multi_coverage)
    write_csv(single["per_ticker"], single["overall_hit"], single["overall_mrr"], multi_coverage)


if __name__ == "__main__":
    asyncio.run(main())
